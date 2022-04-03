#@markdown #**Generator Model**
import random
import torch
import clip
import sys
import torchvision.transforms as transforms
# from celluloid import Camera
import matplotlib.pyplot as plt
import numpy as np
from utilities import save_img, NOUNS

# from dsketch.utils.pyxdrawing import draw_points_lines_crs
from dsketch.experiments.imageopt.imageopt import save_pdf, make_init_params, clamp_params, clamp_colour_params, render

class Generator:
    def __init__(self, args, device, model):
      random.seed(args.seed)
      self.args = args
      self.device = device
      self.model = model
      self.text_features = None

      self.nouns = NOUNS.split(' ')
      noun_prompts = ["a drawing of a " + x for x in self.nouns]

      # Calculate features
      with torch.no_grad():
        self.nouns_features = self.model.encode_text(torch.cat([clip.tokenize(noun_prompts).to(self.device)]))
      
      print("here")
      self.augment = transforms.Compose([
        # transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
        # transforms.RandomPerspective(fill=1, p=0.4, distortion_scale=0.5),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), value=1),
        # transforms.ColorJitter(hue=0.1, saturation=0.1),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
      ])

      r = torch.linspace(-args.grid_row_extent, args.grid_row_extent, 224)
      c = torch.linspace(-args.grid_col_extent, args.grid_col_extent, 224)
      grid = torch.meshgrid(r, c)
      self.grid = torch.stack(grid, dim=2).to(args.device)
      self.coordpairs = torch.stack([torch.arange(0, args.crs_points + 2 - 3, 1),
                              torch.arange(1, args.crs_points + 2 - 2, 1),
                              torch.arange(2, args.crs_points + 2 - 1, 1),
                              torch.arange(3, args.crs_points + 2, 1)], dim=1)
      print("here")
      # scale the sigmas to match the grid defined above, rather than being relative to 1 pixel
      self.args.sf = (2 / 224) ** 2
      self.args.init_sigma2 = args.init_sigma2 * self.args.sf
      self.args.final_sigma2 = args.final_sigma2 * self.args.sf
      self.args.sigma2_current = args.init_sigma2

    def make_init_cparams(self, args):
      if args.colour:
        args.channels = 3
        # cparams = torch.rand((args.points + args.lines + args.crs, 3), device=args.device)
        cparams = torch.full((args.points + args.lines + args.crs, 3), 0.5, device=args.device)
      else:
        args.channels = 1
        cparams = None
      return cparams

    def setup_params(self):
      sigma2params = None
      if self.args.opt_sigma2:
          sigma2params = torch.ones(self.args.points + self.args.lines + self.args.crs, device=self.args.device) * self.args.init_sigma2
          self.args.sigma2_current = sigma2params
      params = make_init_params(self.args)
      cparams = self.make_init_cparams(self.args)
      return sigma2params, params, cparams

    def optimize(self, params, cparams, sigma2params, render_fn):
      losses = []
      min_loss = 10
      predicts = ""
      saved_image = None
      args = self.args
      text_input = clip.tokenize(self.args.prompt).to(self.device)
      text_input_neg1 = clip.tokenize(self.args.neg_prompt).to(self.device)
      text_input_neg2 = clip.tokenize(self.args.neg_prompt_2).to(self.device)

    #   fig = plt.figure(figsize=(15, 7))
    #   camera = Camera(fig)
    #   sp2 = fig.add_subplot(1, 2, 1)
    #   sp2.set_title("Estimate")
    #   sp1 = fig.add_subplot(1, 2, 2)
    #   sp1.set_title("Losses")

      with torch.no_grad():
          text_features = self.model.encode_text(text_input)
          self.text_features = text_features
          text_features_neg1 = self.model.encode_text(text_input_neg1)
          text_features_neg2 = self.model.encode_text(text_input_neg2)

      params.requires_grad = True
      if cparams is not None:
          cparams.requires_grad = True
      if sigma2params is not None:
          sigma2params.requires_grad = True
          sigma2 = sigma2params
      else:
          sigma2 = args.init_sigma2

      # optim = make_optimiser(args, params, cparams, sigma2params)
      params_optim = torch.optim.Adam([params], lr=args.lr)
      sigma2_optim = torch.optim.Adam([sigma2], lr=args.sigma2_lr)
      color_optim = torch.optim.Adam([cparams], lr=args.colour_lr)
    #   itr = args.iters
      # scheduler = torch.optim.lr_scheduler.MultiStepLR(params_optim, verbose=True, milestones=[50,100,150], gamma=0.4)

      #main optimisation loop
      for i in range(args.iters):
          lss, NUM_AUGS, img_augs = 0, 4, []
          params_optim.zero_grad()
          sigma2_optim.zero_grad()
          color_optim.zero_grad()

          est = render_fn(params, cparams, sigma2) # NCHW
          for n in range(NUM_AUGS):
            img_augs.append(self.augment(est))
          im_batch = torch.cat(img_augs)
          image_features = self.model.encode_image(im_batch)
          for n in range(NUM_AUGS):
            lss -= torch.cosine_similarity(text_features, image_features[n:n+1])
            if self.args.use_negative:
              lss += torch.cosine_similarity(text_features_neg1, image_features[n:n+1]) * 0.3
              lss += torch.cosine_similarity(text_features_neg2, image_features[n:n+1]) * 0.3

          if i >= args.last_iters * args.iters and lss.item() < min_loss:
            min_loss = lss.item()
            saved_image = est
            with torch.no_grad():
                im = transforms.ToPILImage()(ras.detach().cpu().squeeze(0)).convert("RGB")
                im_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                noun_norm = self.nouns_features /self. nouns_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * im_norm @ noun_norm.T).softmax(dim=-1)
                values, indices = similarity[0].topk(5)
                predicts = ""
                for value, index in zip(values, indices):
                    predicts += f"{self.nouns[index]:>16s}: {100 * value.item():.2f}%"
                    predicts += "\n"
        
          losses.append(lss.item())
          lss.backward()
          params_optim.step()
          sigma2_optim.step()
          color_optim.step()
          # scheduler.step()
          params = clamp_params(params, args)
          if cparams is not None:
              clamp_colour_params(cparams)

          if sigma2params is not None:
              mask = sigma2params.data < 1e-6

              if args.lines > 0 and args.restarts:
                lparams = params[2 * args.points: 2 * args.points + 4 * args.lines].view(args.lines, 2, 2).data
                for j in range(len(mask)):
                  if mask[j] and i < args.iters / 2:
                    torch.random.seed()
                    # lparams[j] = torch.rand_like(lparams[j]) - 0.5
                    lparams[j] = torch.rand((1, 2, 2))
                    lparams[j, 0, 0] -= 0.5
                    lparams[j, 0, 1] -= 0.5
                    lparams[j, 0, 0] *= 2 * args.grid_row_extent
                    lparams[j, 0, 1] *= 2 * args.grid_col_extent
                    lparams[j, 1, 0] = lparams[j, 0, 0] + 0.2 * (lparams[j, 1, 0] - 0.5)
                    lparams[j, 1, 1] = lparams[j, 0, 1] + 0.2 * (lparams[j, 1, 1] - 0.5)

                    sigma2params.data[j] += args.init_sigma2

              if i < args.iters / 2 and args.restarts:
                  sigma2params.data.clamp_(1e-6, args.init_sigma2)
              else:
                  sigma2params.data.clamp_(1e-6, args.init_sigma2)

          if sigma2params is None:
              if i % args.sigma2_step == 0:
                  sigma2 = sigma2 * args.sigma2_factor
                  if sigma2 < args.final_sigma2:
                      sigma2 = args.final_sigma2

              args.sigma2_current = sigma2
        #     #   itr.set_postfix({'loss': lss.item(), 'sigma^2': sigma2})
        #   else:
        #     #   itr.set_postfix({'loss': lss.item(), 'sigma^2': 'learned'})

          if i % args.snapshots_steps == 0:
            ras = render_fn(params, cparams, sigma2)

            # sp2.imshow(ras.squeeze(0).permute(1, 2, 0).detach().cpu())
            # camera.snap()
            if args.snapshots_path is not None:
              sys.stderr.write(f"iteration: {i}, render:loss: {lss.item()}\n")
              save_img(ras.detach().cpu().squeeze(0), args.snapshots_path + "/snapshot_" + str(i) + ".png")
              save_pdf(params, cparams, args, args.snapshots_path + "/snapshot_" + str(i) + ".pdf")

    #   sp1.plot(np.linspace(1, len(losses), len(losses)), losses)
    #   plt.xlabel("iterations")
    #   plt.ylabel("- sum of cosine similarity of 4 augmented images")
    #   anim = camera.animate(repeat=False)
      anim = None
    #   plt.close()
      return min_loss, saved_image

    def r(self, p, cp, s):
      return 1-render(p, cp, s, self.grid, self.coordpairs, self.args)

    def get_clip_similarity(self, image, text_features):
        normalize = transforms.Compose([transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        norm = normalize(image)
        with torch.no_grad():
            image_features = self.model.encode_image(norm)
        return torch.cosine_similarity(image_features, text_features).item()

    def generate(self, initial_params=None, initial_cparams=None, initial_sigma2params=None):
      sigma2params, params, cparams = self.setup_params()
      if initial_params is not None:
        params = torch.cat((initial_params, params))
        # cparams = torch.cat((initial_cparams, cparams))
        sigma2params = torch.cat((sigma2params, sigma2params))
        # sigma2params = torch.cat(sigma2params, initial_sigma2params)
      print("There")
      if self.args.init_raster is not None:
        ras = self.r(params, cparams, sigma2params)
        save_img(ras.detach().cpu().squeeze(0), self.args.final_raster)
      print("There")
      if self.args.init_pdf is not None:
        save_pdf(params, cparams, self.args, self.args.init_pdf)

      min_loss, saved_image = self.optimize(params, cparams, sigma2params, self.r)

      if self.args.final_raster is not None:
        ras = self.r(params, cparams, self.args.sigma2_current)
        save_img(ras.detach().cpu().squeeze(0), self.args.final_raster)

      if self.args.final_pdf is not None:
        save_pdf(params, cparams, self.args, self.args.final_pdf)
      
    #   if self.args.final_animation is not None:
    #     save_animation(anim, self.args.final_animation)

      if self.args.best_loss_img_path is not None:
        save_img(saved_image.detach().cpu().squeeze(0), self.args.best_loss_img_path)

    #   if self.args.loss_img_path is not None:
    #     plt.plot(np.linspace(1, len(losses), len(losses)), losses)
    #     plt.xlabel("iterations")
    #     plt.ylabel("- sum of cosine similarity of 4 augmented images")
    #     plt.savefig(self.args.loss_img_path)
    #     plt.close()

      with torch.no_grad():
        clip_similarity = self.get_clip_similarity(saved_image, self.text_features)

      return min_loss, clip_similarity