import random
from IPython.core.pylabtools import figsize
import torch
import clip
import torchvision.transforms as transforms
from utils import save_img, nouns, make_video
from stqdm import stqdm
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from dsketch.experiments.imageopt.imageopt import make_init_params, clamp_params, clamp_colour_params, render

class Generator:
    def __init__(self, model, args):
      with st.spinner('Loading Generator...'):
        random.seed(args.seed)
        torch.cuda.empty_cache()
        self.device = torch.device('cuda')
        self.model = model
        self.args = args
        self.text_features = None

        self.nouns = nouns.split(' ')
        noun_prompts = ["a drawing of a " + x for x in self.nouns]

        # Calculate features
        with torch.no_grad():
          self.nouns_features = self.model.encode_text(torch.cat([clip.tokenize(noun_prompts).to(self.device)]))
        
        self.augment = transforms.Compose([
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
      img = st.empty()
      # graph = st.empty()
      predictions = st.empty()
      header = st.empty()
      images = []
      losses = []
      min_loss = 10
      best_iter = 0
      predicts = ""
      saved_image = None
      args = self.args
      text_input = clip.tokenize(self.args.prompt).to(self.device)
      text_input_neg1 = clip.tokenize(self.args.neg_prompt).to(self.device)
      text_input_neg2 = clip.tokenize(self.args.neg_prompt_2).to(self.device)

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

      params_optim = torch.optim.Adam([params], lr=args.lr)
      sigma2_optim = torch.optim.Adam([sigma2], lr=args.sigma2_lr)
      color_optim = torch.optim.Adam([cparams], lr=args.colour_lr)
    
      #main optimisation loop
      for i in stqdm(range(args.iters), desc=args.prompt + " ", backend=True):
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
            best_iter = i
            with torch.no_grad():
                im = transforms.ToPILImage()(ras.detach().cpu().squeeze(0)).convert("RGB")
                im_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                noun_norm = self.nouns_features /self. nouns_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * im_norm @ noun_norm.T).softmax(dim=-1)
                values, indices = similarity[0].topk(5)
                predicts = ""
                for value, index in zip(values, indices):
                  predicts += f"{self.nouns[index]:}: {100 * value.item():.2f}%"
                  predicts += "\n"
        
          losses.append(lss.item())
          lss.backward()
          params_optim.step()
          sigma2_optim.step()
          color_optim.step()
          params = clamp_params(params, args)
          if cparams is not None:
              clamp_colour_params(cparams)

          if sigma2params is not None:
            sigma2params.data.clamp_(1e-6, args.init_sigma2)

          if sigma2params is None:
              if i % args.sigma2_step == 0:
                  sigma2 = sigma2 * args.sigma2_factor
                  if sigma2 < args.final_sigma2:
                      sigma2 = args.final_sigma2

              args.sigma2_current = sigma2
          if i % 10 == 0:
            ras = render_fn(params, cparams, sigma2)
            images.append(ras.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())

          if i % args.snapshots_steps == 0:
            ras = render_fn(params, cparams, sigma2)
            with torch.no_grad():
              # im = transforms.ToPILImage()(ras.detach().cpu().squeeze(0)).convert("RGB")
              im_norm = image_features / image_features.norm(dim=-1, keepdim=True)
              noun_norm = self.nouns_features /self. nouns_features.norm(dim=-1, keepdim=True)
              similarity = (100.0 * im_norm @ noun_norm.T).softmax(dim=-1)
              values, indices = similarity[0].topk(5)
              predicts = ""
              for value, index in zip(values, indices):
                  predicts += f"{self.nouns[index]}: {100 * value.item():.2f}%"
                  predicts += "\n"    
            col1, col2, col3 = st.columns(3)
            with col1:
              header.empty()
              header = st.subheader('Predictions')
              predictions.empty()
              predictions = st.text(predicts)
            with col2:
              img.empty()
              img = st.image(ras.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), clamp=True, width=224, caption=f"iteration: {i}, loss: {lss.item()}\n")
      plt.close()
      with col1:
        header.empty()
        header = st.subheader('Predictions')
        predictions.empty()
        predictions = st.text(predicts)
      with col2:
        img.empty()
        img = st.image(saved_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), clamp=True, width=224, caption=f"iteration: {best_iter}, loss: {min_loss}\n")
      with col3:
        fig, ax = plt.subplots(figsize=(5, 5.4))
        ax.plot(np.linspace(1, len(losses), len(losses)), losses)
        plt.xlabel("iterations")
        plt.ylabel("- sum of cosine similarity of 4 augmented images")
        st.pyplot(fig)
      with col2:
        # video = st.empty()
        # video.empty()
        make_video(images,'out.mp4')
        video_file = open('out.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)        
      return saved_image

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

      saved_image = self.optimize(params, cparams, sigma2params, self.r)

      # if self.args.final_raster is not None:
      #   ras = self.r(params, cparams, self.args.sigma2_current)
      #   save_img(ras.detach().cpu().squeeze(0), self.out_file)

      with torch.no_grad():
        clip_similarity = self.get_clip_similarity(saved_image, self.text_features)

      return clip_similarity