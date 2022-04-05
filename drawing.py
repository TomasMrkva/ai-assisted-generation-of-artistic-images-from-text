import torch
import clip
import argparse
import torch
from datetime import datetime
from generator import Generator
import streamlit as st

def get_args(time, prompt, lines, iters):
  folder = "snaps"
  snapshots = 100
  return argparse.Namespace(
        use_negative = False,
        prompt = prompt,
        neg_prompt = "A badly drawn sketch.",
        neg_prompt_2 = "Text, letters.",
        seed = 1234,
        lines_additional = lines,
        iters = iters,
        lines = lines,
        crs = 0,
        crs_points =2,
        points = 0,
        colour = True,
        lr = 0.002,     #0.004
        colour_lr = 0.003,
        opt_sigma2 = True,
        final_sigma2 = 0.55 ** 2,
        init_sigma2 = 15.0,
        sigma2_lr = 0.001,           #  0.00001
        snapshots_steps = snapshots,
        predict_steps = snapshots*1,
        device = 'cuda:0',
        loss_img_path="{}/{}-{}/loss.png".format(folder, prompt, time),
        best_loss_img_path="{}/{}-{}/best_loss.png".format(folder, prompt, time),
        config_path = "{}/{}-{}".format(folder, prompt, time),
        final_pdf = "{}/{}-{}/{}-final-{}.pdf".format(folder, prompt, time, prompt, time),
        final_animation = "{}/{}-{}/{}-final-{}.mp4".format(folder, prompt, time, prompt, time),
        final_raster = "{}/{}-{}/{}-final-{}.png".format(folder, prompt, time, prompt, time),
        init_raster = "{}/{}-{}/{}-init-{}.png".format(folder, prompt, time, prompt, time),
        init_pdf = "{}/{}-{}/{}-init-{}.pdf".format(folder, prompt, time, prompt, time),
        snapshots_path = "{}/{}-{}".format(folder, prompt, time),
        invert_sketch = True,
        restarts = False,

        sigma2_factor = 0.5, # doesnt matter when opt_sigma2=True
        sigma2_step = 100,  # doesnt matter when opt_sigma2=True
        channels = 3,
        target_shape = torch.Size([1, 3, 224, 224]),
        width = 224,
        sigma2_current = None,
        grid_row_extent = 1,
        grid_col_extent = 1,
        sf = None,
        last_iters = 0.1,
  )

def predict(model, prompt, update, lines=850, iters=1000, initial_sketching=False, row=None):
  params = None
  cparams = None

  print("Generating image for a prompt: " + prompt + " | Parameters updated: " + str(update))
  time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

  args = get_args(time, prompt, lines, iters)
  d = vars(args)

  for sub in d:
    if sub in update:
      if sub == 'lines_additional' and params is not None:
        d['lines'] = int(update[sub] + (params.size()[0]/4))
      elif sub == 'lines_additional':
        raise Exception('Additional lines can only be specified with a reference image saved as tensors of params.')
      d[sub] = update[sub]
    # with st.spinner('Starting up...'):
    clip_similarity = Generator(model, args).generate(initial_params=params, initial_cparams=cparams)
    torch.cuda.empty_cache()
    return clip_similarity


# if __name__ == '__main__':
#   predict("Watercolour painting of a cat", None, 'out.png')
# HTML(anim.to_jshtml())