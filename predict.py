import sys
from datetime import datetime
import tempfile
import argparse
from typing import Any

import torch
import torchvision.transforms as transforms

import clip
from dsketch.utils.pyxdrawing import draw_points_lines_crs
from dsketch.experiments.imageopt.imageopt import save_image, save_pdf, save_vector, make_init_params, clamp_params, exp, clamp_colour_params, render_lines, render_points, render_crs, render

from generator import Generator
from cog import BasePredictor, Path, Input

class Predictor(BasePredictor):
    def setup(self):
        # Load the model
        torch.cuda.empty_cache()
        # self.device = torch.device('cuda')
        self.device = torch.device('cpu')
        self.model, _ = clip.load('ViT-B/32', self.device, jit=False)
        # Image Augmentation Transformation, use_normalized_clip = True
        self.augment_trans = transforms.Compose([
            transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
            transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    # @cog.input("prompt", type=str, default=,
    #            help="prompt for generating image")
    # @cog.input("num_lines", type=int, default=850, help="number of lines")
    # @cog.input("num_iterations", type=int, default=1000, help="number of iterations")
    # @cog.input("snapshots", type=int, default=10, help="display frequency of intermediate images")
    # @cog.input("sigma2", type=float, default=15, help="starting with of the lines")
    def predict(self,  
                prompt: str = Input(description="prompt for generating image", default="Watercolor painting of an underwater submarine"),
                num_lines: int = Input(description="number of lines", default=850),
                num_iterations: int = Input(description="number of iterations", default=1000),
                snapshots: int = Input(description="display frequency of intermediate images", default=10),
                sigma2: float = Input(description="starting with of the lines", default=15.0)
    )->Any:
        assert isinstance(num_lines, int) and num_lines > 0, 'num_paths should be an positive integer'
        assert isinstance(num_iterations, int) and num_iterations > 0, 'num_iterations should be an positive integer'
        folder = "snaps"
        snapshots = 100
        time = datetime.today().strftime('%m_%d_%H_%M_%S')
        args = argparse.Namespace(
            use_negative = False,
            prompt = prompt,
            neg_prompt = "A badly drawn sketch.",
            neg_prompt_2 = "Text, letters.",
            seed = 1234,
            lines_additional = num_lines,
            iters = num_iterations,
            lines = num_lines,
            crs = 0,
            crs_points =2,
            points = 0,
            colour = True,
            lr = 0.002,     #0.004
            colour_lr = 0.003,
            opt_sigma2 = True,
            final_sigma2 = 0.55 ** 2,
            init_sigma2 = sigma2,
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
        output_path = Path(tempfile.mkdtemp(args.config_path))
        min_loss, clip_similarity = Generator(args, self.device, self.model).generate()
        sys.stderr.write(f"Min_loss: {min_loss}, clip_similarity: {clip_similarity}\n")
        return Path(output_path)
