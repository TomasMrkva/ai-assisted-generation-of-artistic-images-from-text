import sys
from datetime import datetime
import tempfile
import argparse
from typing import Any

import torch
import torchvision.transforms as transforms

import clip

from generator import Generator
from cog import BasePredictor, Path, Input

class Predictor(BasePredictor):
    def setup(self):
        torch.cuda.empty_cache()
        self.device = torch.device('cpu')
        self.model, _ = clip.load('ViT-B/32', self.device, jit=False)

    def predict(self,  
                prompt: str = Input(description="prompt for generating image", default="Watercolor painting of an underwater submarine"),
                num_lines: int = Input(description="number of lines", default=10),
                num_iterations: int = Input(description="number of iterations", default=100),
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
            device = 'cpu',
            loss_img_path="{}/{}-{}/loss.png".format(folder, prompt, time),
            best_loss_img_path="{}/{}-{}/best_loss.png".format(folder, prompt, time),
            config_path = "/{}/{}-{}".format(folder, prompt, time),
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
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        min_loss, clip_similarity = Generator(args, self.device, self.model, out_file=str(out_path)).generate()
        sys.stderr.write(f"Min_loss: {min_loss}, clip_similarity: {clip_similarity}\n")
        return Path(out_path)
