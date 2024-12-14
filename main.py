from PIL import Image, ImageFilter
import numpy as np
import os
import torch
from diffusers import DDIMScheduler, StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
from torchvision.io import read_image, ImageReadMode
import torch.nn.functional as F
import cv2
from torchvision.utils import save_image
from torchvision.transforms.functional import gaussian_blur
from pytorch_lightning import seed_everything
from matplotlib import pyplot as plt
from utils import load_image, load_mask, make_redder
from AAS.AAS import AAS_XL
from AAS.AAS_utils import regiter_attention_editor_diffusers
from torchvision.transforms.functional import to_pil_image, to_tensor


class AttentiveEraser:
    def __init__(self, model_path="/root/autodl-tmp/input/stable-diffusion-xl-base-1.0", custom_pipeline="./pipelines/SDXL_inp_pipeline.py"):
        self.dtype = torch.float16
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        self.model_path = model_path
        self.custom_pipeline = custom_pipeline
        self.base = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.model_path,
            custom_pipeline=self.custom_pipeline,
            scheduler=self.scheduler,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.base.enable_attention_slicing()
        self.base.enable_model_cpu_offload()

    def _setup_output_directory(self, sample):
        out_dir = f"./workdir_xl/{sample}/"
        os.makedirs(out_dir, exist_ok=True)
        sample_count = len(os.listdir(out_dir))
        out_dir = os.path.join(out_dir, f"sample_{sample_count}")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def post_process(self, image, mask_an, source_image, out_dir):
        image = image.detach().cpu()
        mask_an = mask_an.detach().cpu()
        source_image = source_image.detach().cpu()

        img = (source_image * 0.5 + 0.5).squeeze(0)
        mask_red = mask_an.squeeze(0)
        img_redder = make_redder(img, mask_red)

        pil_mask = to_pil_image(mask_an.squeeze(0))
        pil_mask_blurred = pil_mask.filter(ImageFilter.GaussianBlur(radius=15))
        mask_blurred = to_tensor(pil_mask_blurred).unsqueeze_(0).to(mask_an.device)
        msak_f = 1 - (1 - mask_an) * (1 - mask_blurred)

        image_1 = image.unsqueeze(0)
        out_tile = msak_f * image_1 + (1 - msak_f) * (source_image * 0.5 + 0.5)
        out_image = torch.concat([img_redder.unsqueeze(0), image_1, out_tile], dim=0)

        self.save_images(out_image, out_dir)
        self.display_image(out_image)

    def __call__(self, source_image, mask_an, sample, prompt="", strength=0.8, num_inference_steps=50, layer=34, end_layer=70, seed=123, out_dir=None):
        self.sample = sample
        self.prompt = prompt
        self.strength = strength
        self.num_inference_steps = num_inference_steps
        self.layer = layer
        self.end_layer = end_layer
        self.seed = int(seed)
        
        self.generator = torch.Generator(self.device).manual_seed(self.seed)
        self.out_dir = out_dir if out_dir is not None else self._setup_output_directory(self.sample)
        self.start_step = 0
        self.end_step = int(self.strength * self.num_inference_steps)
        self.layer_idx = list(range(self.layer, self.end_layer))
        self.editor = AAS_XL(self.start_step, self.end_step, self.layer, self.end_layer, layer_idx=self.layer_idx, mask=mask_an, model_type="SDXL", ss_steps=9, ss_scale=0.3)
        regiter_attention_editor_diffusers(self.base, self.editor)

        image = self.base(
            prompt=self.prompt,
            image=source_image,
            height=1024,
            width=1024,
            rm_guidance_scale=9,
            strength=self.strength,
            mask_image=mask_an,
            generator=self.generator,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=1,
            output_type='pt'
        ).images[0]

        self.post_process(image, mask_an, source_image, self.out_dir)

    def save_images(self, out_image, out_dir):
        save_image(out_image, os.path.join(out_dir, f"all_step{self.end_step}_layer{self.layer}.png"))
        save_image(out_image[0], os.path.join(out_dir, f"source_step{self.end_step}_layer{self.layer}.png"))
        save_image(out_image[1], os.path.join(out_dir, f"anonymous_step{self.end_step}_layer{self.layer}.png"))
        save_image(out_image[2], os.path.join(out_dir, f"anonymous_tile_step{self.end_step}_layer{self.layer}.png"))
        print("Synthesized images are saved in", out_dir)

    def display_image(self, out_image):
        img_ori = to_pil_image(out_image[0])
        plt.figure(figsize=(20, 26))
        plt.imshow(img_ori)
        plt.show()


if __name__ == "__main__":
    eraser = AttentiveEraser()
    
    sample = "an"
    prompt = ""
    strength = 0.8
    num_inference_steps = 50
    layer = 34
    end_layer = 70
    seed = 123
    
    out_dir = eraser._setup_output_directory(sample)
    source_image_path = f"./examples/img/{sample}.png"
    mask_path = f"./examples/mask/{sample}_mask.png"
    source_image = load_image(source_image_path, eraser.device)
    mask_an = load_mask(mask_path, eraser.device)
    
    eraser(source_image, mask_an, sample, prompt, strength, num_inference_steps, layer, end_layer, seed, out_dir)