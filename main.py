from PIL import Image
import numpy as np
import os
import torch
from diffusers import DDIMScheduler,StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
from torchvision.io import read_image, ImageReadMode
import torch.nn.functional as F
import cv2
from torchvision.utils import save_image
from torchvision.io import read_image
from torchvision.transforms.functional import gaussian_blur
from pytorch_lightning import seed_everything
from matplotlib import pyplot as plt
torch.cuda.set_device(0)  # set the GPU device

dtype = torch.float16
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
model_path = "/root/autodl-tmp/input/stable-diffusion-xl-base-1.0"
base = StableDiffusionXLInpaintPipeline.from_pretrained(
    model_path,
    custom_pipeline="./pipelines/SDXL_inp_pipeline.py",
    scheduler=scheduler,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=dtype,
).to(device)
base.enable_attention_slicing()
base.enable_model_cpu_offload()

seed=123 
g = torch.Generator('cuda').manual_seed(seed)
def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    if image.shape[1] != 3:
        image = image.expand(-1, 3, -1, -1)
    image = F.interpolate(image, (1024, 1024))
    image = image.to(dtype).to(device)
    return image

def load_mask(mask_path, device):
    mask = read_image(mask_path,mode=ImageReadMode.GRAY)
    mask = mask.unsqueeze_(0).float() / 255.  # 0 or 1
    mask = F.interpolate(mask, (1024, 1024))
    #mask = gaussian_blur(mask, kernel_size=(77, 77))
    mask[mask < 0.1] = 0
    mask[mask >= 0.1] = 1
    mask = mask.to(dtype).to(device)
    return mask

## check
sample = "an"
prompt = ""
out_dir = f"./workdir_xl/{sample}/"
os.makedirs(out_dir, exist_ok=True)
sample_count = len(os.listdir(out_dir))
out_dir = os.path.join(out_dir, f"sample_{sample_count}")
os.makedirs(out_dir, exist_ok=True)
#SOURCE_IMAGE_PATH = "/hy-tmp/MyREMOVAL/result.png"
SOURCE_IMAGE_PATH = f"./examples/img/{sample}.png"
MASK_PATH = f"./examples/mask/{sample}_mask.png"


source_image = load_image(SOURCE_IMAGE_PATH, device)
mask_an = load_mask(MASK_PATH, device)

from AAS.AAS import AAS_XL
from AAS.AAS_utils import regiter_attention_editor_diffusers
strength = 0.8
num_inference_steps = 50
START_STEP = 0
END_STEP = int(strength*num_inference_steps)
LAYER = 34 #0~23down,24~33mid,34~69up 
END_LAYER = 70
layer_idx=list(range(LAYER, END_LAYER))

# hijack the attention module
editor = AAS_XL(START_STEP, END_STEP, LAYER, END_LAYER,layer_idx= layer_idx, mask=mask_an,model_type="SDXL",ss_steps=9,ss_scale=0.3)
regiter_attention_editor_diffusers(base, editor)

#image_s = Image.open(SOURCE_IMAGE_PATH).convert('RGB')
#mask = Image.open(MASK_PATH)
image = base(
    prompt=prompt,
    image=source_image,
    height=1024,
    width=1024,
    rm_guidance_scale=9,
    strength=strength,
    mask_image=mask_an,
    generator=g,
    num_inference_steps=num_inference_steps,
    guidance_scale=1,
    output_type='pt'
).images[0]

image = image.detach().cpu()
mask_an = mask_an.detach().cpu()
source_image = source_image.detach().cpu()

def make_redder(img, mask, increase_factor=0.4):
    # 创建一个拷贝以避免修改原始图像
    img_redder = img.clone()
    mask_expanded = mask.expand_as(img)
    # 增加红色分量（第一个通道）在 mask 为 1 的区域
    img_redder[0][mask_expanded[0] == 1] = torch.clamp(img_redder[0][mask_expanded[0] == 1] + increase_factor, 0, 1)
    
    return img_redder
img = (source_image* 0.5 + 0.5).squeeze(0)
mask_red = mask_an.squeeze(0)
img_redder = make_redder(img, mask_red)
#save_image(img_redder, os.path.join(out_dir, "redder.png"))

from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image, ImageFilter
pil_mask = to_pil_image(mask_an.squeeze(0))
pil_mask_blurred = pil_mask.filter(ImageFilter.GaussianBlur(radius=15))
mask_blurred = to_tensor(pil_mask_blurred).unsqueeze_(0).to(mask_an.device)
msak_f = 1-(1-mask_an)*(1-mask_blurred)

image_1=image.unsqueeze(0)
out_tile = msak_f * image_1 + (1 - msak_f) * (source_image* 0.5 + 0.5)
out_image = torch.concat([img_redder.unsqueeze(0),
                         image_1,
                         out_tile],
                         dim=0)

save_image(out_image, os.path.join(out_dir, f"all_step{END_STEP}_layer{LAYER}.png"))
save_image(out_image[0], os.path.join(out_dir, f"source_step{END_STEP}_layer{LAYER}.png"))
save_image(out_image[1], os.path.join(out_dir, f"anonymous_step{END_STEP}_layer{LAYER}.png"))
save_image(out_image[2], os.path.join(out_dir, f"anonymous_tile_step{END_STEP}_layer{LAYER}.png"))
print("Syntheiszed images are saved in", out_dir)
img_ori = cv2.imread(os.path.join(out_dir, f"all_step{END_STEP}_layer{LAYER}.png"))
img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20, 26))
plt.imshow(img_ori)