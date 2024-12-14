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

def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    if image.shape[1] != 3:
        image = image.expand(-1, 3, -1, -1)
    image = F.interpolate(image, (1024, 1024))
    image = image.to(torch.float16).to(device)
    return image

def load_mask(mask_path, device):
    mask = read_image(mask_path,mode=ImageReadMode.GRAY)
    mask = mask.unsqueeze_(0).float() / 255.  # 0 or 1
    mask = F.interpolate(mask, (1024, 1024))
    #mask = gaussian_blur(mask, kernel_size=(77, 77))
    mask[mask < 0.1] = 0
    mask[mask >= 0.1] = 1
    mask = mask.to(torch.float16).to(device)
    return mask

def make_redder(img, mask, increase_factor=0.4):
    # 创建一个拷贝以避免修改原始图像
    img_redder = img.clone()
    mask_expanded = mask.expand_as(img)
    # 增加红色分量（第一个通道）在 mask 为 1 的区域
    img_redder[0][mask_expanded[0] == 1] = torch.clamp(img_redder[0][mask_expanded[0] == 1] + increase_factor, 0, 1)
    
    return img_redder
