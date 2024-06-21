import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import json
import tqdm
import pandas as pd
import clip
import csv
import glob
from torchvision.io import read_image,ImageReadMode
from torchvision.transforms.functional import gaussian_blur, to_pil_image, to_tensor
from torchvision.utils import save_image
from PIL import Image, ImageFilter
import torch.nn.functional as F

class InferenceDataset(Dataset):
    def __init__(self, datadir, inference_dir, seeds, img_suffix='.jpg', inpainted_suffix='_removed.png'):
        self.inference_dir = inference_dir
        self.datadir = datadir
        if not datadir.endswith('/'):
            datadir += '/'
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '*mask*.png'), recursive=True)))
        self.ids = [file_name.rsplit('/', 1)[1].rsplit('_mask.png', 1)[0] for file_name in self.mask_filenames]
        self.img_filenames = [os.path.join(datadir, id + img_suffix) for id in self.ids]

        self.file_names_seed1 = [os.path.join(inference_dir, os.path.splitext(fname[len(datadir):])[0] + inpainted_suffix + str(seeds[0]) + '.png')
                                for fname in self.img_filenames]
        self.file_names_seed2 = [os.path.join(inference_dir, os.path.splitext(fname[len(datadir):])[0] + inpainted_suffix + str(seeds[1]) + '.png')
                                for fname in self.img_filenames]
        self.file_names_seed3 = [os.path.join(inference_dir, os.path.splitext(fname[len(datadir):])[0] + inpainted_suffix + str(seeds[2]) + '.png')
                                for fname in self.img_filenames] 
                 
        self.file_names_seed1_ori = [os.path.join(inference_dir, os.path.splitext(fname[len(datadir):])[0] + '_removed_' + str(seeds[0]) + '.png')
                                for fname in self.img_filenames]
        self.file_names_seed2_ori = [os.path.join(inference_dir, os.path.splitext(fname[len(datadir):])[0] + '_removed_' + str(seeds[1]) + '.png')
                                for fname in self.img_filenames]
        self.file_names_seed3_ori = [os.path.join(inference_dir, os.path.splitext(fname[len(datadir):])[0] + '_removed_' + str(seeds[2]) + '.png')
                                for fname in self.img_filenames] 

    def __len__(self):
        return len(self.mask_filenames)

    def load_image(self, image_path,device='cpu'):
        image = read_image(image_path)
        image = image[:3].unsqueeze_(0).float() / 255
        image = F.interpolate(image, (512, 512))
        if image.shape[1] != 3:
            image = image.expand(-1, 3, -1, -1)
        image = image.to(torch.float32).to(device)
        return image
    
    def load_mask(self, mask_path,device='cpu'):
        mask = read_image(mask_path,mode=ImageReadMode.GRAY)
        mask = mask.unsqueeze_(0).float() / 255.  # 0 or 1
        mask = F.interpolate(mask, (512, 512))
        mask = gaussian_blur(mask, kernel_size=(13, 13))
        mask[mask < 0.1] = 0
        mask[mask >= 0.1] = 1
        mask = mask.to(torch.float32).to(device)
        return mask
    
    def __getitem__(self, idx):
        source_image = self.load_image(self.img_filenames[idx])
        mask_image = self.load_mask(self.mask_filenames[idx])
        inpainted_image_seed1 = self.load_image(self.file_names_seed1[idx])
        inpainted_image_seed2 = self.load_image(self.file_names_seed2[idx])
        inpainted_image_seed3 = self.load_image(self.file_names_seed3[idx])
        
        return source_image, mask_image, inpainted_image_seed1, inpainted_image_seed2, inpainted_image_seed3
    
if __name__ == "__main__":
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    datadir = '/hy-tmp/6000_outputs'
    inference_dir = '/hy-tmp/6000_outputs'
    out_dir = '/hy-tmp/6000_outputs_gs'
    os.makedirs(out_dir, exist_ok=True)
    seeds = [123, 321, 777]
    dataset = InferenceDataset(datadir, inference_dir, seeds,img_suffix='.png',inpainted_suffix='_ori_')

    for img_i in tqdm.trange(298,301):
        source_image, mask_image, inpainted_image_seed1, inpainted_image_seed2, inpainted_image_seed3 = dataset[img_i]
        pil_mask = to_pil_image(mask_image.squeeze(0))
        pil_mask_blurred = pil_mask.filter(ImageFilter.GaussianBlur(radius=15))
        mask_blurred = to_tensor(pil_mask_blurred).unsqueeze_(0).to(mask_image.device)
        out_1 = mask_image * inpainted_image_seed1 + (1 - mask_image) * source_image
        out_2 = mask_image * inpainted_image_seed2 + (1 - mask_image) * source_image
        out_3 = mask_image * inpainted_image_seed3 + (1 - mask_image) * source_image
        out_tile_1 = mask_blurred * inpainted_image_seed1 + (1 - mask_blurred) * out_1
        out_tile_2 = mask_blurred * inpainted_image_seed2 + (1 - mask_blurred) * out_2
        out_tile_3 = mask_blurred * inpainted_image_seed3 + (1 - mask_blurred) * out_3
        save_image(out_tile_1, os.path.join(out_dir, dataset.file_names_seed1[img_i].rsplit('/',1)[1]))
        save_image(out_tile_2, os.path.join(out_dir, dataset.file_names_seed2[img_i].rsplit('/',1)[1]))
        save_image(out_tile_3, os.path.join(out_dir, dataset.file_names_seed3[img_i].rsplit('/',1)[1]))
        #save_image(mask_image, os.path.join(out_dir, dataset.mask_filenames[img_i].rsplit('/',1)[1]))
        #save_image(source_image, os.path.join(out_dir, dataset.img_filenames[img_i].rsplit('/',1)[1].rsplit('.',1)[0]+'.png'))