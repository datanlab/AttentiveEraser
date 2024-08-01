import os
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from omegaconf import OmegaConf
from torchvision.utils import save_image
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from evaluation.data import InpaintingDataset, move_to_device
import tqdm
torch.cuda.set_device(0)  # set the GPU device


def main(args):

    config = OmegaConf.load(args.config)
    # Note that you may add your Hugging Face token to get access to the models
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_path = config.model.path 
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        scheduler=scheduler,
        torch_dtype=torch.float32,  
    )
    
    pipe = pipe.to(device)
    out_ext = config.get('out_ext', '.png')
    seed = config.seed
    seed_everything(seed)
    generator=torch.Generator("cuda").manual_seed(seed)
    if not config.dataset.datadir.endswith('/'):
        config.dataset.datadir += '/'
    dataset = InpaintingDataset(**config.dataset)
    for img_i in tqdm.trange(len(dataset)):
        img_fname = dataset.img_filenames[img_i]
        cur_out_fname = os.path.join(
                config.outdir, 
                os.path.splitext(img_fname[len(config.dataset.datadir):])[0] + "_inp_"+ str(seed) + out_ext                                                       
            )
        
        cur_out_fname_ori = os.path.join(
                config.outdir, 
                os.path.splitext(img_fname[len(config.dataset.datadir):])[0] + "_ori_" + str(seed) + out_ext                                                     
            )
        
        os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
        cur_img_fname = os.path.join(
            config.outdir, 
            os.path.splitext(img_fname[len(config.dataset.datadir):])[0] + out_ext                                                                     
        )
        #batch = default_collate([dataset[img_i]])
        batch = dataset[img_i]
        # 计算 mask 中值为 1 的元素的比例
        mask_ratio = torch.sum(batch['mask']).item() / torch.numel(batch['mask'])

        # 如果 mask 中值为 1 的元素的比例小于 0.01，则跳出
        if mask_ratio < 0.0001:
            with open('/hy-tmp/DATA/zero_mask_files.txt', 'w') as f:
                f.write(cur_img_fname + '\n')
            continue

        with torch.no_grad():
            batch = move_to_device(batch, device)

            target_prompt = ""
            image = pipe(prompt=target_prompt, image=batch['image'], mask_image=batch['mask'],num_inference_steps = 50,strength=0.8, guidance_scale = 1,generator = generator,output_type='pt').images[0]

            out_tile = batch['mask'] * image + (1 - batch['mask']) * (batch['image']* 0.5 + 0.5)
        if config.save.result == True:
            save_image(image, cur_out_fname_ori)
        if config.save.tile == True:
            save_image(out_tile, cur_out_fname)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to evaluation config')
    main(parser.parse_args())

    