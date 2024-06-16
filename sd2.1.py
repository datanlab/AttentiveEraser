import os
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
from masactrl.masactrl_utils import regiter_attention_editor_diffusers
from omegaconf import OmegaConf
from torchvision.utils import save_image
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from masactrl.masactrl import MutualSelfAttentionControlMask_An_aug
from evaluation.data import InpaintingDataset, move_to_device
import tqdm
torch.cuda.set_device(0)  # set the GPU device
torch.set_grad_enabled(False)

def main(args):

    config = OmegaConf.load(args.config)
    # Note that you may add your Hugging Face token to get access to the models
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_path = config.model.path 
    custom_pipeline = config.model.pipeline_path
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        scheduler=scheduler,
        custom_pipeline=custom_pipeline,
        torch_dtype=torch.float32
    )
    pipe = pipe.to(device)
    out_ext = config.get('out_ext', '.png')
    out_suffix = config.out_suffix
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
                os.path.splitext(img_fname[len(config.dataset.datadir):])[0] + out_suffix + str(seed) + out_ext                                                       
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
        cur_mask_fname = os.path.join(
            config.outdir, 
            os.path.splitext(img_fname[len(config.dataset.datadir):])[0] + "_mask"+ out_ext                                                                     
        )
        #batch = default_collate([dataset[img_i]])
        batch = dataset[img_i]
        # 计算 mask 中值为 1 的元素的比例
        mask_ratio = torch.sum(batch['mask']).item() / torch.numel(batch['mask'])

        # 如果 mask 中值为 1 的元素的比例小于 0.0001，则跳出
        if mask_ratio < 0.0001:
            with open('/hy-tmp/DATA/zero_mask_files.txt', 'a') as f:
                f.write(cur_img_fname + '\n')
            continue

        
        batch = move_to_device(batch, device)

        source_prompt = ""
        target_prompt = ""
        prompts = [source_prompt, target_prompt]
        # invert the source image
        start_code, x0_latents = pipe.invert(
                                    batch['image'],
                                    batch['mask'],
                                    source_prompt,
                                    guidance_scale=1,
                                    num_inference_steps=50,
                                    generator=generator,
                                    return_intermediates=False)
        
        # inference the synthesized image with MyREMOVAL
        START_STEP = 0
        END_STEP = 50
        LAYER = 7 #0~5down,6mid,7~15up
        END_LAYER = 16

        # hijack the attention module
        editor = MutualSelfAttentionControlMask_An_aug(START_STEP, END_STEP, LAYER, END_LAYER, mask=batch['mask'])
        regiter_attention_editor_diffusers(pipe, editor)

        start_code = start_code.expand(len(prompts), -1, -1, -1)
        #image, pred_x0_list_denoise, latents_list_denoise = pipe(
        image = pipe(
            prompts,
            width=512,
            height=512,
            num_inference_steps=50,
            guidance_scale=1.0,
            latents=start_code,
            x0_latents = x0_latents,
            generator = generator,
            #record_list=list(reversed(latents_list)),
            mask = batch['mask'],
            return_intermediates = False,
        )
        out_tile = batch['mask'] * image[-1:] + (1 - batch['mask']) * (batch['image']* 0.5 + 0.5)

        if config.save.result == True:
            save_image(image[-1], cur_out_fname_ori)
        if config.save.tile == True:
            save_image(out_tile, cur_out_fname)
        if config.save.mask == True:
            save_image(batch['mask'], cur_mask_fname)
        if config.save.resize_img == True:
            save_image(batch['image']* 0.5 + 0.5, cur_img_fname)
        


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to evaluation config')
    main(parser.parse_args())

    