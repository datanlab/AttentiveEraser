import torch
from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.utils import load_image
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, gaussian_blur

if __name__ == "__main__":

    dtype = torch.float16
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    # model_path = "/hy-tmp/stable-diffusion-xl-base-1.0" # change this to the path of the model if you are loading the model offline
    pipeline = DiffusionPipeline.from_pretrained(
        model_path,
        custom_pipeline="./pipelines/pipeline_stable_diffusion_xl_attentive_eraser.py",
        scheduler=scheduler,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=dtype,
    ).to(device)
    pipeline.enable_attention_slicing()
    pipeline.enable_model_cpu_offload()

    def preprocess_image(image_path, device):
        image = to_tensor((load_image(image_path)))
        image = image.unsqueeze_(0).float() * 2 - 1 # [0,1] --> [-1,1]
        if image.shape[1] != 3:
            image = image.expand(-1, 3, -1, -1)
        image = F.interpolate(image, (1024, 1024))
        image = image.to(dtype).to(device)
        return image

    def preprocess_mask(mask_path, device):
        mask = to_tensor((load_image(mask_path, convert_method=lambda img: img.convert('L'))))
        mask = mask.unsqueeze_(0).float()  # 0 or 1
        mask = F.interpolate(mask, (1024, 1024))
        mask = gaussian_blur(mask, kernel_size=(77, 77))
        mask[mask < 0.1] = 0
        mask[mask >= 0.1] = 1
        mask = mask.to(dtype).to(device)
        return mask
    
    prompt = ""
    seed=123 
    generator = torch.Generator(device=device).manual_seed(seed)
    image_name = "276490165110616"
    source_image_path = f"/content/test_mask/513960947746807.jpg"
    # mask_path = f"/content/test_mask/276490165110616_mask_1.png"
    source_image = preprocess_image(source_image_path, device)
    # mask = preprocess_mask(mask_path, device)

    mask_path_list = ["/content/test_mask/513960947746807_mask_1.png", "/content/test_mask/513960947746807_mask_2.png", "/content/test_mask/513960947746807_mask_3.png", "/content/test_mask/513960947746807_mask_4.png", "/content/test_mask/513960947746807_mask_5.png", "/content/test_mask/513960947746807_mask_6.png", "/content/test_mask/513960947746807_mask_7.png"]

    count = 0
    for mask_path in mask_path_list:
      
      print(f"Mask_path: {mask_path}")
      mask = preprocess_mask(mask_path, device)
      image = pipeline(
          prompt=prompt,
          image=source_image,
          mask_image=mask,
          height=1024,
          width=1024,
          AAS=True, # enable AAS
          strength=0.8, # inpainting strength
          rm_guidance_scale=9, # removal guidance scale
          ss_steps = 9, # similarity suppression steps
          ss_scale = 0.3, # similarity suppression scale
          AAS_start_step=0, # AAS start step
          AAS_start_layer=34, # AAS start layer
          AAS_end_layer=70, # AAS end layer
          num_inference_steps=50, # number of inference steps # AAS_end_step = int(strength*num_inference_steps)
          generator=generator,
          guidance_scale=1,
      ).images[0]
      count = count + 1
      image.save(f'./513960947746807_removed_img_{count}.png')
      print(f"Object removal completed - 513960947746807_removed_img_{count}.png")
      source_image = image
    

