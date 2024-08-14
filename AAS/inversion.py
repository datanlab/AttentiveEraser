import torch
import numpy as np
from PIL import Image
import torch.nn.functional as nnf
from torch.optim.adam import Adam

def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None, **kwargs):
            if isinstance(context, dict):  # NOTE: compatible with ELITE (0.11.1)
                context = context['CONTEXT_TENSOR']
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count
    
@torch.no_grad()
def latent2image(model, latents, return_type='np'):
    latents = 1 / 0.18215 * latents.detach()
    image = model.decode(latents)['sample']
    if return_type == 'np':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
    return image

@torch.no_grad()
def image2latent(model, image):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(model.device)
            latents = model.encode(image)['latent_dist'].mean
            latents = latents * 0.18215
    return latents

class DirectInversion:
    
    def prev_step(self, model_output, timestep: int, sample):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        difference_scale_pred_original_sample= - beta_prod_t ** 0.5  / alpha_prod_t ** 0.5
        difference_scale_pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 
        difference_scale = alpha_prod_t_prev ** 0.5 * difference_scale_pred_original_sample + difference_scale_pred_sample_direction
        
        return prev_sample,difference_scale
    
    def next_step(self, model_output, timestep: int, sample):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, guidance_scale, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""]*len(prompt), padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        cond_embeddings=cond_embeddings[[0]]
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent
    
    @torch.no_grad()
    def ddim_null_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings=uncond_embeddings[[0]]
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, uncond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent
    
    @torch.no_grad()
    def ddim_with_guidance_scale_loop(self, latent,guidance_scale):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings=uncond_embeddings[[0]]
        cond_embeddings=cond_embeddings[[0]]
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            uncond_noise_pred = self.get_noise_pred_single(latent, t, uncond_embeddings)
            cond_noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            noise_pred = uncond_noise_pred + guidance_scale * (cond_noise_pred - uncond_noise_pred)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents
    
    @torch.no_grad()
    def ddim_null_inversion(self, image):
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_null_loop(latent)
        return image_rec, ddim_latents
    
    @torch.no_grad()
    def ddim_with_guidance_scale_inversion(self, image,guidance_scale):
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_with_guidance_scale_loop(latent,guidance_scale)
        return image_rec, ddim_latents

    def offset_calculate(self, latents, num_inner_steps, epsilon, guidance_scale):
        noise_loss_list = []
        latent_cur = torch.concat([latents[-1]]*(self.context.shape[0]//2))
        for i in range(self.num_ddim_steps):            
            latent_prev = torch.concat([latents[len(latents) - i - 2]]*latent_cur.shape[0])
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(torch.concat([latent_cur]*2), t, self.context)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec, _ = self.prev_step(noise_pred_w_guidance, t, latent_cur)
                loss = latent_prev - latents_prev_rec
                
            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss
            
        return noise_loss_list
    
    def invert(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5):
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        noise_loss_list = self.offset_calculate(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale)
        return image_gt, image_rec, ddim_latents, noise_loss_list
    
    def invert_without_attn_controller(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5):
        self.init_prompt(prompt)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        noise_loss_list = self.offset_calculate(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale)
        return image_gt, image_rec, ddim_latents, noise_loss_list
    
    def invert_with_guidance_scale_vary_guidance(self, image_gt, prompt, inverse_guidance_scale, forward_guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5):
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        
        image_rec, ddim_latents = self.ddim_with_guidance_scale_inversion(image_gt,inverse_guidance_scale)
        
        noise_loss_list = self.offset_calculate(ddim_latents, num_inner_steps, early_stop_epsilon,forward_guidance_scale)
        return image_gt, image_rec, ddim_latents, noise_loss_list

    def null_latent_calculate(self, latents, num_inner_steps, epsilon, guidance_scale):
        noise_loss_list = []
        latent_cur = torch.concat([latents[-1]]*(self.context.shape[0]//2))
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        for i in range(self.num_ddim_steps):            
            latent_prev = torch.concat([latents[len(latents) - i - 2]]*latent_cur.shape[0])
            t = self.model.scheduler.timesteps[i]

            if num_inner_steps!=0:
                uncond_embeddings = uncond_embeddings.clone().detach()
                uncond_embeddings.requires_grad = True
                optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
                for j in range(num_inner_steps):
                    latents_input = torch.cat([latent_cur] * 2)
                    noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=torch.cat([uncond_embeddings, cond_embeddings]))["sample"]
                    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
                    
                    latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)[0]
                    
                    loss = nnf.mse_loss(latents_prev_rec[[0]], latent_prev[[0]])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_item = loss.item()

                    if loss_item < epsilon + i * 2e-5:
                        break
                    
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(torch.concat([latent_cur]*2), t, self.context)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec, _ = self.prev_step(noise_pred_w_guidance, t, latent_cur)
                
                latent_cur = self.get_noise_pred(latent_cur, t,guidance_scale, False, torch.cat([uncond_embeddings, cond_embeddings]))[0]
                loss = latent_cur - latents_prev_rec
                
            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss
            
        return noise_loss_list
        
    
    def invert_null_latent(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5):
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        latent_list = self.null_latent_calculate(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale)
        return image_gt, image_rec, ddim_latents, latent_list
    
    def offset_calculate_not_full(self, latents, num_inner_steps, epsilon, guidance_scale,scale):
        noise_loss_list = []
        latent_cur = torch.concat([latents[-1]]*(self.context.shape[0]//2))
        for i in range(self.num_ddim_steps):            
            latent_prev = torch.concat([latents[len(latents) - i - 2]]*latent_cur.shape[0])
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(torch.concat([latent_cur]*2), t, self.context)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec, _ = self.prev_step(noise_pred_w_guidance, t, latent_cur)
                loss = latent_prev - latents_prev_rec
                loss=loss*scale
                
            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss
            
        return noise_loss_list
        
    def invert_not_full(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5,scale=1.):
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        noise_loss_list = self.offset_calculate_not_full(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale,scale)
        return image_gt, image_rec, ddim_latents, noise_loss_list
    
    def offset_calculate_skip_step(self, latents, num_inner_steps, epsilon, guidance_scale,skip_step):
        noise_loss_list = []
        latent_cur = torch.concat([latents[-1]]*(self.context.shape[0]//2))
        for i in range(self.num_ddim_steps):            
            latent_prev = torch.concat([latents[len(latents) - i - 2]]*latent_cur.shape[0])
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(torch.concat([latent_cur]*2), t, self.context)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec, _ = self.prev_step(noise_pred_w_guidance, t, latent_cur)
                if (i%skip_step)==0:
                    loss = latent_prev - latents_prev_rec
                else:
                    loss=torch.zeros_like(latent_prev)
                
            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss
            
        return noise_loss_list
    
    
    def invert_skip_step(self, image_gt, prompt, guidance_scale, skip_step,num_inner_steps=10, early_stop_epsilon=1e-5,scale=1.):
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        noise_loss_list = self.offset_calculate_skip_step(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale,skip_step)
        return image_gt, image_rec, ddim_latents, noise_loss_list
    
    
    def __init__(self, model,num_ddim_steps):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.prompt = None
        self.context = None
        self.num_ddim_steps=num_ddim_steps