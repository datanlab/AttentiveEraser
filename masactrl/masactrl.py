import os

import torch
import torch.nn.functional as F
import numpy as np

from einops import rearrange

from .masactrl_utils import AttentionBase

from torchvision.utils import save_image


class MutualSelfAttentionControl(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        print("MasaCtrl at denoising steps: ", self.step_idx)
        print("MasaCtrl at U-Net layers: ", self.layer_idx)

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx: #满足条件则调用父类的forward方法
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u = self.attn_batch(qu, ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c = self.attn_batch(qc, kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
        out = torch.cat([out_u, out_c], dim=0)

        return out


class MutualSelfAttentionControlUnion(MutualSelfAttentionControl):
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"):
        """
        Mutual self-attention control for Stable-Diffusion model with unition source and target [K, V]
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps, model_type)

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        qu_s, qu_t, qc_s, qc_t = q.chunk(4)
        ku_s, ku_t, kc_s, kc_t = k.chunk(4)
        vu_s, vu_t, vc_s, vc_t = v.chunk(4)
        attnu_s, attnu_t, attnc_s, attnc_t = attn.chunk(4)

        # source image branch
        out_u_s = super().forward(qu_s, ku_s, vu_s, sim, attnu_s, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_s = super().forward(qc_s, kc_s, vc_s, sim, attnc_s, is_cross, place_in_unet, num_heads, **kwargs)

        # target image branch, concatenating source and target [K, V]
        out_u_t = self.attn_batch(qu_t, torch.cat([ku_s, ku_t]), torch.cat([vu_s, vu_t]), sim[:num_heads], attnu_t, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_t = self.attn_batch(qc_t, torch.cat([kc_s, kc_t]), torch.cat([vc_s, vc_t]), sim[:num_heads], attnc_t, is_cross, place_in_unet, num_heads, **kwargs)

        out = torch.cat([out_u_s, out_u_t, out_c_s, out_c_t], dim=0)

        return out


class MutualSelfAttentionControlMask(MutualSelfAttentionControl):
    def __init__(self,  start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50,  mask_s=None, mask_t=None, mask_save_dir=None, model_type="SD"):
        """
        Maske-guided MasaCtrl to alleviate the problem of fore- and background confusion
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            mask_s: source mask with shape (h, w)
            mask_t: target mask with same shape as source mask
            mask_save_dir: the path to save the mask image
            model_type: the model type, SD or SDXL
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps, model_type)
        self.mask_s = mask_s  # source mask with shape (h, w)
        self.mask_t = mask_t  # target mask with same shape as source mask
        print("Using mask-guided MasaCtrl")
        if mask_save_dir is not None:
            os.makedirs(mask_save_dir, exist_ok=True)
            save_image(self.mask_s.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask_s.png"))
            save_image(self.mask_t.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask_t.png"))

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if kwargs.get("is_mask_attn") and self.mask_s is not None:
            print("masked attention")
            mask = self.mask_s.unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask, (H, W)).flatten(0).unsqueeze(0)
            mask = mask.flatten()
            # background
            sim_bg = sim + mask.masked_fill(mask == 1, torch.finfo(sim.dtype).min) #所有 mask 张量中等于 1 的元素都被替换为 torch.finfo(sim.dtype).min 极小数
            # object
            sim_fg = sim + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min)
            sim = torch.cat([sim_fg, sim_bg], dim=0)
        attn = sim.softmax(-1)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u_source = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_source = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        out_u_target = self.attn_batch(qu[-num_heads:], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, is_mask_attn=True, **kwargs)
        out_c_target = self.attn_batch(qc[-num_heads:], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, is_mask_attn=True, **kwargs)

        if self.mask_s is not None and self.mask_t is not None:
            out_u_target_fg, out_u_target_bg = out_u_target.chunk(2, 0)
            out_c_target_fg, out_c_target_bg = out_c_target.chunk(2, 0)

            mask = F.interpolate(self.mask_t.unsqueeze(0).unsqueeze(0), (H, W))
            mask = mask.reshape(-1, 1)  # (hw, 1)
            out_u_target = out_u_target_fg * mask + out_u_target_bg * (1 - mask)
            out_c_target = out_c_target_fg * mask + out_c_target_bg * (1 - mask)

        out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)
        return out

class MutualSelfAttentionControlMask_An_aug(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }
    def __init__(self,  start_step=4, end_step= 50, start_layer=10, end_layer=16,layer_idx=None, step_idx=None, total_steps=50,  mask=None, mask_save_dir=None, model_type="SD"):
        """
        Maske-guided MasaCtrl to alleviate the problem of fore- and background confusion
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            mask: source mask with shape (h, w)
            mask_save_dir: the path to save the mask image
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.end_step = end_step
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, end_layer))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, end_step))
        self.mask = mask  # mask with shape (1, 1 ,h, w)
        #print("AN at denoising steps: ", self.step_idx)
        #print("AN at U-Net layers: ", self.layer_idx)
        #print("start to enhance attention")
        self.mask_8 = F.max_pool2d(mask,(512//8,512//8)).round().squeeze().squeeze()
        self.mask_16 = F.max_pool2d(mask,(512//16,512//16)).round().squeeze().squeeze()
        self.mask_32 = F.max_pool2d(mask,(512//32,512//32)).round().squeeze().squeeze()
        self.mask_64 = F.max_pool2d(mask,(512//64,512//64)).round().squeeze().squeeze()
        #self.mask_16 = F.interpolate(mask,(16,16)).round().squeeze().squeeze()
        #self.mask_32 = F.interpolate(mask,(32,32)).round().squeeze().squeeze()
        #self.mask_64 = F.interpolate(mask,(64,64)).round().squeeze().squeeze()
        #self.aug_sim_16 = self.enhance_attention(self.mask_16, 15, 1)#
        self.aug_sim_8 = torch.zeros(1,8*8, 8*8).type_as(self.mask)
        self.aug_sim_16 = torch.zeros(1,16*16, 16*16).type_as(self.mask)
        self.aug_sim_32 = torch.zeros(1,32*32, 32*32).type_as(self.mask)
        self.aug_sim_64 = torch.zeros(1,64*64, 64*64).type_as(self.mask)
        #self.aug_sim_32 = self.enhance_attention(self.mask_32, 6, 1)#2
        #self.aug_sim_64 = self.enhance_attention(self.mask_64, 1, 3)#4

        #print("Using mask-guided AN")
        if mask_save_dir is not None:
            os.makedirs(mask_save_dir, exist_ok=True)
            save_image(self.mask.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask.png"))

    def enhance_attention(self, mask, enhancement_value, range_value, gaussian_sigma=7): #7
        """
        Enhance attention values for pixels inside the mask towards their neighboring non-mask pixels efficiently using PyTorch.
        
        Args:
        - attention_map (torch.Tensor): Attention map of shape [H*W, H*W].
        - mask (torch.Tensor): Binary mask of shape [H, W], where 1 indicates inside the mask and 0 indicates outside.
        - enhancement_factor (float): Factor by which to enhance the attention values for pixels inside the mask.
        
        Returns:
        - enhanced_attention_map (torch.Tensor): Enhanced attention map of shape [H*W, H*W].
        """
        H, W = mask.shape
        aug_sim = torch.zeros(1,H*W, H*W).type_as(mask)
        # Get indices of mask pixels
        mask_indices = torch.nonzero(mask, as_tuple=False).squeeze()
        mask_indices_flat = mask_indices[:, 0] * W + mask_indices[:, 1]
        
        # Get indices of non-mask pixels
        non_mask_indices = torch.nonzero(1 - mask, as_tuple=False).squeeze()
        non_mask_indices_flat = non_mask_indices[:, 0] * W + non_mask_indices[:, 1]
        
        # Calculate enhanced attention values for each mask pixel
        for mask_idx in mask_indices_flat:
            # Get row and column indices of the current mask pixel
            row_idx = mask_idx // W
            col_idx = mask_idx % W
            
            # Get indices of neighboring non-mask pixels
            row_start = max(0, row_idx - range_value)
            row_end = min(H, row_idx + range_value + 1)
            col_start = max(0, col_idx - range_value)
            col_end = min(W, col_idx + range_value + 1)
            
            # Get indices of neighboring non-mask pixels in flattened attention map
            neighbor_indices = non_mask_indices_flat[(non_mask_indices[:, 0] >= row_start) & (non_mask_indices[:, 0] < row_end) & (non_mask_indices[:, 1] >= col_start) & (non_mask_indices[:, 1] < col_end)]
            
            # Update attention values between mask pixel and its neighbors
            for neighbor_idx in neighbor_indices:
                # Convert neighbor index to attention map index
                neighbor_row_idx = neighbor_idx // W
                neighbor_col_idx = neighbor_idx % W
                distance = torch.sqrt((row_idx - neighbor_row_idx) ** 2 + (col_idx - neighbor_col_idx) ** 2)
                weight = torch.exp(- distance / (2 * gaussian_sigma ** 2))
                #weight = 1
                neighbor_attention_idx = neighbor_row_idx * W + neighbor_col_idx
                
                # Update attention value
                aug_sim[:, mask_idx, neighbor_attention_idx] = enhancement_value * weight
        
        return aug_sim
    
    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads,is_mask_attn, mask, aug_sim, **kwargs):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if is_mask_attn:
            mask_flatten = mask.flatten(0)

            # background
            sim_bg = sim + mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min) #所有 mask 张量中等于 1 的元素都被替换为 torch.finfo(sim.dtype).min 极小数
            #sim_bg = sim
            # object 

            #sim_fg = sim + mask_flatten.masked_fill(mask_flatten == 0, 6)
                         
            a=mask_flatten.reshape(-1,1)
            b=mask_flatten.reshape(1,-1)
            C_matrix=a*sim*(1-b)
            C = round(abs(torch.min(C_matrix).item()))
            M = round(torch.max(C_matrix).item())
            sim_fg = sim  + mask_flatten.masked_fill(mask_flatten == 0, -0.1*M)
            #sim_fg = sim  + mask_flatten.masked_fill(mask_flatten == 0, 0)
                                                                         
            #if self.cur_step <= 18 and self.cur_step >= 10:
            if self.cur_step <= 10:
                #sim_fg = torch.clamp(sim_fg, max=M+0.1*C)
                sim_fg = torch.abs(sim_fg)
                #sim_fg = -sim_fg

            """
            if self.cur_step <= 15 : #self.cur_step <= 30
                #sim_fg = sim + aug_sim + mask_flatten.masked_fill(mask_flatten == 0, 10)
                #sim_fg = sim + aug_sim*(10*C) + mask_flatten.masked_fill(mask_flatten == 0, C)
                sim_fg = sim + aug_sim*(10*C) + mask_flatten.masked_fill(mask_flatten == 0, C)
                                                                                    
                if C != 0:
                    #sim_fg = torch.clamp(sim_fg, max=15*C)
                    sim_fg = torch.clamp(sim_fg, max=15*C) 
            else:
                #sim_fg = sim  + mask_flatten.masked_fill(mask_flatten == 0, 10)
                sim_fg = sim  + mask_flatten.masked_fill(mask_flatten == 0, C)"""

            #sim_fg = sim + aug_sim + mask_flatten.masked_fill(mask_flatten == 0, 10)
            #sim_fg = sim +  (aug_sim + mask_flatten.masked_fill(mask_flatten == 0, 10))*torch.exp(- torch.tensor(self.cur_step) / (2 * 5 ** 2))
            #sim_fg = sim +  (aug_sim + mask_flatten.masked_fill(mask_flatten == 0, 6))*1
            #sim_fg += mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min)
                                    
            if self.cur_step <= 50:
                sim_fg += mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min)
            
            """                        
            if self.cur_step <= 30: # 让后续生成也关注一些自己
                #sim_fg = sim + torch.full(mask.shape, 10).to(sim.device)
                sim_fg = sim + mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min)
                sim_fg +=  aug_sim + mask_flatten.masked_fill(mask_flatten == 0, 6)
            else:
                sim_fg = sim + aug_sim
                sim_fg += mask_flatten.masked_fill(mask_flatten == 0, 4) """
            #sim_fg = sim + mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min)
            #sim_fg +=  mask_flatten.masked_fill(mask_flatten == 0, 4) 

            sim = torch.cat([sim_fg, sim_bg], dim=0)
        attn = sim.softmax(-1)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        
        if H == 16:
            mask = self.mask_16.to(sim.device)
            aug_sim = self.aug_sim_16.to(sim.device)
        elif H == 32:
            mask = self.mask_32.to(sim.device)
            aug_sim = self.aug_sim_32.to(sim.device)
        elif H == 8:
            mask = self.mask_8.to(sim.device)
            aug_sim = self.aug_sim_8.to(sim.device)
        else:
            mask = self.mask_64.to(sim.device)
            aug_sim = self.aug_sim_64.to(sim.device) 

        
        q_wo, q_w = q.chunk(2)
        k_wo, k_w = k.chunk(2)
        v_wo, v_w = v.chunk(2)
        sim_wo, sim_w = sim.chunk(2)
        attn_wo, attn_w = attn.chunk(2)

        out_source = self.attn_batch(q_wo, k_wo, v_wo, sim_wo, attn_wo, is_cross, place_in_unet, num_heads,is_mask_attn=False,mask=None,aug_sim=None,**kwargs)
        out_target = self.attn_batch(q_w, k_w, v_w, sim_w, attn_w, is_cross, place_in_unet, num_heads, is_mask_attn=True, mask = mask, aug_sim = aug_sim, **kwargs)

        if self.mask is not None:
            out_target_fg, out_target_bg = out_target.chunk(2, 0)

            #mask = F.interpolate(self.mask.unsqueeze(0).unsqueeze(0), (H, W))
            #mask = F.max_pool2d(self.mask,(512//H,512//W)).round()
            #mask = F.interpolate(self.mask, (H, W))
            mask = mask.reshape(-1, 1)  # (hw, 1)
            out_target = out_target_fg * mask + out_target_bg * (1 - mask)
        
        out = torch.cat([out_source, out_target], dim=0)
        return out

class MutualSelfAttentionControlMask_An(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }
    def __init__(self,  start_step=4, end_step= 50, start_layer=10, end_layer=16,layer_idx=None, step_idx=None, total_steps=50,  mask=None, mask_save_dir=None, model_type="SD"):
        """
        Maske-guided MasaCtrl to alleviate the problem of fore- and background confusion
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            mask: source mask with shape (h, w)
            mask_save_dir: the path to save the mask image
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.end_step = end_step
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, end_layer))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, end_step))
        self.mask = mask  # mask with shape (1, 1 ,h, w)
        #print("AN at denoising steps: ", self.step_idx)
        #print("AN at U-Net layers: ", self.layer_idx)
        #print("start to enhance attention")
        self.mask_8 = F.max_pool2d(mask,(512//8,512//8)).round().squeeze().squeeze()
        self.mask_16 = F.max_pool2d(mask,(512//16,512//16)).round().squeeze().squeeze()
        self.mask_32 = F.max_pool2d(mask,(512//32,512//32)).round().squeeze().squeeze()
        self.mask_64 = F.max_pool2d(mask,(512//64,512//64)).round().squeeze().squeeze()
        #self.mask_16 = F.interpolate(mask,(16,16)).round().squeeze().squeeze()
        #self.mask_32 = F.interpolate(mask,(32,32)).round().squeeze().squeeze()
        #self.mask_64 = F.interpolate(mask,(64,64)).round().squeeze().squeeze()

        #print("Using mask-guided AN")
        if mask_save_dir is not None:
            os.makedirs(mask_save_dir, exist_ok=True)
            save_image(self.mask.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask.png"))
    
    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads,is_mask_attn, mask, **kwargs):
        B = q.shape[0] // num_heads
        #H = W = int(np.sqrt(q.shape[1]))
        #q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        #k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        #v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        #sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if is_mask_attn:
            mask_flatten = mask.flatten(0)

            # background
            sim_bg = sim + mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min) #所有 mask 张量中等于 1 的元素都被替换为 torch.finfo(sim.dtype).min 极小数
            #sim_bg = sim
            # object 

            #sim_fg = sim + mask_flatten.masked_fill(mask_flatten == 0, 6)
                         
            a=mask_flatten.reshape(-1,1)
            b=mask_flatten.reshape(1,-1)
            C_matrix=a*sim*(1-b)
            #C = round(abs(torch.min(C_matrix).item()))
            M = round(torch.max(C_matrix).item())
            #mu = torch.mean(C_matrix).item()
            #sim_fg = sim  + mask_flatten.masked_fill(mask_flatten == 0, -0.1*M)
            #sim_fg = sim  + mask_flatten.masked_fill(mask_flatten == 0, -2*mu)

                                                               
            if self.cur_step <= 10:
                sim_fg = sim  + mask_flatten.masked_fill(mask_flatten == 0, -0.15*M) #0.2
                #sim_fg = (sim_fg  + mask_flatten.masked_fill(mask_flatten == 0, -mu))*0.33 + mask_flatten.masked_fill(mask_flatten == 0, mu)
                sim_fg = torch.abs(sim_fg)
            else:
                sim_fg = sim  + mask_flatten.masked_fill(mask_flatten == 0, 0)

            #sim_fg = sim  + mask_flatten.masked_fill(mask_flatten == 0, 0)
            
            """                                                                          
            #if self.cur_step <= 18 and self.cur_step >= 10:
            if self.cur_step <= 15:
                #sim_fg = torch.clamp(sim_fg, max=M+0.1*C)
                sim_fg = torch.abs(sim_fg)
                #sim_fg = -sim_fg """


            """
            if self.cur_step <= 15 : #self.cur_step <= 30
                #sim_fg = sim + aug_sim + mask_flatten.masked_fill(mask_flatten == 0, 10)
                #sim_fg = sim + aug_sim*(10*C) + mask_flatten.masked_fill(mask_flatten == 0, C)
                sim_fg = sim + aug_sim*(10*C) + mask_flatten.masked_fill(mask_flatten == 0, C)
                                                                                    
                if C != 0:
                    #sim_fg = torch.clamp(sim_fg, max=15*C)
                    sim_fg = torch.clamp(sim_fg, max=15*C) 
            else:
                #sim_fg = sim  + mask_flatten.masked_fill(mask_flatten == 0, 10)
                sim_fg = sim  + mask_flatten.masked_fill(mask_flatten == 0, C)"""

            #sim_fg = sim + aug_sim + mask_flatten.masked_fill(mask_flatten == 0, 10)
            #sim_fg = sim +  (aug_sim + mask_flatten.masked_fill(mask_flatten == 0, 10))*torch.exp(- torch.tensor(self.cur_step) / (2 * 5 ** 2))
            #sim_fg = sim +  (aug_sim + mask_flatten.masked_fill(mask_flatten == 0, 6))*1
            #sim_fg += mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min)
                                    
            if self.cur_step <= 50:
                sim_fg += mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min)

            sim = torch.cat([sim_fg, sim_bg], dim=0)
        attn = sim.softmax(-1)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        
        if H == 16:
            mask = self.mask_16.to(sim.device)
        elif H == 32:
            mask = self.mask_32.to(sim.device)
        elif H == 8:
            mask = self.mask_8.to(sim.device)
        else:
            mask = self.mask_64.to(sim.device)

        
        q_wo, q_w = q.chunk(2)
        k_wo, k_w = k.chunk(2)
        v_wo, v_w = v.chunk(2)
        sim_wo, sim_w = sim.chunk(2)
        attn_wo, attn_w = attn.chunk(2)

        out_source = self.attn_batch(q_wo, k_wo, v_wo, sim_wo, attn_wo, is_cross, place_in_unet, num_heads,is_mask_attn=False,mask=None,**kwargs)
        out_target = self.attn_batch(q_w, k_w, v_w, sim_w, attn_w, is_cross, place_in_unet, num_heads, is_mask_attn=True, mask = mask, **kwargs)

        if self.mask is not None:
            out_target_fg, out_target_bg = out_target.chunk(2, 0)
            
            mask = mask.reshape(-1, 1)  # (hw, 1)
            out_target = out_target_fg * mask + out_target_bg * (1 - mask)
        
        out = torch.cat([out_source, out_target], dim=0)
        return out
    
class MutualSelfAttentionControlMask_An_aug_predict(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }
    def __init__(self,  start_step=4, end_step= 50, start_layer=10, end_layer=16,layer_idx=None, step_idx=None, total_steps=50,  mask=None, mask_save_dir=None, model_type="SD"):
        """
        Maske-guided MasaCtrl to alleviate the problem of fore- and background confusion
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            mask: source mask with shape (h, w)
            mask_save_dir: the path to save the mask image
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.end_step = end_step
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, end_layer))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, end_step))
        self.mask = mask  # mask with shape (1, 1 ,h, w)
        #print("AN at denoising steps: ", self.step_idx)
        #print("AN at U-Net layers: ", self.layer_idx)
        #print("start to enhance attention")
        self.mask_16 = F.max_pool2d(mask,(512//16,512//16)).round().squeeze().squeeze()
        self.mask_32 = F.max_pool2d(mask,(512//32,512//32)).round().squeeze().squeeze()
        self.mask_64 = F.max_pool2d(mask,(512//64,512//64)).round().squeeze().squeeze()
        #self.aug_sim_16 = self.enhance_attention(self.mask_16, 2, 1)#2
        #self.aug_sim_16 = torch.zeros(1,16*16, 16*16).type_as(self.mask)
        #self.aug_sim_32 = torch.zeros(1,32*32, 32*32).type_as(self.mask)
        #self.aug_sim_32 = self.enhance_attention(self.mask_32, 3, 3)
        #self.aug_sim_64 = self.enhance_attention(self.mask_64, 3, 5)
        

        #print("Using mask-guided AN")
        if mask_save_dir is not None:
            os.makedirs(mask_save_dir, exist_ok=True)
            save_image(self.mask.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask.png"))
    
    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads,is_mask_attn, mask, **kwargs):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if is_mask_attn:
            mask_flatten = mask.flatten(0)

            # background
            sim_bg = sim + mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min) #所有 mask 张量中等于 1 的元素都被替换为 torch.finfo(sim.dtype).min 极小数
            #sim_bg = sim
            # object 

            sim_fg = sim  + mask_flatten.masked_fill(mask_flatten == 0, 10)                  
            if self.cur_step <= 50:
                sim_fg += mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min)
            """             
            if self.cur_step <= 10: # 让后续生成也关注一些自己
                #sim_fg = sim + torch.full(mask.shape, 10).to(sim.device)
                sim_fg = sim + aug_sim
                sim_fg += mask_flatten.masked_fill(mask_flatten == 0, 6)
                sim_fg += mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min)
            else:
                sim_fg = sim + aug_sim
                sim_fg += mask_flatten.masked_fill(mask_flatten == 0, 6)
            #sim_fg = sim + mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min)
            #sim_fg +=  mask_flatten.masked_fill(mask_flatten == 0, 4) """

            sim = torch.cat([sim_fg, sim_bg], dim=0)
        attn = sim.softmax(-1)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        
        if H == 16:
            mask = self.mask_16.to(sim.device)
        elif H == 32:
            mask = self.mask_32.to(sim.device)
        else:
            mask = self.mask_64.to(sim.device)

        
        q_wo, q_w = q.chunk(2)
        k_wo, k_w = k.chunk(2)
        v_wo, v_w = v.chunk(2)
        sim_wo, sim_w = sim.chunk(2)
        attn_wo, attn_w = attn.chunk(2)

        out_source = self.attn_batch(q_wo, k_wo, v_wo, sim_wo, attn_wo, is_cross, place_in_unet, num_heads,is_mask_attn=False,mask=None,**kwargs)
        out_target = self.attn_batch(q_w, k_w, v_w, sim_w, attn_w, is_cross, place_in_unet, num_heads, is_mask_attn=True, mask = mask,**kwargs)

        if self.mask is not None:
            out_target_fg, out_target_bg = out_target.chunk(2, 0)

            #mask = F.interpolate(self.mask.unsqueeze(0).unsqueeze(0), (H, W))
            #mask = F.max_pool2d(self.mask,(512//H,512//W)).round()
            #mask = F.interpolate(self.mask, (H, W))
            mask = mask.reshape(-1, 1)  # (hw, 1)
            out_target = out_target_fg * mask + out_target_bg * (1 - mask)
        
        out = torch.cat([out_source, out_target], dim=0)
        return out


class MutualSelfAttentionControlMask_An_aug_XL(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }
    def __init__(self,  start_step=4, end_step= 50, start_layer=10, end_layer=16,layer_idx=None, step_idx=None, total_steps=50,  mask=None, mask_save_dir=None, model_type="SD"):
        """
        Maske-guided MasaCtrl to alleviate the problem of fore- and background confusion
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            mask: source mask with shape (h, w)
            mask_save_dir: the path to save the mask image
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.end_step = end_step
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, end_layer))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, end_step))
        self.mask = mask  # mask with shape (1, 1 ,h, w)
        #print("AN at denoising steps: ", self.step_idx)
        #print("AN at U-Net layers: ", self.layer_idx)
        #print("start to enhance attention")
        self.mask_32 = F.max_pool2d(mask,(1024//32,1024//32)).round().squeeze().squeeze()
        self.mask_64 = F.max_pool2d(mask,(1024//64,1024//64)).round().squeeze().squeeze()
        self.mask_128 = F.max_pool2d(mask,(1024//128,1024//128)).round().squeeze().squeeze()
        self.aug_sim_32 = torch.zeros(1,32*32, 32*32).type_as(self.mask)#4
        self.aug_sim_64 = torch.zeros(1,64*64, 64*64).type_as(self.mask)#4
        self.aug_sim_128 = torch.zeros(1,128*128, 128*128).type_as(self.mask)#4
        #self.aug_sim_64 = self.enhance_attention(self.mask_64, 3, 3)#9
        #self.aug_sim_128 = self.enhance_attention(self.mask_64, 8, 4)#9
        
    
        #print("Using mask-guided AN")
        if mask_save_dir is not None:
            os.makedirs(mask_save_dir, exist_ok=True)
            save_image(self.mask.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask.png"))

    def enhance_attention(self, mask, enhancement_value, range_value, gaussian_sigma=7 ):
        """
        Enhance attention values for pixels inside the mask towards their neighboring non-mask pixels efficiently using PyTorch.
        
        Args:
        - attention_map (torch.Tensor): Attention map of shape [H*W, H*W].
        - mask (torch.Tensor): Binary mask of shape [H, W], where 1 indicates inside the mask and 0 indicates outside.
        - enhancement_factor (float): Factor by which to enhance the attention values for pixels inside the mask.
        
        Returns:
        - enhanced_attention_map (torch.Tensor): Enhanced attention map of shape [H*W, H*W].
        """
        H, W = mask.shape
        aug_sim = torch.zeros(1,H*W, H*W).type_as(mask)
        # Get indices of mask pixels
        mask_indices = torch.nonzero(mask, as_tuple=False).squeeze()
        mask_indices_flat = mask_indices[:, 0] * W + mask_indices[:, 1]
        
        # Get indices of non-mask pixels
        non_mask_indices = torch.nonzero(1 - mask, as_tuple=False).squeeze()
        non_mask_indices_flat = non_mask_indices[:, 0] * W + non_mask_indices[:, 1]
        
        # Calculate enhanced attention values for each mask pixel
        for mask_idx in mask_indices_flat:
            # Get row and column indices of the current mask pixel
            row_idx = mask_idx // W
            col_idx = mask_idx % W
            
            # Get indices of neighboring non-mask pixels
            row_start = max(0, row_idx - range_value)
            row_end = min(H, row_idx + range_value + 1)
            col_start = max(0, col_idx - range_value)
            col_end = min(W, col_idx + range_value + 1)
            
            # Get indices of neighboring non-mask pixels in flattened attention map
            neighbor_indices = non_mask_indices_flat[(non_mask_indices[:, 0] >= row_start) & (non_mask_indices[:, 0] < row_end) & (non_mask_indices[:, 1] >= col_start) & (non_mask_indices[:, 1] < col_end)]
            
            # Update attention values between mask pixel and its neighbors
            for neighbor_idx in neighbor_indices:
                # Convert neighbor index to attention map index
                neighbor_row_idx = neighbor_idx // W
                neighbor_col_idx = neighbor_idx % W
                distance = torch.sqrt((row_idx - neighbor_row_idx) ** 2 + (col_idx - neighbor_col_idx) ** 2)
                weight = torch.exp(- distance / (2 * gaussian_sigma ** 2))
                neighbor_attention_idx = neighbor_row_idx * W + neighbor_col_idx
                
                # Update attention value
                aug_sim[:, mask_idx, neighbor_attention_idx] = enhancement_value * weight
        
        return aug_sim
    
    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads,is_mask_attn, mask,aug_sim, **kwargs):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if is_mask_attn:
            mask_flatten = mask.flatten(0)

            # background
            sim_bg = sim + mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min) #所有 mask 张量中等于 1 的元素都被替换为 torch.finfo(sim.dtype).min 极小数
            #sim_bg = sim
            # object 
            if self.cur_step <= 30 and self.cur_step >=0 :
                sim_fg = sim + aug_sim + mask_flatten.masked_fill(mask_flatten == 0, 10)
            else:
                sim_fg = sim  + mask_flatten.masked_fill(mask_flatten == 0, 10)
            #sim_fg = sim + mask_flatten.masked_fill(mask_flatten == 0, 6)
            sim_fg += mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min)

            sim = torch.cat([sim_fg, sim_bg], dim=0)
        attn = sim.softmax(-1)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        if H == 32:
            mask = self.mask_32.to(sim.device)
            aug_sim = self.aug_sim_32.to(sim.device)
        elif H == 64:
            mask = self.mask_64.to(sim.device)
            aug_sim = self.aug_sim_64.to(sim.device)
        else:
            mask = self.mask_128.to(sim.device)
            aug_sim = self.aug_sim_128.to(sim.device)


        q_wo, q_w = q.chunk(2)
        k_wo, k_w = k.chunk(2)
        v_wo, v_w = v.chunk(2)
        sim_wo, sim_w = sim.chunk(2)
        attn_wo, attn_w = attn.chunk(2)

        out_source = self.attn_batch(q_wo, k_wo, v_wo, sim_wo, attn_wo, is_cross, place_in_unet, num_heads,is_mask_attn=False,mask=None,aug_sim=None,**kwargs)
        out_target = self.attn_batch(q_w, k_w, v_w, sim_w, attn_w, is_cross, place_in_unet, num_heads, is_mask_attn=True, mask = mask,aug_sim=aug_sim,**kwargs)

        if self.mask is not None:
            out_target_fg, out_target_bg = out_target.chunk(2, 0)

            #mask = F.interpolate(self.mask.unsqueeze(0).unsqueeze(0), (H, W))
            #mask = F.max_pool2d(self.mask,(512//H,512//W)).round()
            #mask = F.interpolate(self.mask, (H, W))
            mask = mask.reshape(-1, 1)  # (hw, 1)
            out_target = out_target_fg * mask + out_target_bg * (1 - mask)
        
        out = torch.cat([out_source, out_target], dim=0)
        return out
    
class MutualSelfAttentionControlMask_inp(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }
    def __init__(self,  start_step=4, end_step= 50, start_layer=10, end_layer=16,layer_idx=None, step_idx=None, total_steps=50,  mask=None, mask_save_dir=None, model_type="SD"):
        """
        Maske-guided MasaCtrl to alleviate the problem of fore- and background confusion
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            mask: source mask with shape (h, w)
            mask_save_dir: the path to save the mask image
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.end_step = end_step
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, end_layer))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, end_step))
        self.mask = mask  # mask with shape (h, w)
        print("AN at denoising steps: ", self.step_idx)
        print("AN at U-Net layers: ", self.layer_idx)


        print("Using mask-guided AN")
        if mask_save_dir is not None:
            os.makedirs(mask_save_dir, exist_ok=True)
            save_image(self.mask.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask.png"))

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if kwargs.get("is_mask_attn") and self.mask is not None:
            #print("masked attention")
            #mask = self.mask.unsqueeze(0).unsqueeze(0)
            mask = F.max_pool2d(self.mask,(512//H,512//W)).round().flatten(0).unsqueeze(0).to(sim.device)
            #mask = F.interpolate(self.mask, (H, W)).flatten(0).unsqueeze(0).to(sim.device)
            mask = mask.flatten()

            # background
            sim_bg = sim + mask.masked_fill(mask == 1, torch.finfo(sim.dtype).min) #所有 mask 张量中等于 1 的元素都被替换为 torch.finfo(sim.dtype).min 极小数
            #sim_bg = sim
            # object 
           
            #sim_fg = sim + mask.masked_fill(mask == 0, 20)
                          
            if self.cur_step <= 5: # 让后续生成也关注一些自己
                #sim_fg = sim + torch.full(mask.shape, 10).to(sim.device)
                sim_fg = sim + mask.masked_fill(mask == 0, 10)
                sim_fg +=  mask.masked_fill(mask == 1, torch.finfo(sim.dtype).min)
            else:
                sim_fg = sim + mask.masked_fill(mask == 0, 15) 
            #sim_fg = sim + mask.masked_fill(mask == 1, torch.finfo(sim.dtype).min)
            #sim_fg +=  mask.masked_fill(mask == 0, 15)

            sim = torch.cat([sim_fg, sim_bg], dim=0)
        attn = sim.softmax(-1)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        out_target = self.attn_batch(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, is_mask_attn=True, **kwargs)

        if self.mask is not None:
            out_target_fg, out_target_bg = out_target.chunk(2, 0)

            #mask = F.interpolate(self.mask.unsqueeze(0).unsqueeze(0), (H, W))
            mask = F.interpolate(self.mask, (H, W)).to(out_target.device)
            mask = mask.reshape(-1, 1)  # (hw, 1)
            out_target = out_target_fg * mask + out_target_bg * (1 - mask)

        return out_target

class MutualSelfAttentionControlMaskAuto(MutualSelfAttentionControl):
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, thres=0.1, ref_token_idx=[1], cur_token_idx=[1], mask_save_dir=None, model_type="SD"):
        """
        MasaCtrl with mask auto generation from cross-attention map
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            thres: the thereshold for mask thresholding
            ref_token_idx: the token index list for cross-attention map aggregation
            cur_token_idx: the token index list for cross-attention map aggregation
            mask_save_dir: the path to save the mask image
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps, model_type)
        print("Using MutualSelfAttentionControlMaskAuto")
        self.thres = thres
        self.ref_token_idx = ref_token_idx
        self.cur_token_idx = cur_token_idx

        self.self_attns = []
        self.cross_attns = []

        self.cross_attns_mask = None
        self.self_attns_mask = None

        self.mask_save_dir = mask_save_dir
        if self.mask_save_dir is not None:
            os.makedirs(self.mask_save_dir, exist_ok=True)

    def after_step(self):
        self.self_attns = []
        self.cross_attns = []

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if self.self_attns_mask is not None:
            # binarize the mask
            mask = self.self_attns_mask
            thres = self.thres
            mask[mask >= thres] = 1
            mask[mask < thres] = 0
            sim_fg = sim + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min)
            sim_bg = sim + mask.masked_fill(mask == 1, torch.finfo(sim.dtype).min)
            sim = torch.cat([sim_fg, sim_bg])

        attn = sim.softmax(-1)

        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def aggregate_cross_attn_map(self, idx):
        attn_map = torch.stack(self.cross_attns, dim=1).mean(1)  # (B, N, dim)
        B = attn_map.shape[0]
        res = int(np.sqrt(attn_map.shape[-2]))
        attn_map = attn_map.reshape(-1, res, res, attn_map.shape[-1])
        image = attn_map[..., idx]
        if isinstance(idx, list):
            image = image.sum(-1)
        image_min = image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        image_max = image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        image = (image - image_min) / (image_max - image_min)
        return image

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross:
            # save cross attention map with res 16 * 16
            if attn.shape[1] == 16 * 16:
                self.cross_attns.append(attn.reshape(-1, num_heads, *attn.shape[-2:]).mean(1))

        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u_source = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_source = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        if len(self.cross_attns) == 0:
            self.self_attns_mask = None
            out_u_target = self.attn_batch(qu[-num_heads:], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_target = self.attn_batch(qc[-num_heads:], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
        else:
            mask = self.aggregate_cross_attn_map(idx=self.ref_token_idx)  # (2, H, W)
            mask_source = mask[-2]  # (H, W)
            res = int(np.sqrt(q.shape[1]))
            self.self_attns_mask = F.interpolate(mask_source.unsqueeze(0).unsqueeze(0), (res, res)).flatten()
            if self.mask_save_dir is not None:
                H = W = int(np.sqrt(self.self_attns_mask.shape[0]))
                mask_image = self.self_attns_mask.reshape(H, W).unsqueeze(0)
                save_image(mask_image, os.path.join(self.mask_save_dir, f"mask_s_{self.cur_step}_{self.cur_att_layer}.png"))
            out_u_target = self.attn_batch(qu[-num_heads:], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_target = self.attn_batch(qc[-num_heads:], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        if self.self_attns_mask is not None:
            mask = self.aggregate_cross_attn_map(idx=self.cur_token_idx)  # (2, H, W)
            mask_target = mask[-1]  # (H, W)
            res = int(np.sqrt(q.shape[1]))
            spatial_mask = F.interpolate(mask_target.unsqueeze(0).unsqueeze(0), (res, res)).reshape(-1, 1)
            if self.mask_save_dir is not None:
                H = W = int(np.sqrt(spatial_mask.shape[0]))
                mask_image = spatial_mask.reshape(H, W).unsqueeze(0)
                save_image(mask_image, os.path.join(self.mask_save_dir, f"mask_t_{self.cur_step}_{self.cur_att_layer}.png"))
            # binarize the mask
            thres = self.thres
            spatial_mask[spatial_mask >= thres] = 1
            spatial_mask[spatial_mask < thres] = 0
            out_u_target_fg, out_u_target_bg = out_u_target.chunk(2)
            out_c_target_fg, out_c_target_bg = out_c_target.chunk(2)

            out_u_target = out_u_target_fg * spatial_mask + out_u_target_bg * (1 - spatial_mask)
            out_c_target = out_c_target_fg * spatial_mask + out_c_target_bg * (1 - spatial_mask)

            # set self self-attention mask to None
            self.self_attns_mask = None

        out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)
        return out
