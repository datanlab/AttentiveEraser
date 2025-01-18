# Attentive Eraser: Unleashing Diffusion Model’s Object Removal Potential via Self-Attention Redirection Guidance (AAAI2025 Oral)
[![arXiv](https://img.shields.io/badge/AttentiveEraser-arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.12974) [![arXiv](https://img.shields.io/badge/AttentiveEraser-paper-57babb.svg)](https://arxiv.org/pdf/2412.12974) 
[![GitHub Repo stars](https://img.shields.io/github/stars/Anonym0u3/AttentiveEraser?style=plastic&logo=github)](https://github.com/Anonym0u3/AttentiveEraser)
<p align="center">
  <img src="https://raw.githubusercontent.com/Anonym0u3/Images/refs/heads/main/GifMerge_1024x1024.gif" alt="GIF 1" width="30%">
  <img src="https://raw.githubusercontent.com/Anonym0u3/Images/refs/heads/main/GifMerge_512x512.gif" alt="GIF 2" width="30%">
  <img src="https://raw.githubusercontent.com/Anonym0u3/Images/refs/heads/main/GifMerge_768x768.gif" alt="GIF 3" width="30%">
</p>

## Introduction
Attentive Eraser is a novel tuning-free method that enhances object removal capabilities in pre-trained diffusion models. This official implementation demonstrates the method's efficacy, leveraging altered self-attention mechanisms to prioritize background over foreground in the image generation process.
![Attentive Eraser](http://industry-algo.oss-cn-zhangjiakou.aliyuncs.com/tmp/tiankai/RG.png "The overview of our proposed Attentive Eraser")

## Downloading Pretrained Diffusion Models
The pretrained diffusion models can be downloaded from the link below for offline loading.  
SDXL: <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>  
SD2.1: <https://huggingface.co/stabilityai/stable-diffusion-2-1-base>

## Getting Started
```bash
git clone https://github.com/Anonym0u3/AttentiveEraser.git
cd AttentiveEraser
conda create -n AE python=3.9
conda activate AE
pip install -r requirements.txt
# run SDXL+SIP
python main.py
```

More experimental versions can be found in the `notebook` folder.
## Citation
If you find this project useful in your research, please consider citing it:

```bibtex
@inproceedings{sun2025attentive,
  title={Attentive Eraser: Unleashing Diffusion Model’s Object Removal Potential via Self-Attention Redirection Guidance},
  author={Sun, Wenhao and Cui, Benlei and Dong, Xue-Mei and Tang, Jingqun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```
## Acknowledgments

This repository is built upon and utilizes the following repositories:

- **[MasaCtrl](https://github.com/TencentARC/MasaCtrl)**
- **[diffusers](https://github.com/huggingface/diffusers)**

We would like to express our sincere thanks to the authors and contributors of these repositories for their incredible work, which greatly enhanced the development of this repository.
