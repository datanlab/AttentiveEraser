# Attentive Eraser: Official Implementation
![Attentive Eraser](http://industry-algo.oss-cn-zhangjiakou.aliyuncs.com/tmp/tiankai/RG.png "The overview of our proposed Attentive Eraser")

## Introduction
"Attentive Eraser: Unleashing Diffusion Model’s Object Removal Potential via Self-Attention Redirection Guidance" is a novel tuning-free method that enhances object removal capabilities in pre-trained diffusion models. This official implementation demonstrates the method's efficacy, leveraging altered self-attention mechanisms to prioritize background over foreground in the image generation process.

## Getting Started
```bash
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
cd stable-diffusion-xl-base-1.0
pip install -r requirements.txt
python main.py
