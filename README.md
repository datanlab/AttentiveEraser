# Attentive Eraser: Unleashing Diffusion Model’s Object Removal Potential via Self-Attention Redirection Guidance
![Attentive Eraser](http://industry-algo.oss-cn-zhangjiakou.aliyuncs.com/tmp/tiankai/RG.png "The overview of our proposed Attentive Eraser")

## Introduction
Attentive Eraser is a novel tuning-free method that enhances object removal capabilities in pre-trained diffusion models. This official implementation demonstrates the method's efficacy, leveraging altered self-attention mechanisms to prioritize background over foreground in the image generation process.

## Getting Started
```bash
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
cd stable-diffusion-xl-base-1.0
pip install -r requirements.txt
python main.py
```

## Citation
If you find this project useful in your research, please consider citing it:

```bibtex
@inproceedings{sun2025attentive,
  title={Attentive Eraser: Unleashing Diffusion Model’s Object Removal Potential via Self-Attention Redirection Guidance},
  author={Sun, Wenhao and Cui, Benlei and Dong, Xue-Mei and Tang, Jingqun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}

