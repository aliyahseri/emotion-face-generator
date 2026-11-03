# Emotion Face Generator

## Introduction
This project generates human faces showing a chosen emotion. Three generative models are trained and compared, each conditioned on 8 emotion labels: anger, contempt, disgust, fear, happy, neutral, sad and surprise. A Gradio web app lets you pick a model and an emotion and generate faces instantly.

## Models Used
* Conditional VAE (128×128 output)
* Conditional GAN (64×64 output)
* VAE-GAN (64×64 output)
* ResNet-18 emotion classifier for evaluation

## Training
* Conditional GAN: `python training/train_cgan.py --data_path dataset`
* VAE-GAN: `python training/train_vaegan.py --data_path dataset`
* Conditional VAE: `python training/train_vae.py --data_path dataset`

## Usage
* Web app: `python app/app.py`, then open http://localhost:7860
* Generate from the command line: `python inference/generate.py --model cgan --emotion happy --n 4`
* Evaluate all models (FID, PSNR, SSIM): `python evaluation/evaluate_all.py`

## Dataset
AffectNet-based facial expression images (96×96, 8 emotion classes), placed in `dataset/train/<emotion>/` and `dataset/test/<emotion>/`.

## Requirements
* python 3.10+
* pytorch
* torchvision
* gradio
* numpy
* scipy
* scikit-image
* matplotlib
