# Emotion Face Generator

Emotion-conditioned face generation in PyTorch: three generative models (conditional VAE, conditional GAN and VAE-GAN) trained on an AffectNet-derived dataset with 8 emotion classes, plus a Gradio demo and an evaluation pipeline (FID, PSNR, SSIM, emotion-classifier accuracy).

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-4.x-F97316?logo=gradio&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

The goal is to generate face images for a given emotion and compare how three conditional generative architectures handle the task. Every model receives the emotion label through a learned `nn.Embedding`, so a single network can generate any of the 8 classes: **anger, contempt, disgust, fear, happy, neutral, sad, surprise**.

Key features (all present in the code):

- **Three conditional generators** trained on the same data:
  - **Conditional VAE** (`models/vae.py`): 128×128 input and output, 5-layer conv encoder (3→64→128→256→512→512), latent size 256, 128-d label embedding concatenated to both encoder features and latent vector, loss = MSE + β·KL with β = 4.
  - **Conditional GAN** (`models/cgan_generator.py`, `models/cgan_discriminator.py`): generator takes 128-d noise + 64-d label embedding → FC to 256×4×4 → 4 transposed-conv blocks → 3×64×64 (Tanh). The discriminator projects the label embedding to an extra 64×64 input channel (4 input channels), then 4 conv layers (64→128→256→512) and a sigmoid output.
  - **VAE-GAN** (`models/vaegan.py`): 64×64, latent size 128, 4-layer conv encoder and transposed-conv decoder, both conditioned on a 64-d label embedding, plus a label-conditioned discriminator with the same design as the cGAN discriminator. Trained with MSE + KL (β = 1) + 0.1 × adversarial loss on reconstructions and random samples.
- **Training scripts** for each model with periodic checkpoints, per-emotion sample grids and loss-curve plots.
- **Command-line generation** for one emotion, all emotions, or a latent interpolation between two emotions (VAE and VAE-GAN).
- **Gradio web app** (`app/app.py`): choose model (VAE / cGAN / VAE-GAN), emotion and 1–8 faces, with an optional "blend" slider and generation timing.
- **Evaluation**: FID (InceptionV3 features), PSNR/SSIM, and emotion accuracy of generated images measured with a separately trained classifier; a second Gradio dashboard (`evaluation/eval_gui.py`) shows the comparison side by side.

## Tech stack

| Component | Used for |
|---|---|
| PyTorch, torchvision | Models, training loops, data loading, pretrained InceptionV3 (FID) and ResNet18 (emotion classifier) |
| Gradio | Interactive demo (port 7860) and evaluation dashboard (port 7861) |
| SciPy | Matrix square root for the FID computation |
| scikit-image | SSIM in `evaluate_all.py` |
| torchmetrics | PSNR/SSIM in `evaluate.py` |
| Matplotlib | Loss curves and metric comparison charts |
| NumPy, Pillow, tqdm | Image conversion, I/O, progress bars |

Design choices visible in the code: label conditioning by embedding concatenation (generator/decoder) and by a projected spatial label map (discriminators); label smoothing (real = 0.9) and a lower discriminator learning rate in the GAN setups to keep D from overpowering G; gradient clipping on the cGAN generator.

## Demo / Screenshots

<!-- TODO: add a screenshot at docs/screenshot.png -->
![Gradio app screenshot](docs/screenshot.png)

*Placeholder: capture the Gradio app (`python app/app.py`) after generating a set of faces, and save it as `docs/screenshot.png`.*

Sample grids (one image per emotion, produced during training in `outputs/*_samples/`):

<!-- TODO: add sample grids at docs/samples_vae.png, docs/samples_cgan.png, docs/samples_vaegan.png -->
| VAE | cGAN | VAE-GAN |
|---|---|---|
| ![VAE samples](docs/samples_vae.png) | ![cGAN samples](docs/samples_cgan.png) | ![VAE-GAN samples](docs/samples_vaegan.png) |

*Placeholder: copy a representative grid for each model (for example `outputs/vae_samples/epoch_100.png`, `outputs/cgan_samples/epoch_80.png`, `outputs/vaegan_samples/epoch_60.png`) into `docs/`. Note that grids saved by the training scripts look washed out because of a double denormalization (see Known issues); grids from the Gradio app or `eval_gui.py` are rendered correctly.*

## Dataset

**The dataset is not included in this repository.**

- **Source:** the Kaggle dataset "AffectNet Training Data" (`noamsegal/affectnet-training-data`), an AffectNet-derived, re-labelled subset: https://www.kaggle.com/datasets/noamsegal/affectnet-training-data [TODO: confirm dataset source]. AffectNet itself is subject to its own academic-use terms; check the Kaggle page and the original AffectNet license before use. [TODO: confirm license]
- **Content:** RGB face crops, 96×96 px, 8 emotion classes stored as one folder per class.
- **Size (local copy):** about 16.1k images in `Train` and about 14.5k in `Test`.
- **Extra file:** the download also contains a `labels.csv` (`pth,label,relFCs`). The code does not use it; labels come from the folder names only.

The data loader (`data/dataloader.py`) expects `<data_path>/<split>/<class>/*.jpg|*.jpeg|*.png`, with `split` = `train` or `test` in **lowercase**, resizes every image to 64×64 and normalizes to [-1, 1]. Class indices are assigned by sorted folder name. The Kaggle archive extracts as `Train/` and `Test/`, and some class folders under `Test/` are capitalized (`Anger`, `Contempt`). On Windows (case-insensitive) this works as-is; on Linux/macOS, rename the folders to lowercase.

Place it at the repository root like this:

```
emotion-face-generator/
└── dataset/
    ├── train/
    │   ├── anger/
    │   ├── contempt/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    └── test/
        └── (same 8 folders)
```

### Model checkpoints

**Checkpoints are not included** (each file is roughly 54–256 MB). They are produced by the training scripts, which write to `outputs/` relative to the directory you run them from (run everything from the repository root). The paths the app and evaluation scripts look for:

| File | Produced by | Used by |
|---|---|---|
| `outputs/vae_checkpoints/vae_final.pt` (and `vae_epoch_N.pt`) | `training/train_vae.py` | `app/app.py`, `inference/generate.py` (default for `--model vae`), `evaluation/evaluate_all.py`, `evaluation/eval_gui.py` |
| `outputs/cgan_checkpoints/cgan_epoch_80.pt` | `training/train_cgan.py` (every 10 epochs) | `app/app.py` (preferred), `evaluate_all.py`, `eval_gui.py` |
| `outputs/cgan_checkpoints/generator_final.pt`, `cgan_final.pt` | `training/train_cgan.py` (end of training) | `inference/generate.py` (default for `--model cgan`) |
| `outputs/vaegan_checkpoints/vaegan_epoch_60.pt` | `training/train_vaegan.py` (every 20 epochs) | `app/app.py` (preferred), `evaluate_all.py`, `eval_gui.py` |
| `outputs/vaegan_checkpoints/vaegan_final.pt` | `training/train_vaegan.py` (end of training) | `inference/generate_vaegan.py` (default) |
| `outputs/classifier/emotion_classifier.pt` | `evaluation/evaluate_all.py` | `evaluate_all.py --skip_classifier_training` |

Pretrained weights may be published later as a GitHub Release or on the Hugging Face Hub. [TODO: confirm hosting and add download link]

## Installation

```bash
git clone https://github.com/aliyahseri/emotion-face-generator.git
cd emotion-face-generator

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version first (see https://pytorch.org/get-started/locally/). All scripts fall back to CPU automatically.

## Usage

Run every command from the repository root so that the relative `outputs/` and `dataset/` paths resolve.

### Train

```bash
# Conditional GAN (defaults: 150 epochs, batch 64, lr 1e-4, D lr = 0.3 x lr, checkpoint every 10 epochs)
python training/train_cgan.py --data_path dataset

# Resume a cGAN run
python training/train_cgan.py --data_path dataset --resume outputs/cgan_checkpoints/cgan_epoch_80.pt

# VAE-GAN (defaults: 200 epochs, batch 64, lr 2e-4, latent 128, checkpoint every 20 epochs)
python training/train_vaegan.py --data_path dataset

# Conditional VAE (defaults: 100 epochs, batch 64, lr 5e-4, checkpoint every 10 epochs)
# Note: currently fails with a shape mismatch, see Known issues
python training/train_vae.py --data_path dataset
```

Common flags: `--epochs`, `--batch_size`, `--lr`, `--save_every`, `--output_dir` (default `outputs`).

### Generate from the command line

```bash
# VAE or cGAN, one emotion
python inference/generate.py --model vae --emotion happy --n 4
python inference/generate.py --model cgan --emotion sad --n 6

# All emotions (omit --emotion)
python inference/generate.py --model cgan --n 4 --output_dir all_emotions

# Custom checkpoint
python inference/generate.py --model cgan --checkpoint_path outputs/cgan_checkpoints/cgan_epoch_80.pt --emotion fear

# VAE latent interpolation
python inference/generate.py --model vae --interpolate --from_emotion happy --to_emotion sad --steps 10

# VAE-GAN (defaults to outputs/vaegan_checkpoints/vaegan_final.pt)
python inference/generate_vaegan.py --emotion surprise --n 8
python inference/generate_vaegan.py --interpolate --from_emotion happy --to_emotion sad
```

`generate.py` writes to `generated_samples/` by default; `generate_vaegan.py` writes to `outputs/vaegan_generated/`. Emotion names must match the dataset folder names stored in the checkpoint (e.g. `anger`, not `angry`).

How interpolation works: two random latent vectors are drawn and linearly interpolated, while the emotion label switches from the first emotion to the second at the halfway point (α = 0.5). It is a walk through latent space with a hard label switch, not a gradual emotion morph.

### Web app

```bash
python app/app.py
```

Open http://localhost:7860. Pick a model, an emotion and the number of faces (1–8). The "Blend" controls behave differently per model:

- **VAE:** pixel-space average of two independently sampled batches (primary and blend emotion), weighted by the slider. The two faces are unrelated, so this is an image overlay, not an emotion mix.
- **VAE-GAN:** interpolates between two random latent vectors with the primary label only; the selected blend emotion is not used.

The app listens on `0.0.0.0` (all network interfaces); see Known issues.

### Evaluate

```bash
# Single model with torchmetrics PSNR/SSIM and a small CNN emotion classifier
python evaluation/evaluate.py --model_type cgan --checkpoint_path outputs/cgan_checkpoints/cgan_epoch_80.pt --data_path dataset --num_samples 500

# All three models: FID (InceptionV3, ~500 samples), PSNR/SSIM (skimage), ResNet18 emotion classifier
python evaluation/evaluate_all.py                              # trains the classifier (20 epochs) first
python evaluation/evaluate_all.py --skip_classifier_training   # reuses outputs/classifier/emotion_classifier.pt

# Evaluation dashboard (port 7861)
python evaluation/eval_gui.py
```

`evaluate_all.py` saves `evaluation/results.json` and `evaluation/comparison_plots.png`.

## Results

The only saved artifact is a (truncated) `results.json` from the author's run, which records a VAE FID of about 211.1. The table below reproduces the values hard-coded as `CACHED_RESULTS` in `evaluation/evaluate_all.py`; they come from the author's evaluation run with that script, not from a file shipped in this repo. [TODO: confirm]

| Model | Checkpoint | FID (lower is better) | PSNR (dB) | SSIM |
|---|---|---|---|---|
| VAE | `vae_final.pt` | 211.11 | 10.85 | 0.2724 |
| cGAN | `cgan_epoch_80.pt` | 372.65 | 9.09 | 0.0964 |
| VAE-GAN | `vaegan_epoch_60.pt` | **144.70** | 9.29 | 0.1141 |

How to read these numbers:

- On FID, **VAE-GAN is the best of the three**, the VAE second, and the cGAN clearly worst. All three values are high in absolute terms, so the samples are recognisable as faces but far from realistic.
- The models are **not directly comparable**: the VAE is evaluated at 128×128 against real images resized to 128, while the cGAN and VAE-GAN are evaluated at 64×64.
- FID on ~500 samples (62 per class) is noisy and biased upward compared with the usual 10k–50k-sample FID.
- PSNR/SSIM compare each generated image with an **unrelated** real test image of the same class. For unconditional sampling this mostly rewards blurry, average-looking outputs (which is why the VAE scores highest), so it is not a meaningful quality measure here.
- Emotion-classifier accuracy is computed by the script but was not recorded in `CACHED_RESULTS`. [TODO: confirm / add if available]

## Project structure

```
emotion-face-generator/
├── app/app.py                    # Gradio demo (VAE / cGAN / VAE-GAN), port 7860
├── data/dataloader.py            # EmotionDataset + get_dataloader (64x64, [-1, 1])
├── models/
│   ├── vae.py                    # ConditionalVAE (128x128, latent 256)
│   ├── cgan_generator.py         # ConditionalGANGenerator (noise 128 + label 64 -> 3x64x64)
│   ├── cgan_discriminator.py     # ConditionalGANDiscriminator (label as extra channel)
│   └── vaegan.py                 # VAEGAN encoder / decoder / discriminator (64x64, latent 128)
├── training/
│   ├── train_vae.py
│   ├── train_cgan.py
│   └── train_vaegan.py
├── inference/
│   ├── generate.py               # VAE / cGAN generation and interpolation
│   └── generate_vaegan.py        # VAE-GAN generation and interpolation
├── evaluation/
│   ├── evaluate.py               # FID, torchmetrics PSNR/SSIM, small CNN classifier
│   ├── evaluate_all.py           # FID, skimage PSNR/SSIM, ResNet18 classifier, all 3 models
│   └── eval_gui.py               # Gradio evaluation dashboard, port 7861
├── utils/helpers.py              # seeding, image grids, FID helpers
├── requirements.txt
└── LICENSE
```

## Challenges & what I learned

- **Conditioning a generator on discrete labels.** The project uses two different mechanisms: concatenating a learned label embedding to the noise/latent vector (generators, decoders, encoders) and projecting the embedding into a full-resolution label map fed as an extra image channel (discriminators).
- **Balancing GAN training.** The cGAN script uses one-sided label smoothing, a discriminator learning rate of 0.3× the generator's, and gradient-norm clipping on the generator; the VAE-GAN trains D only every second step at half the learning rate. These are typical counter-measures against a discriminator that wins too early. [TODO: confirm what motivated each change]
- **VAE blur vs. GAN sharpness.** A plain VAE with MSE loss and a strong KL weight (β = 4) gives smooth but blurry samples; adding an adversarial term (VAE-GAN) produced the best FID of the three, which matches the motivation behind VAE-GAN architectures.
- **Evaluating generative models honestly.** FID needs many samples to be stable, image resolution must match across models, and pixel metrics like PSNR/SSIM only make sense for reconstruction, not for free sampling.
- **Working with a noisy, imbalanced dataset.** Class sizes range from about 1.2k (disgust) to 3.1k (sad) training images, and the dataset's own `labels.csv` disagrees with the folder labels for some images. [TODO: confirm]

## Known issues & limitations

- `training/train_vae.py` fails with a shape mismatch: the data loader resizes to 64×64 (`data/dataloader.py:62`) but `ConditionalVAE` expects 128×128 input (`models/vae.py:42`, encoder output 512×4×4).
- `training/train_vae.py` ignores `--beta` and `--latent_dim`; the model is always built with `latent_dim=256, beta=4.0` (`train_vae.py:58-62`).
- `training/train_cgan.py:120-121` skips every odd batch for both D and G (`if batch_idx % 2 != 0: continue`), so half of each epoch is unused.
- `training/train_cgan.py:114` hard-codes the real label as 0.9; the `--real_label` argument is ignored.
- `training/train_cgan.py:87` resumes at `saved_epoch + 1`, but the saved value is already the next epoch index, so one epoch is skipped on resume.
- `utils/helpers.py:42` `save_image_grid` denormalizes again although all callers pass already-denormalized tensors, so saved grids look washed out.
- `app/app.py:66` and `app/app.py:92` have an early `return` in `load_models`: if `vae_final.pt` or `cgan_epoch_80.pt` is missing, the remaining models are never loaded, even when other checkpoints exist.
- `app/app.py:59,85,119` pick the "latest" checkpoint with a text sort, so `epoch_90` sorts after `epoch_100`.
- `app/app.py:197-213` VAE-GAN blending ignores the selected blend emotion; the VAE blend (`app/app.py:184-195`) is a pixel average of two unrelated faces.
- `evaluation/evaluate.py:220` raises `KeyError` for `generator_final.pt` (it reads `generator_state_dict`, but that file stores `state_dict`); use a `cgan_epoch_N.pt` checkpoint instead.
- `evaluation/evaluate.py:264` PSNR/SSIM fail for the VAE because generated images are 128×128 and real images 64×64.
- `evaluation/evaluate.py:345-346` writing the results file fails with a `TypeError` because the nested `per_emotion_accuracy` dict is formatted with `:.4f`.
- `evaluation/evaluate_all.py:155-171` preprocessing mismatch: the ResNet18 classifier is trained on 64×64 images normalized to [-1, 1] but classifies generated images resized to 224 with ImageNet normalization; the "best" classifier epoch is also chosen on training accuracy (`evaluate_all.py:133`), not validation accuracy.
- `evaluation/eval_gui.py:33-51` loads all three checkpoints at import time, so it crashes if any is missing.
- `data/dataloader.py:21` uses lowercase `train`/`test`, but the Kaggle archive has `Train`/`Test` and `Test/Anger`, `Test/Contempt`; the original setup relied on a case-insensitive file system.
- `utils/helpers.py:143,213` use the deprecated `pretrained=True` torchvision argument, and `extract_inception_features` reloads InceptionV3 on every call.
- `app/app.py:390` and `evaluation/eval_gui.py:257` bind to `0.0.0.0`, which exposes the app to the local network; use `127.0.0.1` for local-only use.
- `inference/generate.py` defines `--nz` and `--latent_dim` but never uses them; its fallback label list has only 7 emotions (`angry`, ...), which does not match the 8-class checkpoints.
- Image quality is limited: 64×64 output (128×128 for the VAE) and high FID for all models.

## Future improvements

- Fix the known issues above, starting with the VAE input-size mismatch, the skipped cGAN batches and the double denormalization.
- Make the evaluation fair: same resolution for all models, a held-out validation split for the classifier, matching classifier preprocessing, and FID on several thousand samples (e.g. with `torchmetrics.image.fid` or `pytorch-fid`).
- Replace the pixel-space "blend" with real conditional interpolation (interpolate label embeddings with a fixed latent vector).
- Try stronger GAN training (spectral normalization, hinge loss, projection discriminator) and higher resolution.
- Publish pretrained checkpoints and add a small smoke test for each entry point.

## Credits / Acknowledgements

- Dataset: "AffectNet Training Data" on Kaggle (`noamsegal/affectnet-training-data`), derived from AffectNet: A. Mollahosseini, B. Hasani, M. H. Mahoor, "AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild", IEEE Transactions on Affective Computing, 2017. [TODO: confirm dataset source]
- Architectures follow the ideas of: Kingma & Welling, "Auto-Encoding Variational Bayes" (2014); Higgins et al., "β-VAE" (2017); Mirza & Osindero, "Conditional Generative Adversarial Nets" (2014); Larsen et al., "Autoencoding beyond pixels using a learned similarity metric" (VAE-GAN, 2016).
- FID: Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (2017).
- Pretrained InceptionV3 and ResNet18 weights from torchvision.
- Built with PyTorch and Gradio.

## License

Released under the MIT License; see [LICENSE](LICENSE). The dataset and any pretrained ImageNet weights are subject to their own licenses.

## Author

**Aliyah Seri**

- GitHub: [aliyahseri](https://github.com/aliyahseri)
