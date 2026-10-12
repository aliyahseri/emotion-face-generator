"""
Simple Gradio GUI for Emotion Face Generator evaluation.
Displays results and generates images with all 3 models.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import tempfile

# ABSOLUTE PATHS
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAE_CHECKPOINT = BASE + "/outputs/vae_checkpoints/vae_final.pt"
CGAN_CHECKPOINT = BASE + "/outputs/cgan_checkpoints/cgan_epoch_80.pt"
VAEGAN_CHECKPOINT = BASE + "/outputs/vaegan_checkpoints/vaegan_epoch_60.pt"

# Add BASE to path for imports
sys.path.insert(0, BASE)

from models.vae import ConditionalVAE
from models.cgan_generator import ConditionalGANGenerator
from models.vaegan import VAEGAN
from evaluate_all import CACHED_RESULTS

# Constants
EMOTIONS = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models on startup
print("Loading models...")
vae_model = ConditionalVAE(num_classes=NUM_CLASSES, latent_dim=256).to(DEVICE)
vae_checkpoint = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
vae_model.load_state_dict(vae_checkpoint['model_state_dict'])
vae_model.eval()

cgan_model = ConditionalGANGenerator(nz=128, num_classes=NUM_CLASSES).to(DEVICE)
cgan_checkpoint = torch.load(CGAN_CHECKPOINT, map_location=DEVICE)
cgan_model.load_state_dict(cgan_checkpoint['generator_state_dict'])
cgan_model.eval()

vaegan_model = VAEGAN(latent_dim=128, num_classes=NUM_CLASSES).to(DEVICE)
vaegan_checkpoint = torch.load(VAEGAN_CHECKPOINT, map_location=DEVICE)
vaegan_model.encoder.load_state_dict(vaegan_checkpoint['encoder_state_dict'])
vaegan_model.decoder.load_state_dict(vaegan_checkpoint['decoder_state_dict'])
vaegan_model.encoder.eval()
vaegan_model.decoder.eval()
print("Models loaded successfully!")


def generate_vae_images(label_idx, n):
    """Generate images using VAE."""
    with torch.no_grad():
        z = torch.randn(n, 256).to(DEVICE)
        labels = torch.full((n,), label_idx, dtype=torch.long).to(DEVICE)
        samples = vae_model.decode(z, labels)
    return samples


def generate_cgan_images(label_idx, n):
    """Generate images using cGAN."""
    with torch.no_grad():
        noise = torch.randn(n, 128).to(DEVICE)
        labels = torch.full((n,), label_idx, dtype=torch.long).to(DEVICE)
        samples = cgan_model(noise, labels)
    return samples


def generate_vaegan_images(label_idx, n):
    """Generate images using VAE-GAN."""
    with torch.no_grad():
        z = torch.randn(n, 128).to(DEVICE)
        labels = torch.full((n,), label_idx, dtype=torch.long).to(DEVICE)
        samples = vaegan_model.decoder(z, labels)
    return samples


def tensor_to_pil(images):
    """Convert tensor images to PIL format for Gradio."""
    images = (images + 1.0) / 2.0
    images = torch.clamp(images, 0, 1)
    images_np = images.cpu().numpy()
    images_np = np.transpose(images_np, (0, 2, 3, 1))
    pil_images = []
    for img in images_np:
        pil_images.append((img * 255).astype(np.uint8))
    return pil_images


def create_results_table():
    """Create results table from CACHED_RESULTS."""
    data = [
        ["VAE", CACHED_RESULTS["VAE"]["FID"], CACHED_RESULTS["VAE"]["PSNR"], CACHED_RESULTS["VAE"]["SSIM"], "Best PSNR & SSIM"],
        ["cGAN", CACHED_RESULTS["cGAN"]["FID"], CACHED_RESULTS["cGAN"]["PSNR"], CACHED_RESULTS["cGAN"]["SSIM"], "-"],
        ["VAE-GAN", CACHED_RESULTS["VAE-GAN"]["FID"], CACHED_RESULTS["VAE-GAN"]["PSNR"], CACHED_RESULTS["VAE-GAN"]["SSIM"], "Best FID"]
    ]
    headers = ["Model", "FID ↓", "PSNR (dB) ↑", "SSIM ↑", "Best Metric"]
    return data, headers


def create_comparison_charts():
    """Create 3 bar charts side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = ['VAE', 'cGAN', 'VAE-GAN']
    colors = ['blue', 'orange', 'green']
    
    # FID chart
    fid_values = [CACHED_RESULTS[m]["FID"] for m in models]
    axes[0].bar(models, fid_values, color=colors, alpha=0.7)
    axes[0].set_title('FID Score (lower=better)', fontweight='bold')
    axes[0].set_ylabel('FID')
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(fid_values):
        axes[0].text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # PSNR chart
    psnr_values = [CACHED_RESULTS[m]["PSNR"] for m in models]
    axes[1].bar(models, psnr_values, color=colors, alpha=0.7)
    axes[1].set_title('PSNR in dB (higher=better)', fontweight='bold')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(psnr_values):
        axes[1].text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # SSIM chart
    ssim_values = [CACHED_RESULTS[m]["SSIM"] for m in models]
    axes[2].bar(models, ssim_values, color=colors, alpha=0.7)
    axes[2].set_title('SSIM Score (higher=better)', fontweight='bold')
    axes[2].set_ylabel('SSIM')
    axes[2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(ssim_values):
        axes[2].text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        plt.savefig(f.name, dpi=150, bbox_inches='tight')
        temp_path = f.name
    
    plt.close()
    return temp_path


def generate_images(emotion, num_images):
    """Generate images with all 3 models."""
    emotion_idx = EMOTIONS.index(emotion)
    
    vae_images = generate_vae_images(emotion_idx, num_images)
    cgan_images = generate_cgan_images(emotion_idx, num_images)
    vaegan_images = generate_vaegan_images(emotion_idx, num_images)
    
    vae_pil = tensor_to_pil(vae_images)
    cgan_pil = tensor_to_pil(cgan_images)
    vaegan_pil = tensor_to_pil(vaegan_images)
    
    return vae_pil, cgan_pil, vaegan_pil


def load_sample_image(model_name):
    """Load sample image for a model."""
    sample_paths = {
        'VAE': BASE + '/outputs/vae_samples/epoch_100.png',
        'cGAN': BASE + '/outputs/cgan_samples/epoch_80.png',
        'VAE-GAN': BASE + '/outputs/vaegan_samples/epoch_60.png'
    }
    
    path = sample_paths.get(model_name)
    if path and os.path.exists(path):
        return path
    return None


def create_gui():
    """Create the Gradio GUI interface."""
    
    with gr.Blocks(title="Emotion Face Generator - Evaluation GUI") as interface:
        gr.Markdown("# 🎭 Emotion Face Generator - Evaluation Dashboard")
        gr.Markdown("Compare VAE, cGAN, and VAE-GAN emotion face generation models.")
        
        with gr.Tabs():
            # Tab 1: Results Dashboard
            with gr.Tab("Results Dashboard"):
                gr.Markdown("## Model Evaluation Results")
                
                # Results table
                table_data, table_headers = create_results_table()
                gr.Dataframe(value=table_data, headers=table_headers, label="Evaluation Metrics")
                
                # Comparison charts
                charts_image = gr.Image(value=create_comparison_charts(), label="Metric Comparison Charts")
            
            # Tab 2: Generate & Compare
            with gr.Tab("Generate & Compare"):
                gr.Markdown("## Generate Images with All Models")
                
                with gr.Row():
                    emotion_dropdown = gr.Dropdown(choices=EMOTIONS, label="Select Emotion", value="happy")
                    num_slider = gr.Slider(minimum=1, maximum=4, value=4, step=1, label="Number of Images")
                
                generate_btn = gr.Button("🎨 Generate with All 3 Models", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        vae_label = gr.Markdown(f"### VAE (FID: {CACHED_RESULTS['VAE']['FID']:.2f})")
                        vae_gallery = gr.Gallery(label="VAE Generated Images", columns=2, height=400)
                    
                    with gr.Column():
                        cgan_label = gr.Markdown(f"### cGAN (FID: {CACHED_RESULTS['cGAN']['FID']:.2f})")
                        cgan_gallery = gr.Gallery(label="cGAN Generated Images", columns=2, height=400)
                    
                    with gr.Column():
                        vaegan_label = gr.Markdown(f"### VAE-GAN (FID: {CACHED_RESULTS['VAE-GAN']['FID']:.2f})")
                        vaegan_gallery = gr.Gallery(label="VAE-GAN Generated Images", columns=2, height=400)
                
                generate_btn.click(
                    generate_images,
                    inputs=[emotion_dropdown, num_slider],
                    outputs=[vae_gallery, cgan_gallery, vaegan_gallery]
                )
            
            # Tab 3: Model Comparison
            with gr.Tab("Model Comparison"):
                gr.Markdown("## Metric Explanations")
                
                gr.Markdown("""
                **FID (Fréchet Inception Distance):** Measures how similar generated images are to real images. Lower is better.
                
                **PSNR (Peak Signal-to-Noise Ratio):** Measures pixel-level reconstruction quality in decibels. Higher is better.
                
                **SSIM (Structural Similarity Index):** Measures structural similarity to real images on a scale of 0 to 1. Higher is better.
                """)
                
                gr.Markdown("### Best Sample Images")
                
                with gr.Row():
                    vae_sample = gr.Image(label="VAE - Best Sample (FID: 211.11, PSNR: 10.85, SSIM: 0.2724)")
                    cgan_sample = gr.Image(label="cGAN - Best Sample (FID: 372.65, PSNR: 9.09, SSIM: 0.0964)")
                    vaegan_sample = gr.Image(label="VAE-GAN - Best Sample (FID: 144.70, PSNR: 9.29, SSIM: 0.1141)")
                
                # Load sample images
                vae_sample.value = load_sample_image('VAE')
                cgan_sample.value = load_sample_image('cGAN')
                vaegan_sample.value = load_sample_image('VAE-GAN')
        
        gr.Markdown("---")
        gr.Markdown(f"**Device:** {str(DEVICE)}")
    
    return interface


if __name__ == "__main__":
    gui = create_gui()
    gui.launch(server_name="0.0.0.0", server_port=7861)
