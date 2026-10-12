import os
import time
import torch
import gradio as gr
import numpy as np
import glob
from PIL import Image

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import get_dataloader
from models.vae import ConditionalVAE
from models.cgan_generator import ConditionalGANGenerator
from models.vaegan import VAEGAN
from utils.helpers import set_seed, denormalize, get_label_map


class EmotionFaceGeneratorApp:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_seed(42)
        
        # Initialize models as None
        self.vae_model = None
        self.cgan_model = None
        self.vaegan_model = None
        self.label_to_idx = None
        self.idx_to_label = None
        
        # Load models on startup
        self.load_models()
        
        # Emotion list (8 emotions in alphabetical order)
        self.emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    def load_models(self):
        """Load trained VAE and cGAN models."""
        print("Loading models...")
        
        # Get label mapping
        try:
            self.label_to_idx = get_label_map(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset"))
            self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        except:
            # Default emotion labels (8 emotions in alphabetical order)
            self.emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            self.label_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions)}
            self.idx_to_label = {idx: emotion for emotion, idx in self.label_to_idx.items()}
        
        num_classes = len(self.label_to_idx)
        
        # Load VAE
        try:
            vae_checkpoint_path = "outputs/vae_checkpoints/vae_final.pt"
            if not os.path.exists(vae_checkpoint_path):
                # Find latest VAE checkpoint
                vae_checkpoints = sorted(glob.glob("outputs/vae_checkpoints/vae_epoch_*.pt"))
                if vae_checkpoints:
                    vae_checkpoint_path = vae_checkpoints[-1]
                    print(f"Loading VAE checkpoint: {vae_checkpoint_path}")
                else:
                    print("No VAE checkpoints found")
                    self.vae_model = None
                return
            
            self.vae_model = ConditionalVAE(num_classes=8).to(self.device)
            checkpoint = torch.load(vae_checkpoint_path, map_location=self.device)
            self.vae_model.load_state_dict(checkpoint['model_state_dict'])
            self.vae_model.eval()
            print("VAE model loaded successfully!")
        except Exception as e:
            print(f"VAE failed to load: {e}")
            self.vae_model = None
        
        # Load cGAN
        try:
            # Try to load epoch 80 specifically
            if os.path.exists("outputs/cgan_checkpoints/cgan_epoch_80.pt"):
                cgan_checkpoint_path = "outputs/cgan_checkpoints/cgan_epoch_80.pt"
                print(f"Loading cGAN checkpoint: {cgan_checkpoint_path}")
            else:
                # Fall back to latest available checkpoint
                cgan_checkpoints = sorted(glob.glob("outputs/cgan_checkpoints/cgan_epoch_*.pt"))
                if cgan_checkpoints:
                    cgan_checkpoint_path = cgan_checkpoints[-1]
                    print(f"Epoch 80 not found, loading latest cGAN checkpoint: {cgan_checkpoint_path}")
                else:
                    print("No cGAN checkpoints found")
                    self.cgan_model = None
                return
            
            self.cgan_model = ConditionalGANGenerator(num_classes=8).to(self.device)
            checkpoint = torch.load(cgan_checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                self.cgan_model.load_state_dict(checkpoint['state_dict'])
            elif 'generator_state_dict' in checkpoint:
                self.cgan_model.load_state_dict(checkpoint['generator_state_dict'])
            else:
                self.cgan_model.load_state_dict(checkpoint)
            
            self.cgan_model.eval()
            print("cGAN model loaded successfully!")
        except Exception as e:
            print(f"cGAN failed to load: {e}")
            self.cgan_model = None
        
        # Load VAE-GAN
        try:
            # Try to load epoch 60 checkpoint
            if os.path.exists("outputs/vaegan_checkpoints/vaegan_epoch_60.pt"):
                vaegan_checkpoint_path = "outputs/vaegan_checkpoints/vaegan_epoch_60.pt"
                print(f"Loading VAE-GAN checkpoint: {vaegan_checkpoint_path}")
            else:
                # Fall back to latest available checkpoint
                vaegan_checkpoints = sorted(glob.glob("outputs/vaegan_checkpoints/vaegan_epoch_*.pt"))
                if vaegan_checkpoints:
                    vaegan_checkpoint_path = vaegan_checkpoints[-1]
                    print(f"Epoch 60 not found, loading latest VAE-GAN checkpoint: {vaegan_checkpoint_path}")
                else:
                    print("No VAE-GAN checkpoints found")
                    self.vaegan_model = None
                    return
            
            self.vaegan_model = VAEGAN(num_classes=8).to(self.device)
            checkpoint = torch.load(vaegan_checkpoint_path, map_location=self.device)
            
            # Load state dicts
            self.vaegan_model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.vaegan_model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            self.vaegan_model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            
            self.vaegan_model.eval()
            print("VAE-GAN model loaded successfully!")
        except Exception as e:
            print(f"VAE-GAN failed to load: {e}")
            self.vaegan_model = None
    
    def generate_faces(self, model_type, emotion, num_faces, blend_emotion=None, blend_amount=0.0):
        """
        Generate faces using the specified model and parameters.
        
        Args:
            model_type (str): 'VAE' or 'cGAN'
            emotion (str): Primary emotion
            num_faces (int): Number of faces to generate
            blend_emotion (str): Secondary emotion for blending (VAE only)
            blend_amount (float): Blend amount (0.0 to 1.0)
        
        Returns:
            tuple: (generated_images, generation_time_ms, status_message)
        """
        try:
            # Validate inputs
            if emotion not in self.label_to_idx:
                return [], 0, f"Error: Unknown emotion '{emotion}'"
            
            if blend_emotion and blend_emotion not in self.label_to_idx:
                return [], 0, f"Error: Unknown blend emotion '{blend_emotion}'"
            
            # Get model
            if model_type == 'VAE':
                model = self.vae_model
                if model is None:
                    return [], 0, "Error: VAE model not loaded"
            elif model_type == 'cGAN':
                model = self.cgan_model
                if model is None:
                    return [], 0, "Error: cGAN model not loaded"
            else:  # VAE-GAN
                model = self.vaegan_model
                if model is None:
                    return [], 0, "Error: VAE-GAN model not loaded"
            
            # Get emotion indices
            emotion_idx = self.label_to_idx[emotion]
            
            start_time = time.time()
            
            with torch.no_grad():
                if model_type == 'VAE' and blend_emotion and blend_amount > 0:
                    # VAE with emotion blending
                    blend_idx = self.label_to_idx[blend_emotion]
                    
                    # Generate samples for both emotions
                    samples1 = model.sample(emotion_idx, self.device, num_samples=num_faces)
                    samples2 = model.sample(blend_idx, self.device, num_samples=num_faces)
                    
                    # Blend in latent space
                    # For simplicity, we'll blend the outputs
                    blended = (1 - blend_amount) * samples1 + blend_amount * samples2
                    generated = blended
                    
                elif model_type == 'VAE-GAN' and blend_emotion and blend_amount > 0:
                    # VAE-GAN with interpolation in latent space
                    blend_idx = self.label_to_idx[blend_emotion]
                    
                    # Generate latent vectors for both emotions
                    z1 = torch.randn(num_faces, 128, device=self.device)
                    z2 = torch.randn(num_faces, 128, device=self.device)
                    
                    # Interpolate in latent space
                    z_interp = (1 - blend_amount) * z1 + blend_amount * z2
                    
                    # Generate samples with interpolated latents
                    labels1 = torch.full((num_faces,), emotion_idx, device=self.device, dtype=torch.long)
                    labels2 = torch.full((num_faces,), blend_idx, device=self.device, dtype=torch.long)
                    
                    # Use interpolated latents with primary emotion labels
                    generated = model.decode(z_interp, labels1)
                    
                else:
                    # Standard generation
                    if model_type == 'VAE':
                        generated = model.sample(emotion_idx, self.device, num_samples=num_faces)
                    elif model_type == 'cGAN':
                        generated = model.generate(emotion_idx, self.device, num_samples=num_faces)
                    else:  # VAE-GAN
                        generated = model.sample(emotion_idx, num_faces, self.device)
            
            generation_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Convert to PIL Images
            generated = denormalize(generated)
            generated = torch.clamp(generated, 0, 1)
            
            pil_images = []
            for i in range(generated.shape[0]):
                img_tensor = generated[i].cpu()
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                pil_images.append(pil_img)
            
            status_msg = f"Generated {num_faces} {emotion} faces using {model_type} in {generation_time:.2f}ms"
            if blend_emotion and blend_amount > 0:
                status_msg += f" (blended with {blend_emotion} at {blend_amount:.1f})"
            
            return pil_images, generation_time, status_msg
            
        except Exception as e:
            return [], 0, f"Error during generation: {str(e)}"
    
    def create_interface(self):
        """Create the Gradio interface."""
        
        with gr.Blocks(title="Emotion Face Generator", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎭 Emotion Face Generator")
            gr.Markdown("Generate realistic human facial expressions conditioned on emotion labels using trained VAE and cGAN models.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Model selection
                    model_radio = gr.Radio(
                        choices=["VAE", "cGAN", "VAE-GAN"],
                        value="VAE" if self.vae_model else ("cGAN" if self.cgan_model else "VAE-GAN"),
                        label="Select Model",
                        info="Choose between VAE (supports blending), cGAN, or VAE-GAN (supports interpolation)"
                    )
                    
                    # Emotion selection
                    emotion_dropdown = gr.Dropdown(
                        choices=self.emotions,
                        value="happy",
                        label="Primary Emotion",
                        info="Select the primary emotion to generate"
                    )
                    
                    # Number of faces
                    num_faces_slider = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=4,
                        step=1,
                        label="Number of Faces",
                        info="How many faces to generate"
                    )
                    
                    # VAE-specific blending options
                    with gr.Group(visible=True) as blend_group:
                        gr.Markdown("### VAE Emotion Blending")
                        blend_emotion_dropdown = gr.Dropdown(
                            choices=self.emotions,
                            value="neutral",
                            label="Blend with Emotion",
                            info="Select secondary emotion for blending"
                        )
                        
                        blend_amount_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.1,
                            label="Blend Amount",
                            info="0.0 = primary emotion only, 1.0 = blend emotion only"
                        )
                    
                    # Generate button
                    generate_btn = gr.Button(
                        "🎨 Generate Faces",
                        variant="primary",
                        size="lg"
                    )
                    
                    # Status and timing info
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False,
                        placeholder="Ready to generate..."
                    )
                    
                    timing_text = gr.Textbox(
                        label="Generation Time",
                        interactive=False,
                        placeholder="Generation time will appear here..."
                    )
                
                with gr.Column(scale=2):
                    # Output gallery
                    output_gallery = gr.Gallery(
                        label="Generated Faces",
                        show_label=True,
                        elem_id="output_gallery",
                        columns=4,
                        rows=2,
                        height="auto",
                        object_fit="cover"
                    )
            
            # Toggle blend group based on model selection
            def toggle_blend_group(model_type):
                return gr.Group(visible=(model_type in ["VAE", "VAE-GAN"]))
            
            model_radio.change(
                fn=toggle_blend_group,
                inputs=[model_radio],
                outputs=[blend_group]
            )
            
            # Generation function
            def on_generate(model_type, emotion, num_faces, blend_emotion, blend_amount):
                pil_images, gen_time, status = self.generate_faces(
                    model_type, emotion, num_faces, blend_emotion, blend_amount
                )
                
                timing_info = f"Generation time: {gen_time:.2f} ms" if gen_time > 0 else "Generation failed"
                
                return pil_images, status, timing_info
            
            # Connect generate button
            generate_btn.click(
                fn=on_generate,
                inputs=[
                    model_radio,
                    emotion_dropdown,
                    num_faces_slider,
                    blend_emotion_dropdown,
                    blend_amount_slider
                ],
                outputs=[
                    output_gallery,
                    status_text,
                    timing_text
                ]
            )
            
            # Example generation on load
            interface.load(
                fn=lambda: ([], "Models loaded. Ready to generate!", ""),
                outputs=[output_gallery, status_text, timing_text]
            )
        
        return interface
    
    def launch(self, share=False):
        """Launch the Gradio app."""
        interface = self.create_interface()
        
        print("Launching Emotion Face Generator web app...")
        print(f"Device: {self.device}")
        print(f"VAE Model: {'Loaded' if self.vae_model else 'Not found'}")
        print(f"cGAN Model: {'Loaded' if self.cgan_model else 'Not found'}")
        print(f"VAE-GAN Model: {'Loaded' if self.vaegan_model else 'Not found'}")
        
        interface.launch(
            share=share,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )


def main():
    """Main function to run the Gradio app."""
    app = EmotionFaceGeneratorApp()
    app.launch(share=False)


if __name__ == "__main__":
    main()
