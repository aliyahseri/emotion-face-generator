import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vaegan import VAEGAN
from utils.helpers import denormalize


def interpolate_latent(model, z1, z2, steps=10):
    """Interpolate between two latent vectors"""
    alphas = np.linspace(0, 1, steps)
    interpolated = []
    
    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2
        interpolated.append(z_interp)
    
    return torch.stack(interpolated)


def generate_samples(model, emotion_idx, n_samples, device):
    """Generate samples for a specific emotion"""
    model.eval()
    with torch.no_grad():
        samples = model.sample(emotion_idx, n_samples, device)
    return samples


def save_image_grid(images, path, nrow=None):
    """Save images as a grid"""
    if nrow is None:
        nrow = int(np.sqrt(len(images)))
    
    # Denormalize images from [-1, 1] to [0, 1]
    images = denormalize(images)
    
    # Convert to numpy and create grid
    images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
    
    # Create grid
    nrows = (len(images) + nrow - 1) // nrow
    grid = np.zeros((nrows * 64, nrow * 64, 3))
    
    for i, img in enumerate(images_np):
        row = i // nrow
        col = i % nrow
        grid[row*64:(row+1)*64, col*64:(col+1)*64] = img
    
    # Convert to PIL and save
    grid = (grid * 255).astype(np.uint8)
    Image.fromarray(grid).save(path)


def main():
    parser = argparse.ArgumentParser(description='Generate samples using VAE-GAN')
    
    parser.add_argument('--emotion', type=str, default='happy',
                       choices=['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
                       help='Emotion to generate')
    parser.add_argument('--n', type=int, default=4,
                       help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='outputs/vaegan_generated',
                       help='Output directory')
    parser.add_argument('--checkpoint', type=str, default='outputs/vaegan_checkpoints/vaegan_final.pt',
                       help='Path to checkpoint file')
    parser.add_argument('--interpolate', action='store_true',
                       help='Perform interpolation between emotions')
    parser.add_argument('--from_emotion', type=str, default='happy',
                       choices=['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
                       help='Starting emotion for interpolation')
    parser.add_argument('--to_emotion', type=str, default='sad',
                       choices=['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
                       help='Ending emotion for interpolation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading VAE-GAN from {args.checkpoint}...")
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get label mapping from checkpoint
    label_to_idx = checkpoint.get('label_to_idx', {
        'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 
        'happy': 4, 'neutral': 5, 'sad': 6, 'surprise': 7
    })
    
    # Create model
    model = VAEGAN(latent_dim=128, num_classes=8).to(device)
    
    # Load state dicts
    model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    model.eval()
    print("VAE-GAN loaded successfully!")
    
    if args.interpolate:
        print(f"Interpolating from {args.from_emotion} to {args.to_emotion}...")
        
        # Get emotion indices
        from_idx = label_to_idx[args.from_emotion]
        to_idx = label_to_idx[args.to_emotion]
        
        # Generate latent vectors for both emotions
        with torch.no_grad():
            z_from = torch.randn(1, 128, device=device)
            z_to = torch.randn(1, 128, device=device)
            
            # Create labels
            labels_from = torch.full((1,), from_idx, device=device, dtype=torch.long)
            labels_to = torch.full((1,), to_idx, device=device, dtype=torch.long)
            
            # Generate samples for interpolation
            samples_from = model.decoder(z_from, labels_from)
            samples_to = model.decoder(z_to, labels_to)
            
            # Interpolate in latent space
            interpolated_z = interpolate_latent(model, z_from, z_to, steps=10)
            
            # Generate interpolated samples
            interpolated_samples = []
            for i, z_interp in enumerate(interpolated_z):
                # Linearly interpolate labels as well
                alpha = i / (len(interpolated_z) - 1)
                if alpha < 0.5:
                    label = labels_from
                else:
                    label = labels_to
                
                sample = model.decoder(z_interp, label)
                interpolated_samples.append(sample)
            
            interpolated_samples = torch.cat(interpolated_samples, dim=0)
        
        # Save interpolation
        output_path = os.path.join(args.output_dir, f'interpolate_{args.from_emotion}_to_{args.to_emotion}.png')
        save_image_grid(interpolated_samples, output_path, nrow=10)
        print(f"Interpolation saved to {output_path}")
        
    else:
        print(f"Generating {args.n} samples for {args.emotion}...")
        
        # Get emotion index
        emotion_idx = label_to_idx[args.emotion]
        
        # Generate samples
        samples = generate_samples(model, emotion_idx, args.n, device)
        
        # Save samples
        output_path = os.path.join(args.output_dir, f'{args.emotion}_samples.png')
        save_image_grid(samples, output_path, nrow=min(args.n, 4))
        print(f"Samples saved to {output_path}")
    
    print("Generation completed!")


if __name__ == "__main__":
    main()
