import os
import argparse
import time
import torch
import torch.nn as nn

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import get_dataloader
from models.vae import ConditionalVAE
from models.cgan_generator import ConditionalGANGenerator
from utils.helpers import set_seed, save_image_grid, denormalize, get_label_map


def load_model(model_type, checkpoint_path, device):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_type (str): 'vae' or 'cgan'
        checkpoint_path (str): Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded model and label mappings
    """
    print(f"Loading {model_type.upper()} model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get label mappings
    label_to_idx = checkpoint.get('label_to_idx', None)
    idx_to_label = checkpoint.get('idx_to_label', None)
    
    if label_to_idx is None:
        print("Warning: Label mapping not found in checkpoint. Using default mapping.")
        # Try to get label mapping from dataset
        try:
            label_to_idx = get_label_map(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset"))
            idx_to_label = {v: k for k, v in label_to_idx.items()}
        except:
            # Default emotion labels
            emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            label_to_idx = {emotion: idx for idx, emotion in enumerate(emotions)}
            idx_to_label = {idx: emotion for emotion, idx in label_to_idx.items()}
    
    num_classes = len(label_to_idx)
    
    # Create and load model
    if model_type == 'vae':
        model = ConditionalVAE(num_classes=num_classes).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif model_type == 'cgan':
        model = ConditionalGANGenerator(num_classes=num_classes).to(device)
        
        # Handle different checkpoint formats
        if 'generator_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['generator_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    print(f"Model loaded successfully!")
    print(f"Label mapping: {label_to_idx}")
    
    return model, label_to_idx, idx_to_label


def generate_images(model, model_type, emotion_idx, device, num_samples=4):
    """
    Generate images using the specified model.
    
    Args:
        model: Loaded model
        model_type (str): 'vae' or 'cgan'
        emotion_idx (int): Emotion label index
        device: Device to generate on
        num_samples (int): Number of samples to generate
    
    Returns:
        Generated images tensor
    """
    print(f"Generating {num_samples} samples for emotion index {emotion_idx}")
    
    start_time = time.time()
    
    with torch.no_grad():
        if model_type == 'vae':
            generated = model.sample(emotion_idx, device, num_samples=num_samples)
        elif model_type == 'cgan':
            generated = model.generate(emotion_idx, device, num_samples=num_samples)
    
    generation_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    print(f"Generation completed in {generation_time:.2f} ms")
    
    return generated, generation_time


def interpolate_emotions(model, model_type, from_emotion_idx, to_emotion_idx, device, steps=10):
    """
    Generate interpolation between two emotions (VAE only).
    
    Args:
        model: Loaded VAE model
        model_type (str): Must be 'vae'
        from_emotion_idx (int): Starting emotion index
        to_emotion_idx (int): Ending emotion index
        device: Device to generate on
        steps (int): Number of interpolation steps
    
    Returns:
        Interpolated images tensor
    """
    if model_type != 'vae':
        raise ValueError("Interpolation is only supported for VAE models")
    
    print(f"Generating interpolation from emotion {from_emotion_idx} to {to_emotion_idx} in {steps} steps")
    
    start_time = time.time()
    
    with torch.no_grad():
        interpolated = model.interpolate(from_emotion_idx, to_emotion_idx, steps=steps, device=device)
    
    generation_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    print(f"Interpolation completed in {generation_time:.2f} ms")
    
    return interpolated, generation_time


def main():
    parser = argparse.ArgumentParser(description='Generate Emotion Faces using Trained Models')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True, choices=['vae', 'cgan'],
                       help='Model type to use for generation')
    parser.add_argument('--checkpoint_path', type=str,
                       default='outputs/vae_checkpoints/vae_final.pt' if 'vae' in sys.argv else 'outputs/cgan_checkpoints/generator_final.pt',
                       help='Path to model checkpoint')
    
    # Generation arguments
    parser.add_argument('--emotion', type=str, required=False,
                       help='Emotion name to generate (e.g., happy, sad, angry)')
    parser.add_argument('--emotion_idx', type=int, required=False,
                       help='Emotion index to generate (0-6)')
    parser.add_argument('--nz', type=int, default=256,
                       help='Size of noise vector for GAN')
    parser.add_argument('--latent_dim', type=int, default=256,
                       help='Latent dimension for VAE')
    parser.add_argument('--n', type=int, default=4,
                       help='Number of images to generate')
    parser.add_argument('--output_dir', type=str, default='generated_samples',
                       help='Output directory for generated images')
    
    # Interpolation arguments (VAE only)
    parser.add_argument('--interpolate', action='store_true',
                       help='Generate emotion interpolation (VAE only)')
    parser.add_argument('--from_emotion', type=str,
                       help='Starting emotion for interpolation')
    parser.add_argument('--to_emotion', type=str,
                       help='Ending emotion for interpolation')
    parser.add_argument('--steps', type=int, default=10,
                       help='Number of interpolation steps')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set device and seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)
    
    print(f"Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Device: {device}")
    print(f"  Output Dir: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, label_to_idx, idx_to_label = load_model(args.model, args.checkpoint_path, device)
    
    # Handle emotion selection
    if args.interpolate:
        # Interpolation mode
        if not args.from_emotion or not args.to_emotion:
            raise ValueError("Both --from_emotion and --to_emotion must be specified for interpolation")
        
        if args.from_emotion not in label_to_idx:
            raise ValueError(f"Unknown emotion: {args.from_emotion}. Available: {list(label_to_idx.keys())}")
        if args.to_emotion not in label_to_idx:
            raise ValueError(f"Unknown emotion: {args.to_emotion}. Available: {list(label_to_idx.keys())}")
        
        from_idx = label_to_idx[args.from_emotion]
        to_idx = label_to_idx[args.to_emotion]
        
        # Generate interpolation
        interpolated, generation_time = interpolate_emotions(
            model, args.model, from_idx, to_idx, device, args.steps
        )
        
        # Save interpolation
        output_path = os.path.join(args.output_dir, f"interpolation_{args.from_emotion}_to_{args.to_emotion}.png")
        save_image_grid(denormalize(interpolated), output_path, nrow=args.steps)
        
        print(f"Interpolation saved to {output_path}")
        print(f"Generation time: {generation_time:.2f} ms")
        
    else:
        # Single emotion generation mode
        if args.emotion:
            if args.emotion not in label_to_idx:
                raise ValueError(f"Unknown emotion: {args.emotion}. Available: {list(label_to_idx.keys())}")
            emotion_idx = label_to_idx[args.emotion]
            emotion_name = args.emotion
        elif args.emotion_idx is not None:
            if args.emotion_idx < 0 or args.emotion_idx >= len(label_to_idx):
                raise ValueError(f"Emotion index must be between 0 and {len(label_to_idx)-1}")
            emotion_idx = args.emotion_idx
            emotion_name = idx_to_label[emotion_idx]
        else:
            # Generate all emotions if none specified
            print("No emotion specified, generating all emotions...")
            for emotion_name, emotion_idx in label_to_idx.items():
                generated, generation_time = generate_images(
                    model, args.model, emotion_idx, device, num_samples=args.n
                )
                
                output_path = os.path.join(args.output_dir, f"{args.model}_{emotion_name}_{args.n}samples.png")
                save_image_grid(denormalize(generated), output_path, nrow=args.n)
                
                print(f"Generated {args.n} {emotion_name} faces saved to {output_path}")
                print(f"Generation time: {generation_time:.2f} ms")
            
            return
        
        # Generate images for specified emotion
        generated, generation_time = generate_images(
            model, args.model, emotion_idx, device, num_samples=args.n
        )
        
        # Save generated images
        output_path = os.path.join(args.output_dir, f"{args.model}_{emotion_name}_{args.n}samples.png")
        save_image_grid(denormalize(generated), output_path, nrow=args.n)
        
        print(f"Generated {args.n} {emotion_name} faces saved to {output_path}")
        print(f"Generation time: {generation_time:.2f} ms")


if __name__ == "__main__":
    main()
