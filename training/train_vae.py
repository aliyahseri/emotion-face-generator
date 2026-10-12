import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import get_dataloader
from models.vae import ConditionalVAE
from utils.helpers import set_seed, save_image_grid, denormalize


def train_vae(args):
    """
    Train the Conditional VAE model.
    
    Args:
        args: Command line arguments
    """
    print("Starting VAE Training...")
    print(f"Device: {args.device}")
    print(f"Data path: {args.data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Save every: {args.save_every} epochs")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    train_loader, train_dataset = get_dataloader(
        args.data_path, 
        split='train', 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4
    )
    
    num_classes = train_dataset.get_num_classes()
    print(f"Number of classes: {num_classes}")
    
    # Initialize model
    print("Initializing VAE model...")
    model = ConditionalVAE(
        latent_dim=256,
        num_classes=num_classes,
        beta=4.0
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    # Training loop
    print("Starting training loop...")
    model.train()
    
    train_losses = []
    recon_losses = []
    kl_losses = []
    
    for epoch in range(args.epochs):
        epoch_train_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (images, labels, label_names) in enumerate(progress_bar):
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, mu, log_var = model(images, labels)
            
            # Compute loss
            total_loss, recon_loss, kl_loss = model.loss_function(
                reconstructed, images, mu, log_var
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Update running losses
            epoch_train_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Total Loss': f'{total_loss.item():.4f}',
                'Recon Loss': f'{recon_loss.item():.4f}',
                'KL Loss': f'{kl_loss.item():.4f}'
            })
        
        # Calculate average losses for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        avg_kl_loss = epoch_kl_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Recon Loss: {avg_recon_loss:.4f}, "
              f"KL Loss: {avg_kl_loss:.4f}")
        
        # Save samples and checkpoint every save_every epochs
        if (epoch + 1) % args.save_every == 0:
            print(f"Saving checkpoint and samples for epoch {epoch+1}...")
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.checkpoint_dir, f'vae_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'recon_loss': avg_recon_loss,
                'kl_loss': avg_kl_loss,
                'label_to_idx': train_dataset.label_to_idx,
                'idx_to_label': train_dataset.idx_to_label
            }, checkpoint_path)
            
            # Generate and save samples
            model.eval()
            with torch.no_grad():
                # Generate one sample for each emotion
                samples = []
                for emotion_idx in range(num_classes):
                    sample = model.sample(emotion_idx, args.device, num_samples=1)
                    samples.append(sample)
                
                samples = torch.cat(samples, dim=0)  # [num_classes, 3, 64, 64]
                
                # Save sample grid
                sample_path = os.path.join(args.sample_dir, f'epoch_{epoch+1}.png')
                save_image_grid(denormalize(samples), sample_path, nrow=8)
            
            model.train()
            print(f"Checkpoint saved to {checkpoint_path}")
            print(f"Samples saved to {sample_path}")
    
    # Save final model
    print("Saving final model...")
    final_checkpoint_path = os.path.join(args.checkpoint_dir, 'vae_final.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1] if train_losses else 0,
        'recon_loss': recon_losses[-1] if recon_losses else 0,
        'kl_loss': kl_losses[-1] if kl_losses else 0,
        'label_to_idx': train_dataset.label_to_idx,
        'idx_to_label': train_dataset.idx_to_label,
        'train_losses': train_losses,
        'recon_losses': recon_losses,
        'kl_losses': kl_losses
    }, final_checkpoint_path)
    
    # Plot and save loss curves
    print("Saving loss curves...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(recon_losses)
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(kl_losses)
    plt.title('KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    loss_plot_path = os.path.join(args.output_dir, 'vae_loss_curves.png')
    plt.savefig(loss_plot_path)
    plt.close()
    
    print(f"Training completed!")
    print(f"Final model saved to {final_checkpoint_path}")
    print(f"Loss curves saved to {loss_plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Conditional VAE for Emotion Face Generation')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                       default='dataset',
                       help='Path to the dataset')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='KL divergence weight')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent dimension')
    
    # Saving arguments
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--output_dir', type=str, 
                       default='outputs',
                       help='Output directory')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set directory paths
    args.checkpoint_dir = os.path.join(args.output_dir, 'vae_checkpoints')
    args.sample_dir = os.path.join(args.output_dir, 'vae_samples')
    
    # Start training
    train_vae(args)


if __name__ == "__main__":
    main()
