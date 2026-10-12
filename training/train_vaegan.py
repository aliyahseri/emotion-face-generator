import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import get_dataloader
from models.vaegan import VAEGAN
from utils.helpers import set_seed, save_image_grid, denormalize


def train_vaegan(args):
    """
    Train the VAE-GAN model.
    
    Args:
        args: Command line arguments
    """
    print("Starting VAE-GAN Training...")
    print(f"Device: {args.device}")
    print(f"Data path: {args.data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Latent dimension: {args.latent_dim}")
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
    print("Initializing VAE-GAN...")
    model = VAEGAN(latent_dim=args.latent_dim, num_classes=num_classes).to(args.device)
    
    print(f"Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"Decoder parameters: {sum(p.numel() for p in model.decoder.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in model.discriminator.parameters()):,}")
    
    # Initialize optimizers
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.999))
    optimizer_VAE = optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), 
                             lr=args.lr, betas=(0.5, 0.999))
    
    # Loss functions
    recon_criterion = nn.MSELoss()
    gan_criterion = nn.BCELoss()
    beta = 1.0  # KL loss weight
    
    # Training loop
    print("Starting training loop...")
    
    recon_losses = []
    kl_losses = []
    gan_losses = []
    d_losses = []
    
    for epoch in range(args.epochs):
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_gan_loss = 0.0
        epoch_d_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (images, labels, label_names) in enumerate(progress_bar):
            batch_size = images.size(0)
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            # Adversarial ground truths
            real_labels = torch.full((batch_size, 1), 0.9, device=args.device)
            fake_labels = torch.full((batch_size, 1), 0.1, device=args.device)
            
            # ---------------------
            # Train VAE-GAN (Encoder + Decoder) - Always train
            # ---------------------
            optimizer_VAE.zero_grad()
            
            # Encode and decode (create fresh computation graph)
            mu, logvar = model.encode(images, labels)
            z = model.reparameterize(mu, logvar)
            recon_images = model.decode(z, labels)
            
            # Reconstruction loss
            recon_loss = recon_criterion(recon_images, images)
            
            # KL loss
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            # GAN loss (generator wants discriminator to think reconstructions are real)
            gan_loss_recon = gan_criterion(model.discriminator(recon_images, labels), real_labels)
            
            # Generate new random samples for VAE-GAN training
            new_random_z = torch.randn(batch_size, args.latent_dim, device=args.device)
            new_random_images = model.decode(new_random_z, labels)
            gan_loss_random = gan_criterion(model.discriminator(new_random_images, labels), real_labels)
            gan_loss = gan_loss_recon + gan_loss_random
            
            # Total VAE-GAN loss
            total_vae_loss = recon_loss + beta * kl_loss + 0.1 * gan_loss
            total_vae_loss.backward()
            optimizer_VAE.step()
            
            # ---------------------
            # Train Discriminator (every 2 steps)
            # ---------------------
            if batch_idx % 2 == 0:  # Train D every 2 steps
                optimizer_D.zero_grad()
                
                # Real images
                outputs_real = model.discriminator(images, labels)
                d_loss_real = gan_criterion(outputs_real, real_labels)
                
                # Reconstructed images (fake) - use detach to break graph
                outputs_recon = model.discriminator(recon_images.detach(), labels)
                d_loss_recon = gan_criterion(outputs_recon, fake_labels)
                
                # Random z samples (fake) - use detach to break graph
                outputs_random = model.discriminator(new_random_images.detach(), labels)
                d_loss_random = gan_criterion(outputs_random, fake_labels)
                
                # Total discriminator loss
                d_loss = d_loss_real + d_loss_recon + d_loss_random
                d_loss.backward()
                optimizer_D.step()
                
                epoch_d_loss += d_loss.item()
            
            # Update running losses
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_gan_loss += gan_loss.item()
            
            # Update progress bar
            if batch_idx % 50 == 0:
                progress_bar.set_postfix({
                    'Recon': f'{recon_loss.item():.4f}',
                    'KL': f'{kl_loss.item():.4f}',
                    'GAN': f'{gan_loss.item():.4f}',
                    'D Loss': f'{d_loss.item() if batch_idx % 2 == 0 else 0:.4f}',
                    'D Real': f'{outputs_real.mean().item():.4f}',
                    'D Fake': f'{outputs_recon.mean().item():.4f}'
                })
        
        # Calculate average losses for the epoch
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        avg_kl_loss = epoch_kl_loss / len(train_loader)
        avg_gan_loss = epoch_gan_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / (len(train_loader) // 2)  # D trained every 2 steps
        
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)
        gan_losses.append(avg_gan_loss)
        d_losses.append(avg_d_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Recon Loss: {avg_recon_loss:.4f}, "
              f"KL Loss: {avg_kl_loss:.4f}, "
              f"GAN Loss: {avg_gan_loss:.4f}, "
              f"D Loss: {avg_d_loss:.4f}")
        
        # Save samples and checkpoint every save_every epochs
        if (epoch + 1) % args.save_every == 0:
            print(f"Saving checkpoint and samples for epoch {epoch+1}...")
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.checkpoint_dir, f'vaegan_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': model.encoder.state_dict(),
                'decoder_state_dict': model.decoder.state_dict(),
                'discriminator_state_dict': model.discriminator.state_dict(),
                'optimizer_VAE_state_dict': optimizer_VAE.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'recon_loss': avg_recon_loss,
                'kl_loss': avg_kl_loss,
                'gan_loss': avg_gan_loss,
                'd_loss': avg_d_loss,
                'label_to_idx': train_dataset.label_to_idx,
                'idx_to_label': train_dataset.idx_to_label
            }, checkpoint_path)
            
            # Generate and save samples
            model.eval()
            with torch.no_grad():
                # Generate one sample for each emotion using random z
                samples = []
                for emotion_idx in range(num_classes):
                    sample = model.sample(emotion_idx, 1, args.device)
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
    final_checkpoint_path = os.path.join(args.checkpoint_dir, 'vaegan_final.pt')
    torch.save({
        'epoch': args.epochs,
        'encoder_state_dict': model.encoder.state_dict(),
        'decoder_state_dict': model.decoder.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
        'optimizer_VAE_state_dict': optimizer_VAE.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'recon_loss': recon_losses[-1],
        'kl_loss': kl_losses[-1],
        'gan_loss': gan_losses[-1],
        'd_loss': d_losses[-1],
        'label_to_idx': train_dataset.label_to_idx,
        'idx_to_label': train_dataset.idx_to_label,
        'recon_losses': recon_losses,
        'kl_losses': kl_losses,
        'gan_losses': gan_losses,
        'd_losses': d_losses
    }, final_checkpoint_path)
    
    # Plot and save loss curves
    print("Saving loss curves...")
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(recon_losses, label='Reconstruction Loss')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(kl_losses, label='KL Loss')
    plt.title('KL Divergence Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(gan_losses, label='GAN Loss')
    plt.title('GAN Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(d_losses, label='Discriminator Loss')
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    loss_plot_path = os.path.join(args.output_dir, 'vaegan_loss_curves.png')
    plt.savefig(loss_plot_path)
    plt.close()
    
    print(f"Training completed!")
    print(f"Final model saved to {final_checkpoint_path}")
    print(f"Loss curves saved to {loss_plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Train VAE-GAN for Emotion Face Generation')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                       default='dataset',
                       help='Path to the dataset')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent dimension')
    
    # Saving arguments
    parser.add_argument('--save_every', type=int, default=20,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--output_dir', type=str, 
                       default='outputs',
                       help='Output directory')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set directory paths
    args.checkpoint_dir = os.path.join(args.output_dir, 'vaegan_checkpoints')
    args.sample_dir = os.path.join(args.output_dir, 'vaegan_samples')
    
    # Start training
    train_vaegan(args)


if __name__ == "__main__":
    main()
