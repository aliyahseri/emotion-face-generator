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
from models.cgan_generator import ConditionalGANGenerator
from models.cgan_discriminator import ConditionalGANDiscriminator
from utils.helpers import set_seed, save_image_grid, denormalize


def train_cgan(args):
    """
    Train the Conditional GAN model.
    
    Args:
        args: Command line arguments
    """
    print("Starting Conditional GAN Training...")
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
    
    # Initialize models
    print("Initializing Generator and Discriminator...")
    generator = ConditionalGANGenerator(
        nz=args.nz,
        num_classes=num_classes
    ).to(args.device)
    
    discriminator = ConditionalGANDiscriminator(
        num_classes=num_classes,
        img_size=64,
        channels=3
    ).to(args.device)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Initialize optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr * 0.3, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Checkpoint loading logic
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}")
    else:
        start_epoch = 0
    
    # Training loop
    print("Starting training loop...")
    
    g_losses = []
    d_losses = []
    prev_g_loss = 0.0  # Track previous G loss for conditional training
    prev_d_loss = 1.0  # Track previous D loss for conditional training
    
    for epoch in range(start_epoch, args.epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        g_updates = 0
        d_updates = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (images, labels, label_names) in enumerate(progress_bar):
            batch_size = images.size(0)
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            # Adversarial ground truths with label smoothing
            real_labels = torch.full((batch_size, 1), 0.9, device=args.device)
            fake_labels = torch.full((batch_size, 1), args.fake_label, device=args.device)
            
            # ---------------------
            #  Train Discriminator (only every 2 steps to let G keep up)
            # ---------------------
            if batch_idx % 2 != 0:
                continue
            
            optimizer_D.zero_grad()
            
            # Loss for real images
            outputs_real = discriminator(images, labels)
            d_loss_real = criterion(outputs_real, real_labels)
            
            # Loss for fake images
            noise = torch.randn(batch_size, args.nz, device=args.device)
            fake_images = generator(noise, labels)
            outputs_fake = discriminator(fake_images.detach(), labels)
            d_loss_fake = criterion(outputs_fake, fake_labels)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate fake images
            noise = torch.randn(batch_size, args.nz, device=args.device)
            fake_images = generator(noise, labels)
            
            # Generator wants discriminator to think fake images are real
            outputs = discriminator(fake_images, labels)
            g_loss = criterion(outputs, real_labels)
            
            g_loss.backward()
            
            # Add gradient clipping for generator
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            
            optimizer_G.step()
            
            
            # Update previous G loss for conditional training
            prev_g_loss = g_loss.item()
            prev_d_loss = d_loss.item()
            
            # Update running losses
            epoch_g_loss += g_loss.item()
            g_updates += 1
            epoch_d_loss += d_loss.item()
            d_updates += 1
            
            # Update progress bar
            if batch_idx % 50 == 0:
                progress_bar.set_postfix({
                    'D Loss': f'{d_loss.item():.4f}',
                    'G Loss': f'{g_loss.item():.4f}',
                    'D Real': f'{outputs_real.mean().item():.4f}',
                    'D Fake': f'{outputs_fake.mean().item():.4f}'
                })
        
        # Calculate average losses for the epoch
        avg_g_loss = epoch_g_loss / max(g_updates, 1)
        avg_d_loss = epoch_d_loss / max(d_updates, 1)
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"G Loss: {avg_g_loss:.4f}, "
              f"D Loss: {avg_d_loss:.4f}")
        
        # Save samples and checkpoint every save_every epochs
        if (epoch + 1) % args.save_every == 0:
            print(f"Saving checkpoint and samples for epoch {epoch+1}...")
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.checkpoint_dir, f'cgan_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss,
                'label_to_idx': train_dataset.label_to_idx,
                'idx_to_label': train_dataset.idx_to_label
            }, checkpoint_path)
            
            # Generate and save samples
            generator.eval()
            with torch.no_grad():
                # Generate one sample for each emotion
                samples = []
                for emotion_idx in range(num_classes):
                    sample = generator.generate(emotion_idx, args.device, num_samples=1)
                    samples.append(sample)
                
                samples = torch.cat(samples, dim=0)  # [num_classes, 3, 64, 64]
                
                # Save sample grid
                sample_path = os.path.join(args.sample_dir, f'epoch_{epoch+1}.png')
                save_image_grid(denormalize(samples), sample_path, nrow=8)
            
            generator.train()
            print(f"Checkpoint saved to {checkpoint_path}")
            print(f"Samples saved to {sample_path}")
    
    # Save final models
    print("Saving final models...")
    final_checkpoint_path = os.path.join(args.checkpoint_dir, 'cgan_final.pt')
    torch.save({
        'epoch': args.epochs,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'g_loss': g_losses[-1] if g_losses else 0,
        'd_loss': d_losses[-1] if d_losses else 0,
        'label_to_idx': train_dataset.label_to_idx,
        'idx_to_label': train_dataset.idx_to_label,
        'g_losses': g_losses,
        'd_losses': d_losses
    }, final_checkpoint_path)
    
    # Also save models separately for easier loading
    generator_path = os.path.join(args.checkpoint_dir, 'generator_final.pt')
    discriminator_path = os.path.join(args.checkpoint_dir, 'discriminator_final.pt')
    
    torch.save({
        'state_dict': generator.state_dict(),
        'label_to_idx': train_dataset.label_to_idx,
        'idx_to_label': train_dataset.idx_to_label
    }, generator_path)
    
    torch.save({
        'state_dict': discriminator.state_dict(),
        'label_to_idx': train_dataset.label_to_idx,
        'idx_to_label': train_dataset.idx_to_label
    }, discriminator_path)
    
    # Plot and save loss curves
    print("Saving loss curves...")
    plt.figure(figsize=(10, 5))
    
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    loss_plot_path = os.path.join(args.output_dir, 'cgan_loss_curves.png')
    plt.savefig(loss_plot_path)
    plt.close()
    
    print(f"Training completed!")
    print(f"Final models saved to {final_checkpoint_path}")
    print(f"Generator saved to {generator_path}")
    print(f"Discriminator saved to {discriminator_path}")
    print(f"Loss curves saved to {loss_plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Conditional GAN for Emotion Face Generation')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                       default='dataset',
                       help='Path to the dataset')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--nz', type=int, default=128,
                       help='Size of noise vector')
    
    # Label smoothing
    parser.add_argument('--real_label', type=float, default=0.9,
                       help='Label for real images (for smoothing)')
    parser.add_argument('--fake_label', type=float, default=0.0,
                       help='Label for fake images')
    
    # Saving arguments
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--output_dir', type=str, 
                       default='outputs',
                       help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set directory paths
    args.checkpoint_dir = os.path.join(args.output_dir, 'cgan_checkpoints')
    args.sample_dir = os.path.join(args.output_dir, 'cgan_samples')
    
    # Start training
    train_cgan(args)


if __name__ == "__main__":
    main()
