import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=256, num_classes=7, beta=4.0):
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.beta = beta
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, 128)
        
        # Encoder: 5 conv layers (3→64→128→256→512→512)
        self.encoder = nn.Sequential(
            # Input: 3x128x128
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 64x64x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128x32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256x16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 512x8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 512x4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Calculate flattened size
        self.encoder_output_size = 512 * 4 * 4
        
        # FC layers for mu and log_var
        self.fc_mu = nn.Linear(self.encoder_output_size + 128, latent_dim)
        self.fc_log_var = nn.Linear(self.encoder_output_size + 128, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim + 128, self.encoder_output_size)
        
        self.decoder = nn.Sequential(
            # Input: 512x4x4
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # 512x8x8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 256x16x16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128x32x32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x64x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x128x128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),  # 3x128x128
            nn.Tanh()  # Output in [-1, 1] range
        )
    
    def encode(self, x, labels):
        """
        Encode images to mu and log_var.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            labels: Emotion labels [batch_size]
        
        Returns:
            mu, log_var: Parameters of the latent Gaussian distribution
        """
        # Encode images
        encoded = self.encoder(x)  # [batch_size, 512, 4, 4]
        encoded = encoded.view(encoded.size(0), -1)  # [batch_size, 512*4*4]
        
        # Get label embeddings
        label_emb = self.label_embedding(labels)  # [batch_size, 128]
        
        # Concatenate encoded features with label embeddings
        combined = torch.cat([encoded, label_emb], dim=1)  # [batch_size, 512*4*4 + 128]
        
        # Get mu and log_var
        mu = self.fc_mu(combined)
        log_var = self.fc_log_var(combined)
        
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from the latent space.
        
        Args:
            mu: Mean of the latent Gaussian [batch_size, latent_dim]
            log_var: Log variance of the latent Gaussian [batch_size, latent_dim]
        
        Returns:
            z: Sampled latent vectors [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z, labels):
        """
        Decode latent vectors back to images.
        
        Args:
            z: Latent vectors [batch_size, latent_dim]
            labels: Emotion labels [batch_size]
        
        Returns:
            reconstructed: Reconstructed images [batch_size, channels, height, width]
        """
        # Get label embeddings
        label_emb = self.label_embedding(labels)  # [batch_size, 128]
        
        # Concatenate latent vectors with label embeddings
        combined = torch.cat([z, label_emb], dim=1)  # [batch_size, latent_dim + 128]
        
        # FC layer
        decoded = self.decoder_fc(combined)  # [batch_size, 512*4*4]
        decoded = decoded.view(decoded.size(0), 512, 4, 4)  # [batch_size, 512, 4, 4]
        
        # Decoder layers
        reconstructed = self.decoder(decoded)  # [batch_size, channels, 128, 128]
        
        return reconstructed
    
    def forward(self, x, labels):
        """
        Forward pass of the VAE.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            labels: Emotion labels [batch_size]
        
        Returns:
            reconstructed: Reconstructed images [batch_size, channels, height, width]
            mu: Mean of the latent Gaussian [batch_size, latent_dim]
            log_var: Log variance of the latent Gaussian [batch_size, latent_dim]
        """
        mu, log_var = self.encode(x, labels)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z, labels)
        
        return reconstructed, mu, log_var
    
    def loss_function(self, reconstructed, x, mu, log_var):
        """
        Compute VAE loss: reconstruction loss + KL divergence.
        
        Args:
            reconstructed: Reconstructed images [batch_size, channels, height, width]
            x: Original images [batch_size, channels, height, width]
            mu: Mean of the latent Gaussian [batch_size, latent_dim]
            log_var: Log variance of the latent Gaussian [batch_size, latent_dim]
        
        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence loss
        """
        # Reconstruction loss (BCE)
        recon_loss = F.mse_loss(reconstructed, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def sample(self, label_idx, device, num_samples=1):
        """
        Generate samples for a specific emotion.
        
        Args:
            label_idx: Emotion label index
            device: Device to generate on
            num_samples: Number of samples to generate
        
        Returns:
            samples: Generated images [num_samples, channels, height, width]
        """
        self.eval()
        with torch.no_grad():
            # Sample random latent vectors
            z = torch.randn(num_samples, self.latent_dim).to(device)
            
            # Create label tensor
            labels = torch.full((num_samples,), label_idx, dtype=torch.long).to(device)
            
            # Generate samples
            samples = self.decode(z, labels)
            
        return samples
    
    def interpolate(self, label_idx1, label_idx2, steps=10, device='cpu'):
        """
        Interpolate between two emotions in latent space.
        
        Args:
            label_idx1: Starting emotion label index
            label_idx2: Ending emotion label index
            steps: Number of interpolation steps
            device: Device to generate on
        
        Returns:
            interpolated: Interpolated images [steps, channels, height, width]
        """
        self.eval()
        with torch.no_grad():
            # Sample random latent vectors for both emotions
            z1 = torch.randn(1, self.latent_dim).to(device)
            z2 = torch.randn(1, self.latent_dim).to(device)
            
            # Create label tensors
            label1 = torch.full((1,), label_idx1, dtype=torch.long).to(device)
            label2 = torch.full((1,), label_idx2, dtype=torch.long).to(device)
            
            interpolated = []
            
            for i in range(steps):
                # Linear interpolation in latent space
                alpha = i / (steps - 1) if steps > 1 else 0
                z_interp = (1 - alpha) * z1 + alpha * z2
                
                # Linear interpolation of labels (use weighted average of embeddings)
                if alpha < 0.5:
                    label = label1
                else:
                    label = label2
                
                # Generate interpolated image
                img = self.decode(z_interp, label)
                interpolated.append(img)
            
            interpolated = torch.cat(interpolated, dim=0)
            
        return interpolated


if __name__ == "__main__":
    # Test the VAE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing VAE on device: {device}")
    
    # Create VAE
    vae = ConditionalVAE().to(device)
    print(f"VAE created with {sum(p.numel() for p in vae.parameters())} parameters")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 128, 128).to(device)
    labels = torch.randint(0, 7, (batch_size,)).to(device)
    
    reconstructed, mu, log_var = vae(x, labels)
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Log_var shape: {log_var.shape}")
    
    # Test loss
    total_loss, recon_loss, kl_loss = vae.loss_function(reconstructed, x, mu, log_var)
    print(f"Total loss: {total_loss:.4f}, Recon loss: {recon_loss:.4f}, KL loss: {kl_loss:.4f}")
    
    # Test sampling
    samples = vae.sample(0, device, num_samples=2)
    print(f"Sample shape: {samples.shape}")
    
    # Test interpolation
    interpolated = vae.interpolate(0, 1, steps=5, device=device)
    print(f"Interpolated shape: {interpolated.shape}")
    
    print("VAE test completed successfully!")
