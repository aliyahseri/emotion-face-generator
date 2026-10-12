import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEGAN_Encoder(nn.Module):
    """VAE-GAN Encoder for 64x64 images"""
    
    def __init__(self, latent_dim=128, num_classes=8):
        super(VAEGAN_Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, 64)
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # 3x64x64 -> 64x32x32
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 64x32x32 -> 128x16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 128x16x16 -> 256x8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 256x8x8 -> 512x4x4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        # Fully connected layers for mu and logvar
        self.fc_mu = nn.Linear(512 * 4 * 4 + 64, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4 + 64, latent_dim)
    
    def forward(self, x, labels):
        # Convolutional encoding
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Get label embedding
        label_emb = self.label_embedding(labels)  # [batch_size, 64]
        
        # Concatenate features with label embedding
        x = torch.cat([x, label_emb], dim=1)
        
        # Get mu and logvar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class VAEGAN_Decoder(nn.Module):
    """VAE-GAN Decoder (acts as Generator)"""
    
    def __init__(self, latent_dim=128, num_classes=8):
        super(VAEGAN_Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, 64)
        
        # Fully connected layer
        self.fc = nn.Linear(latent_dim + 64, 512 * 4 * 4)
        
        # Deconvolutional layers
        self.deconv_layers = nn.Sequential(
            # 512x4x4 -> 256x8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 256x8x8 -> 128x16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 128x16x16 -> 64x32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 64x32x32 -> 3x64x64
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        # Get label embedding
        label_emb = self.label_embedding(labels)  # [batch_size, 64]
        
        # Concatenate latent vector with label embedding
        x = torch.cat([z, label_emb], dim=1)
        
        # Fully connected
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)  # Reshape
        
        # Deconvolution
        x = self.deconv_layers(x)
        
        return x


class VAEGAN_Discriminator(nn.Module):
    """VAE-GAN Discriminator (same as cGAN discriminator)"""
    
    def __init__(self, num_classes=8, img_size=64, channels=3):
        super(VAEGAN_Discriminator, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.channels = channels
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, 64)
        
        # FC layer to project label embedding to spatial dimensions
        self.label_fc = nn.Linear(64, img_size * img_size)
        
        # Discriminator layers: 4 conv layers
        # Input: 4x64x64 (3 image channels + 1 label channel)
        self.conv1 = nn.Conv2d(channels + 1, 64, kernel_size=4, stride=2, padding=1)  # 64x32x32
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 128x16x16
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 256x8x8
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # 512x4x4
        
        # Final fully connected layer
        self.fc = nn.Linear(512 * 4 * 4, 1)
        
    def forward(self, img, labels):
        # Get label embeddings
        label_emb = self.label_embedding(labels)  # [batch_size, 64]
        
        # Project label embedding to spatial dimensions
        label_map = self.label_fc(label_emb)  # [batch_size, img_size * img_size]
        label_map = label_map.view(label_map.size(0), 1, self.img_size, self.img_size)  # [batch_size, 1, 64, 64]
        
        # Concatenate image with label map
        combined = torch.cat([img, label_map], dim=1)  # [batch_size, channels + 1, 64, 64]
        
        # Pass through discriminator layers
        x = F.leaky_relu(self.conv1(combined), 0.2)  # [batch_size, 64, 32, 32]
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)  # [batch_size, 128, 16, 16]
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)  # [batch_size, 256, 8, 8]
        
        x = self.conv4(x)  # [batch_size, 512, 4, 4]
        x = x.view(x.size(0), -1)  # [batch_size, 8192]
        
        # Final classification
        validity = self.fc(x)  # [batch_size, 1]
        validity = torch.sigmoid(validity)  # [batch_size, 1]
        
        return validity


class VAEGAN(nn.Module):
    """Complete VAE-GAN model"""
    
    def __init__(self, latent_dim=128, num_classes=8):
        super(VAEGAN, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.encoder = VAEGAN_Encoder(latent_dim, num_classes)
        self.decoder = VAEGAN_Decoder(latent_dim, num_classes)
        self.discriminator = VAEGAN_Discriminator(num_classes)
        
    def forward(self, x, labels):
        mu, log_var = self.encode(x, labels)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z, labels)
        return recon, mu, log_var
    
    def encode(self, x, labels):
        return self.encoder(x, labels)
    
    def decode(self, z, labels):
        return self.decoder(z, labels)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def sample(self, label_idx, n, device):
        """Generate n samples for given emotion label"""
        self.eval()
        with torch.no_grad():
            # Sample random latent vectors
            z = torch.randn(n, self.latent_dim, device=device)
            labels = torch.full((n,), label_idx, device=device, dtype=torch.long)
            samples = self.decode(z, labels)
        return samples


if __name__ == "__main__":
    import torch
    model = VAEGAN(num_classes=8, latent_dim=128)
    x = torch.randn(4, 3, 64, 64)
    labels = torch.randint(0, 8, (4,))
    recon, mu, logvar = model(x, labels)
    print(f"Input: {x.shape}")
    print(f"Output: {recon.shape}")
    print(f"mu: {mu.shape}")
    print(f"Test passed!")
