import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalGANDiscriminator(nn.Module):
    """
    Conditional Discriminator for 64x64 emotion face classification.
    
    Architecture:
    - Input: 3x64x64 image + label embedding tiled as extra channel
    - Label embedding: nn.Embedding(8, 64) → Linear(64, 64*64) → reshape to 1x64x64
    - Concat image + label channel → 4x64x64
    - 4 conv layers: 4→64→128→256→512, stride=2, kernel=4, padding=1
    - LeakyReLU(0.2), BatchNorm on layers 2-3 only
    - Flatten → Linear(512*4*4, 1) → Sigmoid
    """
    
    def __init__(self, num_classes=8, img_size=64, channels=3):
        super(ConditionalGANDiscriminator, self).__init__()
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
        """
        Forward pass of the conditional discriminator.
        
        Args:
            img: Input images [batch_size, channels, height, width]
            labels: Emotion labels [batch_size]
        
        Returns:
            validity: Real/fake probability [batch_size, 1]
        """
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


if __name__ == "__main__":
    # Test the Discriminator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing Conditional GAN Discriminator on device: {device}")
    
    # Create discriminator
    discriminator = ConditionalGANDiscriminator(num_classes=8, img_size=64).to(device)
    print(f"Discriminator created with {sum(p.numel() for p in discriminator.parameters())} parameters")
    
    # Test forward pass
    batch_size = 4
    img = torch.randn(batch_size, 3, 64, 64).to(device)
    labels = torch.randint(0, 8, (batch_size,)).to(device)
    
    validity = discriminator(img, labels)
    print(f"Image shape: {img.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Validity shape: {validity.shape}")
    print(f"Validity range: [{validity.min().item():.3f}, {validity.max().item():.3f}]")
    
    # Test with real and fake labels
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)
    
    print(f"Real labels shape: {real_labels.shape}")
    print(f"Fake labels shape: {fake_labels.shape}")
    
    print("Discriminator test completed successfully!")
