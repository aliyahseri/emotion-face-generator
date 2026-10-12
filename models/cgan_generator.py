import torch
import torch.nn as nn


class ConditionalGANGenerator(nn.Module):
    """
    Conditional Generator for 64x64 emotion face generation.
    
    Architecture:
    - Input: noise (nz=128) + label embedding (64)
    - FC: Linear(128+64, 256*4*4)
    - 4 ConvTranspose2d blocks: 256→128→64→32→3
    - Output: 3x64x64 images
    """
    
    def __init__(self, nz=128, num_classes=8):
        """
        Initialize the Conditional Generator.
        
        Args:
            nz (int): Size of the noise vector
            num_classes (int): Number of emotion classes
        """
        super(ConditionalGANGenerator, self).__init__()
        
        self.nz = nz
        self.num_classes = num_classes
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, 64)
        
        # Fully connected layer to project concatenated input to initial feature map
        self.fc = nn.Linear(nz + 64, 256 * 4 * 4)
        
        # Main generator network with ConvTranspose2d blocks
        self.main = nn.Sequential(
            # Block 1: 256 -> 128, 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128x8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Block 2: 128 -> 64, 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Block 3: 64 -> 32, 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # Block 4: 32 -> 3, 32x32 -> 64x64
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 3x64x64
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        """
        Forward pass of the conditional generator.
        
        Args:
            noise: Random noise vectors [batch_size, nz]
            labels: Emotion labels [batch_size]
        
        Returns:
            generated: Generated images [batch_size, channels, height, width]
        """
        # Get label embeddings
        label_emb = self.label_embedding(labels)  # [batch_size, 64]
        
        # Concatenate noise and label embedding
        x = torch.cat([noise, label_emb], dim=1)
        
        # Pass through fully connected layer
        x = self.fc(x)
        x = x.view(x.size(0), 256, 4, 4)  # Reshape to 256x4x4 feature map
        
        # Generate image
        generated = self.main(x)  # [batch_size, 3, 64, 64]
        
        return generated
    
    def generate(self, label_idx, device, num_samples=1):
        """
        Generate samples for a specific emotion.
        
        Args:
            label_idx: Emotion label index
            device: Device to generate on
            num_samples: Number of samples to generate
        
        Returns:
            Generated images tensor
        """
        self.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.nz, device=device)
            labels = torch.full((num_samples,), label_idx, device=device, dtype=torch.long)
            generated = self.forward(noise, labels)
        return generated


if __name__ == "__main__":
    # Test the Generator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing Conditional GAN Generator on device: {device}")
    
    # Create generator
    generator = ConditionalGANGenerator(nz=128, num_classes=8).to(device)
    print(f"Generator created with {sum(p.numel() for p in generator.parameters())} parameters")
    
    # Test forward pass
    batch_size = 4
    noise = torch.randn(batch_size, 128).to(device)
    labels = torch.randint(0, 8, (batch_size,)).to(device)
    
    generated = generator(noise, labels)
    print(f"Noise shape: {noise.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Generated shape: {generated.shape}")
    print(f"Generated range: [{generated.min().item():.3f}, {generated.max().item():.3f}]")
    
    # Test with specific emotion
    happy_faces = generator.generate(label_idx=4, device=device, num_samples=2)
    print(f"Happy faces shape: {happy_faces.shape}")
    
    print("Generator test completed successfully!")
