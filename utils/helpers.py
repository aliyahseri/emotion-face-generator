import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import torchvision.transforms as transforms


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def save_image_grid(tensor_list, path, nrow=8, normalize=True):
    """
    Save a grid of tensors as a PNG image.
    
    Args:
        tensor_list: List of tensors or single tensor [N, C, H, W]
        path (str): Output file path
        nrow (int): Number of images per row
        normalize (bool): Whether to normalize tensors to [0, 1]
    """
    if isinstance(tensor_list, list):
        tensor_list = torch.cat(tensor_list, dim=0)
    
    if normalize:
        tensor_list = denormalize(tensor_list)
    
    # Ensure values are in [0, 1]
    tensor_list = torch.clamp(tensor_list, 0, 1)
    
    # Create grid and save
    grid = make_grid(tensor_list, nrow=nrow, padding=2, normalize=False)
    save_image(grid, path)
    print(f"Image grid saved to {path}")


def denormalize(tensor):
    """
    Convert tensor from [-1, 1] to [0, 1] range.
    
    Args:
        tensor: Input tensor in [-1, 1] range
    
    Returns:
        Tensor in [0, 1] range
    """
    return (tensor + 1.0) / 2.0


def get_label_map(data_path):
    """
    Get label mapping from dataset folder names.
    
    Args:
        data_path (str): Path to dataset root
    
    Returns:
        dict: Mapping from emotion name to index
    """
    train_path = os.path.join(data_path, 'train')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data directory not found: {train_path}")
    
    # Get sorted folder names
    emotion_folders = sorted([f for f in os.listdir(train_path) 
                             if os.path.isdir(os.path.join(train_path, f))])
    
    # Create mapping
    label_to_idx = {emotion: idx for idx, emotion in enumerate(emotion_folders)}
    
    print(f"Label mapping: {label_to_idx}")
    return label_to_idx


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_losses(g_losses, d_losses, save_path):
    """
    Plot and save generator and discriminator loss curves.
    
    Args:
        g_losses (list): List of generator losses
        d_losses (list): List of discriminator losses
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss', color='blue')
    plt.plot(d_losses, label='Discriminator Loss', color='red')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {save_path}")


def create_emotion_classifier(num_classes=7, pretrained=True):
    """
    Create a simple emotion classifier based on ResNet18.
    
    Args:
        num_classes (int): Number of emotion classes
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        nn.Module: Emotion classifier model
    """
    import torchvision.models as models
    
    # Load pretrained ResNet18
    model = models.resnet18(pretrained=pretrained)
    
    # Modify the final layer for emotion classification
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def calculate_fid(real_features, fake_features):
    """
    Calculate Fréchet Inception Distance (FID) between real and fake features.
    
    Args:
        real_features (torch.Tensor): Features from real images [N, D]
        fake_features (torch.Tensor): Features from fake images [N, D]
    
    Returns:
        float: FID score
    """
    import scipy.linalg
    
    # Convert to numpy
    mu_real = real_features.mean(dim=0).cpu().numpy()
    mu_fake = fake_features.mean(dim=0).cpu().numpy()
    
    sigma_real = torch.cov(real_features.T).cpu().numpy()
    sigma_fake = torch.cov(fake_features.T).cpu().numpy()
    
    # Calculate FID
    diff = mu_real - mu_fake
    
    # Calculate square root of product of covariances
    covmean, _ = scipy.linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    
    # Handle numerical errors
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % 1e-6
        print(msg)
        offset = np.eye(sigma_real.shape[0]) * 1e-6
        covmean = scipy.linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    
    # Calculate FID
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * tr_covmean
    
    return float(fid)


def extract_inception_features(images, device='cpu'):
    """
    Extract features using InceptionV3 model.
    
    Args:
        images (torch.Tensor): Input images [N, 3, H, W] in [-1, 1] range
        device (str): Device to run on
    
    Returns:
        torch.Tensor: Features [N, D]
    """
    import torchvision.models as models
    
    # Load InceptionV3
    inception = models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = nn.Identity()  # Remove final classification layer
    inception.eval()
    inception.to(device)
    
    # Denormalize images from [-1, 1] to [0, 1] and resize to 299x299
    images = denormalize(images)
    resize_transform = transforms.Resize(299)
    images = resize_transform(images)
    
    # Normalize with ImageNet stats
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
    images = normalize(images)
    
    # Extract features
    with torch.no_grad():
        features = inception(images)
    
    return features


def print_metrics_table(metrics_dict):
    """
    Print a formatted table of metrics.
    
    Args:
        metrics_dict (dict): Dictionary of metric names and values
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    for metric_name, metric_value in metrics_dict.items():
        if isinstance(metric_value, float):
            print(f"{metric_name:25s}: {metric_value:.4f}")
        elif isinstance(metric_value, dict):
            print(f"\n{metric_name}:")
            for sub_name, sub_value in metric_value.items():
                if isinstance(sub_value, float):
                    print(f"  {sub_name:20s}: {sub_value:.4f}")
                else:
                    print(f"  {sub_name:20s}: {sub_value}")
        else:
            print(f"{metric_name:25s}: {metric_value}")
    
    print("="*60)


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test set_seed
    set_seed(42)
    
    # Test tensor operations
    test_tensor = torch.randn(4, 3, 64, 64) * 2 - 1  # [-1, 1] range
    denorm_tensor = denormalize(test_tensor)
    print(f"Original range: [{test_tensor.min():.3f}, {test_tensor.max():.3f}]")
    print(f"Denormalized range: [{denorm_tensor.min():.3f}, {denorm_tensor.max():.3f}]")
    
    # Test model parameter counting
    from models.vae import ConditionalVAE
    model = ConditionalVAE()
    param_count = count_parameters(model)
    print(f"VAE parameters: {param_count:,}")
    
    print("Utility functions test completed!")
