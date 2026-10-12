import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import get_dataloader
from models.vae import ConditionalVAE
from models.cgan_generator import ConditionalGANGenerator
from utils.helpers import (
    set_seed, denormalize, extract_inception_features, 
    calculate_fid, create_emotion_classifier, print_metrics_table
)

# Import torchmetrics
try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    print("Warning: torchmetrics not available. PSNR and SSIM will be skipped.")
    TORCHMETRICS_AVAILABLE = False


class EmotionClassifier(nn.Module):
    """
    Simple emotion classifier for evaluation.
    """
    def __init__(self, num_classes=7):
        super(EmotionClassifier, self).__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_emotion_classifier(train_loader, device, num_classes=7, epochs=10):
    """
    Train a simple emotion classifier on the training set.
    
    Args:
        train_loader: Training data loader
        device: Device to train on
        num_classes: Number of emotion classes
        epochs: Number of training epochs
    
    Returns:
        Trained classifier model
    """
    print("Training emotion classifier...")
    
    classifier = EmotionClassifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    
    classifier.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels, _ in tqdm(train_loader, desc=f"Classifier Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}, Accuracy = {accuracy:.2f}%")
    
    classifier.eval()
    print("Emotion classifier training completed!")
    return classifier


def evaluate_emotion_accuracy(classifier, generated_images, target_labels, device):
    """
    Evaluate emotion classification accuracy on generated images.
    
    Args:
        classifier: Trained emotion classifier
        generated_images: Generated images [N, 3, H, W]
        target_labels: Target emotion labels [N]
        device: Device to evaluate on
    
    Returns:
        dict: Per-emotion accuracy and overall accuracy
    """
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(generated_images)
        _, predicted = outputs.max(1)
    
    # Calculate overall accuracy
    overall_accuracy = (predicted == target_labels).float().mean().item()
    
    # Calculate per-emotion accuracy
    per_emotion_accuracy = {}
    unique_labels = torch.unique(target_labels)
    
    for label in unique_labels:
        mask = target_labels == label
        if mask.sum() > 0:
            emotion_accuracy = (predicted[mask] == target_labels[mask]).float().mean().item()
            per_emotion_accuracy[f"class_{label.item()}"] = emotion_accuracy
    
    return {
        "overall_accuracy": overall_accuracy,
        "per_emotion_accuracy": per_emotion_accuracy
    }


def calculate_psnr_ssim(real_images, generated_images, device):
    """
    Calculate PSNR and SSIM metrics.
    
    Args:
        real_images: Real images [N, 3, H, W]
        generated_images: Generated images [N, 3, H, W]
        device: Device to calculate on
    
    Returns:
        dict: PSNR and SSIM scores
    """
    if not TORCHMETRICS_AVAILABLE:
        return {"psnr": None, "ssim": None}
    
    # Denormalize images to [0, 1]
    real_images = denormalize(real_images)
    generated_images = denormalize(generated_images)
    
    # Initialize metrics
    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)
    
    # Calculate metrics
    with torch.no_grad():
        psnr_score = psnr_metric(generated_images, real_images).item()
        ssim_score = ssim_metric(generated_images, real_images).item()
    
    return {"psnr": psnr_score, "ssim": ssim_score}


def evaluate_model(model_type, checkpoint_path, data_path, device, num_samples=500):
    """
    Evaluate a trained model (VAE or cGAN).
    
    Args:
        model_type (str): 'vae' or 'cgan'
        checkpoint_path (str): Path to model checkpoint
        data_path (str): Path to dataset
        device: Device to evaluate on
        num_samples (int): Number of samples to generate for evaluation
    
    Returns:
        dict: Evaluation metrics
    """
    print(f"Evaluating {model_type.upper()} model...")
    
    # Load test data
    _, test_dataset = get_dataloader(data_path, split='test', batch_size=32, shuffle=False)
    num_classes = test_dataset.get_num_classes()
    label_to_idx, idx_to_label = test_dataset.get_label_mapping()
    
    # Load model
    if model_type == 'vae':
        model = ConditionalVAE(num_classes=num_classes).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Generate samples
        model.eval()
        generated_images = []
        target_labels = []
        
        with torch.no_grad():
            for emotion_idx in range(num_classes):
                samples_per_emotion = num_samples // num_classes
                samples = model.sample(emotion_idx, device, num_samples=samples_per_emotion)
                generated_images.append(samples)
                target_labels.extend([emotion_idx] * samples_per_emotion)
        
        generated_images = torch.cat(generated_images, dim=0)
        target_labels = torch.tensor(target_labels).to(device)
        
    elif model_type == 'cgan':
        model = ConditionalGANGenerator(num_classes=num_classes).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['generator_state_dict'])
        
        # Generate samples
        model.eval()
        generated_images = []
        target_labels = []
        
        with torch.no_grad():
            for emotion_idx in range(num_classes):
                samples_per_emotion = num_samples // num_classes
                samples = model.generate(emotion_idx, device, num_samples=samples_per_emotion)
                generated_images.append(samples)
                target_labels.extend([emotion_idx] * samples_per_emotion)
        
        generated_images = torch.cat(generated_images, dim=0)
        target_labels = torch.tensor(target_labels).to(device)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Generated {len(generated_images)} samples for evaluation")
    
    # Load real test images for comparison
    test_loader, _ = get_dataloader(data_path, split='test', batch_size=32, shuffle=False)
    real_images = []
    real_labels = []
    
    for images, labels, _ in test_loader:
        real_images.append(images)
        real_labels.append(labels)
        if len(real_images) * 32 >= num_samples:
            break
    
    real_images = torch.cat(real_images, dim=0)[:len(generated_images)]
    real_labels = torch.cat(real_labels, dim=0)[:len(generated_images)]
    
    # Calculate FID score
    print("Calculating FID score...")
    real_features = extract_inception_features(real_images, device)
    fake_features = extract_inception_features(generated_images, device)
    fid_score = calculate_fid(real_features, fake_features)
    
    # Calculate PSNR and SSIM
    print("Calculating PSNR and SSIM...")
    psnr_ssim_metrics = calculate_psnr_ssim(real_images, generated_images, device)
    
    # Train emotion classifier and evaluate accuracy
    print("Training emotion classifier...")
    train_loader, _ = get_dataloader(data_path, split='train', batch_size=32, shuffle=True)
    classifier = train_emotion_classifier(train_loader, device, num_classes, epochs=5)
    
    print("Evaluating emotion classification accuracy...")
    emotion_accuracy = evaluate_emotion_accuracy(classifier, generated_images, target_labels, device)
    
    # Compile metrics
    metrics = {
        "model_type": model_type.upper(),
        "num_samples": len(generated_images),
        "fid_score": fid_score,
        "psnr": psnr_ssim_metrics["psnr"],
        "ssim": psnr_ssim_metrics["ssim"],
        "emotion_accuracy": emotion_accuracy
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Emotion Face Generation Models')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, required=True, choices=['vae', 'cgan'],
                       help='Type of model to evaluate')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                       default='dataset',
                       help='Path to the dataset')
    
    # Evaluation arguments
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Number of samples to generate for evaluation')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set device and seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)
    
    print(f"Evaluation Configuration:")
    print(f"  Model Type: {args.model_type}")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Data Path: {args.data_path}")
    print(f"  Device: {device}")
    print(f"  Num Samples: {args.num_samples}")
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    # Evaluate model
    metrics = evaluate_model(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        data_path=args.data_path,
        device=device,
        num_samples=args.num_samples
    )
    
    # Print results
    print_metrics_table(metrics)
    
    # Save results to file
    results_path = f"evaluation_results_{args.model_type}.txt"
    with open(results_path, 'w') as f:
        f.write(f"Evaluation Results for {args.model_type.upper()} Model\n")
        f.write("="*60 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key:25s}: {value:.4f}\n")
            elif isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for sub_key, sub_value in value.items():
                    f.write(f"  {sub_key:20s}: {sub_value:.4f}\n")
            else:
                f.write(f"{key:25s}: {value}\n")
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
