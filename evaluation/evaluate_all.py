"""
Complete evaluation script for Emotion Face Generator project.
Evaluates VAE, cGAN, and VAE-GAN models using FID, PSNR, SSIM, and Emotion Classification Accuracy.
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
import numpy as np
import scipy.linalg
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# ABSOLUTE PATHS
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "dataset")
VAE_CHECKPOINT = BASE + "/outputs/vae_checkpoints/vae_final.pt"
CGAN_CHECKPOINT = BASE + "/outputs/cgan_checkpoints/cgan_epoch_80.pt"
VAEGAN_CHECKPOINT = BASE + "/outputs/vaegan_checkpoints/vaegan_epoch_60.pt"
CLASSIFIER_CHECKPOINT = BASE + "/outputs/classifier/emotion_classifier.pt"
RESULTS_PATH = BASE + "/evaluation/results.json"
PLOTS_PATH = BASE + "/evaluation/comparison_plots.png"

# Add BASE to path for imports
sys.path.insert(0, BASE)

from data.dataloader import get_dataloader
from models.vae import ConditionalVAE
from models.cgan_generator import ConditionalGANGenerator
from models.vaegan import VAEGAN

# Constants
EMOTIONS = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = 8
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cached results for quick access
CACHED_RESULTS = {
    "VAE": {"FID": 211.11, "PSNR": 10.85, "SSIM": 0.2724},
    "cGAN": {"FID": 372.65, "PSNR": 9.09, "SSIM": 0.0964},
    "VAE-GAN": {"FID": 144.70, "PSNR": 9.29, "SSIM": 0.1141}
}


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class EmotionClassifier(nn.Module):
    """
    Emotion classifier using ResNet18 backbone.
    """
    def __init__(self, num_classes=8):
        super(EmotionClassifier, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace final FC layer for emotion classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)
    
    def train_classifier(self, data_path, save_path, epochs=20):
        """
        Train the emotion classifier on real training data.
        """
        print(f"Training emotion classifier for {epochs} epochs...")
        
        # Load training data
        train_loader, train_dataset = get_dataloader(data_path, split='train', batch_size=32, shuffle=True)
        num_classes = train_dataset.get_num_classes()
        
        # Move to device
        self.to(DEVICE)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, labels, _ in pbar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
            
            accuracy = 100. * correct / total
            avg_loss = total_loss / len(train_loader)
            scheduler.step()
            
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'model_state_dict': self.state_dict(),
                    'accuracy': accuracy,
                    'num_classes': num_classes
                }, save_path)
                print(f"Saved best model with accuracy {accuracy:.2f}%")
        
        print(f"Training completed! Best accuracy: {best_accuracy:.2f}%")
        return self
    
    def load_classifier(self, checkpoint_path):
        """Load a trained classifier from checkpoint."""
        print(f"Loading emotion classifier from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.to(DEVICE)
        print(f"Classifier loaded successfully (accuracy: {checkpoint.get('accuracy', 'N/A'):.2f}%)")
        return self
    
    def classify_batch(self, images):
        """
        Classify a batch of images.
        Images should be in [-1, 1] range.
        """
        self.eval()
        
        # Denormalize from [-1, 1] to [0, 1]
        images = (images + 1.0) / 2.0
        
        # Resize to 224x224 for ResNet
        resize = transforms.Resize(224)
        images = resize(images)
        
        # Normalize with ImageNet stats
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images = normalize(images)
        
        with torch.no_grad():
            outputs = self(images)
            _, predicted = outputs.max(1)
        
        return predicted


def load_vae_model(checkpoint_path):
    """Load VAE model from checkpoint."""
    print(f"Loading VAE from {checkpoint_path}...")
    model = ConditionalVAE(num_classes=NUM_CLASSES, latent_dim=256).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("VAE loaded successfully")
    return model


def load_cgan_model(checkpoint_path):
    """Load cGAN generator from checkpoint."""
    print(f"Loading cGAN from {checkpoint_path}...")
    model = ConditionalGANGenerator(nz=128, num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.eval()
    print("cGAN loaded successfully")
    return model


def load_vaegan_model(checkpoint_path):
    """Load VAE-GAN model from checkpoint."""
    print(f"Loading VAE-GAN from {checkpoint_path}...")
    model = VAEGAN(latent_dim=128, num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Load encoder and decoder separately
    model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    model.encoder.eval()
    model.decoder.eval()
    print("VAE-GAN loaded successfully")
    return model


def generate_vae_images(vae_model, label_idx, n, device):
    """Generate images using VAE."""
    vae_model.eval()
    with torch.no_grad():
        z = torch.randn(n, 256).to(device)
        labels = torch.full((n,), label_idx, dtype=torch.long).to(device)
        samples = vae_model.decode(z, labels)
    return samples  # (n, 3, 128, 128) in [-1, 1]


def generate_cgan_images(generator, label_idx, n, device):
    """Generate images using cGAN."""
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(n, 128).to(device)
        labels = torch.full((n,), label_idx, dtype=torch.long).to(device)
        samples = generator(noise, labels)
    return samples  # (n, 3, 64, 64) in [-1, 1]


def generate_vaegan_images(encoder, decoder, label_idx, n, device):
    """Generate images using VAE-GAN."""
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        z = torch.randn(n, 128).to(device)
        labels = torch.full((n,), label_idx, dtype=torch.long).to(device)
        samples = decoder(z, labels)
    return samples  # (n, 3, 64, 64) in [-1, 1]


def load_real_images(split='test', n_per_emotion=50, img_size=64):
    """Load real images from dataset."""
    print(f"Loading {n_per_emotion} real images per emotion from {split} split (size: {img_size}x{img_size})...")
    
    # Create custom transform for the requested size
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load dataset
    from data.dataloader import EmotionDataset
    dataset = EmotionDataset(DATA_PATH, split)
    dataset.transform = transform
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Collect images per emotion
    images_per_emotion = {i: [] for i in range(NUM_CLASSES)}
    
    for images, labels, _ in dataloader:
        for img, label in zip(images, labels):
            label_idx = label.item()
            if len(images_per_emotion[label_idx]) < n_per_emotion:
                images_per_emotion[label_idx].append(img)
        
        if all(len(v) >= n_per_emotion for v in images_per_emotion.values()):
            break
    
    # Concatenate
    real_images = []
    for emotion_idx in range(NUM_CLASSES):
        emotion_images = torch.stack(images_per_emotion[emotion_idx][:n_per_emotion])
        real_images.append(emotion_images)
    
    real_images = torch.cat(real_images, dim=0)
    print(f"Loaded {len(real_images)} real images total")
    return real_images


def extract_inception_features(images, inception_model):
    """Extract features using InceptionV3."""
    # Denormalize from [-1, 1] to [0, 1]
    images = (images + 1.0) / 2.0
    
    # Resize to 299x299 for InceptionV3
    resize = transforms.Resize(299)
    images = resize(images)
    
    # Normalize with ImageNet stats
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    images = normalize(images)
    
    # Extract features in batches
    features = []
    with torch.no_grad():
        for i in range(0, len(images), BATCH_SIZE):
            batch = images[i:i+BATCH_SIZE].to(DEVICE)
            feat = inception_model(batch)
            features.append(feat.cpu())
    
    features = torch.cat(features, dim=0)
    return features


def compute_fid(real_images, fake_images, inception_model):
    """Compute Fréchet Inception Distance."""
    print("Computing FID...")
    
    # Extract features
    real_features = extract_inception_features(real_images, inception_model)
    fake_features = extract_inception_features(fake_images, inception_model)
    
    # Compute statistics
    mu_real = real_features.mean(dim=0).cpu().numpy()
    mu_fake = fake_features.mean(dim=0).cpu().numpy()
    
    sigma_real = torch.cov(real_features.T).cpu().numpy()
    sigma_fake = torch.cov(fake_features.T).cpu().numpy()
    
    # Compute FID
    diff = mu_real - mu_fake
    covmean, _ = scipy.linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    
    # Handle numerical errors
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * 1e-6
        covmean = scipy.linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean)
    
    print(f"FID: {fid:.2f}")
    return float(fid)


def compute_psnr(real_images, fake_images):
    """Compute PSNR (Peak Signal-to-Noise Ratio)."""
    print("Computing PSNR...")
    
    # Denormalize to [0, 1]
    real_np = ((real_images + 1.0) / 2.0).cpu().numpy()
    fake_np = ((fake_images + 1.0) / 2.0).cpu().numpy()
    
    # Resize fake to match real if needed
    if real_np.shape[2:] != fake_np.shape[2:]:
        resize = transforms.Resize(real_images.shape[2])
        fake_images_resized = resize(fake_images)
        fake_np = ((fake_images_resized + 1.0) / 2.0).cpu().numpy()
    
    # Compute per emotion
    psnr_scores = {}
    n_per_emotion = len(real_images) // NUM_CLASSES
    
    for emotion_idx in range(NUM_CLASSES):
        start = emotion_idx * n_per_emotion
        end = start + n_per_emotion
        
        real_emotion = real_np[start:end]
        fake_emotion = fake_np[start:end]
        
        emotion_psnr = []
        for i in range(n_per_emotion):
            mse = np.mean((real_emotion[i] - fake_emotion[i]) ** 2)
            if mse > 0:
                psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            else:
                psnr = 100.0
            emotion_psnr.append(psnr)
        
        avg_psnr = np.mean(emotion_psnr)
        psnr_scores[EMOTIONS[emotion_idx]] = avg_psnr
        print(f"  {EMOTIONS[emotion_idx]}: {avg_psnr:.2f} dB")
    
    overall_psnr = np.mean(list(psnr_scores.values()))
    psnr_scores['average'] = overall_psnr
    print(f"Average PSNR: {overall_psnr:.2f} dB")
    
    return psnr_scores


def compute_ssim(real_images, fake_images):
    """Compute Structural Similarity Index."""
    print("Computing SSIM...")
    
    # Denormalize to [0, 1]
    real_np = ((real_images + 1.0) / 2.0).cpu().numpy()
    fake_np = ((fake_images + 1.0) / 2.0).cpu().numpy()
    
    # Resize fake to match real if needed
    if real_np.shape[2:] != fake_np.shape[2:]:
        resize = transforms.Resize(real_images.shape[2])
        fake_images_resized = resize(fake_images)
        fake_np = ((fake_images_resized + 1.0) / 2.0).cpu().numpy()
    
    # Compute per emotion
    ssim_scores = {}
    n_per_emotion = len(real_images) // NUM_CLASSES
    
    for emotion_idx in range(NUM_CLASSES):
        start = emotion_idx * n_per_emotion
        end = start + n_per_emotion
        
        real_emotion = real_np[start:end]
        fake_emotion = fake_np[start:end]
        
        emotion_ssim = []
        for i in range(n_per_emotion):
            img1 = np.transpose(real_emotion[i], (1, 2, 0))
            img2 = np.transpose(fake_emotion[i], (1, 2, 0))
            ssim_val = ssim(img1, img2, data_range=1.0, channel_axis=2)
            emotion_ssim.append(ssim_val)
        
        avg_ssim = np.mean(emotion_ssim)
        ssim_scores[EMOTIONS[emotion_idx]] = avg_ssim
        print(f"  {EMOTIONS[emotion_idx]}: {avg_ssim:.4f}")
    
    overall_ssim = np.mean(list(ssim_scores.values()))
    ssim_scores['average'] = overall_ssim
    print(f"Average SSIM: {overall_ssim:.4f}")
    
    return ssim_scores


def compute_emotion_accuracy(model, model_type, classifier, n_per_emotion=50):
    """Compute emotion classification accuracy on generated images."""
    print(f"Computing emotion classification accuracy for {model_type}...")
    
    # Generate images
    generated_images = []
    for emotion_idx in range(NUM_CLASSES):
        if model_type == 'VAE':
            samples = generate_vae_images(model, emotion_idx, n_per_emotion, DEVICE)
        elif model_type == 'cGAN':
            samples = generate_cgan_images(model, emotion_idx, n_per_emotion, DEVICE)
        elif model_type == 'VAE-GAN':
            samples = generate_vaegan_images(model.encoder, model.decoder, emotion_idx, n_per_emotion, DEVICE)
        generated_images.append(samples)
    
    generated_images = torch.cat(generated_images, dim=0)
    
    # Create target labels
    target_labels = []
    for emotion_idx in range(NUM_CLASSES):
        target_labels.extend([emotion_idx] * n_per_emotion)
    target_labels = torch.tensor(target_labels).to(DEVICE)
    
    # Classify
    predicted_labels = classifier.classify_batch(generated_images)
    
    # Compute accuracy
    correct = (predicted_labels == target_labels).sum().item()
    total = len(target_labels)
    overall_accuracy = 100.0 * correct / total
    
    # Per-emotion accuracy
    per_emotion_acc = {}
    for emotion_idx in range(NUM_CLASSES):
        start = emotion_idx * n_per_emotion
        end = start + n_per_emotion
        
        emotion_preds = predicted_labels[start:end]
        emotion_targets = target_labels[start:end]
        
        emotion_correct = (emotion_preds == emotion_targets).sum().item()
        emotion_acc = 100.0 * emotion_correct / n_per_emotion
        
        per_emotion_acc[EMOTIONS[emotion_idx]] = emotion_acc
        print(f"  {EMOTIONS[emotion_idx]}: {emotion_acc:.2f}%")
    
    print(f"Overall emotion accuracy: {overall_accuracy:.2f}%")
    
    return {
        'overall': overall_accuracy,
        'per_emotion': per_emotion_acc
    }


def evaluate_model(model_name, model, model_type, classifier, inception_model):
    """Evaluate a single model on all metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    results = {}
    
    try:
        # Determine image size
        if model_type == 'VAE':
            img_size = 128
        else:
            img_size = 64
        
        # Generate images for FID (500 total = 62-63 per emotion)
        print("\n--- FID Evaluation ---")
        n_fid = 500 // NUM_CLASSES
        fake_images = []
        for emotion_idx in range(NUM_CLASSES):
            if model_type == 'VAE':
                samples = generate_vae_images(model, emotion_idx, n_fid, DEVICE)
            elif model_type == 'cGAN':
                samples = generate_cgan_images(model, emotion_idx, n_fid, DEVICE)
            elif model_type == 'VAE-GAN':
                samples = generate_vaegan_images(model.encoder, model.decoder, emotion_idx, n_fid, DEVICE)
            fake_images.append(samples)
        fake_images = torch.cat(fake_images, dim=0)
        
        real_images = load_real_images(split='test', n_per_emotion=n_fid, img_size=img_size)
        
        # Resize if needed
        if fake_images.shape[2:] != real_images.shape[2:]:
            resize = transforms.Resize(real_images.shape[2])
            fake_images = resize(fake_images)
        
        fid_score = compute_fid(real_images, fake_images, inception_model)
        results['FID'] = fid_score
        
        # PSNR and SSIM (50 per emotion)
        print("\n--- PSNR Evaluation ---")
        n_psnr = 50
        fake_images_psnr = []
        for emotion_idx in range(NUM_CLASSES):
            if model_type == 'VAE':
                samples = generate_vae_images(model, emotion_idx, n_psnr, DEVICE)
            elif model_type == 'cGAN':
                samples = generate_cgan_images(model, emotion_idx, n_psnr, DEVICE)
            elif model_type == 'VAE-GAN':
                samples = generate_vaegan_images(model.encoder, model.decoder, emotion_idx, n_psnr, DEVICE)
            fake_images_psnr.append(samples)
        fake_images_psnr = torch.cat(fake_images_psnr, dim=0)
        
        real_images_psnr = load_real_images(split='test', n_per_emotion=n_psnr, img_size=img_size)
        psnr_scores = compute_psnr(real_images_psnr, fake_images_psnr)
        results['PSNR'] = psnr_scores['average']
        
        print("\n--- SSIM Evaluation ---")
        ssim_scores = compute_ssim(real_images_psnr, fake_images_psnr)
        results['SSIM'] = ssim_scores['average']
        
        # Emotion classification accuracy
        print("\n--- Emotion Classification Accuracy ---")
        emotion_acc = compute_emotion_accuracy(model, model_type, classifier, n_per_emotion=50)
        results['emotion_accuracy'] = emotion_acc['overall']
        results['per_emotion_acc'] = emotion_acc['per_emotion']
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
    
    return results


def print_comparison_table(results):
    """Print a formatted comparison table."""
    print("\n" + "╔" + "═" * 12 + "╦" + "═" * 10 + "╦" + "═" * 11 + "╦" + "═" * 10 + "╦" + "═" * 18 + "╗")
    print("║ Model        ║ FID ↓    ║ PSNR ↑    ║ SSIM ↑   ║ Emotion Acc ↑    ║")
    print("╠" + "═" * 12 + "╬" + "═" * 10 + "╬" + "═" * 11 + "╬" + "═" * 10 + "╬" + "═" * 18 + "╣")
    
    for model_name in ['VAE', 'cGAN', 'VAE-GAN']:
        if model_name in results and 'error' not in results[model_name]:
            r = results[model_name]
            fid = r.get('FID', float('nan'))
            psnr = r.get('PSNR', float('nan'))
            ssim = r.get('SSIM', float('nan'))
            acc = r.get('emotion_accuracy', float('nan'))
            
            print(f"║ {model_name:<12} ║ {fid:8.2f} ║ {psnr:9.2f} ║ {ssim:8.4f} ║ {acc:16.2f}% ║")
        else:
            print(f"║ {model_name:<12} ║ {'N/A':>8} ║ {'N/A':>9} ║ {'N/A':>8} ║ {'N/A':>16} ║")
    
    print("╚" + "═" * 12 + "╩" + "═" * 10 + "╩" + "═" * 11 + "╩" + "═" * 10 + "╩" + "═" * 18 + "╝")


def save_comparison_plots(results, save_path):
    """Save comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Comparison Metrics', fontsize=16, fontweight='bold')
    
    models = []
    fid_scores = []
    psnr_scores = []
    ssim_scores = []
    
    for model_name in ['VAE', 'cGAN', 'VAE-GAN']:
        if model_name in results and 'error' not in results[model_name]:
            models.append(model_name)
            fid_scores.append(results[model_name].get('FID', 0))
            psnr_scores.append(results[model_name].get('PSNR', 0))
            ssim_scores.append(results[model_name].get('SSIM', 0))
    
    if not models:
        print("No valid results to plot")
        return
    
    # Plot 1: FID
    axes[0, 0].bar(models, fid_scores, color='red', alpha=0.7)
    axes[0, 0].set_title('FID Score (Lower is Better)', fontweight='bold')
    axes[0, 0].set_ylabel('FID')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(fid_scores):
        axes[0, 0].text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # Plot 2: PSNR
    axes[0, 1].bar(models, psnr_scores, color='green', alpha=0.7)
    axes[0, 1].set_title('PSNR (Higher is Better)', fontweight='bold')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(psnr_scores):
        axes[0, 1].text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # Plot 3: SSIM
    axes[1, 0].bar(models, ssim_scores, color='blue', alpha=0.7)
    axes[1, 0].set_title('SSIM (Higher is Better)', fontweight='bold')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(ssim_scores):
        axes[1, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    # Plot 4: Emotion accuracy per emotion
    axes[1, 1].set_title('Emotion Classification Accuracy per Emotion', fontweight='bold')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_xlabel('Emotion')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    x = np.arange(len(EMOTIONS))
    width = 0.25
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, model_name in enumerate(models):
        if model_name in results and 'per_emotion_acc' in results[model_name]:
            per_emotion = results[model_name]['per_emotion_acc']
            acc_values = [per_emotion.get(emotion, 0) for emotion in EMOTIONS]
            axes[1, 1].bar(x + idx * width, acc_values, width, label=model_name, color=colors[idx], alpha=0.7)
    
    axes[1, 1].set_xticks(x + width)
    axes[1, 1].set_xticklabels(EMOTIONS, rotation=45, ha='right')
    axes[1, 1].legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plots saved to {save_path}")
    plt.close()


def run_evaluation(skip_classifier_training=False):
    """Run evaluation on all models."""
    print(f"Evaluation Configuration:")
    print(f"  Data Path: {DATA_PATH}")
    print(f"  Device: {DEVICE}")
    print(f"  Skip classifier training: {skip_classifier_training}")
    
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Initialize InceptionV3 for FID
    inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    inception.fc = nn.Identity()
    inception.eval()
    inception.to(DEVICE)
    
    # Train or load classifier
    if skip_classifier_training and os.path.exists(CLASSIFIER_CHECKPOINT):
        classifier = EmotionClassifier(num_classes=NUM_CLASSES)
        classifier.load_classifier(CLASSIFIER_CHECKPOINT)
    else:
        classifier = EmotionClassifier(num_classes=NUM_CLASSES)
        classifier.train_classifier(DATA_PATH, CLASSIFIER_CHECKPOINT, epochs=20)
    
    results = {}
    
    # Evaluate VAE
    try:
        vae_model = load_vae_model(VAE_CHECKPOINT)
        results['VAE'] = evaluate_model('VAE', vae_model, 'VAE', classifier, inception)
    except Exception as e:
        print(f"Failed to evaluate VAE: {e}")
        import traceback
        traceback.print_exc()
        results['VAE'] = {'error': str(e)}
    
    # Evaluate cGAN
    try:
        cgan_model = load_cgan_model(CGAN_CHECKPOINT)
        results['cGAN'] = evaluate_model('cGAN', cgan_model, 'cGAN', classifier, inception)
    except Exception as e:
        print(f"Failed to evaluate cGAN: {e}")
        import traceback
        traceback.print_exc()
        results['cGAN'] = {'error': str(e)}
    
    # Evaluate VAE-GAN
    try:
        vaegan_model = load_vaegan_model(VAEGAN_CHECKPOINT)
        results['VAE-GAN'] = evaluate_model('VAE-GAN', vaegan_model, 'VAE-GAN', classifier, inception)
    except Exception as e:
        print(f"Failed to evaluate VAE-GAN: {e}")
        import traceback
        traceback.print_exc()
        results['VAE-GAN'] = {'error': str(e)}
    
    # Print table
    print_comparison_table(results)
    
    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {RESULTS_PATH}")
    
    # Save plots
    save_comparison_plots(results, PLOTS_PATH)
    
    print("\nEvaluation completed!")
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Emotion Face Generation Models')
    parser.add_argument('--skip_classifier_training', action='store_true',
                       help='Skip training classifier if checkpoint exists')
    args = parser.parse_args()
    
    run_evaluation(skip_classifier_training=args.skip_classifier_training)


if __name__ == "__main__":
    main()
