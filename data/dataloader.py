import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import defaultdict
import random


class EmotionDataset(Dataset):
    def __init__(self, root_path, split='train'):
        """
        PyTorch Dataset for AffectNet emotion dataset.
        
        Args:
            root_path (str): Root path to dataset folder
            split (str): 'train' or 'test'
        """
        self.root_path = os.path.join(root_path, split)
        self.split = split
        
        # Get emotion labels (folder names)
        self.emotion_folders = sorted([f for f in os.listdir(self.root_path) 
                                     if os.path.isdir(os.path.join(self.root_path, f))])
        
        # Create label to index mapping
        self.label_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotion_folders)}
        self.idx_to_label = {idx: emotion for emotion, idx in self.label_to_idx.items()}
        
        # Collect all image paths and labels
        self.samples = []
        self.class_counts = defaultdict(int)
        
        print(f"Loading {split} dataset from {self.root_path}")
        
        for emotion_folder in self.emotion_folders:
            emotion_path = os.path.join(self.root_path, emotion_folder)
            emotion_idx = self.label_to_idx[emotion_folder]
            
            # Get all image files
            image_files = []
            valid_exts = ('.jpg', '.jpeg', '.png')
            image_files.extend([f for f in os.listdir(emotion_path) 
                              if f.lower().endswith(valid_exts)])
            
            # Add samples
            for img_file in image_files:
                img_path = os.path.join(emotion_path, img_file)
                self.samples.append((img_path, emotion_idx, emotion_folder))
                self.class_counts[emotion_folder] += 1
        
        print(f"Dataset loaded successfully!")
        print(f"Total samples: {len(self.samples)}")
        print("Class distribution:")
        for emotion, count in self.class_counts.items():
            print(f"  {emotion}: {count} samples")
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label_idx, label_name = self.samples[idx]
        
        try:
            # Load and transform image
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            
            return image, label_idx, label_name
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random valid sample as fallback
            return self[random.randint(0, len(self) - 1)]
    
    def get_num_classes(self):
        return len(self.emotion_folders)
    
    def get_label_mapping(self):
        return self.label_to_idx, self.idx_to_label


def get_dataloader(root_path, split='train', batch_size=32, shuffle=None, num_workers=4):
    """
    Create a DataLoader for the emotion dataset.
    
    Args:
        root_path (str): Root path to dataset
        split (str): 'train' or 'test'
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data. If None, defaults to True for train, False for test
        num_workers (int): Number of worker processes
    
    Returns:
        DataLoader: PyTorch DataLoader
    """
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = EmotionDataset(root_path, split)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader, dataset


if __name__ == "__main__":
    # Test the dataloader
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
    
    print("Testing train dataloader...")
    train_loader, train_dataset = get_dataloader(data_path, 'train', batch_size=4)
    
    print("\nTesting test dataloader...")
    test_loader, test_dataset = get_dataloader(data_path, 'test', batch_size=4)
    
    # Test loading a batch
    print("\nTesting batch loading...")
    for images, labels, label_names in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        print(f"Label names: {label_names}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        break
    
    print("\nDataloader test completed successfully!")
