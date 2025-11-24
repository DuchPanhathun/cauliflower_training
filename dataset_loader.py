"""Dataset loader for cauliflower disease classification."""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


class CauliflowerDataset(Dataset):
    """Custom dataset for cauliflower disease images."""
    
    def __init__(
        self,
        csv_file: str | Path,
        dataset_root: str | Path,
        transform: Optional[transforms.Compose] = None,
        label_to_idx: Optional[dict] = None
    ):
        """
        Args:
            csv_file: Path to the CSV file with image_path and label columns
            dataset_root: Root directory of the dataset
            transform: Optional transform to be applied on images
            label_to_idx: Optional mapping from label names to indices
        """
        self.df = pd.read_csv(csv_file)
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        
        # Create label to index mapping
        if label_to_idx is None:
            unique_labels = sorted(self.df['label'].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx
        
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get image and label at index."""
        row = self.df.iloc[idx]
        img_path = self.dataset_root / row['image_path']
        label = row['label']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert label to index
        label_idx = self.label_to_idx[label]
        
        return image, label_idx
    
    def get_class_names(self) -> list[str]:
        """Return list of class names."""
        return [self.idx_to_label[i] for i in range(len(self.idx_to_label))]


def get_transforms(is_training: bool = True, img_size: int = 224) -> transforms.Compose:
    """Get image transforms for training or validation."""
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders(
    csv_file: str | Path,
    dataset_root: str | Path,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    img_size: int = 224,
    num_workers: int = 4,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create train, validation, and test data loaders.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, label_mapping)
    """
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Create label mapping
    unique_labels = sorted(df['label'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Split dataset: train/temp -> train/val/test
    train_df, temp_df = train_test_split(
        df, 
        test_size=(val_split + test_split),
        stratify=df['label'],
        random_state=random_seed
    )
    
    val_size = val_split / (val_split + test_split)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        stratify=temp_df['label'],
        random_state=random_seed
    )
    
    # Save splits to CSV
    train_df.to_csv('train_split.csv', index=False)
    val_df.to_csv('val_split.csv', index=False)
    test_df.to_csv('test_split.csv', index=False)
    
    print(f"Dataset split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Create datasets
    train_dataset = CauliflowerDataset(
        'train_split.csv',
        dataset_root,
        transform=get_transforms(is_training=True, img_size=img_size),
        label_to_idx=label_to_idx
    )
    
    val_dataset = CauliflowerDataset(
        'val_split.csv',
        dataset_root,
        transform=get_transforms(is_training=False, img_size=img_size),
        label_to_idx=label_to_idx
    )
    
    test_dataset = CauliflowerDataset(
        'test_split.csv',
        dataset_root,
        transform=get_transforms(is_training=False, img_size=img_size),
        label_to_idx=label_to_idx
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, label_to_idx

