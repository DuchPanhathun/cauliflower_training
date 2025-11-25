"""TensorFlow/Keras dataset loader for cauliflower disease classification."""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_and_preprocess_image(
    image_path: str,
    label: int,
    img_size: int = 224,
    is_training: bool = True
) -> Tuple[tf.Tensor, int]:
    """Load and preprocess a single image."""
    # Read image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    
    if is_training:
        # Training augmentations
        image = tf.image.resize(image, [img_size + 32, img_size + 32])
        image = tf.image.random_crop(image, [img_size, img_size, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.1)
    else:
        # Validation/test preprocessing
        image = tf.image.resize(image, [img_size, img_size])
    
    # Normalize to [0, 1]
    image = image / 255.0
    
    # Normalize using ImageNet statistics
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    return image, label


def create_dataset(
    image_paths: list,
    labels: list,
    batch_size: int = 32,
    img_size: int = 224,
    is_training: bool = True,
    shuffle: bool = True
) -> tf.data.Dataset:
    """Create a TensorFlow dataset."""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    
    # Map preprocessing function
    dataset = dataset.map(
        lambda x, y: load_and_preprocess_image(x, y, img_size, is_training),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def prepare_data(
    csv_file: str | Path,
    dataset_root: str | Path,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    img_size: int = 224,
    random_seed: int = 42
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict, dict]:
    """
    Prepare train, validation, and test datasets.
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, label_to_idx, idx_to_label)
    """
    # Read CSV
    df = pd.read_csv(csv_file)
    dataset_root = Path(dataset_root)
    
    # Create label mapping
    unique_labels = sorted(df['label'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    # Convert to full paths and numeric labels
    df['full_path'] = df['image_path'].apply(lambda x: str(dataset_root / x))
    df['label_idx'] = df['label'].map(label_to_idx)
    
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
    
    print(f"Dataset split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Classes: {unique_labels}\n")
    
    # Create datasets
    train_dataset = create_dataset(
        train_df['full_path'].tolist(),
        train_df['label_idx'].tolist(),
        batch_size=batch_size,
        img_size=img_size,
        is_training=True,
        shuffle=True
    )
    
    val_dataset = create_dataset(
        val_df['full_path'].tolist(),
        val_df['label_idx'].tolist(),
        batch_size=batch_size,
        img_size=img_size,
        is_training=False,
        shuffle=False
    )
    
    test_dataset = create_dataset(
        test_df['full_path'].tolist(),
        test_df['label_idx'].tolist(),
        batch_size=batch_size,
        img_size=img_size,
        is_training=False,
        shuffle=False
    )
    
    return train_dataset, val_dataset, test_dataset, label_to_idx, idx_to_label


def get_dataset_size(csv_file: str | Path, split_ratio: float = 0.7) -> int:
    """Get approximate number of training samples."""
    df = pd.read_csv(csv_file)
    return int(len(df) * split_ratio)

