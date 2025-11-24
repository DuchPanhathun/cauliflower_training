"""Evaluation script for trained model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from dataset_loader import create_data_loaders
from model import create_model


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: list[str]
) -> dict:
    """Evaluate model on given data loader."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()
    
    # Classification report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'classification_report': report,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: Path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")


def plot_training_history(history_path: Path, save_dir: Path):
    """Plot training history."""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / 'training_history.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved to {save_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--csv-file', type=str, default='dataset_labels.csv',
                        help='Path to CSV file with labels')
    parser.add_argument('--dataset-root', type=str, default='dataset',
                        help='Root directory of dataset')
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        choices=['efficientnet_b0', 'resnet50', 'mobilenet_v3'],
                        help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load label mapping
    label_mapping_path = Path('checkpoints/label_mapping.json')
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    
    class_names = sorted(label_mapping.keys(), key=lambda x: label_mapping[x])
    num_classes = len(class_names)
    
    print(f"Classes: {class_names}\n")
    
    # Create data loaders
    print("Loading data...")
    _, _, test_loader, _ = create_data_loaders(
        csv_file=args.csv_file,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print(f"Loading model from {args.checkpoint}...")
    model = create_model(
        num_classes=num_classes,
        model_name=args.model,
        pretrained=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    print("✓ Model loaded successfully\n")
    
    # Evaluate
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device, class_names)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"{'='*60}\n")
    
    # Print classification report
    print("Classification Report:")
    print('-' * 60)
    report = results['classification_report']
    for class_name in class_names:
        metrics = report[class_name]
        print(f"{class_name:25s} - Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1-score']:.4f}")
    
    print('-' * 60)
    print(f"{'Overall Metrics':25s} - Accuracy: {report['accuracy']:.4f}, "
          f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"{'':25s}   Weighted F1: {report['weighted avg']['f1-score']:.4f}")
    
    # Save results
    results_to_save = {
        'accuracy': float(results['accuracy']),
        'classification_report': results['classification_report']
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        results['confusion_matrix'],
        class_names,
        output_dir / 'confusion_matrix.png'
    )
    
    # Plot training history if available
    history_path = Path('checkpoints/training_history.json')
    if history_path.exists():
        plot_training_history(history_path, output_dir)
    
    print(f"\n✓ Evaluation complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()

