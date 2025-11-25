"""Keras evaluation script for trained model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from dataset_loader_keras import prepare_data


def evaluate_model(
    model: keras.Model,
    dataset: tf.data.Dataset,
    class_names: list[str]
) -> dict:
    """Evaluate model on given dataset."""
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating model...")
    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        
        all_probs.extend(predictions)
        all_preds.extend(np.argmax(predictions, axis=1))
        all_labels.extend(np.argmax(labels, axis=1))
    
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
    plt.close()


def plot_training_history(history_path: Path, save_dir: Path):
    """Plot training history."""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    if 'loss' in history:
        ax1.plot(history['loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    if 'accuracy' in history:
        ax2.plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    if 'val_accuracy' in history:
        ax2.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / 'training_history.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved to {save_path}")
    plt.close()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained Keras model')
    
    parser.add_argument('--model', type=str, default='checkpoints_keras/best_model.h5',
                        help='Path to Keras model (.h5 file)')
    parser.add_argument('--csv-file', type=str, default='dataset_labels.csv',
                        help='Path to CSV file with labels')
    parser.add_argument('--dataset-root', type=str, default='dataset',
                        help='Root directory of dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output-dir', type=str, default='evaluation_results_keras',
                        help='Directory to save evaluation results')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Using GPU: {gpus[0].name}\n")
    else:
        print("Using CPU\n")
    
    # Load label mapping
    label_mapping_path = Path('checkpoints_keras/idx_to_label.json')
    with open(label_mapping_path, 'r') as f:
        idx_to_label = json.load(f)
    
    class_names = [idx_to_label[str(i)] for i in range(len(idx_to_label))]
    print(f"Classes: {class_names}\n")
    
    # Prepare test dataset
    print("Loading data...")
    _, _, test_dataset, _, _ = prepare_data(
        csv_file=args.csv_file,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size
    )
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = keras.models.load_model(args.model)
    print("✓ Model loaded successfully\n")
    
    # Evaluate
    results = evaluate_model(model, test_dataset, class_names)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"{'='*60}\n")
    
    # Print classification report
    print("Classification Report:")
    print('-' * 60)
    report = results['classification_report']
    for class_name in class_names:
        if class_name in report:
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
    history_path = Path('checkpoints_keras/training_history.json')
    if history_path.exists():
        plot_training_history(history_path, output_dir)
    
    print(f"\n✓ Evaluation complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()

