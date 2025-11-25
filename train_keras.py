"""TensorFlow/Keras training script for cauliflower disease classification."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from dataset_loader_keras import prepare_data, get_dataset_size
from model_keras import create_model, compile_model


def get_callbacks(save_dir: Path, patience: int = 10) -> list:
    """Create training callbacks."""
    callbacks = [
        # Model checkpoint - save best model
        keras.callbacks.ModelCheckpoint(
            filepath=str(save_dir / 'best_model.h5'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Model checkpoint - save weights only
        keras.callbacks.ModelCheckpoint(
            filepath=str(save_dir / 'best_weights.h5'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            mode='max',
            verbose=1,
            restore_best_weights=True
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # CSV logger
        keras.callbacks.CSVLogger(
            str(save_dir / 'training_log.csv'),
            append=True
        ),
        
        # TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=str(save_dir / 'logs'),
            histogram_freq=0,
            write_graph=False
        )
    ]
    
    return callbacks


def setup_gpu():
    """Configure GPU settings."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"\n✓ GPU(s) detected: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            print()
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("\n⚠ No GPU detected, using CPU\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train cauliflower disease classifier (Keras/H5)')
    
    # Data parameters
    parser.add_argument('--csv-file', type=str, default='dataset_labels.csv',
                        help='Path to CSV file with labels')
    parser.add_argument('--dataset-root', type=str, default='dataset',
                        help='Root directory of dataset')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        choices=['efficientnet_b0', 'resnet50', 'mobilenet_v3'],
                        help='Model architecture')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Do not use pretrained weights')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Loss parameters
    parser.add_argument('--use-focal-loss', action='store_true',
                        help='Use focal loss instead of categorical crossentropy')
    
    # Other parameters
    parser.add_argument('--save-dir', type=str, default='checkpoints_keras',
                        help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds
    tf.random.set_seed(args.seed)
    
    # Setup GPU
    setup_gpu()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, val_dataset, test_dataset, label_to_idx, idx_to_label = prepare_data(
        csv_file=args.csv_file,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        random_seed=args.seed
    )
    
    num_classes = len(label_to_idx)
    
    # Save label mapping
    with open(save_dir / 'label_mapping.json', 'w') as f:
        json.dump(label_to_idx, f, indent=2)
    
    with open(save_dir / 'idx_to_label.json', 'w') as f:
        json.dump(idx_to_label, f, indent=2)
    
    # Create model
    print(f"Creating model: {args.model}")
    model = create_model(
        num_classes=num_classes,
        model_name=args.model,
        img_size=args.img_size,
        pretrained=not args.no_pretrained
    )
    
    # Compile model
    model = compile_model(
        model,
        learning_rate=args.lr,
        use_focal_loss=args.use_focal_loss
    )
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Get callbacks
    callbacks = get_callbacks(save_dir, patience=args.patience)
    
    # Train model
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*60}\n")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    print("\nSaving final model...")
    model.save(save_dir / 'final_model.h5')
    print(f"✓ Final model saved to {save_dir / 'final_model.h5'}")
    
    # Save training history
    history_dict = {
        'loss': [float(x) for x in history.history.get('loss', [])],
        'accuracy': [float(x) for x in history.history.get('accuracy', [])],
        'val_loss': [float(x) for x in history.history.get('val_loss', [])],
        'val_accuracy': [float(x) for x in history.history.get('val_accuracy', [])],
    }
    
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(test_dataset, verbose=1)
    test_metrics = dict(zip(model.metrics_names, test_results))
    
    print(f"\n{'='*60}")
    print("Test Set Results:")
    for metric_name, value in test_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    print(f"{'='*60}\n")
    
    # Save test results
    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)
    
    print(f"✓ Training complete! Models saved to: {save_dir}")
    print(f"\nModel files:")
    print(f"  - best_model.h5 (full model - use this for deployment)")
    print(f"  - final_model.h5 (last epoch model)")
    print(f"  - best_weights.h5 (weights only)")
    print(f"  - label_mapping.json (class labels)")


if __name__ == '__main__':
    main()

