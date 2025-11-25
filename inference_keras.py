"""Keras inference script for single image prediction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image


def load_model(model_path: str) -> keras.Model:
    """Load trained Keras model from .h5 file."""
    model = keras.models.load_model(model_path)
    return model


def preprocess_image(image_path: str, img_size: int = 224) -> tf.Tensor:
    """Preprocess image for inference."""
    # Read image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((img_size, img_size))
    
    # Convert to array and normalize
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    
    # Normalize using ImageNet statistics
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std
    
    # Add batch dimension
    image_tensor = tf.convert_to_tensor(image_array)
    image_tensor = tf.expand_dims(image_tensor, 0)
    
    return image_tensor


def predict_image(
    image_path: str,
    model: keras.Model,
    class_names: list[str],
    img_size: int = 224,
    top_k: int = 3
) -> dict:
    """Predict disease class for a single image."""
    # Preprocess image
    image_tensor = preprocess_image(image_path, img_size)
    
    # Predict
    predictions = model.predict(image_tensor, verbose=0)[0]
    
    # Get top-k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'class': class_names[idx],
            'probability': float(predictions[idx]),
            'confidence_percent': float(predictions[idx] * 100)
        })
    
    return {
        'image_path': image_path,
        'predictions': results,
        'top_prediction': results[0]
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict disease class using Keras model')
    
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file')
    parser.add_argument('--model', type=str, default='checkpoints_keras/best_model.h5',
                        help='Path to Keras model (.h5 file)')
    parser.add_argument('--label-mapping', type=str, default='checkpoints_keras/idx_to_label.json',
                        help='Path to idx_to_label mapping JSON')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Number of top predictions to show')
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Using GPU: {gpus[0].name}\n")
    else:
        print("Using CPU\n")
    
    # Load label mapping
    with open(args.label_mapping, 'r') as f:
        idx_to_label = json.load(f)
    
    # Convert string keys to int and create ordered list
    class_names = [idx_to_label[str(i)] for i in range(len(idx_to_label))]
    
    # Load model
    print("Loading model...")
    model = load_model(args.model)
    print("✓ Model loaded successfully\n")
    
    # Predict
    print(f"Predicting for image: {args.image}")
    result = predict_image(args.image, model, class_names, args.img_size, args.top_k)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Top Prediction: {result['top_prediction']['class']}")
    print(f"Confidence: {result['top_prediction']['confidence_percent']:.2f}%")
    print(f"{'='*60}\n")
    
    print(f"Top {args.top_k} Predictions:")
    print('-' * 60)
    for i, pred in enumerate(result['predictions'], 1):
        print(f"{i}. {pred['class']:25s} - {pred['confidence_percent']:5.2f}%")
    print('-' * 60)


if __name__ == '__main__':
    main()

