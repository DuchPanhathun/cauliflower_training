"""ONNX inference script for universal deployment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_onnx_model(model_path: str):
    """Load ONNX model."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("Please install onnxruntime: pip install onnxruntime-gpu (or onnxruntime for CPU)")
    
    session = ort.InferenceSession(model_path)
    return session


def preprocess_image(image_path: str, img_size: int = 224) -> np.ndarray:
    """Preprocess image for inference."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((img_size, img_size))
    
    # Convert to array
    image_array = np.array(image, dtype=np.float32)
    
    # Normalize to [0, 1]
    image_array = image_array / 255.0
    
    # Normalize using ImageNet statistics
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std
    
    # Transpose to CHW format (PyTorch/ONNX format)
    image_array = image_array.transpose(2, 0, 1)
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, 0)
    
    return image_array.astype(np.float32)


def predict_image(
    image_path: str,
    session,
    class_names: list[str],
    img_size: int = 224,
    top_k: int = 3
) -> dict:
    """Predict disease class for a single image."""
    # Preprocess image
    image_tensor = preprocess_image(image_path, img_size)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Run inference
    outputs = session.run(None, {input_name: image_tensor})
    predictions = outputs[0][0]
    
    # Apply softmax
    exp_preds = np.exp(predictions - np.max(predictions))
    probabilities = exp_preds / exp_preds.sum()
    
    # Get top-k predictions
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'class': class_names[idx],
            'probability': float(probabilities[idx]),
            'confidence_percent': float(probabilities[idx] * 100)
        })
    
    return {
        'image_path': image_path,
        'predictions': results,
        'top_prediction': results[0]
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict disease class using ONNX model')
    
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file')
    parser.add_argument('--model', type=str, default='checkpoints/model.onnx',
                        help='Path to ONNX model file')
    parser.add_argument('--label-mapping', type=str, default='checkpoints/label_mapping.json',
                        help='Path to label mapping JSON')
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
    
    # Load label mapping
    with open(args.label_mapping, 'r') as f:
        label_mapping = json.load(f)
    
    class_names = sorted(label_mapping.keys(), key=lambda x: label_mapping[x])
    
    # Load model
    print("Loading ONNX model...")
    session = load_onnx_model(args.model)
    print("Model loaded successfully\n")
    
    # Predict
    print(f"Predicting for image: {args.image}")
    result = predict_image(args.image, session, class_names, args.img_size, args.top_k)
    
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

