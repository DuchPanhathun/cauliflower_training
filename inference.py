"""Inference script for single image prediction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import create_model


def load_model(checkpoint_path: str, model_name: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    model = create_model(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=False
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    return model


def get_inference_transform(img_size: int = 224) -> transforms.Compose:
    """Get image transforms for inference."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_image(
    image_path: str,
    model: torch.nn.Module,
    transform: transforms.Compose,
    class_names: list[str],
    device: torch.device,
    top_k: int = 3
) -> dict:
    """Predict disease class for a single image."""
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    # Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities, min(top_k, len(class_names)))
    
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        predictions.append({
            'class': class_names[idx],
            'probability': float(prob),
            'confidence_percent': float(prob * 100)
        })
    
    return {
        'image_path': image_path,
        'predictions': predictions,
        'top_prediction': predictions[0]
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict disease class for an image')
    
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--label-mapping', type=str, default='checkpoints/label_mapping.json',
                        help='Path to label mapping JSON')
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        choices=['efficientnet_b0', 'resnet50', 'mobilenet_v3'],
                        help='Model architecture')
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
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load label mapping
    with open(args.label_mapping, 'r') as f:
        label_mapping = json.load(f)
    
    class_names = sorted(label_mapping.keys(), key=lambda x: label_mapping[x])
    num_classes = len(class_names)
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, args.model, num_classes, device)
    print("✓ Model loaded successfully\n")
    
    # Get transform
    transform = get_inference_transform(args.img_size)
    
    # Predict
    print(f"Predicting for image: {args.image}")
    result = predict_image(args.image, model, transform, class_names, device, args.top_k)
    
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

