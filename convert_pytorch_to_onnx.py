"""Convert PyTorch model to ONNX format for universal deployment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.onnx

from model import create_model


def convert_to_onnx(
    pytorch_model_path: str,
    onnx_output_path: str,
    num_classes: int,
    model_name: str = 'efficientnet_b0',
    img_size: int = 224
):
    """Convert PyTorch model to ONNX format."""
    # Load PyTorch model
    print(f"Loading PyTorch model from {pytorch_model_path}...")
    model = create_model(num_classes=num_classes, model_name=model_name, pretrained=False)
    model.load_state_dict(torch.load(pytorch_model_path, map_location='cpu', weights_only=True))
    model.eval()
    print("Model loaded successfully\n")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # Export to ONNX
    print(f"Converting to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {onnx_output_path}\n")
    
    # Verify the model
    try:
        import onnx
        onnx_model = onnx.load(onnx_output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed")
    except ImportError:
        print("ONNX not installed, skipping verification (pip install onnx)")
    except Exception as e:
        print(f"ONNX verification warning: {e}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    
    parser.add_argument('--pytorch-model', type=str, default='checkpoints/best_model.pth',
                        help='Path to PyTorch model file')
    parser.add_argument('--onnx-output', type=str, default='checkpoints/model.onnx',
                        help='Output path for ONNX model')
    parser.add_argument('--label-mapping', type=str, default='checkpoints/label_mapping.json',
                        help='Path to label mapping JSON')
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        choices=['efficientnet_b0', 'resnet50', 'mobilenet_v3'],
                        help='Model architecture')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size')
    
    return parser.parse_args()


def main():
    """Main conversion function."""
    args = parse_args()
    
    # Load label mapping to get number of classes
    with open(args.label_mapping, 'r') as f:
        label_mapping = json.load(f)
    
    num_classes = len(label_mapping)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {list(label_mapping.keys())}\n")
    
    # Convert model
    convert_to_onnx(
        pytorch_model_path=args.pytorch_model,
        onnx_output_path=args.onnx_output,
        num_classes=num_classes,
        model_name=args.model,
        img_size=args.img_size
    )
    
    print("\n" + "="*60)
    print("Conversion complete!")
    print("="*60)
    print(f"\nYou can now use {args.onnx_output} with:")
    print("  - ONNX Runtime (Python, C++, C#, Java)")
    print("  - TensorFlow (via onnx-tf)")
    print("  - TensorRT (NVIDIA)")
    print("  - CoreML (Apple)")
    print("  - Mobile devices")
    print("  - Web browsers")


if __name__ == '__main__':
    main()

