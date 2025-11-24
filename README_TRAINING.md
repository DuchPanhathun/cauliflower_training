# Cauliflower Disease Classification Training Guide

This project trains a deep learning model to classify cauliflower diseases using your NVIDIA T600 Laptop GPU.

## Dataset Structure

The dataset contains 6 classes with 16,814 images:
- **Alternaria Leaf Spot**: 1,508 images
- **Bacterial Soft Rot**: 1,401 images
- **Black Rot**: 800 images
- **Black Spot**: 251 images
- **Downy Mildew**: 1,287 images
- **Healthy**: 3,160 images

## Setup Instructions

### 1. Install PyTorch with CUDA Support

For your NVIDIA T600, install PyTorch with CUDA support:

```bash
# Install PyTorch with CUDA 11.8 (recommended for T600)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify CUDA Installation

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Training the Model

### Quick Start (Recommended Settings)

```bash
python train.py --epochs 50 --batch-size 32 --model efficientnet_b0
```

### Training Options

```bash
python train.py \
    --csv-file dataset_labels.csv \
    --dataset-root dataset \
    --model efficientnet_b0 \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --img-size 224 \
    --save-dir checkpoints
```

**Model Options:**
- `efficientnet_b0` (default) - Best balance of accuracy and speed
- `resnet50` - Classic architecture, good accuracy
- `mobilenet_v3` - Fastest inference, good for deployment

**Key Parameters:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32, adjust based on GPU memory)
- `--lr`: Learning rate (default: 0.001)
- `--use-focal-loss`: Use focal loss for class imbalance
- `--no-amp`: Disable automatic mixed precision (slower but uses less memory)

### Training with Focal Loss (for Class Imbalance)

Since your dataset has imbalanced classes (Healthy: 3160 vs Black Spot: 251), you can use focal loss:

```bash
python train.py --use-focal-loss --epochs 50
```

### Adjusting Batch Size

If you get out-of-memory errors, reduce the batch size:

```bash
python train.py --batch-size 16
```

## Monitoring Training

The training script will:
- Display real-time progress bars with loss and accuracy
- Save the best model based on validation accuracy
- Save training history to `checkpoints/training_history.json`
- Print epoch summaries with train/val metrics

**Output Files:**
- `checkpoints/best_model.pth` - Best model weights
- `checkpoints/best_checkpoint.pth` - Full checkpoint with optimizer state
- `checkpoints/last_checkpoint.pth` - Latest checkpoint
- `checkpoints/label_mapping.json` - Class label mapping
- `checkpoints/training_history.json` - Training metrics history

## Evaluating the Model

After training, evaluate on the test set:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

This will generate:
- Test accuracy and per-class metrics
- Confusion matrix visualization
- Classification report with precision/recall/F1
- Results saved to `evaluation_results/`

## Making Predictions

### Single Image Prediction

```bash
python inference.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pth
```

### Example Output

```
Using device: cuda

Loading model...
✓ Model loaded successfully

Predicting for image: test_image.jpg

============================================================
Top Prediction: Alternaria Leaf Spot
Confidence: 95.32%
============================================================

Top 3 Predictions:
------------------------------------------------------------
1. Alternaria Leaf Spot      - 95.32%
2. Black Spot                 -  3.21%
3. Healthy                    -  1.15%
------------------------------------------------------------
```

## GPU Optimization Tips

### For NVIDIA T600 (4GB VRAM):

1. **Use Mixed Precision Training** (default, automatic):
   - Reduces memory usage by ~50%
   - Speeds up training by ~2-3x
   - Already enabled by default

2. **Optimal Batch Sizes**:
   - EfficientNet-B0: batch_size=32 (default)
   - ResNet50: batch_size=24
   - MobileNet-V3: batch_size=48

3. **Monitor GPU Usage**:
   ```bash
   # In another terminal, run:
   nvidia-smi -l 1
   ```

4. **If Out of Memory**:
   - Reduce batch size: `--batch-size 16`
   - Use smaller model: `--model mobilenet_v3`
   - Disable AMP: `--no-amp` (slower but uses less memory)

## Expected Training Time

On NVIDIA T600 (4GB VRAM):
- **EfficientNet-B0**: ~45-60 minutes for 50 epochs
- **ResNet50**: ~60-75 minutes for 50 epochs
- **MobileNet-V3**: ~30-40 minutes for 50 epochs

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python train.py --batch-size 16

# Or use smaller model
python train.py --model mobilenet_v3
```

### CUDA Not Available
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Training Too Slow
- Ensure you're using CUDA (check device output at training start)
- Verify AMP is enabled (check training output)
- Close other GPU-intensive applications

## Advanced Usage

### Resume Training from Checkpoint

```python
# Modify train.py to load from last_checkpoint.pth
checkpoint = torch.load('checkpoints/last_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

### Export Model for Deployment

```python
import torch
from model import create_model

model = create_model(num_classes=6, model_name='efficientnet_b0', pretrained=False)
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

# Export to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')
```

## Results Visualization

After training completes, check:
- `evaluation_results/confusion_matrix.png` - Visual confusion matrix
- `evaluation_results/training_history.png` - Loss and accuracy curves
- `evaluation_results/evaluation_results.json` - Detailed metrics

## Tips for Better Accuracy

1. **Train for more epochs**: Try 75-100 epochs
2. **Use focal loss**: `--use-focal-loss` helps with class imbalance
3. **Adjust learning rate**: Try `--lr 0.0005` for more stable training
4. **Ensemble models**: Train multiple models and average predictions
5. **Data augmentation**: Already included (rotation, flip, color jitter)

## File Structure

```
cauliflower_training/
├── dataset/                    # Your image dataset
├── dataset_labels.csv          # Generated labels
├── train_split.csv            # Training set (generated)
├── val_split.csv              # Validation set (generated)
├── test_split.csv             # Test set (generated)
├── checkpoints/               # Saved models
├── evaluation_results/        # Evaluation outputs
├── create_dataset_labels.py   # Label generation script
├── dataset_loader.py          # Dataset and data loader
├── model.py                   # Model definitions
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── inference.py               # Inference script
├── requirements.txt           # Dependencies
└── config.yaml                # Configuration file
```

## Quick Command Reference

```bash
# 1. Generate labels (already done)
python create_dataset_labels.py

# 2. Train model
python train.py --epochs 50 --batch-size 32

# 3. Evaluate model
python evaluate.py

# 4. Predict single image
python inference.py --image path/to/image.jpg

# 5. Monitor GPU
nvidia-smi -l 1
```

## Support

If you encounter issues:
1. Check GPU availability: `nvidia-smi`
2. Verify CUDA in PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reduce batch size if OOM errors occur
4. Check training logs for error messages

