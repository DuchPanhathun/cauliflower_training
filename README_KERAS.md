# Cauliflower Disease Classification - Keras/H5 Training Guide

This guide covers training with **TensorFlow/Keras** to generate `.h5` model files for deployment.

## Why Use Keras/H5 Format?

- **Universal compatibility**: Works with TensorFlow.js, TensorFlow Lite, ONNX
- **Easy deployment**: Compatible with web, mobile, and edge devices
- **Standard format**: Widely supported in production environments
- **Telegram bots**: Easy integration with Python Telegram bots

## Setup Instructions

### 1. Install TensorFlow with GPU Support

```bash
# For NVIDIA T600 GPU (CUDA support)
pip install tensorflow[and-cuda]>=2.13.0

# OR install from requirements
pip install -r requirements_keras.txt
```

### 2. Verify GPU Installation

```bash
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"
```

## Training the Model

### Quick Start (Recommended)

```bash
python train_keras.py --epochs 50 --batch-size 32 --model efficientnet_b0
```

### Full Training Options

```bash
python train_keras.py \
    --csv-file dataset_labels.csv \
    --dataset-root dataset \
    --model efficientnet_b0 \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --patience 10 \
    --save-dir checkpoints_keras
```

**Model Options:**
- `efficientnet_b0` (default) - Best balance, recommended
- `resnet50` - Classic architecture
- `mobilenet_v3` - Fastest inference

**Key Parameters:**
- `--epochs`: Maximum training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 10)
- `--use-focal-loss`: Use focal loss for class imbalance

### Training with Class Imbalance Handling

```bash
python train_keras.py --use-focal-loss --epochs 50
```

## Output Files

After training, you'll get:

```
checkpoints_keras/
├── best_model.h5          ← Use this for deployment! (full model)
├── final_model.h5         ← Last epoch model
├── best_weights.h5        ← Weights only
├── label_mapping.json     ← Class names mapping
├── idx_to_label.json      ← Index to label mapping
├── training_history.json  ← Training metrics
├── training_log.csv       ← Detailed training log
└── logs/                  ← TensorBoard logs
```

**For Telegram Bot**: Use `best_model.h5` + `idx_to_label.json`

## Expected Training Time

On **NVIDIA T600 (4GB VRAM)** with 8,407 images:

| Model | Time per Epoch | Total (50 epochs) |
|-------|----------------|-------------------|
| EfficientNet-B0 | ~45-60 seconds | **35-50 minutes** |
| ResNet50 | ~55-70 seconds | **45-60 minutes** |
| MobileNet-V3 | ~30-45 seconds | **25-40 minutes** |

*Note: Includes automatic early stopping - may finish sooner!*

## Evaluating the Model

```bash
python evaluate_keras.py --model checkpoints_keras/best_model.h5
```

**Outputs:**
- Test accuracy and metrics
- Confusion matrix (PNG)
- Training history plots (PNG)
- Detailed classification report

## Making Predictions

### Single Image Prediction

```bash
python inference_keras.py --image path/to/image.jpg --model checkpoints_keras/best_model.h5
```

### Example Output

```
Loading model...
✓ Model loaded successfully

Predicting for image: test_image.jpg

============================================================
Top Prediction: Healthy
Confidence: 96.83%
============================================================

Top 3 Predictions:
------------------------------------------------------------
1. Healthy                    - 96.83%
2. Alternaria Leaf Spot       -  2.14%
3. Black Spot                 -  0.78%
------------------------------------------------------------
```

## Using H5 Model in Telegram Bot

### Basic Integration Example

```python
import tensorflow as tf
from tensorflow import keras
import json
import numpy as np
from PIL import Image

# Load model and labels
model = keras.models.load_model('checkpoints_keras/best_model.h5')
with open('checkpoints_keras/idx_to_label.json', 'r') as f:
    idx_to_label = json.load(f)
    class_names = [idx_to_label[str(i)] for i in range(len(idx_to_label))]

def predict_disease(image_path):
    """Predict disease from image."""
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, 0)
    
    # Predict
    predictions = model.predict(img_array)[0]
    top_idx = np.argmax(predictions)
    
    return {
        'disease': class_names[top_idx],
        'confidence': float(predictions[top_idx] * 100)
    }

# Use in Telegram bot handler
def handle_image(update, context):
    # Download image from telegram
    file = context.bot.get_file(update.message.photo[-1].file_id)
    file.download('temp_image.jpg')
    
    # Predict
    result = predict_disease('temp_image.jpg')
    
    # Send result
    update.message.reply_text(
        f"🌿 Disease: {result['disease']}\n"
        f"📊 Confidence: {result['confidence']:.2f}%"
    )
```

### Full Telegram Bot Example

```python
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import tensorflow as tf
import json
import numpy as np
from PIL import Image

# Load model once at startup
MODEL = tf.keras.models.load_model('checkpoints_keras/best_model.h5')
with open('checkpoints_keras/idx_to_label.json', 'r') as f:
    idx_to_label = json.load(f)
    CLASS_NAMES = [idx_to_label[str(i)] for i in range(len(idx_to_label))]

def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "🌿 Cauliflower Disease Detector Bot\n\n"
        "Send me a photo of a cauliflower plant and I'll identify any diseases!"
    )

def predict_image(image_path: str) -> dict:
    """Predict disease from image."""
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, 0)
    
    predictions = MODEL.predict(img_array, verbose=0)[0]
    top_3_idx = np.argsort(predictions)[-3:][::-1]
    
    return {
        'top': CLASS_NAMES[top_3_idx[0]],
        'confidence': float(predictions[top_3_idx[0]] * 100),
        'all': [(CLASS_NAMES[i], float(predictions[i] * 100)) for i in top_3_idx]
    }

def handle_photo(update: Update, context: CallbackContext):
    # Download image
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('temp.jpg')
    
    # Predict
    result = predict_image('temp.jpg')
    
    # Format response
    message = f"🔍 Analysis Results:\n\n"
    message += f"🌿 **Disease:** {result['top']}\n"
    message += f"📊 **Confidence:** {result['confidence']:.2f}%\n\n"
    message += "Top 3 Predictions:\n"
    for i, (disease, conf) in enumerate(result['all'], 1):
        message += f"{i}. {disease}: {conf:.2f}%\n"
    
    update.message.reply_text(message, parse_mode='Markdown')

def main():
    updater = Updater("YOUR_BOT_TOKEN", use_context=True)
    dp = updater.dispatcher
    
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))
    
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
```

## GPU Optimization for T600

### Optimal Settings

```bash
# Best performance on T600 (4GB VRAM)
python train_keras.py \
    --model efficientnet_b0 \
    --batch-size 32 \
    --epochs 50
```

### If Out of Memory

```bash
# Reduce batch size
python train_keras.py --batch-size 16

# OR use smaller model
python train_keras.py --model mobilenet_v3 --batch-size 48
```

### Monitor GPU Usage

```bash
# Windows (in another terminal)
nvidia-smi -l 1

# Check TensorFlow GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Model Export Options

### 1. For Telegram Bot (Recommended)
```python
# Use best_model.h5 directly - already optimized!
model = keras.models.load_model('checkpoints_keras/best_model.h5')
```

### 2. For TensorFlow Lite (Mobile)
```python
import tensorflow as tf

model = tf.keras.models.load_model('checkpoints_keras/best_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### 3. For TensorFlow.js (Web)
```bash
pip install tensorflowjs
tensorflowjs_converter \
    --input_format=keras \
    checkpoints_keras/best_model.h5 \
    tfjs_model/
```

### 4. For ONNX (Universal)
```bash
pip install tf2onnx
python -m tf2onnx.convert \
    --keras checkpoints_keras/best_model.h5 \
    --output model.onnx
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir checkpoints_keras/logs
```

Then open: `http://localhost:6006`

### Training Callbacks

The training includes:
- **ModelCheckpoint**: Saves best model automatically
- **EarlyStopping**: Stops if no improvement (patience=10)
- **ReduceLROnPlateau**: Reduces learning rate when stuck
- **CSVLogger**: Logs all metrics to CSV
- **TensorBoard**: Real-time visualization

## Troubleshooting

### TensorFlow Not Using GPU

```bash
# Check CUDA installation
nvidia-smi

# Reinstall TensorFlow with GPU
pip uninstall tensorflow
pip install tensorflow[and-cuda]>=2.13.0
```

### Out of Memory Error

```bash
# Option 1: Reduce batch size
python train_keras.py --batch-size 16

# Option 2: Use smaller model
python train_keras.py --model mobilenet_v3

# Option 3: Reduce image size
python train_keras.py --img-size 192
```

### Slow Training

- Verify GPU is being used (check logs at training start)
- Close other GPU applications
- Ensure TensorFlow has GPU support installed

## File Size Comparison

| Format | Size | Use Case |
|--------|------|----------|
| `best_model.h5` | ~15-20 MB | **Telegram Bot, API** |
| `best_weights.h5` | ~10-15 MB | Custom loading |
| `model.tflite` | ~5-8 MB | Mobile apps |
| `tfjs_model/` | ~20-25 MB | Web apps |

## Quick Command Reference

```bash
# 1. Train model (generates H5 file)
python train_keras.py --epochs 50 --batch-size 32

# 2. Evaluate model
python evaluate_keras.py

# 3. Test single image
python inference_keras.py --image test.jpg

# 4. Monitor with TensorBoard
tensorboard --logdir checkpoints_keras/logs

# 5. Check GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Model Performance Tips

1. **Use EfficientNet-B0**: Best accuracy/speed balance
2. **Enable Early Stopping**: Saves time, prevents overfitting
3. **Monitor TensorBoard**: Watch training in real-time
4. **Use Focal Loss**: If class imbalance is severe
5. **Fine-tune Learning Rate**: Try 0.0005 if 0.001 is unstable

## Deployment Checklist

- [ ] Model file: `best_model.h5`
- [ ] Label mapping: `idx_to_label.json`
- [ ] Test inference script works
- [ ] Verify predictions are correct
- [ ] Check model size (<50MB ideal)
- [ ] Test on various images
- [ ] Document expected accuracy

## Support

For issues:
1. Check GPU: `nvidia-smi` and TensorFlow GPU detection
2. Verify file paths in commands
3. Ensure dataset is clean (run `clean_dataset.py` if needed)
4. Check training logs in `checkpoints_keras/`

