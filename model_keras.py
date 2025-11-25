"""Keras model definitions for cauliflower disease classification."""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Literal


def create_model(
    num_classes: int,
    model_name: Literal['efficientnet_b0', 'resnet50', 'mobilenet_v3'] = 'efficientnet_b0',
    img_size: int = 224,
    pretrained: bool = True
) -> Model:
    """
    Create a Keras model for disease classification.
    
    Args:
        num_classes: Number of output classes
        model_name: Architecture to use
        img_size: Input image size
        pretrained: Whether to use ImageNet pretrained weights
        
    Returns:
        Keras model
    """
    weights = 'imagenet' if pretrained else None
    input_shape = (img_size, img_size, 3)
    
    if model_name == 'efficientnet_b0':
        # EfficientNet-B0 (good balance of accuracy and speed)
        base_model = keras.applications.EfficientNetB0(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            pooling='avg'
        )
        
    elif model_name == 'resnet50':
        # ResNet50 (classic architecture, good accuracy)
        base_model = keras.applications.ResNet50(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            pooling='avg'
        )
        
    elif model_name == 'mobilenet_v3':
        # MobileNetV3 (fast inference, good for deployment)
        base_model = keras.applications.MobileNetV3Small(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            pooling='avg'
        )
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Build model
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    
    # Add dropout for regularization
    x = layers.Dropout(0.2)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name=f'{model_name}_classifier')
    
    return model


def create_model_with_fine_tuning(
    num_classes: int,
    model_name: Literal['efficientnet_b0', 'resnet50', 'mobilenet_v3'] = 'efficientnet_b0',
    img_size: int = 224,
    pretrained: bool = True,
    trainable_layers: int = 20
) -> Model:
    """
    Create a model with option to fine-tune top layers.
    
    Args:
        num_classes: Number of output classes
        model_name: Architecture to use
        img_size: Input image size
        pretrained: Whether to use ImageNet pretrained weights
        trainable_layers: Number of top layers to make trainable
        
    Returns:
        Keras model ready for fine-tuning
    """
    model = create_model(num_classes, model_name, img_size, pretrained)
    
    # Freeze base model layers except top trainable_layers
    base_model = model.layers[1]
    base_model.trainable = True
    
    # Freeze all layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Unfreeze top layers
    for layer in base_model.layers[-trainable_layers:]:
        layer.trainable = True
    
    return model


class FocalLoss(keras.losses.Loss):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate focal loss
        weight = self.alpha * tf.pow(1 - y_pred, self.gamma)
        focal_loss = weight * cross_entropy
        
        return tf.reduce_sum(focal_loss, axis=-1)


def compile_model(
    model: Model,
    learning_rate: float = 0.001,
    use_focal_loss: bool = False
) -> Model:
    """
    Compile the model with optimizer and loss.
    
    Args:
        model: Keras model to compile
        learning_rate: Initial learning rate
        use_focal_loss: Whether to use focal loss instead of categorical crossentropy
        
    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    if use_focal_loss:
        loss = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        loss = keras.losses.CategoricalCrossentropy()
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')
        ]
    )
    
    return model

