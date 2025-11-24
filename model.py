"""Model definition for cauliflower disease classification."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from typing import Literal


def create_model(
    num_classes: int,
    model_name: Literal['efficientnet_b0', 'resnet50', 'mobilenet_v3'] = 'efficientnet_b0',
    pretrained: bool = True
) -> nn.Module:
    """
    Create a CNN model for disease classification.
    
    Args:
        num_classes: Number of output classes
        model_name: Architecture to use
        pretrained: Whether to use ImageNet pretrained weights
        
    Returns:
        PyTorch model
    """
    if model_name == 'efficientnet_b0':
        # EfficientNet-B0 (good balance of accuracy and speed)
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'resnet50':
        # ResNet50 (classic architecture, good accuracy)
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == 'mobilenet_v3':
        # MobileNetV3 (fast inference, good for deployment)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

