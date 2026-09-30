"""
FreshGuard AI — Model 2: ConvNeXt-Tiny
Pretrained torchvision ConvNeXt-Tiny with standardized classification head.
"""
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from .efficientnetv2_s import FreshGuardClassifierHead

def get_convnext_tiny(num_classes=6, pretrained=True):
    """
    Constructs ConvNeXt-Tiny model with pretrained ImageNet weights.
    """
    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)
    
    # Extract feature dimension from classifier head
    in_features = model.classifier[2].in_features  # 768 for ConvNeXt-Tiny
    
    # Replace classifier with standardized head
    model.classifier[2] = FreshGuardClassifierHead(in_features, num_classes=num_classes)
    return model

if __name__ == "__main__":
    m = get_convnext_tiny(num_classes=6, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    print("ConvNeXt-Tiny output shape:", out.shape)
