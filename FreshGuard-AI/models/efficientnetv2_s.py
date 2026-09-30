"""
FreshGuard AI — Model 1: EfficientNetV2-S
Pretrained torchvision EfficientNetV2-S with standardized classification head.
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class FreshGuardClassifierHead(nn.Module):
    """
    Standardized classification head across all 3 benchmark models:
    LayerNorm -> Linear(in_features, 256) -> GELU -> Dropout(0.3) -> Linear(256, num_classes)
    """
    def __init__(self, in_features, num_classes=6):
        super(FreshGuardClassifierHead, self).__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.head(x)

def get_efficientnetv2_s(num_classes=6, pretrained=True):
    """
    Constructs EfficientNetV2-S model with pretrained ImageNet weights.
    """
    weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    model = efficientnet_v2_s(weights=weights)
    
    # Extract feature dimension from existing classifier
    in_features = model.classifier[1].in_features  # 1280 for EfficientNetV2-S
    
    # Replace classifier with standardized head
    model.classifier = FreshGuardClassifierHead(in_features, num_classes=num_classes)
    return model

if __name__ == "__main__":
    m = get_efficientnetv2_s(num_classes=6, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    print("EfficientNetV2-S output shape:", out.shape)
