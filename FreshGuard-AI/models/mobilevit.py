"""
FreshGuard AI — Model 3: MobileViT (Mobile Vision Transformer)
Pretrained MobileViT-S with standardized classification head.
"""
import torch
import torch.nn as nn
from .efficientnetv2_s import FreshGuardClassifierHead

def get_mobilevit(num_classes=6, pretrained=True):
    """
    Constructs MobileViT-S model with pretrained weights using timm or torchvision.
    """
    model = None
    try:
        import timm
        # Try loading via timm (mobilevit_s)
        model = timm.create_model('mobilevit_s', pretrained=pretrained, num_classes=0)
        in_features = model.num_features
        class MobileViTWrapper(nn.Module):
            def __init__(self, backbone, in_feat, num_cls):
                super().__init__()
                self.backbone = backbone
                self.classifier = FreshGuardClassifierHead(in_feat, num_classes=num_cls)
            def forward(self, x):
                feat = self.backbone(x)
                return self.classifier(feat)
        return MobileViTWrapper(model, in_features, num_classes)
    except Exception:
        pass

    try:
        from torchvision.models import mobilevit_s, MobileViT_S_Weights
        weights = MobileViT_S_Weights.DEFAULT if pretrained else None
        model = mobilevit_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = FreshGuardClassifierHead(in_features, num_classes=num_classes)
        return model
    except Exception as e:
        raise RuntimeError(f"Could not load MobileViT model via timm or torchvision: {e}")

if __name__ == "__main__":
    m = get_mobilevit(num_classes=6, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    print("MobileViT output shape:", out.shape)
