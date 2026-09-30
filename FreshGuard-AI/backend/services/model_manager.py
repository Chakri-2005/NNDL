import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as root_config
from models import get_efficientnetv2_s, get_convnext_tiny, get_mobilevit

class ModelManager:
    _instance = None
    
    def __init__(self):
        self.device = root_config.DEVICE
        self.models = {}
        self.registry = {
            "efficientnetv2_s": {
                "name": "EfficientNetV2-S",
                "get_fn": get_efficientnetv2_s,
                "ckpt": os.path.join(root_config.CHECKPOINTS_DIR, "efficientnetv2_s_best.pth")
            },
            "convnext_tiny": {
                "name": "ConvNeXt-Tiny",
                "get_fn": get_convnext_tiny,
                "ckpt": os.path.join(root_config.CHECKPOINTS_DIR, "convnext_tiny_best.pth")
            },
            "mobilevit": {
                "name": "MobileViT",
                "get_fn": get_mobilevit,
                "ckpt": os.path.join(root_config.CHECKPOINTS_DIR, "mobilevit_best.pth")
            }
        }
        
        self.preprocess = transforms.Compose([
            transforms.Resize((root_config.IMAGE_SIZE, root_config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self._load_models()
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def _load_models(self):
        for m_key, m_info in self.registry.items():
            ckpt_path = m_info["ckpt"]
            model = m_info["get_fn"](num_classes=root_config.NUM_CLASSES, pretrained=False).to(self.device)
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
                model.eval()
                self.models[m_key] = model
                print(f"Loaded {m_info['name']}")
            else:
                print(f"Warning: {ckpt_path} not found.")
                
    def get_available_models(self):
        return [{"key": k, "name": v["name"], "loaded": k in self.models} for k, v in self.registry.items()]
                
    def predict(self, model_key, image):
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not loaded.")
            
        model = self.models[model_key]
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        import time
        start_t = time.time()
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
        inference_time_ms = (time.time() - start_t) * 1000
        
        confidence, pred_idx = torch.max(probs, dim=0)
        pred_class = root_config.CLASS_NAMES[pred_idx.item()]
        
        class_probs = {
            root_config.CLASS_NAMES[i]: round(probs[i].item() * 100, 2)
            for i in range(len(root_config.CLASS_NAMES))
        }
        
        return {
            "predicted_class": pred_class,
            "confidence": confidence.item(),
            "probabilities": class_probs,
            "inference_time_ms": inference_time_ms
        }
