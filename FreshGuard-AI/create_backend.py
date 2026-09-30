import os

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

backend_dir = r"C:\NNDL\FreshGuard-AI\backend"

# requirements.txt
write_file(os.path.join(backend_dir, "requirements.txt"), """flask>=3.0.0
flask-cors>=4.0.0
torch>=2.1.0
torchvision>=0.16.0
Pillow>=10.0.0
pandas>=2.0.0
""")

# app.py
write_file(os.path.join(backend_dir, "app.py"), """from flask import Flask
from flask_cors import CORS
import sys
import os

# Ensure we can import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.routes.prediction_routes import prediction_bp
from backend.routes.model_routes import model_bp
from backend.routes.metrics_routes import metrics_bp
from backend.services.model_manager import ModelManager

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Initialize models
    with app.app_context():
        ModelManager.get_instance()

    app.register_blueprint(prediction_bp, url_prefix='/api')
    app.register_blueprint(model_bp, url_prefix='/api')
    app.register_blueprint(metrics_bp, url_prefix='/api')

    @app.route('/api/health')
    def health():
        return {"status": "ok", "message": "FreshGuard API is running."}

    return app

if __name__ == '__main__':
    app = create_app()
    print("Starting Flask backend on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
""")

# services/model_manager.py
write_file(os.path.join(backend_dir, "services", "model_manager.py"), """import os
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
""")

# services/prediction_service.py
write_file(os.path.join(backend_dir, "services", "prediction_service.py"), """import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as root_config
from backend.services.model_manager import ModelManager

def get_prediction_explanation(predicted_class, confidence):
    fruit, status = predicted_class.split("_")
    if confidence < root_config.CONFIDENCE_THRESHOLD:
        return "The model is uncertain. A clearer image may produce a more reliable prediction."
    
    if status == "Fresh":
        return f"Based on the visual characteristics learned by the model, this fruit was classified as Fresh with high confidence. The visual features in the image were more consistent with the Fresh class than the Rotten class."
    else:
        return f"The model detected visual characteristics more consistent with the Rotten class. This is an AI-based visual estimate and is not a food-safety certification."

def process_prediction(model_key, image):
    manager = ModelManager.get_instance()
    result = manager.predict(model_key, image)
    
    pred_class = result["predicted_class"]
    confidence = result["confidence"]
    fruit, status = pred_class.split("_")
    
    explanation = get_prediction_explanation(pred_class, confidence)
    
    return {
        "success": True,
        "model": manager.registry[model_key]["name"],
        "predicted_class": pred_class,
        "fruit": fruit,
        "freshness": status,
        "confidence": round(confidence * 100, 2),
        "probabilities": result["probabilities"],
        "inference_time_ms": round(result["inference_time_ms"], 2),
        "explanation": explanation,
        "is_low_confidence": confidence < root_config.CONFIDENCE_THRESHOLD
    }
""")

# routes/prediction_routes.py
write_file(os.path.join(backend_dir, "routes", "prediction_routes.py"), """from flask import Blueprint, request, jsonify
from backend.services.prediction_service import process_prediction
from backend.services.model_manager import ModelManager
import io
from PIL import Image

prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    file = request.files['image']
    model_key = request.form.get('model', 'efficientnetv2_s')
    
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        result = process_prediction(model_key, image)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@prediction_bp.route('/compare', methods=['POST'])
def compare():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    file = request.files['image']
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        manager = ModelManager.get_instance()
        results = []
        for model_key in manager.models.keys():
            res = process_prediction(model_key, image)
            results.append(res)
            
        return jsonify({"success": True, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
""")

# routes/model_routes.py
write_file(os.path.join(backend_dir, "routes", "model_routes.py"), """from flask import Blueprint, jsonify
from backend.services.model_manager import ModelManager

model_bp = Blueprint('models', __name__)

@model_bp.route('/models', methods=['GET'])
def get_models():
    manager = ModelManager.get_instance()
    return jsonify({
        "models": manager.get_available_models()
    })
""")

# routes/metrics_routes.py
write_file(os.path.join(backend_dir, "routes", "metrics_routes.py"), """from flask import Blueprint, jsonify
import os
import json
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as root_config

metrics_bp = Blueprint('metrics', __name__)

@metrics_bp.route('/metrics', methods=['GET'])
def get_metrics():
    metrics_dir = root_config.METRICS_DIR
    per_class_dir = root_config.PER_CLASS_DIR
    
    models = ["efficientnetv2_s", "convnext_tiny", "mobilevit"]
    all_metrics = {}
    
    for model in models:
        json_path = os.path.join(metrics_dir, f"{model}_metrics.json")
        csv_path = os.path.join(per_class_dir, f"{model}.csv")
        
        model_data = {}
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                model_data["overall"] = json.load(f)
                
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            model_data["per_class"] = df.to_dict(orient='records')
            
        all_metrics[model] = model_data
        
    return jsonify({"metrics": all_metrics})
""")

print("Backend files created successfully!")
