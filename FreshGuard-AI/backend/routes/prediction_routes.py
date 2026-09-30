from flask import Blueprint, request, jsonify
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
