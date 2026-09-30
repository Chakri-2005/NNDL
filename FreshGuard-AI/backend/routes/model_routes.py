from flask import Blueprint, jsonify
from backend.services.model_manager import ModelManager

model_bp = Blueprint('models', __name__)

@model_bp.route('/models', methods=['GET'])
def get_models():
    manager = ModelManager.get_instance()
    return jsonify({
        "models": manager.get_available_models()
    })
