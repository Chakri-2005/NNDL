import sys
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
