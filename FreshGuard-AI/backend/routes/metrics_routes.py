from flask import Blueprint, jsonify
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
