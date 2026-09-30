from flask import Flask
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
