"""
Integration Tests for Flask Backend API Endpoints.
"""
import io
import pytest
from PIL import Image

from backend.app import create_app

@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Tests GET /api/health returns HTTP 200 and status ok."""
    resp = client.get('/api/health')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('status') == 'ok'

def test_models_endpoint(client):
    """Tests GET /api/models returns the available model list."""
    resp = client.get('/api/models')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'models' in data
    model_keys = [m['key'] for m in data['models']]
    assert 'efficientnetv2_s' in model_keys
    assert 'convnext_tiny' in model_keys
    assert 'mobilevit' in model_keys

def test_metrics_endpoint(client):
    """Tests GET /api/metrics returns recorded metrics."""
    resp = client.get('/api/metrics')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'metrics' in data

def test_predict_endpoint_missing_image(client):
    """Tests POST /api/predict without image returns HTTP 400."""
    resp = client.post('/api/predict', data={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert 'error' in data

def test_predict_endpoint_with_valid_image(client, sample_image_bytes):
    """Tests POST /api/predict with valid image returns prediction and confidence."""
    data = {
        'image': (io.BytesIO(sample_image_bytes), 'test.jpg'),
        'model': 'efficientnetv2_s'
    }
    resp = client.post('/api/predict', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200
    res = resp.get_json()
    assert 'predicted_class' in res
    assert 'confidence' in res
    assert 'probabilities' in res
    assert 'fruit' in res
    assert 'freshness' in res

def test_compare_endpoint_with_valid_image(client, sample_image_bytes):
    """Tests POST /api/compare evaluates all three models."""
    data = {
        'image': (io.BytesIO(sample_image_bytes), 'test.jpg')
    }
    resp = client.post('/api/compare', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200
    res = resp.get_json()
    assert res.get('success') is True
    assert len(res.get('results', [])) >= 3
