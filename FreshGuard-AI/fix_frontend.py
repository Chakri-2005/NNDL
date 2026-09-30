import os

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

frontend_src = r"C:\NNDL\FreshGuard-AI\frontend\src"

# 1. Update Scan.jsx to show the actual result even if low confidence
scan_jsx_path = os.path.join(frontend_src, "pages", "Scan.jsx")
with open(scan_jsx_path, 'r', encoding='utf-8') as f:
    scan_code = f.read()

# Replace the conditional rendering for low confidence
old_render = """            {result.is_low_confidence ? (
              <div className="low-confidence-warning">
                <AlertTriangle size={32} />
                <h3>Low Confidence Prediction</h3>
                <p>{result.explanation}</p>
                <button className="btn-primary" onClick={retakePhoto}>Try Again</button>
              </div>
            ) : ("""

new_render = """            <>
              {result.is_low_confidence && (
                <div style={{ background: 'var(--warning-light)', padding: '1rem', borderRadius: 'var(--radius)', marginBottom: '1.5rem', color: 'var(--warning-dark)', display: 'flex', alignItems: 'center', gap: '0.75rem', border: '1px solid var(--warning)' }}>
                  <AlertTriangle size={24} />
                  <div>
                    <strong>Low Confidence Prediction</strong>
                    <p style={{ margin: 0, fontSize: '0.9rem' }}>The model is uncertain. A clearer image may produce a more reliable prediction.</p>
                  </div>
                </div>
              )}"""

scan_code = scan_code.replace(old_render, new_render)
# Remove the closing brace of the old ternary operator
scan_code = scan_code.replace("""                </button>
              </>
            )}
          </div>""", """                </button>
              </>
          </div>""")
write_file(scan_jsx_path, scan_code)


# 2. Create UploadPredict.jsx
upload_jsx = """import React, { useState, useRef } from 'react';
import { Upload, ChevronRight, AlertTriangle } from 'lucide-react';
import { predict } from '../services/api';

const UploadPredict = () => {
  const [imageSrc, setImageSrc] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelKey, setModelKey] = useState('efficientnetv2_s');
  const fileInputRef = useRef(null);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setImageSrc(e.target.result);
        setResult(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const dataURLtoBlob = (dataurl) => {
    let arr = dataurl.split(','), mime = arr[0].match(/:(.*?);/)[1],
        bstr = atob(arr[1]), n = bstr.length, u8arr = new Uint8Array(n);
    while(n--){
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], {type:mime});
  }

  const analyzeImage = async () => {
    if (!imageSrc) return;
    setLoading(true);
    try {
      const blob = dataURLtoBlob(imageSrc);
      const file = new File([blob], 'upload.jpg', { type: 'image/jpeg' });
      const res = await predict(file, modelKey);
      setResult(res);
      
      // Save to local storage history
      const historyStr = localStorage.getItem('freshguard_history');
      const history = historyStr ? JSON.parse(historyStr) : [];
      history.unshift({
        date: new Date().toISOString(),
        fruit: res.fruit,
        freshness: res.freshness,
        confidence: res.confidence,
        model: res.model,
        thumbnail: imageSrc
      });
      localStorage.setItem('freshguard_history', JSON.stringify(history.slice(0, 10)));
      
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container animate-fade-in" style={{ paddingBottom: '4rem' }}>
      <div className="scan-header" style={{ marginTop: '2rem' }}>
        <h1 className="section-title">Upload & Predict</h1>
        <p className="section-subtitle">Upload an image and select a model to make a prediction.</p>
      </div>

      <div className="card text-center" style={{ padding: '3rem', maxWidth: '600px', margin: '0 auto 2rem auto' }}>
        {!imageSrc ? (
          <div onClick={() => fileInputRef.current.click()} style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
            <div className="feature-icon"><Upload size={32} /></div>
            <h3>Upload an Image</h3>
            <p className="text-muted">Supports JPG, PNG, WEBP</p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.5rem' }}>
            <img src={imageSrc} style={{ maxHeight: '300px', borderRadius: 'var(--radius-lg)' }} alt="Upload" />
            
            {!result && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', width: '100%', maxWidth: '300px' }}>
                <select 
                  style={{ padding: '0.75rem', borderRadius: 'var(--radius)', border: '1px solid var(--border)' }}
                  value={modelKey} 
                  onChange={(e) => setModelKey(e.target.value)}
                >
                  <option value="efficientnetv2_s">EfficientNetV2-S (Recommended)</option>
                  <option value="convnext_tiny">ConvNeXt-Tiny</option>
                  <option value="mobilevit">MobileViT (Fastest)</option>
                </select>
                
                <div style={{ display: 'flex', gap: '1rem' }}>
                  <button className="btn-secondary" style={{ flex: 1 }} onClick={() => setImageSrc(null)}>Change Image</button>
                  <button className="btn-primary" style={{ flex: 1, justifyContent: 'center' }} onClick={analyzeImage} disabled={loading}>
                    {loading ? 'Analyzing...' : 'Predict'} <ChevronRight size={18} />
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
        <input type="file" accept="image/*" ref={fileInputRef} onChange={handleFileUpload} style={{ display: 'none' }} />
      </div>

      {result && (
        <div className="card animate-fade-in" style={{ padding: '2rem', maxWidth: '600px', margin: '0 auto' }}>
          {result.is_low_confidence && (
            <div style={{ background: 'var(--warning-light)', padding: '1rem', borderRadius: 'var(--radius)', marginBottom: '1.5rem', color: 'var(--warning-dark)', display: 'flex', alignItems: 'center', gap: '0.75rem', border: '1px solid var(--warning)' }}>
              <AlertTriangle size={24} />
              <div>
                <strong>Low Confidence Prediction</strong>
                <p style={{ margin: 0, fontSize: '0.9rem' }}>The model is uncertain. A clearer image may produce a more reliable prediction.</p>
              </div>
            </div>
          )}
          
          <div className="result-header">
            <h2 className="result-title">
              {result.fruit} <span className={result.freshness === 'Fresh' ? 'text-success' : 'text-danger'}>• {result.freshness}</span>
            </h2>
            <div className="confidence-badge">
              {result.confidence}% Confidence
            </div>
          </div>
          
          <div className="confidence-bar-container">
            <div 
              className={`confidence-bar ${result.freshness === 'Fresh' ? 'bg-success' : 'bg-danger'}`}
              style={{ width: `${result.confidence}%` }}
            ></div>
          </div>
          
          <div className="result-details">
            <div className="detail-item">
              <span className="detail-label">Model</span>
              <span className="detail-value">{result.model}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Inference Time</span>
              <span className="detail-value">{result.inference_time_ms} ms</span>
            </div>
          </div>

          <div className="probabilities-box mt-4">
            <h4>Class Probabilities</h4>
            {Object.entries(result.probabilities)
              .sort((a,b) => b[1] - a[1])
              .map(([cls, prob]) => (
              <div key={cls} className="prob-row">
                <span className="prob-label">{cls.replace('_', ' ')}</span>
                <div className="prob-bar-wrapper">
                  <div className="prob-bar" style={{ width: `${prob}%` }}></div>
                </div>
                <span className="prob-value">{prob}%</span>
              </div>
            ))}
          </div>
          
          <button className="btn-secondary" style={{ marginTop: '1.5rem', width: '100%' }} onClick={() => { setImageSrc(null); setResult(null); }}>
            Upload Another Image
          </button>
        </div>
      )}
    </div>
  );
};
export default UploadPredict;
"""
write_file(os.path.join(frontend_src, "pages", "UploadPredict.jsx"), upload_jsx)


# 3. Update App.jsx to include UploadPredict
app_jsx_path = os.path.join(frontend_src, "App.jsx")
with open(app_jsx_path, 'r', encoding='utf-8') as f:
    app_code = f.read()

app_code = app_code.replace("import Scan from './pages/Scan';", "import Scan from './pages/Scan';\nimport UploadPredict from './pages/UploadPredict';")
app_code = app_code.replace('<Route path="/scan" element={<Scan />} />', '<Route path="/scan" element={<Scan />} />\n            <Route path="/upload" element={<UploadPredict />} />')
write_file(app_jsx_path, app_code)

# 4. Update Navbar.jsx to include the link
nav_jsx_path = os.path.join(frontend_src, "components", "Navbar.jsx")
with open(nav_jsx_path, 'r', encoding='utf-8') as f:
    nav_code = f.read()

nav_code = nav_code.replace('<Link to="/" className={`nav-link ${isActive(\'/\') ? \'active\' : \'\'}`}>Home</Link>', '<Link to="/" className={`nav-link ${isActive(\'/\') ? \'active\' : \'\'}`}>Home</Link>\n          <Link to="/upload" className={`nav-link ${isActive(\'/upload\') ? \'active\' : \'\'}`}>Upload & Predict</Link>')
write_file(nav_jsx_path, nav_code)

print("Updates applied successfully.")
