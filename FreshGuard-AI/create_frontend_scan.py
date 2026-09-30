import os

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

frontend_src = r"C:\NNDL\FreshGuard-AI\frontend\src"

# 1. pages/Home.jsx
write_file(os.path.join(frontend_src, "pages", "Home.jsx"), """import React from 'react';
import { Link } from 'react-router-dom';
import { Camera, Activity, Zap, CheckCircle, Shield } from 'lucide-react';
import '../styles/home.css';

const Home = () => {
  return (
    <div className="home-container animate-fade-in">
      <section className="hero">
        <div className="hero-content">
          <div className="badge">AI-Powered Food Safety</div>
          <h1 className="hero-title">Smarter freshness for healthier living.</h1>
          <p className="hero-subtitle">
            Advanced visual food freshness detection using state-of-the-art deep learning.
            Upload or scan an image of an apple, banana, or orange, and let FreshGuard analyze its quality.
          </p>
          <div className="hero-buttons">
            <Link to="/scan" className="btn-primary btn-large">
              <Camera size={20} /> Scan Your Fruit
            </Link>
            <Link to="/dashboard" className="btn-secondary btn-large">
              Explore Models
            </Link>
          </div>
        </div>
        <div className="hero-image">
          <div className="hero-glow"></div>
          <img src="https://images.unsplash.com/photo-1610832958506-aa56368176cf?q=80&w=600&auto=format&fit=crop" alt="Fresh Fruits" className="hero-img-element" />
        </div>
      </section>

      <section className="features container">
        <h2 className="section-title">Why FreshGuard AI?</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon"><Activity size={24} /></div>
            <h3>Three Powerful Models</h3>
            <p>Powered by EfficientNetV2-S, ConvNeXt-Tiny, and MobileViT for diverse performance profiles.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon"><Zap size={24} /></div>
            <h3>Real-Time Inference</h3>
            <p>Optimized architectures delivering sub-100ms predictions natively on CPU.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon"><Shield size={24} /></div>
            <h3>Leakage-Safe Data</h3>
            <p>Trained on a rigorously audited dataset with dHash near-duplicate removal ensuring valid metrics.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon"><CheckCircle size={24} /></div>
            <h3>Explainable AI</h3>
            <p>Human-readable model interpretations and confidence analyses for every prediction.</p>
          </div>
        </div>
      </section>
      
      <section className="supported-classes container">
        <h2 className="section-title">Supported Classes</h2>
        <p className="section-subtitle">Our models are explicitly trained to detect freshness across three major fruit domains.</p>
        <div className="classes-grid">
          <div className="class-card">🍎 Apple (Fresh / Rotten)</div>
          <div className="class-card">🍌 Banana (Fresh / Rotten)</div>
          <div className="class-card">🍊 Orange (Fresh / Rotten)</div>
        </div>
      </section>
    </div>
  );
};
export default Home;
""")

# 2. styles/home.css
write_file(os.path.join(frontend_src, "styles", "home.css"), """.home-container {
  padding-bottom: 4rem;
}

.hero {
  display: flex;
  align-items: center;
  justify-content: space-between;
  max-width: 1200px;
  margin: 4rem auto;
  padding: 0 1.5rem;
  gap: 4rem;
}

.hero-content {
  flex: 1;
}

.badge {
  display: inline-block;
  padding: 0.25rem 0.75rem;
  background-color: var(--success-light);
  color: var(--success-dark);
  border-radius: 999px;
  font-size: 0.875rem;
  font-weight: 600;
  margin-bottom: 1.5rem;
}

.hero-title {
  font-size: 3.5rem;
  font-weight: 700;
  line-height: 1.1;
  color: var(--text-main);
  margin-bottom: 1.5rem;
  letter-spacing: -0.02em;
}

.hero-subtitle {
  font-size: 1.125rem;
  color: var(--text-muted);
  margin-bottom: 2.5rem;
  line-height: 1.6;
}

.hero-buttons {
  display: flex;
  gap: 1rem;
}

.btn-large {
  padding: 0.875rem 1.5rem;
  font-size: 1rem;
  border-radius: var(--radius-lg);
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 600;
  transition: all 0.2s ease;
}

.btn-secondary {
  background-color: white;
  color: var(--text-main);
  border: 1px solid var(--border);
  box-shadow: var(--shadow-sm);
}

.btn-secondary:hover {
  background-color: var(--bg-color);
  border-color: var(--text-muted);
}

.hero-image {
  flex: 1;
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
}

.hero-img-element {
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-hover);
  position: relative;
  z-index: 2;
  width: 100%;
  max-width: 500px;
  object-fit: cover;
}

.hero-glow {
  position: absolute;
  width: 120%;
  height: 120%;
  background: radial-gradient(circle, var(--success-light) 0%, transparent 70%);
  z-index: 1;
  opacity: 0.6;
  top: -10%;
  left: -10%;
}

.section-title {
  text-align: center;
  font-size: 2.25rem;
  font-weight: 700;
  margin-bottom: 1rem;
  color: var(--text-main);
}

.section-subtitle {
  text-align: center;
  color: var(--text-muted);
  margin-bottom: 3rem;
  font-size: 1.125rem;
}

.features {
  margin-top: 6rem;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
  margin-top: 3rem;
}

.feature-card {
  background: var(--card-bg);
  padding: 2rem;
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow);
  border: 1px solid var(--border);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.feature-card:hover {
  transform: translateY(-5px);
  box-shadow: var(--shadow-lg);
}

.feature-icon {
  width: 3rem;
  height: 3rem;
  background-color: var(--success-light);
  color: var(--success-dark);
  border-radius: var(--radius-lg);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 1.25rem;
}

.feature-card h3 {
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 0.75rem;
}

.feature-card p {
  color: var(--text-muted);
  line-height: 1.5;
}

.supported-classes {
  margin-top: 6rem;
  background: linear-gradient(to bottom, transparent, var(--success-light));
  padding: 4rem 1.5rem;
  border-radius: var(--radius-xl);
}

.classes-grid {
  display: flex;
  justify-content: center;
  gap: 1.5rem;
  flex-wrap: wrap;
}

.class-card {
  background: white;
  padding: 1rem 2rem;
  border-radius: var(--radius-lg);
  font-weight: 600;
  color: var(--success-dark);
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--success-light);
  font-size: 1.125rem;
}

@media (max-width: 900px) {
  .hero {
    flex-direction: column;
    text-align: center;
    margin: 2rem auto;
  }
  .hero-buttons {
    justify-content: center;
  }
}
""")

# 3. pages/Scan.jsx (The core camera page)
write_file(os.path.join(frontend_src, "pages", "Scan.jsx"), """import React, { useState, useRef, useEffect } from 'react';
import { Camera, Upload, RefreshCw, ChevronRight, AlertTriangle } from 'lucide-react';
import { predict } from '../services/api';
import '../styles/scan.css';

const Scan = () => {
  const [stream, setStream] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [modelKey, setModelKey] = useState('efficientnetv2_s');
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, []);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } 
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setError(null);
    } catch (err) {
      console.error("Camera error:", err);
      setError("Camera access denied or unavailable. Please upload an image instead.");
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
      setImageSrc(dataUrl);
      stopCamera();
    }
  };

  const retakePhoto = () => {
    setImageSrc(null);
    setResult(null);
    setError(null);
    startCamera();
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setImageSrc(e.target.result);
        stopCamera();
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
    setError(null);
    try {
      const blob = dataURLtoBlob(imageSrc);
      const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
      
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
      // Keep only last 10
      localStorage.setItem('freshguard_history', JSON.stringify(history.slice(0, 10)));
      
    } catch (err) {
      setError("Failed to analyze image. Ensure the backend server is running.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="scan-container container animate-fade-in">
      <div className="scan-header">
        <h1 className="section-title">Scan Your Fruit</h1>
        <p className="section-subtitle">Position the fruit clearly in the frame.</p>
      </div>

      <div className="scan-main">
        <div className="camera-section card">
          {!imageSrc ? (
            <div className="camera-view">
              {error ? (
                <div className="camera-error">
                  <AlertTriangle size={32} />
                  <p>{error}</p>
                </div>
              ) : (
                <>
                  <video ref={videoRef} autoPlay playsInline muted className="video-element"></video>
                  <div className="camera-overlay">
                    <div className="guide-box"></div>
                    <p className="guide-text">Place fruit here</p>
                  </div>
                </>
              )}
              <canvas ref={canvasRef} style={{ display: 'none' }}></canvas>
              
              <div className="camera-controls">
                <button className="btn-capture" onClick={capturePhoto} disabled={!!error}>
                  <div className="capture-inner"></div>
                </button>
                <div className="upload-wrapper">
                  <input 
                    type="file" 
                    accept="image/*" 
                    className="file-input" 
                    ref={fileInputRef}
                    onChange={handleFileUpload}
                  />
                  <button className="btn-icon" onClick={() => fileInputRef.current.click()}>
                    <Upload size={24} />
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="preview-view">
              <img src={imageSrc} alt="Captured" className="preview-image" />
              
              {!result && !loading && (
                <div className="preview-controls">
                  <button className="btn-secondary" onClick={retakePhoto}>
                    <RefreshCw size={18} /> Retake
                  </button>
                  
                  <select 
                    className="model-select"
                    value={modelKey} 
                    onChange={(e) => setModelKey(e.target.value)}
                  >
                    <option value="efficientnetv2_s">EfficientNetV2-S (Recommended)</option>
                    <option value="convnext_tiny">ConvNeXt-Tiny</option>
                    <option value="mobilevit">MobileViT (Fastest)</option>
                  </select>

                  <button className="btn-primary" onClick={analyzeImage}>
                    Analyze <ChevronRight size={18} />
                  </button>
                </div>
              )}
              
              {loading && (
                <div className="loading-state">
                  <div className="spinner"></div>
                  <p>Analyzing visual features...</p>
                </div>
              )}
            </div>
          )}
        </div>

        {result && (
          <div className="result-section card animate-fade-in">
            {result.is_low_confidence ? (
              <div className="low-confidence-warning">
                <AlertTriangle size={32} />
                <h3>Low Confidence Prediction</h3>
                <p>{result.explanation}</p>
                <button className="btn-primary" onClick={retakePhoto}>Try Again</button>
              </div>
            ) : (
              <>
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
                
                <div className="explanation-box">
                  <h4>Model Interpretation</h4>
                  <p>{result.explanation}</p>
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

                <div className="probabilities-box">
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
                
                <button className="btn-secondary mt-4 w-full" onClick={retakePhoto}>
                  Scan Another Fruit
                </button>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
export default Scan;
""")

# 4. styles/scan.css
write_file(os.path.join(frontend_src, "styles", "scan.css"), """.scan-container {
  margin-top: 2rem;
  padding-bottom: 4rem;
}

.scan-header {
  margin-bottom: 2rem;
}

.scan-main {
  display: flex;
  flex-direction: column;
  gap: 2rem;
  max-width: 800px;
  margin: 0 auto;
}

.card {
  background: var(--card-bg);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow);
  border: 1px solid var(--border);
  overflow: hidden;
}

.camera-view {
  position: relative;
  background-color: #000;
  aspect-ratio: 4/3;
  display: flex;
  align-items: center;
  justify-content: center;
}

.video-element {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.camera-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  pointer-events: none;
}

.guide-box {
  width: 60%;
  height: 60%;
  border: 2px dashed rgba(255,255,255,0.7);
  border-radius: var(--radius-lg);
  box-shadow: 0 0 0 4000px rgba(0,0,0,0.3);
}

.guide-text {
  color: white;
  margin-top: 1rem;
  font-weight: 500;
  text-shadow: 0 1px 2px rgba(0,0,0,0.8);
}

.camera-controls {
  position: absolute;
  bottom: 2rem;
  left: 0;
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 2rem;
}

.btn-capture {
  width: 4rem;
  height: 4rem;
  border-radius: 50%;
  background: rgba(255,255,255,0.3);
  border: 3px solid white;
  display: flex;
  align-items: center;
  justify-content: center;
}

.capture-inner {
  width: 3rem;
  height: 3rem;
  border-radius: 50%;
  background: white;
  transition: transform 0.1s;
}

.btn-capture:active .capture-inner {
  transform: scale(0.9);
}

.upload-wrapper {
  position: absolute;
  right: 2rem;
}

.file-input {
  display: none;
}

.btn-icon {
  background: rgba(0,0,0,0.5);
  color: white;
  width: 3rem;
  height: 3rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.btn-icon:hover {
  background: rgba(0,0,0,0.7);
}

.camera-error {
  color: white;
  text-align: center;
  padding: 2rem;
}

.preview-view {
  display: flex;
  flex-direction: column;
}

.preview-image {
  width: 100%;
  max-height: 50vh;
  object-fit: contain;
  background: #f1f5f9;
}

.preview-controls {
  padding: 1.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  background: white;
}

.model-select {
  padding: 0.75rem;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  flex: 1;
  font-family: inherit;
  font-size: 0.95rem;
  outline: none;
}

.loading-state {
  padding: 3rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  color: var(--text-muted);
}

.spinner {
  width: 2.5rem;
  height: 2.5rem;
  border: 3px solid var(--border);
  border-top-color: var(--success);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.result-section {
  padding: 2rem;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 1rem;
}

.result-title {
  font-size: 2rem;
  font-weight: 700;
}

.text-success { color: var(--success-dark); }
.text-danger { color: var(--danger-dark); }
.bg-success { background-color: var(--success); }
.bg-danger { background-color: var(--danger); }

.confidence-badge {
  background: var(--bg-color);
  padding: 0.5rem 1rem;
  border-radius: var(--radius-lg);
  font-weight: 600;
  border: 1px solid var(--border);
}

.confidence-bar-container {
  height: 0.75rem;
  background: var(--bg-color);
  border-radius: 999px;
  overflow: hidden;
  margin-bottom: 2rem;
}

.confidence-bar {
  height: 100%;
  border-radius: 999px;
  transition: width 1s ease-out;
}

.explanation-box {
  background: var(--success-light);
  padding: 1.25rem;
  border-radius: var(--radius);
  margin-bottom: 1.5rem;
  border-left: 4px solid var(--success);
}
.explanation-box h4 {
  color: var(--success-dark);
  margin-bottom: 0.5rem;
}
.explanation-box p {
  color: var(--text-main);
}

.result-details {
  display: flex;
  gap: 2rem;
  margin-bottom: 2rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid var(--border);
}

.detail-item {
  display: flex;
  flex-direction: column;
}

.detail-label {
  font-size: 0.875rem;
  color: var(--text-muted);
}

.detail-value {
  font-weight: 600;
}

.probabilities-box h4 {
  margin-bottom: 1rem;
}

.prob-row {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 0.75rem;
}

.prob-label {
  width: 120px;
  font-size: 0.875rem;
  color: var(--text-muted);
}

.prob-bar-wrapper {
  flex: 1;
  height: 0.5rem;
  background: var(--bg-color);
  border-radius: 999px;
  overflow: hidden;
}

.prob-bar {
  height: 100%;
  background: var(--primary);
  border-radius: 999px;
}

.prob-value {
  width: 40px;
  text-align: right;
  font-size: 0.875rem;
  font-weight: 600;
}

.w-full {
  width: 100%;
  justify-content: center;
}
.mt-4 {
  margin-top: 1rem;
}

.low-confidence-warning {
  text-align: center;
  padding: 2rem;
  color: var(--warning-dark);
}
.low-confidence-warning h3 { margin: 1rem 0; }
.low-confidence-warning p { margin-bottom: 1.5rem; color: var(--text-muted); }

@media (max-width: 600px) {
  .preview-controls {
    flex-direction: column;
  }
  .preview-controls button, .preview-controls select {
    width: 100%;
  }
}
""")

print("Scan and Home pages created.")
