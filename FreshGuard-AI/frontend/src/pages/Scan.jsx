import React, { useState, useRef, useEffect } from 'react';
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
            <>
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
          </div>
        )}
      </div>
    </div>
  );
};
export default Scan;
