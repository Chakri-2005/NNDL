import os

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

frontend_src = r"C:\NNDL\FreshGuard-AI\frontend\src"

# 1. pages/Dashboard.jsx
write_file(os.path.join(frontend_src, "pages", "Dashboard.jsx"), """import React, { useEffect, useState } from 'react';
import { getMetrics } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import '../styles/dashboard.css';

const Dashboard = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const data = await getMetrics();
        setMetrics(data.metrics);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchMetrics();
  }, []);

  if (loading) return <div className="loading-state"><div className="spinner"></div><p>Loading Dashboard...</p></div>;
  if (!metrics) return <div className="container mt-4">Failed to load metrics. Is the backend running?</div>;

  const compareData = [
    {
      name: 'Accuracy',
      'EfficientNetV2-S': (metrics.efficientnetv2_s.overall.accuracy * 100).toFixed(1),
      'ConvNeXt-Tiny': (metrics.convnext_tiny.overall.accuracy * 100).toFixed(1),
      'MobileViT': (metrics.mobilevit.overall.accuracy * 100).toFixed(1)
    },
    {
      name: 'F1 Score',
      'EfficientNetV2-S': (metrics.efficientnetv2_s.overall.f1_macro * 100).toFixed(1),
      'ConvNeXt-Tiny': (metrics.convnext_tiny.overall.f1_macro * 100).toFixed(1),
      'MobileViT': (metrics.mobilevit.overall.f1_macro * 100).toFixed(1)
    }
  ];

  return (
    <div className="dashboard-container container animate-fade-in">
      <h1 className="section-title">Performance Dashboard</h1>
      <p className="section-subtitle">Real metrics loaded directly from the model evaluation results.</p>
      
      <div className="metrics-grid">
        {Object.entries(metrics).map(([key, data]) => (
          <div key={key} className="metric-card card">
            <h3 className="card-title">{data.overall.model_name}</h3>
            <div className="stat-row"><span>Accuracy</span><strong>{(data.overall.accuracy * 100).toFixed(2)}%</strong></div>
            <div className="stat-row"><span>F1 Score (Macro)</span><strong>{(data.overall.f1_macro * 100).toFixed(2)}%</strong></div>
            <div className="stat-row"><span>Parameters</span><strong>{(data.overall.total_parameters / 1000000).toFixed(1)}M</strong></div>
            <div className="stat-row"><span>Inference Time</span><strong>{data.overall.inference_time_ms.toFixed(1)} ms</strong></div>
          </div>
        ))}
      </div>

      <div className="charts-section card mt-4">
        <h3 className="card-title">Model Comparison</h3>
        <div style={{ height: 400, width: '100%' }}>
          <ResponsiveContainer>
            <BarChart data={compareData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[90, 100]} />
              <Tooltip />
              <Legend />
              <Bar dataKey="EfficientNetV2-S" fill="var(--primary)" />
              <Bar dataKey="ConvNeXt-Tiny" fill="var(--success)" />
              <Bar dataKey="MobileViT" fill="var(--warning)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      <div className="per-class-section mt-4">
        <h2 className="section-title">Per-Class Performance (EfficientNetV2-S)</h2>
        <div className="table-responsive card">
          <table className="data-table">
            <thead>
              <tr>
                <th>Class</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>Support</th>
              </tr>
            </thead>
            <tbody>
              {metrics.efficientnetv2_s.per_class.map((cls, idx) => (
                <tr key={idx}>
                  <td>{cls.class_name}</td>
                  <td>{(cls.precision * 100).toFixed(1)}%</td>
                  <td>{(cls.recall * 100).toFixed(1)}%</td>
                  <td>{(cls.f1_score * 100).toFixed(1)}%</td>
                  <td>{cls.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
export default Dashboard;
""")

# 2. pages/CompareModels.jsx
write_file(os.path.join(frontend_src, "pages", "CompareModels.jsx"), """import React, { useState, useRef } from 'react';
import { Upload, ChevronRight, Activity } from 'lucide-react';
import { compareModels } from '../services/api';

const CompareModels = () => {
  const [imageSrc, setImageSrc] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setImageSrc(e.target.result);
        setResults(null);
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

  const analyzeAll = async () => {
    if (!imageSrc) return;
    setLoading(true);
    try {
      const blob = dataURLtoBlob(imageSrc);
      const file = new File([blob], 'compare.jpg', { type: 'image/jpeg' });
      const res = await compareModels(file);
      setResults(res.results);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container animate-fade-in" style={{ paddingBottom: '4rem' }}>
      <div className="scan-header" style={{ marginTop: '2rem' }}>
        <h1 className="section-title">Model Agreement Analysis</h1>
        <p className="section-subtitle">Run all three benchmark models simultaneously on the same image.</p>
      </div>

      <div className="card text-center" style={{ padding: '3rem', maxWidth: '600px', margin: '0 auto 2rem auto' }}>
        {!imageSrc ? (
          <div onClick={() => fileInputRef.current.click()} style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
            <div className="feature-icon"><Upload size={32} /></div>
            <h3>Upload an Image to Compare</h3>
            <p className="text-muted">Supports JPG, PNG, WEBP</p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
            <img src={imageSrc} style={{ maxHeight: '300px', borderRadius: 'var(--radius-lg)' }} alt="Upload" />
            <div style={{ display: 'flex', gap: '1rem' }}>
              <button className="btn-secondary" onClick={() => setImageSrc(null)}>Change Image</button>
              <button className="btn-primary" onClick={analyzeAll} disabled={loading}>
                {loading ? 'Analyzing...' : 'Run All Models'} <ChevronRight size={18} />
              </button>
            </div>
          </div>
        )}
        <input type="file" accept="image/*" ref={fileInputRef} onChange={handleFileUpload} style={{ display: 'none' }} />
      </div>

      {results && (
        <div>
          <h2 className="section-title" style={{ marginTop: '4rem' }}>Comparison Results</h2>
          <div className="features-grid">
            {results.map((res, idx) => (
              <div key={idx} className="card" style={{ padding: '1.5rem', borderTop: `4px solid ${res.freshness === 'Fresh' ? 'var(--success)' : 'var(--danger)'}` }}>
                <h3 style={{ marginBottom: '1rem', color: 'var(--primary-dark)' }}>{res.model}</h3>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                  {res.fruit} • <span style={{ color: res.freshness === 'Fresh' ? 'var(--success-dark)' : 'var(--danger-dark)' }}>{res.freshness}</span>
                </div>
                <div className="stat-row"><span>Confidence</span><strong>{res.confidence}%</strong></div>
                <div className="stat-row"><span>Inference Time</span><strong>{res.inference_time_ms} ms</strong></div>
              </div>
            ))}
          </div>
          
          <div className="card mt-4" style={{ padding: '2rem', textAlign: 'center', background: 'var(--primary-light)', border: '1px solid var(--primary)' }}>
            <h3><Activity size={20} style={{ display: 'inline', marginRight: '0.5rem', verticalAlign: 'middle' }}/> Ensemble Agreement</h3>
            <p style={{ marginTop: '0.5rem', fontSize: '1.125rem' }}>
              {new Set(results.map(r => r.predicted_class)).size === 1 
                ? "✅ All 3 models agree on the prediction." 
                : "⚠️ Models disagree. Consider retaking the image with better lighting."}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
export default CompareModels;
""")

# 3. styles/dashboard.css
write_file(os.path.join(frontend_src, "styles", "dashboard.css"), """.dashboard-container {
  padding-bottom: 4rem;
  margin-top: 2rem;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
}

.metric-card {
  padding: 1.5rem;
}

.card-title {
  font-size: 1.25rem;
  color: var(--primary-dark);
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}

.stat-row {
  display: flex;
  justify-content: space-between;
  padding: 0.75rem 0;
  border-bottom: 1px solid var(--border);
}

.stat-row:last-child {
  border-bottom: none;
}

.stat-row span {
  color: var(--text-muted);
}

.table-responsive {
  overflow-x: auto;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
}

.data-table th, .data-table td {
  padding: 1rem;
  text-align: left;
  border-bottom: 1px solid var(--border);
}

.data-table th {
  background-color: var(--bg-color);
  font-weight: 600;
  color: var(--text-muted);
}

.data-table tr:hover {
  background-color: var(--primary-light);
}
""")

# 4. pages/History.jsx
write_file(os.path.join(frontend_src, "pages", "History.jsx"), """import React, { useState, useEffect } from 'react';
import { Trash2 } from 'lucide-react';

const HistoryPage = () => {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const historyStr = localStorage.getItem('freshguard_history');
    if (historyStr) {
      setHistory(JSON.parse(historyStr));
    }
  }, []);

  const clearHistory = () => {
    localStorage.removeItem('freshguard_history');
    setHistory([]);
  };

  return (
    <div className="container animate-fade-in" style={{ marginTop: '2rem', paddingBottom: '4rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <h1 className="section-title" style={{ margin: 0 }}>Prediction History</h1>
        {history.length > 0 && (
          <button className="btn-secondary" onClick={clearHistory} style={{ display: 'flex', gap: '0.5rem', color: 'var(--danger-dark)', borderColor: 'var(--danger-light)' }}>
            <Trash2 size={18} /> Clear
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="card text-center" style={{ padding: '4rem', color: 'var(--text-muted)' }}>
          <p>No recent predictions found.</p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {history.map((item, idx) => (
            <div key={idx} className="card" style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', padding: '1rem' }}>
              <img src={item.thumbnail} alt="thumbnail" style={{ width: '80px', height: '80px', objectFit: 'cover', borderRadius: 'var(--radius)' }} />
              <div style={{ flex: 1 }}>
                <h3 style={{ fontSize: '1.25rem', marginBottom: '0.25rem' }}>{item.fruit} • <span style={{ color: item.freshness === 'Fresh' ? 'var(--success-dark)' : 'var(--danger-dark)' }}>{item.freshness}</span></h3>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                  Model: {item.model} | Confidence: {item.confidence}% | Date: {new Date(item.date).toLocaleString()}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
export default HistoryPage;
""")

# 5. pages/Research.jsx
write_file(os.path.join(frontend_src, "pages", "Research.jsx"), """import React from 'react';

const Research = () => {
  return (
    <div className="container animate-fade-in" style={{ marginTop: '2rem', maxWidth: '800px', paddingBottom: '4rem' }}>
      <h1 className="section-title">Research & Methodology</h1>
      
      <div className="card mt-4" style={{ padding: '2rem' }}>
        <h2 style={{ color: 'var(--primary-dark)', marginBottom: '1rem' }}>Dataset Curation</h2>
        <p style={{ marginBottom: '1rem', lineHeight: '1.6' }}>
          The initial dataset contained 12,000 images, which was rigorously scoped to 6,000 images covering Apple, Banana, and Orange classes. 
          A core contribution of this project is the <strong>Leakage-Safe Data Pipeline</strong>. We implemented byte-level deduplication and dHash near-duplicate clustering via Union-Find to ensure that near-identical images never cross the Train/Validation/Test boundaries.
        </p>
        
        <h2 style={{ color: 'var(--primary-dark)', margin: '2rem 0 1rem 0' }}>Three Benchmark Architectures</h2>
        <ul style={{ paddingLeft: '1.5rem', lineHeight: '1.8' }}>
          <li><strong>EfficientNetV2-S:</strong> Designed for high parameter efficiency and accuracy. Achieved the highest overall F1 score.</li>
          <li><strong>ConvNeXt-Tiny:</strong> A modernized pure CNN architecture offering excellent representational power.</li>
          <li><strong>MobileViT:</strong> A hybrid Vision Transformer designed for mobile environments, offering the lowest inference latency (~36ms).</li>
        </ul>

        <h2 style={{ color: 'var(--primary-dark)', margin: '2rem 0 1rem 0' }}>Fair Evaluation Framework</h2>
        <p style={{ lineHeight: '1.6' }}>
          To ensure scientific validity, all three architectures utilize an identical standardized classification head (LayerNorm → Linear → GELU → Dropout → Linear). 
          The models were fine-tuned using a 2-stage frozen/unfrozen transfer learning strategy, prioritizing reproducibility and preventing artificial accuracy inflation.
        </p>
      </div>
    </div>
  );
};
export default Research;
""")

# 6. pages/About.jsx
write_file(os.path.join(frontend_src, "pages", "About.jsx"), """import React from 'react';

const About = () => {
  return (
    <div className="container animate-fade-in" style={{ marginTop: '2rem', maxWidth: '800px', paddingBottom: '4rem', textAlign: 'center' }}>
      <h1 className="section-title">About FreshGuard AI</h1>
      <p className="section-subtitle">Smarter freshness for healthier living.</p>
      
      <div className="card mt-4" style={{ padding: '3rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2rem' }}>
        <div style={{ width: '80px', height: '80px', background: 'var(--success-light)', color: 'var(--success-dark)', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"></path><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"></path></svg>
        </div>
        <div>
          <h3>Our Mission</h3>
          <p style={{ color: 'var(--text-muted)', marginTop: '1rem', lineHeight: '1.6' }}>
            FreshGuard AI was developed to provide an accessible, AI-powered tool for assessing food freshness. 
            By leveraging state-of-the-art deep learning architectures, we aim to reduce food waste and improve food safety at the consumer level.
          </p>
        </div>
        <div style={{ width: '100%', height: '1px', background: 'var(--border)' }}></div>
        <div>
          <h4 style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Technology Stack</h4>
          <p><strong>Frontend:</strong> React, Vite, Recharts, Lucide Icons</p>
          <p><strong>Backend:</strong> Flask, PyTorch, Torchvision</p>
        </div>
      </div>
    </div>
  );
};
export default About;
""")

print("Dashboard, CompareModels, History, Research, and About pages created.")
