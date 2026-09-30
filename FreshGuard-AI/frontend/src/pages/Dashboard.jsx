import React, { useEffect, useState } from 'react';
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

    </div>
  );
};
export default Dashboard;
