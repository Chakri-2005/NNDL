import React, { useState, useEffect } from 'react';
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
