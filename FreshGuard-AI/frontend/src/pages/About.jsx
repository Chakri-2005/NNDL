import React from 'react';

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
