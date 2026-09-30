import React from 'react';
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
