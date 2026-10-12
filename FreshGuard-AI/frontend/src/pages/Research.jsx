import React from 'react';
import { Link } from 'react-router-dom';

const Research = () => {
  return (
    <div className="container animate-fade-in" style={{ marginTop: '2rem', maxWidth: '860px', paddingBottom: '4rem' }}>
      <h1 className="section-title">Research &amp; Methodology</h1>

      {/* Dataset */}
      <div className="card mt-4" style={{ padding: '2rem' }}>
        <h2 style={{ color: 'var(--primary-dark)', marginBottom: '1rem' }}>Dataset Curation</h2>
        <p style={{ marginBottom: '1rem', lineHeight: '1.6' }}>
          The initial dataset contained 12,000 images, which was rigorously scoped to 6,000 images covering
          Apple, Banana, and Orange classes (Fresh / Rotten). A core contribution of this project is the{' '}
          <strong>Leakage-Safe Data Pipeline</strong>: byte-level deduplication and dHash near-duplicate
          clustering via Union-Find ensure that near-identical images never cross Train/Validation/Test boundaries.
        </p>
      </div>

      {/* Three Classifiers */}
      <div className="card mt-4" style={{ padding: '2rem' }}>
        <h2 style={{ color: 'var(--primary-dark)', marginBottom: '1rem' }}>Three Benchmark Classifiers</h2>
        <ul style={{ paddingLeft: '1.5rem', lineHeight: '1.9' }}>
          <li>
            <strong>⚡ EfficientNetV2-S:</strong> Designed for high parameter efficiency and accuracy.
            Achieved the highest overall F1 score. Paired with the <em>Residual DCGAN</em>.
          </li>
          <li>
            <strong>🧱 ConvNeXt-Tiny:</strong> A modernised pure-CNN architecture offering excellent
            representational power. Paired with the <em>Inverted-Bottleneck DCGAN</em>.
          </li>
          <li>
            <strong>🔭 MobileViT-S:</strong> A hybrid Vision Transformer designed for mobile environments
            (~36 ms inference). Paired with the <em>Transformer-Augmented GAN</em>.
          </li>
        </ul>
      </div>

      {/* Fair Evaluation */}
      <div className="card mt-4" style={{ padding: '2rem' }}>
        <h2 style={{ color: 'var(--primary-dark)', marginBottom: '1rem' }}>Fair Evaluation Framework</h2>
        <p style={{ lineHeight: '1.6' }}>
          All three architectures share an identical standardised classification head
          (LayerNorm → Linear → GELU → Dropout → Linear). Models were fine-tuned using a 2-stage
          frozen/unfrozen transfer-learning strategy, prioritising reproducibility and preventing
          artificial accuracy inflation.
        </p>
      </div>

      {/* Three Dedicated GANs */}
      <div className="card mt-4" style={{ padding: '2rem', borderTop: '3px solid #6c63ff' }}>
        <h2 style={{ color: 'var(--primary-dark)', marginBottom: '1rem' }}>🧬 Three Dedicated GANs</h2>
        <p style={{ lineHeight: '1.6', marginBottom: '1rem' }}>
          A key novelty of this project is the use of <strong>three independent GANs</strong> — one dedicated
          to each classifier — rather than a single shared generative model. This allows each GAN to be tuned
          for the representational complexity of its paired classifier and provides a scientifically valid
          per-model augmentation condition.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', marginTop: '1.2rem' }}>
          {[
            { icon: '⚡', name: 'EfficientNet GAN', arch: 'Residual DCGAN', color: '#6c63ff' },
            { icon: '🧱', name: 'ConvNeXt GAN', arch: 'Inverted-Bottleneck DCGAN', color: '#00b894' },
            { icon: '🔭', name: 'MobileViT GAN', arch: 'Transformer-Augmented GAN', color: '#e17055' },
          ].map((g) => (
            <div
              key={g.name}
              style={{
                background: `${g.color}11`,
                border: `1px solid ${g.color}44`,
                borderRadius: '12px',
                padding: '1rem',
              }}
            >
              <div style={{ fontSize: '1.4rem', marginBottom: '0.3rem' }}>{g.icon}</div>
              <strong style={{ color: g.color }}>{g.name}</strong>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted, #94a3b8)', marginTop: '0.3rem' }}>
                {g.arch}
              </p>
            </div>
          ))}
        </div>

        <p style={{ marginTop: '1.2rem', lineHeight: '1.6', fontSize: '0.9rem', color: 'var(--text-muted, #64748b)' }}>
          All three GANs were trained on the same dataset, evaluated independently for FID, KID, perceptual
          diversity, and mode-collapse detection. View full results and synthetic image grids on the{' '}
          <Link to="/gan-lab" style={{ color: '#6c63ff', fontWeight: '600' }}>🧬 GAN Laboratory</Link> page.
        </p>
      </div>

      {/* Robustness & Forensics */}
      <div className="card mt-4" style={{ padding: '2rem' }}>
        <h2 style={{ color: 'var(--primary-dark)', marginBottom: '1rem' }}>Robustness &amp; Forensics</h2>
        <p style={{ lineHeight: '1.6' }}>
          Beyond classification accuracy, each model was subjected to a comprehensive robustness suite:
        </p>
        <ul style={{ paddingLeft: '1.5rem', lineHeight: '1.9', marginTop: '0.8rem' }}>
          <li><strong>Adversarial Robustness:</strong> FGSM attacks at multiple epsilon levels.</li>
          <li><strong>Distribution Shift:</strong> Gaussian noise, blur, and brightness perturbations.</li>
          <li><strong>Data Leakage Detection:</strong> Byte-level and perceptual-hash checks across splits.</li>
          <li><strong>Calibration:</strong> Expected Calibration Error (ECE) and reliability diagrams.</li>
          <li><strong>Label Smoothing Analysis:</strong> Effect on overconfidence and ECE.</li>
        </ul>
      </div>
    </div>
  );
};
export default Research;
