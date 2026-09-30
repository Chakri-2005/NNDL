import React from 'react';

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
