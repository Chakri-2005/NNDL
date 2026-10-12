import React, { useEffect, useState, useCallback } from 'react';
import {
  getGanAll,
  getGanSummary,
  getGanSamplesUrl,
  getGanLossCurveUrl,
} from '../services/api';
import '../styles/gan_lab.css';

// ─── Constants ────────────────────────────────────────────────────────────────
const MODEL_KEYS = ['efficientnetv2_s', 'convnext_tiny', 'mobilevit_s'];

const MODEL_META = {
  efficientnetv2_s: {
    label: 'EfficientNetV2-S',
    ganLabel: 'EfficientNet GAN',
    color: '#6c63ff',
    accent: '#a29bfe',
    icon: '⚡',
    ganId: 'efficientnet_gan',
    description: 'Residual DCGAN — 4-layer residual generator matched to EfficientNet scale and depth.',
  },
  convnext_tiny: {
    label: 'ConvNeXt-Tiny',
    ganLabel: 'ConvNeXt GAN',
    color: '#00b894',
    accent: '#55efc4',
    icon: '🧱',
    ganId: 'convnext_gan',
    description: 'Inverted-Bottleneck DCGAN — depth-wise separable blocks mirroring ConvNeXt stage structure.',
  },
  mobilevit_s: {
    label: 'MobileViT-S',
    ganLabel: 'MobileViT GAN',
    color: '#e17055',
    accent: '#fab1a0',
    icon: '🔭',
    ganId: 'mobilevit_gan',
    description: 'Transformer-augmented GAN — lightweight self-attention in the generator bottleneck.',
  },
};

const METRIC_CONFIG = [
  { key: 'fid_score',                   label: 'FID ↓',           fmt: (v) => v?.toFixed(1) ?? '—',    tip: 'Fréchet Inception Distance — lower means samples are closer to real images.' },
  { key: 'kid_score',                   label: 'KID ↓',           fmt: (v) => v?.toFixed(3) ?? '—',    tip: 'Kernel Inception Distance — lower is better.' },
  { key: 'classifier_consistency',      label: 'Class Consist. ↑', fmt: (v) => v !== undefined ? `${(v*100).toFixed(1)}%` : '—', tip: 'Fraction of synthetic images the classifier predicts with high confidence.' },
  { key: 'mean_perceptual_diversity_bits', label: 'Diversity ↑', fmt: (v) => v?.toFixed(2) ?? '—',   tip: 'Mean perceptual hash diversity in bits — higher means less mode collapse.' },
  { key: 'latency_ms_per_image',        label: 'Latency (ms)',    fmt: (v) => v?.toFixed(2) ?? '—',    tip: 'Generator inference time per image.' },
  { key: 'throughput_imgs_per_sec',     label: 'Throughput',      fmt: (v) => v ? `${v.toFixed(0)} img/s` : '—', tip: 'Images generated per second.' },
  { key: 'epochs_trained',              label: 'Epochs',          fmt: (v) => v ?? '—',               tip: 'Training epochs completed.' },
  { key: 'evaluated_samples',           label: 'Eval Samples',    fmt: (v) => v ?? '—',               tip: 'Number of synthetic images evaluated.' },
];

// ─── Sub-components ───────────────────────────────────────────────────────────

const MetricBadge = ({ label, value, tip, highlight }) => (
  <div className={`glab-metric-badge ${highlight ? 'glab-metric-badge--highlight' : ''}`} title={tip}>
    <span className="glab-metric-label">{label}</span>
    <span className="glab-metric-value">{value}</span>
  </div>
);

const Tooltip = ({ text }) => (
  <span className="glab-tooltip" title={text}>ⓘ</span>
);

const TabBar = ({ tabs, active, onSelect, color }) => (
  <div className="glab-tabbar">
    {tabs.map((t) => (
      <button
        key={t.key}
        className={`glab-tab ${active === t.key ? 'glab-tab--active' : ''}`}
        style={active === t.key ? { borderColor: color, color } : {}}
        onClick={() => onSelect(t.key)}
      >
        {t.label}
      </button>
    ))}
  </div>
);

const ImageViewer = ({ src, alt, caption }) => {
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(false);

  return (
    <div className="glab-img-wrapper">
      {!loaded && !error && <div className="glab-img-skeleton" />}
      {error ? (
        <div className="glab-img-error">⚠ Image not available (backend offline?)</div>
      ) : (
        <img
          src={src}
          alt={alt}
          onLoad={() => setLoaded(true)}
          onError={() => setError(true)}
          style={{ display: loaded ? 'block' : 'none' }}
          className="glab-img"
        />
      )}
      {caption && <p className="glab-img-caption">{caption}</p>}
    </div>
  );
};

const SummaryTable = ({ summary }) => {
  if (!summary?.length) return null;
  return (
    <div className="glab-table-wrap">
      <table className="glab-table">
        <thead>
          <tr>
            <th>GAN</th>
            <th>FID ↓</th>
            <th>KID ↓</th>
            <th>Consistency ↑</th>
            <th>Diversity ↑</th>
            <th>Latency (ms)</th>
            <th>Mode Collapse</th>
          </tr>
        </thead>
        <tbody>
          {summary.map((row) => {
            const meta = MODEL_META[row.model_key];
            return (
              <tr key={row.model_key}>
                <td>
                  <span className="glab-table-icon" style={{ color: meta.color }}>{meta.icon}</span>
                  {meta.label}
                </td>
                <td>{row.fid_score?.toFixed(1) ?? '—'}</td>
                <td>{row.kid_score?.toFixed(3) ?? '—'}</td>
                <td>{row.classifier_consistency !== undefined ? `${(row.classifier_consistency * 100).toFixed(1)}%` : '—'}</td>
                <td>{row.mean_perceptual_diversity_bits?.toFixed(2) ?? '—'}</td>
                <td>{row.latency_ms_per_image?.toFixed(2) ?? '—'}</td>
                <td>
                  <span className={`glab-badge ${row.mode_collapse_detected ? 'glab-badge--warn' : 'glab-badge--ok'}`}>
                    {row.mode_collapse_detected ? 'Detected' : 'None'}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

const DiversityBar = ({ classDiversity }) => {
  if (!classDiversity) return null;
  const entries = Object.entries(classDiversity);
  const max = Math.max(...entries.map(([, v]) => v));

  return (
    <div className="glab-diversity">
      <h4 className="glab-section-label">Per-Class Diversity (bits)</h4>
      <div className="glab-diversity-bars">
        {entries.map(([cls, val]) => (
          <div key={cls} className="glab-bar-row">
            <span className="glab-bar-label">{cls.replace('_', ' ')}</span>
            <div className="glab-bar-track">
              <div
                className="glab-bar-fill"
                style={{ width: `${(val / max) * 100}%` }}
              />
            </div>
            <span className="glab-bar-val">{val.toFixed(1)}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

// ─── Model Card ───────────────────────────────────────────────────────────────
const GANCard = ({ modelKey, metrics }) => {
  const meta = MODEL_META[modelKey];
  const [activeTab, setActiveTab] = useState('metrics');

  const tabs = [
    { key: 'metrics', label: '📊 Metrics' },
    { key: 'samples', label: '🖼 Synthetic Samples' },
    { key: 'loss',    label: '📉 Loss Curves' },
  ];

  return (
    <div className="glab-card" style={{ '--card-accent': meta.color }}>
      {/* Card Header */}
      <div className="glab-card-header" style={{ background: `linear-gradient(135deg, ${meta.color}22, ${meta.accent}11)` }}>
        <div className="glab-card-title-row">
          <span className="glab-card-icon">{meta.icon}</span>
          <div>
            <h3 className="glab-card-title" style={{ color: meta.color }}>{meta.ganLabel}</h3>
            <p className="glab-card-subtitle">Paired with {meta.label}</p>
          </div>
          {metrics?.mode_collapse_detected === false && (
            <span className="glab-badge glab-badge--ok glab-badge--float">✓ No Mode Collapse</span>
          )}
        </div>
        <p className="glab-card-desc">{meta.description}</p>
      </div>

      {/* Tab Navigation */}
      <TabBar tabs={tabs} active={activeTab} onSelect={setActiveTab} color={meta.color} />

      {/* Tab Content */}
      <div className="glab-card-body">
        {activeTab === 'metrics' && (
          <>
            <div className="glab-metrics-grid">
              {METRIC_CONFIG.map(({ key, label, fmt, tip }) => (
                <MetricBadge
                  key={key}
                  label={label}
                  value={metrics ? fmt(metrics[key]) : '—'}
                  tip={tip}
                  highlight={key === 'fid_score' || key === 'mean_perceptual_diversity_bits'}
                />
              ))}
            </div>
            {metrics?.diversity_by_class && (
              <DiversityBar classDiversity={metrics.diversity_by_class} />
            )}
            {metrics?.timestamp && (
              <p className="glab-meta-note">Evaluated: {metrics.timestamp} · Checkpoint: {metrics.checkpoint_file}</p>
            )}
          </>
        )}

        {activeTab === 'samples' && (
          <ImageViewer
            src={getGanSamplesUrl(modelKey)}
            alt={`Synthetic samples from ${meta.ganLabel}`}
            caption={`Grid of 96 synthetic images generated by ${meta.ganLabel}. Each row = one fruit class.`}
          />
        )}

        {activeTab === 'loss' && (
          <ImageViewer
            src={getGanLossCurveUrl(modelKey)}
            alt={`Training loss curves for ${meta.ganLabel}`}
            caption={`Generator vs. Discriminator loss over training epochs for ${meta.ganLabel}.`}
          />
        )}
      </div>
    </div>
  );
};

// ─── Main Page ────────────────────────────────────────────────────────────────
const GANLaboratory = () => {
  const [allMetrics, setAllMetrics] = useState(null);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeSection, setActiveSection] = useState('overview');

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [metricsRes, summaryRes] = await Promise.all([
        getGanAll(),
        getGanSummary(),
      ]);
      setAllMetrics(metricsRes.gan_metrics);
      setSummary(summaryRes.gan_summary);
    } catch (err) {
      setError('Could not load GAN data. Please ensure the Flask backend is running on port 5000.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const sections = [
    { key: 'overview', label: '🔬 Overview' },
    { key: 'comparison', label: '📋 Comparison Table' },
    { key: 'models', label: '🧪 Individual GANs' },
  ];

  return (
    <div className="glab-page animate-fade-in">
      {/* Hero Banner */}
      <div className="glab-hero">
        <div className="glab-hero-bg" />
        <div className="container glab-hero-content">
          <div className="glab-hero-badge">🧬 Generative AI</div>
          <h1 className="glab-hero-title">GAN Laboratory</h1>
          <p className="glab-hero-sub">
            Three dedicated Generative Adversarial Networks — one per classifier — trained to
            synthesize fruit images and augment each model's training distribution independently.
          </p>
          <div className="glab-hero-stats">
            {MODEL_KEYS.map((k) => {
              const m = allMetrics?.[k];
              const meta = MODEL_META[k];
              return (
                <div key={k} className="glab-hero-stat" style={{ borderColor: meta.color }}>
                  <span className="glab-hero-stat-icon">{meta.icon}</span>
                  <span className="glab-hero-stat-name" style={{ color: meta.color }}>{meta.ganLabel}</span>
                  {m && !m.error && (
                    <span className="glab-hero-stat-val">FID {m.fid_score?.toFixed(0)}</span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="container glab-body">
        {/* Section Nav */}
        <div className="glab-section-nav">
          {sections.map((s) => (
            <button
              key={s.key}
              className={`glab-section-btn ${activeSection === s.key ? 'glab-section-btn--active' : ''}`}
              onClick={() => setActiveSection(s.key)}
            >
              {s.label}
            </button>
          ))}
          <button className="glab-refresh-btn" onClick={fetchData} disabled={loading}>
            {loading ? '⟳ Loading…' : '⟳ Refresh'}
          </button>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="glab-loading">
            <div className="glab-spinner" />
            <p>Fetching GAN evaluation data from backend…</p>
          </div>
        )}

        {/* Error State */}
        {error && !loading && (
          <div className="glab-error">
            <span>⚠</span>
            <p>{error}</p>
            <button className="glab-retry-btn" onClick={fetchData}>Retry</button>
          </div>
        )}

        {/* Content */}
        {!loading && !error && (
          <>
            {/* Overview */}
            {activeSection === 'overview' && (
              <section className="glab-section">
                <h2 className="glab-section-title">Architecture Overview</h2>
                <div className="glab-overview-grid">
                  {MODEL_KEYS.map((k) => {
                    const meta = MODEL_META[k];
                    const m = allMetrics?.[k];
                    return (
                      <div key={k} className="glab-overview-card" style={{ '--oc-color': meta.color }}>
                        <div className="glab-oc-icon">{meta.icon}</div>
                        <h3 className="glab-oc-title" style={{ color: meta.color }}>{meta.ganLabel}</h3>
                        <p className="glab-oc-desc">{meta.description}</p>
                        <div className="glab-oc-stats">
                          <span>FID: <strong>{m?.fid_score?.toFixed(1) ?? '—'}</strong></span>
                          <span>Epochs: <strong>{m?.epochs_trained ?? '—'}</strong></span>
                          <span>Mode Collapse: <strong>{m?.mode_collapse_detected === false ? 'None ✓' : 'Detected ⚠'}</strong></span>
                        </div>
                        <p className="glab-oc-note">Paired with <em>{meta.label}</em></p>
                      </div>
                    );
                  })}
                </div>

                <div className="glab-methodology card">
                  <h3>📐 Methodology</h3>
                  <ul>
                    <li><strong>Independent Training:</strong> Each GAN was trained with its own seed, optimizer state, and latent noise distribution — no shared weights.</li>
                    <li><strong>Architecture Matching:</strong> Generator complexity is proportional to the paired classifier's representational depth.</li>
                    <li><strong>Leakage-Safe Evaluation:</strong> Synthetic samples were never used in baseline training; GAN augmentation is evaluated as a separate experimental condition.</li>
                    <li><strong>FID / KID:</strong> Measured against the real test split using InceptionV3 feature statistics.</li>
                    <li><strong>Mode Collapse Detection:</strong> Triggered when mean perceptual diversity drops below 10 bits across any class.</li>
                  </ul>
                </div>
              </section>
            )}

            {/* Comparison Table */}
            {activeSection === 'comparison' && (
              <section className="glab-section">
                <h2 className="glab-section-title">Cross-GAN Comparison</h2>
                <p className="glab-section-desc">
                  Side-by-side evaluation metrics for all three dedicated GANs. Lower FID/KID indicates higher visual fidelity. Higher diversity prevents mode collapse.
                </p>
                <SummaryTable summary={summary} />

                {/* Mini metric bars */}
                {summary?.length > 0 && (
                  <div className="glab-compare-bars">
                    <h3 className="glab-section-label">FID Score Comparison (lower is better)</h3>
                    {summary.map((row) => {
                      const meta = MODEL_META[row.model_key];
                      const maxFid = Math.max(...summary.map((r) => r.fid_score ?? 0));
                      return (
                        <div key={row.model_key} className="glab-bar-row">
                          <span className="glab-bar-label">
                            {meta.icon} {meta.label}
                          </span>
                          <div className="glab-bar-track">
                            <div
                              className="glab-bar-fill"
                              style={{ width: `${((row.fid_score ?? 0) / maxFid) * 100}%`, background: meta.color }}
                            />
                          </div>
                          <span className="glab-bar-val">{row.fid_score?.toFixed(1)}</span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </section>
            )}

            {/* Individual GAN Cards */}
            {activeSection === 'models' && (
              <section className="glab-section">
                <h2 className="glab-section-title">Individual GAN Reports</h2>
                <p className="glab-section-desc">
                  Explore metrics, synthetic sample grids, and loss curves for each dedicated GAN.
                </p>
                <div className="glab-cards-grid">
                  {MODEL_KEYS.map((k) => (
                    <GANCard key={k} modelKey={k} metrics={allMetrics?.[k]} />
                  ))}
                </div>
              </section>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default GANLaboratory;
