import React, { useState, useEffect, useCallback } from 'react';
import './index.css';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  ReferenceLine, Area, AreaChart
} from 'recharts';
import axios from 'axios';

const API = 'http://localhost:8000';

const MONTHS_FULL  = ['June','July','August','September','October','November'];
const MONTHS_SHORT = ['Jun','Jul','Aug','Sep','Oct','Nov'];
const CURRENT_YEAR = new Date().getFullYear();

const EXP_LABELS = {
  exp1:'Exp1 — Tabular (XGBoost)',
  exp2:'Exp2 — Neural (no images)',
  exp3:'Exp3 — Neural + Images',
  exp4:'Exp4 — PINN',
  exp5:'Exp5 — PINN + Stress',
  exp6:'Exp6 — Ablation',
};
const EXP_COLORS = ['#4ade80','#34d399','#22c55e','#86efac','#bbf7d0','#6ee7b7'];

const THEME_KEY = 'agropinn_theme';

// ─── Helpers ────────────────────────────────────────────────────────────────

const confClass   = (l) => ({ High:'high', Medium:'medium', Low:'low' }[l] || 'low');
const stressBadge = (l) => {
  if (l === 'Low')      return 'badge-green';
  if (l === 'Moderate') return 'badge-amber';
  if (l === 'High')     return 'badge-red';
  return 'badge-red';
};
const r2Class = (v) => {
  if (v == null) return '';
  if (v >= 0.65) return 'r2-good';
  if (v >= 0.50) return 'r2-mid';
  return 'r2-low';
};

function ThemeToggle({ theme, onToggle }) {
  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={onToggle}
      aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
      title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
    >
      <span className="theme-toggle-icon">{theme === 'dark' ? '☀️' : '🌙'}</span>
      <span className="theme-toggle-text">{theme === 'dark' ? 'Light' : 'Dark'}</span>
    </button>
  );
}

// ─── SVG Arc Gauge ──────────────────────────────────────────────────────────

function ArcGauge({ value, level }) {
  const pct    = Math.min(1, Math.max(0, value));
  const radius = 70;
  const cx = 90, cy = 85;
  const startAngle = Math.PI;
  const sweepAngle = Math.PI;
  const angle      = startAngle + sweepAngle * pct;

  const arcX = (a) => cx + radius * Math.cos(a);
  const arcY = (a) => cy + radius * Math.sin(a);

  const trackD = `M ${arcX(startAngle)} ${arcY(startAngle)}
    A ${radius} ${radius} 0 0 1 ${arcX(startAngle + sweepAngle)} ${arcY(startAngle + sweepAngle)}`;
  const fillD  = pct > 0
    ? `M ${arcX(startAngle)} ${arcY(startAngle)}
       A ${radius} ${radius} 0 ${pct > 0.5 ? 1 : 0} 1 ${arcX(angle)} ${arcY(angle)}`
    : '';

  const colors = { Low:'#4ade80', Moderate:'#fbbf24', High:'#fb923c', Severe:'#ef4444' };
  const color  = colors[level] || 'var(--text-soft)';

  const nx = arcX(angle), ny = arcY(angle);

  return (
    <svg width="180" height="100" viewBox="0 0 180 100" className="gauge-svg">
      <path d={trackD} fill="none" stroke="var(--gauge-track)" strokeWidth="12" strokeLinecap="round"/>
      {fillD && (
        <path d={fillD} fill="none" stroke={color} strokeWidth="12" strokeLinecap="round"
              style={{ filter:`drop-shadow(0 0 6px ${color}80)` }}/>
      )}
      {/* needle dot */}
      <circle cx={nx} cy={ny} r="6" fill={color} style={{ filter:`drop-shadow(0 0 4px ${color})` }}/>
      {/* zone labels */}
      <text x="22"  y="98" textAnchor="middle" fontSize="9" fill="rgba(34,197,94,0.7)">Low</text>
      <text x="90"  y="22" textAnchor="middle" fontSize="9" fill="rgba(251,191,36,0.7)">Mod</text>
      <text x="158" y="98" textAnchor="middle" fontSize="9" fill="rgba(239,68,68,0.7)">High</text>
      {/* center value */}
      <text x="90" y="80" textAnchor="middle" className="gauge-label-num" fontFamily="JetBrains Mono,monospace"
        fontSize="22" fontWeight="700" fill="var(--text)">
        {(pct * 100).toFixed(0)}%
      </text>
      <text x="90" y="97" textAnchor="middle" className="gauge-label-txt"
            fontSize="10" fill="var(--text-soft)" fontFamily="Inter,sans-serif">
        {level || '—'}
      </text>
    </svg>
  );
}

// ─── NDVI Custom Tooltip ─────────────────────────────────────────────────────

const NDVITooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background:'var(--tooltip-bg)', border:'1px solid var(--tooltip-border)',
                  borderRadius:8, padding:'8px 12px', fontSize:'0.82rem' }}>
      <div style={{ color:'var(--text-soft)', marginBottom:2 }}>{label}</div>
      <div style={{ color:'#4ade80', fontFamily:'JetBrains Mono,monospace', fontWeight:600 }}>
        NDVI: {payload[0].value.toFixed(3)}
      </div>
    </div>
  );
};

// ─── Loading Steps ───────────────────────────────────────────────────────────

const STEPS = [
  { id: 1, icon: '📡', text: (d) => `Fetching satellite imagery for ${d}…` },
  { id: 2, icon: '🌦️', text: (d, y) => `Analyzing weather patterns for ${y} season…` },
  { id: 3, icon: '🧪', text: () => 'Looking up soil conditions…' },
  { id: 4, icon: '🧠', text: () => 'Running physics-informed prediction model…' },
];

function LoadingOverlay({ district, year, currentStep }) {
  return (
    <div className="card fade-up" style={{ marginTop: 20 }}>
      <div className="card-title"><span className="icon">⚡</span> Processing</div>
      <div className="loading-steps">
        {STEPS.map((s) => {
          const done   = currentStep > s.id;
          const active = currentStep === s.id;
          return (
            <div key={s.id} className={`loading-step${active ? ' active' : done ? ' done' : ''}`}>
              {active
                ? <div className="spin" />
                : <span style={{ width:18, textAlign:'center', fontSize:'1rem' }}>
                    {done ? '✅' : '○'}
                  </span>
              }
              <span>{s.icon}&nbsp; {s.text(district, year)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── PAGE: Predict ───────────────────────────────────────────────────────────

function PredictPage() {
  const [districts,      setDistricts]      = useState([]);
  const [availableCrops, setAvailableCrops] = useState([]);

  const [crop,     setCrop]     = useState('Rice');
  const [district, setDistrict] = useState('');
  const [year,     setYear]     = useState(2022);

  const [loadingStep, setLoadingStep] = useState(0);
  const [loading,     setLoading]     = useState(false);
  const [result,      setResult]      = useState(null);
  const [error,       setError]       = useState('');

  // Fetch all districts on mount
  useEffect(() => {
    axios.get(`${API}/api/districts`).then(r => {
      const dists = r.data.districts || [];
      setDistricts(dists);
      if (dists.length) {
        const first = dists[0];
        setDistrict(first.name);
        setAvailableCrops(first.available_crops);
        if (first.available_crops.length) setCrop(first.available_crops[0]);
      }
    }).catch(() => setError('Could not load district list. Is the backend running?'));
  }, []);

  // When district changes, reset available crops
  const handleDistrictChange = (name) => {
    const d = districts.find(x => x.name === name);
    setDistrict(name);
    const crops = d?.available_crops || ['Rice','Jowar','Bajra','Soyabean','Cotton(lint)'];
    setAvailableCrops(crops);
    if (!crops.includes(crop)) setCrop(crops[0]);
    setResult(null); setError('');
  };

  // Year hint
  const yearHint = () => {
    if (year < 2000) return { cls:'warn', msg:'⚠ Satellite data not available before 2000' };
    if (year > CURRENT_YEAR) return { cls:'info', msg:'ℹ Prediction is for upcoming season' };
    return { cls:'ok', msg:'✓ Historical + satellite data available' };
  };
  const hint = yearHint();

  // Animated loading steps
  const runLoadingAnimation = useCallback(async () => {
    for (let s = 1; s <= 4; s++) {
      setLoadingStep(s);
      await new Promise(r => setTimeout(r, s === 4 ? 800 : 600));
    }
  }, []);

  const handlePredict = async () => {
    if (!district) return;
    setLoading(true); setError(''); setResult(null); setLoadingStep(0);

    const animPromise = runLoadingAnimation();

    try {
      const token = localStorage.getItem('agropinn_token') || '';
      const r = await axios.post(
        `${API}/api/predict`,
        { crop, district, year },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      await animPromise;
      setResult(r.data);
      setLoadingStep(0);
    } catch(e) {
      await animPromise;
      const detail = e.response?.data?.detail || 'Prediction failed. Is the backend running?';
      setError(detail);
      setLoadingStep(0);
    } finally {
      setLoading(false);
    }
  };

  const ndviChartData = result
    ? result.ndvi_profile.values.map((v,i) => ({
        month: MONTHS_SHORT[i],
        ndvi: +v.toFixed(4),
      }))
    : [];

  return (
    <div className="page">
      {/* Hero */}
      <div className="hero">
        <div className="hero-pill">🌾 Physics-Informed Multimodal Model</div>
        <h1>Maharashtra Crop Yield AI</h1>
        <p>Select your crop, district and year — we handle satellite imagery, weather &amp; soil automatically.</p>
      </div>

      {/* Input form */}
      <div className="card fade-up" style={{ marginBottom: 20 }}>
        <div className="card-title"><span className="icon">⚙️</span> Prediction Inputs</div>
        <div className="form-row">
          <div className="field">
            <label>District</label>
            <select value={district} onChange={e => handleDistrictChange(e.target.value)} disabled={loading}>
              {districts.map(d => <option key={d.name} value={d.name}>{d.name}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Crop</label>
            <select value={crop} onChange={e => { setCrop(e.target.value); setResult(null); }} disabled={loading}>
              {availableCrops.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Year</label>
            <input type="number" min="2000" max={CURRENT_YEAR + 2}
              value={year}
              onChange={e => { setYear(parseInt(e.target.value)||2022); setResult(null); }}
              disabled={loading} />
            <div className={`year-hint ${hint.cls}`}>{hint.msg}</div>
          </div>
        </div>
        <button className="btn-predict" onClick={handlePredict}
                disabled={loading || !district || year < 2000}>
          {loading ? <><div className="spin"/>Predicting…</> : '🔍 Predict Yield'}
        </button>
      </div>

      {/* Loading overlay */}
      {loading && <LoadingOverlay district={district} year={year} currentStep={loadingStep} />}

      {/* Error */}
      {error && !loading && (
        <div className="err-box fade-up" style={{ marginTop: 20 }}>
          <span>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <div className="fade-up">
          {/* Top 3 cards */}
          <div className="results-top" style={{ marginTop: 20 }}>

            {/* Yield Card */}
            <div className="card">
              <div className="card-title"><span className="icon">🌾</span> Predicted Yield</div>
              <div className="yield-value">{result.prediction.predicted_yield}</div>
              <div className="yield-unit">{result.prediction.yield_unit}</div>

              {/* Yield range bar */}
              <div style={{ marginBottom: 6, fontSize:'0.72rem', color:'var(--text-soft)', letterSpacing:'0.04em' }}>
                PREDICTED RANGE
              </div>
              <YieldRangeBar
                low={result.prediction.yield_range.low}
                high={result.prediction.yield_range.high}
                value={result.prediction.predicted_yield}
              />

              <div style={{ display:'flex', gap:8, marginTop: 14, flexWrap:'wrap' }}>
                <span className={`badge ${stressBadge(result.stress.level)}`}>{result.stress.level} Stress</span>
                <span className="badge badge-blue">🤖 {result.metadata.model_version}</span>
              </div>
            </div>

            {/* Stress Gauge */}
            <div className="card" style={{ textAlign:'center' }}>
              <div className="card-title"><span className="icon">🌡️</span> Stress Index</div>
              <ArcGauge value={result.stress.overall_index} level={result.stress.level} />
              <div className="divider"/>
              <div className="stress-bars">
                <StressBar label="Thermal" value={result.stress.thermal_stress} type="thermal" />
                <StressBar label="Water"   value={result.stress.water_stress}   type="water" />
              </div>
            </div>

            {/* Confidence */}
            <div className="card">
              <div className="card-title"><span className="icon">🎯</span> Confidence</div>
              <div className={`conf-score ${confClass(result.confidence.level)}`}>
                {Math.round(result.confidence.score * 100)}%
              </div>
              <div className={`conf-level badge ${
                result.confidence.level === 'High' ? 'badge-green' :
                result.confidence.level === 'Medium' ? 'badge-amber' : 'badge-red'}`}>
                {result.confidence.level}
              </div>
              <div className="conf-factors">
                {result.confidence.factors.map((f, i) => (
                  <div className="conf-factor" key={i}>
                    <div className="conf-dot" />
                    <span>{f}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Bottom: NDVI chart + Weather */}
          <div className="results-bottom">
            {/* NDVI chart */}
            <div className="card">
              <div className="card-title">
                <span className="icon">📈</span> NDVI Seasonal Profile
                &nbsp;
                <span className="badge badge-green" style={{ marginLeft:'auto' }}>
                  {result.ndvi_profile.health_status}
                </span>
                &nbsp;&nbsp;
                <span style={{ fontSize:'0.72rem', color:'var(--text-dim)', fontWeight:400 }}>
                  Peak: {result.ndvi_profile.peak_month}
                </span>
              </div>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={ndviChartData} margin={{ top:8, right:8, bottom:0, left:-20 }}>
                    <defs>
                      <linearGradient id="ndviGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%"  stopColor="#22c55e" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#22c55e" stopOpacity={0.02}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
                    {/* Healthy range band */}
                    <ReferenceLine y={0.7} stroke="rgba(34,197,94,0.2)" strokeDasharray="4 4" label={{ value:'Healthy', fill:'rgba(34,197,94,0.4)', fontSize:9, position:'right' }}/>
                    <ReferenceLine y={0.4} stroke="rgba(34,197,94,0.2)" strokeDasharray="4 4" />
                    <XAxis dataKey="month" tick={{ fontSize:11, fill:'var(--chart-tick)' }} axisLine={false} tickLine={false}/>
                    <YAxis domain={[0,1]} tick={{ fontSize:10, fill:'var(--chart-tick)' }} axisLine={false} tickLine={false}/>
                    <Tooltip content={<NDVITooltip />} />
                    <Area type="monotone" dataKey="ndvi"
                      stroke="#22c55e" strokeWidth={2.5}
                      fill="url(#ndviGrad)"
                      dot={{ r:4, fill:'#22c55e', stroke:'var(--surface-contrast)', strokeWidth:2 }}
                      activeDot={{ r:6, fill:'#4ade80' }}/>
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="stress-desc" style={{ marginTop: 8 }}>
                ℹ {result.stress.description}
              </div>
            </div>

            {/* Weather */}
            <div className="card">
              <div className="card-title"><span className="icon">🌦️</span> Weather Summary</div>
              <div className="wx-stats">
                <div className="wx-stat">
                  <div className="wx-stat-val">{result.weather_summary.avg_temperature}°C</div>
                  <div className="wx-stat-lbl">Avg Temp</div>
                </div>
                <div className="wx-stat">
                  <div className="wx-stat-val">{result.weather_summary.max_temperature}°C</div>
                  <div className="wx-stat-lbl">Max Temp</div>
                </div>
                <div className="wx-stat">
                  <div className="wx-stat-val">{result.weather_summary.total_rainfall}</div>
                  <div className="wx-stat-lbl">Rainfall (mm)</div>
                </div>
              </div>
              {result.weather_summary.dry_weeks > 0 && (
                <div style={{ marginBottom:10 }}>
                  <span className="badge badge-amber">⚠ {result.weather_summary.dry_weeks} dry weeks</span>
                </div>
              )}
              <div className="wx-desc">{result.weather_summary.description}</div>
            </div>
          </div>

          {/* Data sources bar */}
          <div className="sources-bar">
            <span className="sources-bar-label">Data Sources:</span>
            {result.metadata.data_sources.map(s => <span key={s} className="source-chip">{s}</span>)}
            <span style={{ marginLeft:'auto', fontSize:'0.72rem', color:'var(--text-dim)' }}>
              {result.metadata.processing_time_ms}ms
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Sub-components ──────────────────────────────────────────────────────────

function YieldRangeBar({ low, high, value }) {
  const range    = high - low || 1;
  const dotPct   = ((value - low) / range) * 100;
  const leftPct  = 5;
  const rightPct = 5;
  return (
    <div style={{ marginBottom: 8 }}>
      <div className="yield-range-bar">
        <div className="yield-range-fill"
             style={{ left:`${leftPct}%`, width:`${100 - leftPct - rightPct}%` }}/>
        <div className="yield-range-dot"
             style={{ left:`${leftPct + dotPct * (100 - leftPct - rightPct) / 100}%` }}/>
      </div>
      <div className="yield-range-labels">
        <span>{low}</span>
        <span style={{ color:'var(--green-400)', fontWeight:600 }}>▲ {value}</span>
        <span>{high}</span>
      </div>
    </div>
  );
}

function StressBar({ label, value, type }) {
  const pct = Math.round(value * 100);
  return (
    <div className="stress-bar-row">
      <div className="stress-bar-header">
        <span className="label">{label} Stress</span>
        <span className="pct">{pct}%</span>
      </div>
      <div className="stress-bar-track">
        <div className={`stress-bar-fill ${type}`} style={{ width:`${pct}%` }}/>
      </div>
    </div>
  );
}

// ─── PAGE: Results ───────────────────────────────────────────────────────────

function ResultsPage() {
  const [rows,    setRows]    = useState([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState('');

  useEffect(() => {
    axios.get(`${API}/api/results`).then(r => {
      setRows(r.data || []);
      setLoading(false);
    }).catch(e => {
      setError(e.response?.data?.detail || 'Could not load results.');
      setLoading(false);
    });
  }, []);

  const overallRows = rows.filter(r => r.split === 'oof' && r.crop === 'overall');
  const expOrder    = ['exp5','exp4','exp3','exp2','exp1'];
  const bestExp     = expOrder.find(e => overallRows.some(r => r.experiment === e));
  const perCropRows = rows.filter(r => r.split === 'oof' && r.experiment === bestExp && r.crop !== 'overall');

  const barData = overallRows.map(r => ({
    name: EXP_LABELS[r.experiment] || r.experiment,
    R2: parseFloat((r.r2 || 0).toFixed(3)),
    exp: r.experiment,
  }));

  if (loading) return (
    <div className="page" style={{ textAlign:'center', paddingTop:60 }}>
      <div className="spin" style={{ margin:'0 auto 16px', width:28, height:28, borderWidth:3 }}/>
      <p style={{ color:'var(--text-soft)' }}>Loading experiment results…</p>
    </div>
  );

  if (error) return (
    <div className="page">
      <div className="err-box"><span>⚠️</span><span>{error}</span></div>
    </div>
  );

  return (
    <div className="page fade-up">
      <div className="section-title">📊 Experiment Results</div>
      <div className="section-sub">
        Out-of-fold R² from 5-fold cross-validation across all experiments.
      </div>

      {barData.length > 0 && (
        <div className="card" style={{ marginBottom: 16 }}>
          <div className="card-title"><span className="icon">📈</span> Overall OOF R² by Experiment</div>
          <div style={{ height: 220 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} margin={{ top:4, right:8, bottom:24, left:-20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
                <XAxis dataKey="name" tick={{ fontSize:9, fill:'var(--chart-tick)' }}
                       angle={-12} textAnchor="end" interval={0} axisLine={false} tickLine={false}/>
                <YAxis domain={[0,1]} tick={{ fontSize:10, fill:'var(--chart-tick)' }} axisLine={false} tickLine={false}/>
                <Tooltip
                  contentStyle={{ background:'var(--tooltip-bg)', border:'1px solid var(--tooltip-border)', borderRadius:8 }}
                  labelStyle={{ color:'var(--text-soft)', fontSize:'0.8rem' }}
                  formatter={(v) => [v.toFixed(4), 'OOF R²']}
                />
                <Bar dataKey="R2" radius={[4,4,0,0]}>
                  {barData.map((_, i) => (
                    <Cell key={i} fill={EXP_COLORS[i % EXP_COLORS.length]}/>
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div className="card" style={{ marginBottom: 16, overflowX:'auto' }}>
        <div className="card-title"><span className="icon">📋</span> Summary Table</div>
        <table className="results-table">
          <thead>
            <tr>
              <th>Experiment</th><th>OOF R²</th><th>MAE</th><th>RMSE</th>
            </tr>
          </thead>
          <tbody>
            {overallRows.map((r,i) => (
              <tr key={i}>
                <td className="model-name">{EXP_LABELS[r.experiment] || r.model}</td>
                <td className={r2Class(r.r2)}>{r.r2?.toFixed(4)}</td>
                <td>{r.mae?.toFixed(4)}</td>
                <td>{r.rmse?.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {perCropRows.length > 0 && (
        <div className="card">
          <div className="card-title">
            <span className="icon">🌱</span>
            Per-Crop OOF R² — {EXP_LABELS[bestExp] || bestExp}
          </div>
          <div style={{ height:180 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={perCropRows.map(r => ({ crop: r.crop, R2: parseFloat((r.r2||0).toFixed(3)) }))}
                margin={{ top:4, right:8, bottom:0, left:-20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)"/>
                <XAxis dataKey="crop" tick={{ fontSize:11, fill:'var(--chart-tick)' }} axisLine={false} tickLine={false}/>
                <YAxis domain={[0,1]} tick={{ fontSize:10, fill:'var(--chart-tick)' }} axisLine={false} tickLine={false}/>
                <Tooltip
                  contentStyle={{ background:'var(--tooltip-bg)', border:'1px solid var(--tooltip-border)', borderRadius:8 }}
                  formatter={(v) => [v.toFixed(4), 'OOF R²']}
                />
                <Bar dataKey="R2" radius={[4,4,0,0]}>
                  {perCropRows.map((_,i) => <Cell key={i} fill={EXP_COLORS[i % EXP_COLORS.length]}/>)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── PAGE: Validate ──────────────────────────────────────────────────────────

const ValTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background:'var(--tooltip-bg)', border:'1px solid var(--tooltip-border)',
                  borderRadius:8, padding:'8px 12px', fontSize:'0.82rem' }}>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, fontFamily:'JetBrains Mono,monospace' }}>
          {p.name}: {p.value}
        </div>
      ))}
    </div>
  );
};

function ValidatePage() {
  const [districts,      setDistricts]      = useState([]);
  const [availableCrops, setAvailableCrops] = useState([]);
  const [district,       setDistrict]       = useState('');
  const [crop,           setCrop]           = useState('Rice');
  const [years,          setYears]          = useState(8);
  const [loading,        setLoading]        = useState(false);
  const [result,         setResult]         = useState(null);
  const [error,          setError]          = useState('');

  useEffect(() => {
    axios.get(`${API}/api/districts`).then(r => {
      const dists = r.data.districts || [];
      setDistricts(dists);
      if (dists.length) {
        setDistrict(dists[0].name);
        const crops = dists[0].available_crops || ['Rice'];
        setAvailableCrops(crops);
        setCrop(crops[0]);
      }
    }).catch(() => setError('Could not load districts.'));
  }, []);

  const handleDistrictChange = (name) => {
    const d = districts.find(x => x.name === name);
    setDistrict(name);
    const crops = d?.available_crops || ['Rice'];
    setAvailableCrops(crops);
    if (!crops.includes(crop)) setCrop(crops[0]);
    setResult(null); setError('');
  };

  const handleValidate = async () => {
    if (!district) return;
    setLoading(true); setError(''); setResult(null);
    try {
      const r = await axios.get(`${API}/api/validate`, {
        params: { district, crop, years }
      });
      setResult(r.data);
    } catch (e) {
      setError(e.response?.data?.detail || 'Validation failed.');
    } finally {
      setLoading(false);
    }
  };

  const chartData = result?.rows?.map(r => ({
    year:      r.year,
    Actual:    r.actual,
    Predicted: r.predicted,
  })) || [];

  const s = result?.summary;

  return (
    <div className="page fade-up">
      <div className="hero" style={{ marginBottom:24 }}>
        <div className="hero-pill">🔬 Ground Truth vs Model</div>
        <h1>Prediction Validation</h1>
        <p>Compare model output against real recorded yields from the dataset.</p>
      </div>

      {/* Controls */}
      <div className="card" style={{ marginBottom:20 }}>
        <div className="card-title"><span className="icon">⚙️</span> Validation Settings</div>
        <div className="form-row">
          <div className="field">
            <label>District</label>
            <select value={district} onChange={e => handleDistrictChange(e.target.value)} disabled={loading}>
              {districts.map(d => <option key={d.name} value={d.name}>{d.name}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Crop</label>
            <select value={crop} onChange={e => { setCrop(e.target.value); setResult(null); }} disabled={loading}>
              {availableCrops.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Years</label>
            <input type="number" min={1} max={23} value={years}
              onChange={e => setYears(parseInt(e.target.value)||5)}
              disabled={loading} />
          </div>
        </div>
        <button className="btn-predict" onClick={handleValidate} disabled={loading || !district}>
          {loading ? <><div className="spin"/>Validating…</> : '🔬 Run Validation'}
        </button>
      </div>

      {error && !loading && (
        <div className="err-box fade-up"><span>⚠️</span><span>{error}</span></div>
      )}

      {result && !loading && (
        <div className="fade-up">

          {/* Summary metric cards */}
          {s && (
            <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:12, marginBottom:16 }}>
              {[
                { label:'OOF R²',     val: s.r2,                    good: s.r2 >= 0.65,    unit:'' },
                { label:'MAE',        val: s.mae,                   good: s.mae < 0.5,     unit: result.yield_unit.split('/')[0] },
                { label:'RMSE',       val: s.rmse,                  good: s.rmse < 0.7,    unit: result.yield_unit.split('/')[0] },
                { label:'±15% Acc',   val: s.within_15pct_pct+'%', good: s.within_15pct_pct >= 60, unit:'' },
              ].map(m => (
                <div className="card" key={m.label} style={{ textAlign:'center', padding:16 }}>
                  <div style={{ fontSize:'0.7rem', textTransform:'uppercase', letterSpacing:'0.06em',
                                color:'var(--text-soft)', marginBottom:6 }}>{m.label}</div>
                  <div style={{ fontFamily:'JetBrains Mono,monospace', fontSize:'1.6rem', fontWeight:700,
                                color: m.good ? 'var(--green-400)' : 'var(--amber-400)' }}>
                    {m.val}
                  </div>
                  {m.unit && <div style={{ fontSize:'0.7rem', color:'var(--text-dim)' }}>{m.unit}</div>}
                </div>
              ))}
            </div>
          )}

          {/* Side-by-side bar chart */}
          <div className="card" style={{ marginBottom:16 }}>
            <div className="card-title">
              <span className="icon">📊</span>
              Actual vs Predicted — {result.crop} in {result.district}
              <span style={{ marginLeft:'auto', fontSize:'0.72rem', color:'var(--text-dim)' }}>
                {result.yield_unit}
              </span>
            </div>
            <div style={{ height: 220 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top:4, right:8, bottom:0, left:-20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)"/>
                  <XAxis dataKey="year" tick={{ fontSize:11, fill:'var(--chart-tick)' }} axisLine={false} tickLine={false}/>
                  <YAxis tick={{ fontSize:10, fill:'var(--chart-tick)' }} axisLine={false} tickLine={false}/>
                  <Tooltip content={<ValTooltip />}/>
                  <Bar dataKey="Actual"    fill="var(--chart-tick)" radius={[3,3,0,0]} opacity={0.75}/>
                  <Bar dataKey="Predicted" fill="#22c55e" radius={[3,3,0,0]} opacity={0.85}/>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div style={{ display:'flex', gap:16, justifyContent:'center', marginTop:8 }}>
              {[['Actual','var(--chart-tick)'],['Predicted','#22c55e']].map(([l, c]) => (
                <div key={l} style={{ display:'flex', alignItems:'center', gap:5, fontSize:'0.78rem', color:'var(--text-soft)' }}>
                  <div style={{ width:10, height:10, borderRadius:2, background:c }}/> {l}
                </div>
              ))}
            </div>
          </div>

          {/* Per-row table */}
          <div className="card" style={{ overflowX:'auto' }}>
            <div className="card-title"><span className="icon">📋</span> Year-by-Year Breakdown</div>
            <table className="results-table">
              <thead>
                <tr>
                  <th>Year</th>
                  <th>Actual</th>
                  <th>Predicted</th>
                  <th>Error</th>
                  <th>% Error</th>
                  <th>Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {result.rows.map(r => {
                  const good = r.within_15pct;
                  return (
                    <tr key={r.year}>
                      <td style={{ color:'var(--text)'}}>{r.year}</td>
                      <td>{r.actual} <span style={{ fontSize:'0.7rem', color:'var(--text-dim)' }}>{result.yield_unit}</span></td>
                      <td style={{ color: r.predicted !== null ? 'var(--green-400)' : 'var(--red-500)' }}>
                        {r.predicted ?? '—'}
                      </td>
                      <td style={{ color: r.error === null ? '' : (Math.abs(r.error) < 0.3 ? 'var(--green-400)' : 'var(--amber-400)') }}>
                        {r.error !== null ? (r.error > 0 ? `+${r.error}` : r.error) : '—'}
                      </td>
                      <td style={{ color: r.pct_error === null ? '' : (r.pct_error < 15 ? 'var(--green-400)' : r.pct_error < 30 ? 'var(--amber-400)' : '#fca5a5') }}>
                        {r.pct_error !== null ? `${r.pct_error}%` : '—'}
                      </td>
                      <td>
                        <span className={`badge ${good ? 'badge-green' : 'badge-amber'}`}>
                          {good ? '✓ ≤15%' : '≈ >15%'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Prof-ready note */}
          <div className="sources-bar" style={{ marginTop:12 }}>
            <span className="sources-bar-label">Validation Method:</span>
            <span className="source-chip">Historical data from Maharashtra Agri Census (2000–2022)</span>
            <span className="source-chip">Out-of-fold — model never trained on these specific rows</span>
          </div>
        </div>
      )}
    </div>
  );
}

function HistoryPage() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const token = localStorage.getItem('agropinn_token') || '';
    axios.get(`${API}/api/history`, {
      headers: { Authorization: `Bearer ${token}` },
    }).then((r) => {
      setItems(r.data?.items || []);
      setLoading(false);
    }).catch((e) => {
      setError(e.response?.data?.detail || 'Could not load prediction history.');
      setLoading(false);
    });
  }, []);

  return (
    <div className="page fade-up">
      <div className="hero" style={{ marginBottom:24 }}>
        <div className="hero-pill">🧾 Farmer Prediction Log</div>
        <h1>Prediction History</h1>
        <p>Your previous yield predictions with stress indicators.</p>
      </div>

      {loading && (
        <div className="card" style={{ textAlign:'center' }}>
          <div className="spin" style={{ margin:'0 auto 16px', width:28, height:28, borderWidth:3 }}/>
          <p style={{ color:'var(--text-soft)' }}>Loading history…</p>
        </div>
      )}

      {error && !loading && (
        <div className="err-box fade-up"><span>⚠️</span><span>{error}</span></div>
      )}

      {!loading && !error && items.length === 0 && (
        <div className="card" style={{ textAlign:'center' }}>
          <div className="card-title" style={{ justifyContent:'center' }}>
            <span className="icon">🌱</span> No History Yet
          </div>
          <p style={{ color:'var(--text-soft)' }}>
            Run your first prediction from the Predict tab.
          </p>
        </div>
      )}

      {!loading && !error && items.length > 0 && (
        <div className="card" style={{ overflowX:'auto' }}>
          <div className="card-title"><span className="icon">📚</span> Recent Predictions</div>
          <table className="results-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>District</th>
                <th>Crop</th>
                <th>Year</th>
                <th>Predicted Yield</th>
                <th>Stress</th>
              </tr>
            </thead>
            <tbody>
              {items.map((it) => (
                <tr key={it.id}>
                  <td>{new Date(it.created_at).toLocaleString()}</td>
                  <td>{it.district}</td>
                  <td>{it.crop}</td>
                  <td>{it.year}</td>
                  <td style={{ color:'var(--green-400)', fontFamily:'JetBrains Mono,monospace' }}>
                    {it.predicted_yield.toFixed(3)}
                    <span style={{ color:'var(--text-dim)', fontSize:'0.72rem', marginLeft:6 }}>{it.yield_unit}</span>
                  </td>
                  <td>
                    <span className={`badge ${stressBadge(it.stress_level)}`}>
                      {it.stress_level} ({Math.round((it.stress_index || 0) * 100)}%)
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function AuthPage({ onAuthSuccess, theme, onToggleTheme }) {
  const [mode, setMode] = useState('login');
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const postAuth = async (action, payload) => {
    const paths = action === 'register'
      ? ['/api/auth/register', '/auth/register']
      : ['/api/auth/login', '/auth/login'];

    let lastError = null;
    for (const path of paths) {
      try {
        return await axios.post(`${API}${path}`, payload);
      } catch (e) {
        lastError = e;
        if (e?.response?.status !== 404) {
          throw e;
        }
      }
    }
    throw lastError;
  };

  const submit = async () => {
    if (!name.trim() || !password.trim()) {
      setError('Please enter both name and password.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const endpoint = mode === 'login' ? 'login' : 'register';
      const res = await postAuth(endpoint, {
        name: name.trim(),
        password,
      });
      onAuthSuccess(res.data.token, res.data.farmer);
    } catch (e) {
      const detail = e.response?.data?.detail;
      if (detail === 'Not Found') {
        setError('Auth endpoint not found. Please restart backend and try again.');
      } else {
        setError(detail || 'Authentication failed.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="landing-home">
      <nav className="landing-nav">
        <a href="#home" className="landing-brand" aria-label="AgroPINN Home">
          <span className="leaf">🌱</span>
          AgroPINN
        </a>
        <div className="landing-links">
          <a href="#home">Home</a>
          <a href="#what-is">About Us</a>
          <a href="#services">Services</a>
          <a href="#access" className="landing-cta">Login</a>
        </div>
        <ThemeToggle theme={theme} onToggle={onToggleTheme} />
      </nav>

      <section
        id="home"
        className="auth-viewport"
        style={{ backgroundImage: `linear-gradient(120deg, var(--auth-overlay-start), var(--auth-overlay-end)), url(${process.env.PUBLIC_URL}/home.jpg)` }}
      >
        <div className="auth-hero-layout fade-up">
          <div className="auth-hero-copy">
            <div className="hero-pill">🌾 Maharastra Based System</div>
            <h1>Better Crop Planning, in Simple Steps</h1>
            <p>
              AgroPINN helps you estimate crop yield using weather, satellite and soil data.
              You choose district, crop and year, then get a clear prediction in seconds.
            </p>
            <div className="auth-quick-points">
              <div className="auth-quick-point">Easy for first-time smartphone users</div>
              <div className="auth-quick-point">Clear stress and confidence insights</div>
              <div className="auth-quick-point">Built for Kharif farming conditions</div>
            </div>
          </div>

          <div className="auth-panel" id="access">
            <div className="auth-badge">Farmer Access</div>
            <h1>AgroPINN Farmer Portal</h1>
            <p>Login to continue, or register if you are a new farmer.</p>

            <div className="auth-switch">
              <button className={`auth-switch-btn ${mode === 'login' ? 'active' : ''}`} onClick={() => setMode('login')}>
                Login
              </button>
              <button className={`auth-switch-btn ${mode === 'register' ? 'active' : ''}`} onClick={() => setMode('register')}>
                Register if not
              </button>
            </div>

            <div className="auth-form">
              <div className="field">
                <label>Farmer Name</label>
                <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Enter your name" />
              </div>
              <div className="field">
                <label>Password</label>
                <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Enter password" />
              </div>
              {error && <div className="err-box" style={{ marginTop: 8 }}><span>⚠️</span><span>{error}</span></div>}
              <button className="btn-predict" onClick={submit} disabled={loading} style={{ marginTop: 14 }}>
                {loading ? <><div className="spin"/>Please wait…</> : (mode === 'login' ? 'Login' : 'Create Account')}
              </button>
            </div>
          </div>
        </div>
      </section>

      <section id="what-is" className="landing-section fade-up">
        <div className="landing-wrap">
          <div className="landing-head">
            <span className="hero-pill">About Us</span>
            <h2>AgroPINN</h2>
            <p>
              AgroPINN is a smart digital helper for farmers. It does not replace your farming knowledge,
              it supports your decisions with data. You simply choose your district, crop, and year.
              Then AgroPINN checks important factors like rainfall, temperature, crop greenness from satellite
              images, and soil condition. After combining all of this, it gives one clear output: expected yield,
              crop stress level, and confidence score.
            </p>
            <p>
              In practical terms, this means you can plan better before the season and during crop growth.
              You can understand whether conditions are favorable, whether stress is building, and whether
              your expected production may go up or down. This helps with planning irrigation, fertilizer use,
              labor, storage, and market timing. The goal is simple: reduce guesswork and help you take
              better, more confident decisions with easy-to-understand insights.
            </p>
          </div>
          <div className="simple-steps">
            <div className="simple-step">
              <div className="step-num">1</div>
              <h3>Choose your details</h3>
              <p>Select district, crop and year in a few taps.</p>
            </div>
            <div className="simple-step">
              <div className="step-num">2</div>
              <h3>Model analyzes conditions</h3>
              <p>Weather, satellite and soil data are checked together.</p>
            </div>
            <div className="simple-step">
              <div className="step-num">3</div>
              <h3>Get clear output</h3>
              <p>See predicted yield, stress level, and confidence score.</p>
            </div>
          </div>
        </div>
      </section>

      <section id="services" className="landing-section fade-up" style={{ paddingBottom: 84 }}>
        <div className="landing-wrap">
          <div className="landing-head">
            <span className="hero-pill">Our Services</span>
            <h2>Everything in one farmer-friendly dashboard</h2>
          </div>
          <div className="services-grid">
            <div className="service-tile">
              <div className="service-icon">🌿</div>
              <h3>Yield Prediction</h3>
              <p>District-wise crop yield forecast before harvest season.</p>
            </div>
            <div className="service-tile">
              <div className="service-icon">🌡️</div>
              <h3>Stress Monitoring</h3>
              <p>Thermal and water stress indicators with easy color-coded labels.</p>
            </div>
            <div className="service-tile">
              <div className="service-icon">📈</div>
              <h3>NDVI Crop Health</h3>
              <p>Month-by-month vegetation health profile from satellite signals.</p>
            </div>
            <div className="service-tile">
              <div className="service-icon">🔬</div>
              <h3>Validation Insights</h3>
              <p>Compare model predictions with historical outcomes for trust.</p>
            </div>
            <div className="service-tile">
              <div className="service-icon">🧾</div>
              <h3>Prediction History</h3>
              <p>Store past predictions and review trends over time.</p>
            </div>
            <div className="service-tile">
              <div className="service-icon">📱</div>
              <h3>Simple UI</h3>
              <p>Clean, readable interface designed for mobile and desktop use.</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}


export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem(THEME_KEY) || 'dark');
  const [token, setToken] = useState(() => localStorage.getItem('agropinn_token') || '');
  const [farmer, setFarmer] = useState(() => {
    const raw = localStorage.getItem('agropinn_farmer');
    if (!raw) return null;
    try { return JSON.parse(raw); } catch { return null; }
  });
  const [page, setPage] = useState('predict');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    document.documentElement.style.colorScheme = theme;
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((curr) => (curr === 'dark' ? 'light' : 'dark'));
  };

  const onAuthSuccess = (newToken, farmerInfo) => {
    localStorage.setItem('agropinn_token', newToken);
    localStorage.setItem('agropinn_farmer', JSON.stringify(farmerInfo));
    setToken(newToken);
    setFarmer(farmerInfo);
  };

  const logout = () => {
    localStorage.removeItem('agropinn_token');
    localStorage.removeItem('agropinn_farmer');
    setToken('');
    setFarmer(null);
  };

  if (!token) {
    return (
      <>
        <AuthPage onAuthSuccess={onAuthSuccess} theme={theme} onToggleTheme={toggleTheme} />
      </>
    );
  }

  const pages = [
    { id:'predict',  label:'🌿 Predict'  },
    { id:'validate', label:'🔬 Validate' },
    { id:'history',  label:'🧾 History'  },
    { id:'results',  label:'📊 Results'  },
  ];

  return (
    <>
      <nav className="nav">
        <div className="nav-brand">
          <span className="leaf">🌱</span>
          AgroPINN
        </div>
        <div className="nav-links">
          {pages.map(p => (
            <button key={p.id}
              className={`nav-link ${page === p.id ? 'active' : ''}`}
              onClick={() => setPage(p.id)}>
              {p.label}
            </button>
          ))}
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
          <button className="nav-link" onClick={logout}>Logout</button>
        </div>
      </nav>

      {page === 'predict'  && <PredictPage />}
      {page === 'validate' && <ValidatePage />}
      {page === 'history'  && <HistoryPage />}
      {page === 'results'  && <ResultsPage />}
      <footer className="footer">
        {farmer?.name ? `Farmer: ${farmer.name} · ` : ''}
        AgroPINN — Physics-Informed Multimodal Crop Yield Prediction &nbsp;·&nbsp;
        Maharashtra Kharif Season &nbsp;·&nbsp; 2024
      </footer>
    </>
  );
}
