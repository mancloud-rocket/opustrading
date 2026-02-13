// OPUS Trading Bot - ML Prediction Dashboard
// Loads chart_data.json and renders interactive charts

let DATA = null;

async function loadData() {
  // Method 1: Try global variable from chart_data.js (works with file://)
  if (typeof CHART_DATA !== 'undefined') {
    DATA = CHART_DATA;
    renderAll();
    return;
  }

  // Method 2: Try fetch (works with HTTP server)
  try {
    const resp = await fetch('./chart_data.json');
    DATA = await resp.json();
    renderAll();
  } catch (e) {
    document.getElementById('app').innerHTML =
      '<p style="color:#ef4444;text-align:center;padding:40px;">' +
      'Error loading data. Run:<br><code>cd python && python generate_chart_data.py</code></p>';
  }
}

function renderAll() {
  renderHeader();
  initMinuteSelector();
  initLiveMinuteSelector();
  renderLiveScatterChart('7');
  renderScatterChart('7');
  renderCalibrationChart();
  renderDistributionChart();
  renderPnlCurves();
  renderFeatureImportance();
  renderMinuteAccuracy();
  renderFooter();
}

// ---- Header stats ----
function renderHeader() {
  const s = DATA.model_stats;
  document.getElementById('stat-auc').textContent = s.cv_auc.toFixed(3);
  document.getElementById('stat-acc').textContent = s.cv_accuracy.toFixed(1) + '%';
  document.getElementById('stat-brier').textContent = s.cv_brier.toFixed(4);
  document.getElementById('stat-markets').textContent = s.n_markets;
  document.getElementById('stat-samples').textContent = s.n_samples.toLocaleString();
}

// ---- 1. Scatter: P(UP) vs Actual Outcome (with minute selector) ----
let scatterChart = null;
let currentScatterMinute = '7';

function getPointsForMinute(minute) {
  const byMin = DATA.market_predictions_by_minute;
  if (byMin && byMin[minute]) return byMin[minute];
  // Fallback: use default market_predictions
  return DATA.market_predictions || [];
}

function computeMinuteStats(pts) {
  if (!pts.length) return '';
  const correct = pts.filter(p => {
    const predUp = p.p_up > 0.5;
    return predUp === (p.actual === 1);
  }).length;
  const acc = (correct / pts.length * 100).toFixed(1);
  const avgConf = (pts.reduce((s, p) => s + Math.max(p.p_up, 1 - p.p_up), 0) / pts.length * 100).toFixed(1);
  return `${pts.length} markets | Acc: ${acc}% | Conf: ${avgConf}%`;
}

function renderScatterChart(minute) {
  if (!minute) minute = currentScatterMinute;
  currentScatterMinute = minute;

  const pts = getPointsForMinute(minute);

  // Update stats text
  const statsEl = document.getElementById('minuteStats');
  if (statsEl) statsEl.textContent = computeMinuteStats(pts);

  const upWon = pts.filter(p => p.actual === 1).map(p => ({
    x: p.p_up, y: p.up_price, r: Math.max(4, Math.abs(p.btc_ret) * 20)
  }));
  const downWon = pts.filter(p => p.actual === 0).map(p => ({
    x: p.p_up, y: p.up_price, r: Math.max(4, Math.abs(p.btc_ret) * 20)
  }));

  // Destroy previous chart if exists
  if (scatterChart) {
    scatterChart.destroy();
    scatterChart = null;
  }

  const ctx = document.getElementById('scatterChart').getContext('2d');
  scatterChart = new Chart(ctx, {
    type: 'bubble',
    data: {
      datasets: [
        {
          label: 'UP Won',
          data: upWon,
          backgroundColor: 'rgba(34, 197, 94, 0.5)',
          borderColor: 'rgba(34, 197, 94, 0.8)',
          borderWidth: 1,
        },
        {
          label: 'DOWN Won',
          data: downWon,
          backgroundColor: 'rgba(239, 68, 68, 0.5)',
          borderColor: 'rgba(239, 68, 68, 0.8)',
          borderWidth: 1,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, labels: { color: '#94a3b8', font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const currPts = getPointsForMinute(currentScatterMinute);
              const p = currPts.find(
                m => Math.abs(m.p_up - ctx.raw.x) < 0.001 && Math.abs(m.up_price - ctx.raw.y) < 0.001
              );
              if (!p) return '';
              return [
                `P(UP): ${(p.p_up * 100).toFixed(1)}%`,
                `UP Price: $${p.up_price.toFixed(2)}`,
                `BTC ret: ${p.btc_ret.toFixed(3)}%`,
                `Result: ${p.actual === 1 ? 'UP Won' : 'DOWN Won'}`
              ];
            }
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'P(UP) - Model Prediction', color: '#64748b' },
          min: 0, max: 1,
          grid: { color: 'rgba(148,163,184,0.08)' },
          ticks: { color: '#64748b' }
        },
        y: {
          title: { display: true, text: 'UP Token Price', color: '#64748b' },
          min: 0, max: 1,
          grid: { color: 'rgba(148,163,184,0.08)' },
          ticks: { color: '#64748b' }
        }
      },
      annotation: {
        annotations: {
          line1: { type: 'line', xMin: 0.5, xMax: 0.5, borderColor: '#475569', borderWidth: 1, borderDash: [5, 5] }
        }
      }
    }
  });
}

function initMinuteSelector() {
  const select = document.getElementById('minuteSelect');
  if (select) {
    select.addEventListener('change', (e) => {
      renderScatterChart(e.target.value);
    });
  }
}

// ---- 0. LIVE Scatter: Real P(UP) from bot ----
let liveScatterChart = null;
let currentLiveMinute = '7';

function getLivePointsForMinute(minute) {
  const byMin = DATA.live_predictions_by_minute;
  if (byMin && byMin[minute]) return byMin[minute];
  return [];
}

function computeLiveMinuteStats(pts) {
  if (!pts.length) return 'Sin datos live para este minuto';
  const correct = pts.filter(p => {
    const predUp = p.p_up > 0.5;
    return predUp === (p.actual === 1);
  }).length;
  const acc = (correct / pts.length * 100).toFixed(1);
  const avgConf = (pts.reduce((s, p) => s + p.confidence, 0) / pts.length * 100).toFixed(1);
  return `${pts.length} markets | Acc REAL: ${acc}% | Conf: ${avgConf}%`;
}

function renderLiveScatterChart(minute) {
  if (!minute) minute = currentLiveMinute;
  currentLiveMinute = minute;

  const pts = getLivePointsForMinute(minute);

  // Update stats
  const statsEl = document.getElementById('liveMinuteStats');
  if (statsEl) statsEl.textContent = computeLiveMinuteStats(pts);

  const upWon = pts.filter(p => p.actual === 1).map(p => ({
    x: p.p_up, y: p.up_price, r: Math.max(5, Math.abs(p.btc_ret) * 25)
  }));
  const downWon = pts.filter(p => p.actual === 0).map(p => ({
    x: p.p_up, y: p.up_price, r: Math.max(5, Math.abs(p.btc_ret) * 25)
  }));

  if (liveScatterChart) {
    liveScatterChart.destroy();
    liveScatterChart = null;
  }

  const canvas = document.getElementById('liveScatterChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  liveScatterChart = new Chart(ctx, {
    type: 'bubble',
    data: {
      datasets: [
        {
          label: 'UP Won',
          data: upWon,
          backgroundColor: 'rgba(34, 197, 94, 0.6)',
          borderColor: 'rgba(34, 197, 94, 0.9)',
          borderWidth: 2,
        },
        {
          label: 'DOWN Won',
          data: downWon,
          backgroundColor: 'rgba(239, 68, 68, 0.6)',
          borderColor: 'rgba(239, 68, 68, 0.9)',
          borderWidth: 2,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, labels: { color: '#94a3b8', font: { size: 12 } } },
        title: {
          display: !pts.length,
          text: 'Sin datos live con ML para este minuto. El bot necesita mas tiempo corriendo.',
          color: '#f97316',
          font: { size: 14 },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const currPts = getLivePointsForMinute(currentLiveMinute);
              const p = currPts.find(
                m => Math.abs(m.p_up - ctx.raw.x) < 0.002 && Math.abs(m.up_price - ctx.raw.y) < 0.002
              );
              if (!p) return '';
              return [
                `Live P(UP): ${(p.p_up * 100).toFixed(1)}%`,
                `Confidence: ${(p.confidence * 100).toFixed(1)}%`,
                `UP Price: $${p.up_price.toFixed(2)}`,
                `BTC ret: ${p.btc_ret.toFixed(3)}%`,
                `Result: ${p.actual === 1 ? 'UP Won' : 'DOWN Won'}`,
                `Ticks: ${p.n_ticks}`,
              ];
            }
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Live P(UP) - Real Bot Prediction', color: '#f97316', font: { weight: 'bold' } },
          min: 0, max: 1,
          grid: { color: 'rgba(148,163,184,0.08)' },
          ticks: { color: '#64748b' }
        },
        y: {
          title: { display: true, text: 'UP Token Price', color: '#64748b' },
          min: 0, max: 1,
          grid: { color: 'rgba(148,163,184,0.08)' },
          ticks: { color: '#64748b' }
        }
      }
    }
  });
}

function initLiveMinuteSelector() {
  const select = document.getElementById('liveMinuteSelect');
  if (select) {
    select.addEventListener('change', (e) => {
      renderLiveScatterChart(e.target.value);
    });
  }
}

// ---- 2. Calibration Curve ----
function renderCalibrationChart() {
  const ctx = document.getElementById('calibrationChart').getContext('2d');
  const cal = DATA.calibration;

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: cal.map(c => c.bin),
      datasets: [
        {
          label: 'Predicted P(UP)',
          data: cal.map(c => c.predicted),
          borderColor: '#818cf8',
          backgroundColor: 'rgba(129, 140, 248, 0.1)',
          fill: true,
          tension: 0.3,
          pointRadius: 5,
          pointBackgroundColor: '#818cf8',
        },
        {
          label: 'Actual Frequency',
          data: cal.map(c => c.actual),
          borderColor: '#22c55e',
          backgroundColor: 'rgba(34, 197, 94, 0.1)',
          fill: true,
          tension: 0.3,
          pointRadius: 5,
          pointBackgroundColor: '#22c55e',
        },
        {
          label: 'Perfect Calibration',
          data: cal.map(c => c.bin_mid),
          borderColor: '#475569',
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { size: 11 } } },
        tooltip: {
          callbacks: {
            afterLabel: (ctx) => {
              const c = cal[ctx.dataIndex];
              return c ? `(${c.count} samples)` : '';
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(148,163,184,0.08)' },
          ticks: { color: '#64748b' }
        },
        y: {
          min: 0, max: 1,
          title: { display: true, text: 'Probability', color: '#64748b' },
          grid: { color: 'rgba(148,163,184,0.08)' },
          ticks: { color: '#64748b' }
        }
      }
    }
  });
}

// ---- 3. Distribution of P(UP) ----
function renderDistributionChart() {
  const ctx = document.getElementById('distributionChart').getContext('2d');

  // Create histogram bins
  const bins = 20;
  const binWidth = 1.0 / bins;
  const upHist = new Array(bins).fill(0);
  const downHist = new Array(bins).fill(0);

  DATA.distribution_up_won.forEach(p => {
    const idx = Math.min(Math.floor(p / binWidth), bins - 1);
    upHist[idx]++;
  });
  DATA.distribution_down_won.forEach(p => {
    const idx = Math.min(Math.floor(p / binWidth), bins - 1);
    downHist[idx]++;
  });

  const labels = Array.from({ length: bins }, (_, i) => ((i + 0.5) * binWidth).toFixed(2));

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'UP Won',
          data: upHist,
          backgroundColor: 'rgba(34, 197, 94, 0.6)',
          borderColor: 'rgba(34, 197, 94, 0.8)',
          borderWidth: 1,
          borderRadius: 3,
        },
        {
          label: 'DOWN Won',
          data: downHist,
          backgroundColor: 'rgba(239, 68, 68, 0.6)',
          borderColor: 'rgba(239, 68, 68, 0.8)',
          borderWidth: 1,
          borderRadius: 3,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { size: 11 } } },
      },
      scales: {
        x: {
          title: { display: true, text: 'P(UP)', color: '#64748b' },
          grid: { color: 'rgba(148,163,184,0.08)' },
          ticks: { color: '#64748b', maxTicksLimit: 10 }
        },
        y: {
          title: { display: true, text: 'Count', color: '#64748b' },
          grid: { color: 'rgba(148,163,184,0.08)' },
          ticks: { color: '#64748b' }
        }
      }
    }
  });
}

// ---- 4. PnL Curves by Confidence ----
function renderPnlCurves() {
  const ctx = document.getElementById('pnlChart').getContext('2d');
  const curves = DATA.pnl_curves;

  const colors = {
    '55%': '#ef4444',
    '60%': '#f97316',
    '65%': '#22c55e',
    '70%': '#3b82f6',
    '75%': '#a855f7',
    '80%': '#06b6d4',
  };

  const datasets = Object.entries(curves).map(([key, points]) => ({
    label: `>= ${key}`,
    data: points.map(p => ({ x: p.trade, y: p.pnl })),
    borderColor: colors[key] || '#94a3b8',
    backgroundColor: 'transparent',
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.2,
  }));

  new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: $${ctx.raw.y.toFixed(2)}`
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'Trade #', color: '#64748b' },
          grid: { color: 'rgba(148,163,184,0.08)' },
          ticks: { color: '#64748b' }
        },
        y: {
          title: { display: true, text: 'Cumulative PnL ($)', color: '#64748b' },
          grid: { color: 'rgba(148,163,184,0.08)' },
          ticks: { color: '#64748b', callback: v => '$' + v }
        }
      }
    }
  });
}

// ---- 5. Feature Importance (GBM weights) ----
function renderFeatureImportance() {
  const container = document.getElementById('featureBars');
  const features = DATA.feature_importance;
  const maxCoef = Math.max(...features.map(f => f.abs_coefficient));

  let html = '';
  features.forEach(f => {
    const pct = (f.abs_coefficient / maxCoef * 100).toFixed(1);
    const cls = 'bar-positive';  // GBM importances are always positive
    const val = (f.abs_coefficient * 100).toFixed(1);
    html += `
      <div class="feature-bar">
        <div class="name">${f.feature}</div>
        <div class="bar-track">
          <div class="bar-fill ${cls}" style="width:${pct}%">
            <span>${val}%</span>
          </div>
        </div>
      </div>
    `;
  });
  container.innerHTML = html;
}

// ---- 6. Minute-by-Minute Accuracy ----
function renderMinuteAccuracy() {
  const container = document.getElementById('minuteGrid');
  const stats = DATA.minute_accuracy;

  let html = '';
  stats.forEach(m => {
    const color = m.accuracy >= 80 ? '#22c55e'
               : m.accuracy >= 70 ? '#3b82f6'
               : m.accuracy >= 60 ? '#f97316'
               : '#ef4444';
    const bg = m.accuracy >= 80 ? 'rgba(34,197,94,0.1)'
             : m.accuracy >= 70 ? 'rgba(59,130,246,0.1)'
             : m.accuracy >= 60 ? 'rgba(249,115,22,0.1)'
             : 'rgba(239,68,68,0.1)';
    html += `
      <div class="minute-cell" style="background:${bg};border:1px solid ${color}33">
        <div class="min-label">Min ${m.minute}</div>
        <div class="min-acc" style="color:${color}">${m.accuracy}%</div>
        <div class="min-conf">${m.avg_confidence}% conf</div>
        <div class="min-conf">${m.count} pts</div>
      </div>
    `;
  });
  container.innerHTML = html;
}

// ---- Footer ----
function renderFooter() {
  const gen = new Date(DATA.generated_at).toLocaleString();
  document.getElementById('footer-time').textContent = gen;
}

// ---- Init ----
document.addEventListener('DOMContentLoaded', loadData);

