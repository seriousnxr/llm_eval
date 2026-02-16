#!/usr/bin/env python3
"""Build a self-contained HTML dashboard from evaluation results."""

import argparse
import json
from pathlib import Path

import yaml

OUT_PATH = Path("dashboard.html")


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Build evaluation dashboard")
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML (reads output.results_dir). Default: config.yaml",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    results_dir = Path(cfg.get("output", {}).get("results_dir", "results"))

    summary = json.loads((results_dir / "summary_report.json").read_text())
    gpqa = load_jsonl(results_dir / "gpqa_results.jsonl")
    math500 = load_jsonl(results_dir / "math500_results.jsonl")
    lcb = load_jsonl(results_dir / "livecodebench_results.jsonl")
    errors = load_jsonl(results_dir / "error_log.jsonl")

    # Truncate long fields for browser performance
    for rec in gpqa + math500 + lcb:
        if rec.get("prompt") and len(rec["prompt"]) > 600:
            rec["prompt"] = rec["prompt"][:600] + "…"
        if rec.get("raw_response") and len(rec["raw_response"]) > 1000:
            rec["raw_response"] = rec["raw_response"][:1000] + "…"
        if rec.get("generated_code") and len(rec["generated_code"]) > 1000:
            rec["generated_code"] = rec["generated_code"][:1000] + "…"

    for e in errors:
        if e.get("message") and len(e["message"]) > 400:
            e["message"] = e["message"][:400] + "…"

    html = HTML_TEMPLATE.replace("__SUMMARY__", json.dumps(summary))
    html = html.replace("__GPQA__", json.dumps(gpqa))
    html = html.replace("__MATH500__", json.dumps(math500))
    html = html.replace("__LCB__", json.dumps(lcb))
    html = html.replace("__ERRORS__", json.dumps(errors))

    OUT_PATH.write_text(html)
    print(f"Dashboard written to {OUT_PATH}  ({len(html)//1024} KB)")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Evaluation Dashboard</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922; --orange: #db6d28;
    --radius: 10px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.5; }
  .container { max-width: 1400px; margin: 0 auto; padding: 20px; }

  /* Header */
  header { text-align: center; padding: 30px 0 20px; }
  header h1 { font-size: 28px; font-weight: 700; }
  header .sub { color: var(--muted); font-size: 14px; margin-top: 4px; }

  /* Cards */
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin: 20px 0; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
          padding: 20px; }
  .card h2 { font-size: 14px; text-transform: uppercase; color: var(--muted); letter-spacing: 1px; margin-bottom: 12px; }
  .card .big { font-size: 42px; font-weight: 700; }
  .card .label { font-size: 13px; color: var(--muted); }

  /* Accuracy bars */
  .acc-row { display: flex; align-items: center; gap: 12px; margin: 8px 0; }
  .acc-row .name { width: 120px; font-size: 14px; font-weight: 600; }
  .acc-bar-bg { flex: 1; height: 26px; background: #21262d; border-radius: 6px; overflow: hidden; position: relative; }
  .acc-bar { height: 100%; border-radius: 6px; transition: width 0.6s ease; }
  .acc-bar-label { position: absolute; right: 8px; top: 50%; transform: translateY(-50%); font-size: 12px; font-weight: 600; }
  .acc-bar.gpqa { background: linear-gradient(90deg, #1f6feb, #58a6ff); }
  .acc-bar.math500 { background: linear-gradient(90deg, #238636, #3fb950); }
  .acc-bar.livecodebench { background: linear-gradient(90deg, #9e6a03, #d29922); }

  /* Error breakdown */
  .err-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 12px 0; }
  .err-item { background: #21262d; border-radius: 8px; padding: 12px 16px; display: flex; justify-content: space-between; align-items: center; }
  .err-item .cat { font-size: 13px; }
  .err-item .cnt { font-size: 20px; font-weight: 700; }

  /* Tabs */
  .tabs { display: flex; gap: 4px; margin: 24px 0 0; border-bottom: 1px solid var(--border); }
  .tab-btn { background: none; border: none; color: var(--muted); font-size: 14px; padding: 10px 18px;
             cursor: pointer; border-bottom: 2px solid transparent; font-weight: 500; transition: .2s; }
  .tab-btn:hover { color: var(--text); }
  .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }

  /* Sample browser */
  .sample-section { background: var(--surface); border: 1px solid var(--border); border-radius: 0 0 var(--radius) var(--radius);
                    padding: 16px; margin-bottom: 24px; }
  .sample-controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 16px; }
  .sample-controls select, .sample-controls input {
    background: #21262d; border: 1px solid var(--border); color: var(--text);
    padding: 6px 10px; border-radius: 6px; font-size: 13px;
  }
  .sample-controls input[type=text] { width: 200px; }
  .sample-nav { display: flex; gap: 6px; align-items: center; margin-left: auto; }
  .sample-nav button { background: #21262d; border: 1px solid var(--border); color: var(--text);
                       padding: 5px 12px; border-radius: 6px; cursor: pointer; font-size: 13px; }
  .sample-nav button:hover { background: #30363d; }
  .sample-nav span { font-size: 13px; color: var(--muted); min-width: 80px; text-align: center; }

  .sample-detail { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .sample-detail.full { grid-template-columns: 1fr; }
  .detail-box { background: #0d1117; border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
  .detail-box h4 { font-size: 12px; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; letter-spacing: .5px; }
  .detail-box pre { font-size: 12px; white-space: pre-wrap; word-break: break-word; max-height: 300px; overflow-y: auto;
                    line-height: 1.5; }
  .detail-meta { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; }
  .meta-tag { font-size: 12px; padding: 3px 10px; border-radius: 20px; font-weight: 600; }
  .meta-tag.correct { background: rgba(63,185,80,.15); color: var(--green); }
  .meta-tag.wrong { background: rgba(248,81,73,.15); color: var(--red); }
  .meta-tag.error { background: rgba(219,109,40,.15); color: var(--orange); }
  .meta-tag.info { background: rgba(88,166,255,.1); color: var(--accent); }

  /* Test results for LCB */
  .test-results { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 8px; }
  .test-pip { width: 22px; height: 22px; border-radius: 4px; display: flex; align-items: center; justify-content: center;
              font-size: 10px; font-weight: 700; }
  .test-pip.pass { background: rgba(63,185,80,.2); color: var(--green); }
  .test-pip.fail { background: rgba(248,81,73,.2); color: var(--red); }

  /* Error log table */
  .err-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
  .err-table th, .err-table td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }
  .err-table th { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .5px; background: #0d1117; position: sticky; top: 0; }
  .err-table-wrap { max-height: 400px; overflow-y: auto; border-radius: 8px; border: 1px solid var(--border); }

  /* Latency dot chart */
  .latency-chart { height: 120px; position: relative; margin: 12px 0; background: #0d1117; border-radius: 8px;
                   border: 1px solid var(--border); overflow: hidden; }
  .lat-dot { position: absolute; width: 4px; height: 4px; border-radius: 50%; opacity: .7; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* Donut */
  .donut-wrap { display: flex; gap: 24px; flex-wrap: wrap; justify-content: center; }
  .donut-item { text-align: center; }
  .donut-item .donut-label { font-size: 13px; color: var(--muted); margin-top: 6px; }
  svg.donut { transform: rotate(-90deg); }

  @media (max-width: 768px) {
    .sample-detail { grid-template-columns: 1fr; }
    .cards { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>LLM Evaluation Results</h1>
    <div class="sub" id="timestamp"></div>
  </header>

  <!-- Summary cards -->
  <div class="cards" id="summary-cards"></div>

  <!-- Accuracy bars -->
  <div class="card" id="accuracy-section">
    <h2>Accuracy by Task</h2>
    <div id="accuracy-bars"></div>
  </div>

  <!-- Error breakdown donut + latency -->
  <div class="cards">
    <div class="card">
      <h2>Error Breakdown</h2>
      <div class="donut-wrap" id="donut-area"></div>
      <div class="err-grid" id="overall-errors"></div>
    </div>
    <div class="card">
      <h2>Latency Distribution (ms)</h2>
      <div id="latency-area"></div>
    </div>
  </div>

  <!-- Sample browser tabs -->
  <div class="tabs" id="tabs">
    <button class="tab-btn active" data-tab="gpqa">GPQA Diamond (198)</button>
    <button class="tab-btn" data-tab="math500">MATH500 (500)</button>
    <button class="tab-btn" data-tab="livecodebench">LiveCodeBench (400)</button>
    <button class="tab-btn" data-tab="errors">Error Log (59)</button>
  </div>
  <div class="sample-section" id="sample-area"></div>
</div>

<script>
// ─── Data ────────────────────────────────────────────────────────────
const SUMMARY = __SUMMARY__;
const DATA = {
  gpqa: __GPQA__,
  math500: __MATH500__,
  livecodebench: __LCB__,
};
const ERRORS = __ERRORS__;

// ─── Render summary cards ────────────────────────────────────────────
document.getElementById('timestamp').textContent =
    'Run: ' + new Date(SUMMARY.timestamp).toLocaleString();

const overall = SUMMARY.overall;
const cardsEl = document.getElementById('summary-cards');
const cardData = [
  { label: 'Total Samples', value: overall.total_samples, color: 'var(--accent)' },
  { label: 'Overall Accuracy', value: (overall.overall_accuracy * 100).toFixed(2) + '%', color: overall.overall_accuracy > 0.1 ? 'var(--green)' : 'var(--red)' },
  { label: 'Correct', value: overall.total_correct, color: 'var(--green)' },
  { label: 'Tasks', value: Object.keys(SUMMARY.tasks).length, color: 'var(--yellow)' },
];
cardData.forEach(c => {
  const d = document.createElement('div');
  d.className = 'card';
  d.innerHTML = `<h2>${c.label}</h2><div class="big" style="color:${c.color}">${c.value}</div>`;
  cardsEl.appendChild(d);
});

// ─── Accuracy bars ───────────────────────────────────────────────────
const barsEl = document.getElementById('accuracy-bars');
const COLORS = { gpqa: 'gpqa', math500: 'math500', livecodebench: 'livecodebench' };
const LABELS = { gpqa: 'GPQA', math500: 'MATH 500', livecodebench: 'LiveCode' };
Object.entries(SUMMARY.tasks).forEach(([name, t]) => {
  const pct = (t.accuracy * 100).toFixed(2);
  const barWidth = Math.max(t.accuracy * 100, 1.5);
  const row = document.createElement('div');
  row.className = 'acc-row';
  row.innerHTML = `
    <span class="name">${LABELS[name] || name}</span>
    <div class="acc-bar-bg">
      <div class="acc-bar ${COLORS[name]}" style="width:${barWidth}%"></div>
      <span class="acc-bar-label">${pct}%  (${t.correct}/${t.total_samples})</span>
    </div>`;
  barsEl.appendChild(row);
});

// ─── Error breakdown ─────────────────────────────────────────────────
// Aggregate across all tasks
const allErrors = {};
Object.values(SUMMARY.tasks).forEach(t => {
  Object.entries(t.error_breakdown).forEach(([k, v]) => {
    allErrors[k] = (allErrors[k] || 0) + v;
  });
});
const errEl = document.getElementById('overall-errors');
const errColors = { none: 'var(--green)', truncated_json: 'var(--yellow)', parse_failure: 'var(--orange)',
                    compilation_error: 'var(--red)', runtime_error: '#f47067', server_error: '#da3633',
                    api_error: '#da3633', timeout: 'var(--orange)', wrong_answer: 'var(--red)' };
Object.entries(allErrors).sort((a,b) => b[1]-a[1]).forEach(([cat, cnt]) => {
  const d = document.createElement('div');
  d.className = 'err-item';
  d.innerHTML = `<span class="cat" style="color:${errColors[cat]||'var(--muted)'}">${cat}</span><span class="cnt">${cnt}</span>`;
  errEl.appendChild(d);
});

// Donuts per task
const donutArea = document.getElementById('donut-area');
Object.entries(SUMMARY.tasks).forEach(([name, t]) => {
  const entries = Object.entries(t.error_breakdown);
  const total = entries.reduce((s, e) => s + e[1], 0);
  let cumulative = 0;
  const R = 40, C = 2 * Math.PI * R;
  let arcs = '';
  const dColors = ['#3fb950','#d29922','#f85149','#f47067','#da3633','#8b949e','#58a6ff'];
  entries.forEach(([cat, cnt], i) => {
    const pct = cnt / total;
    const offset = C * cumulative;
    const len = C * pct;
    arcs += `<circle r="${R}" cx="50" cy="50" fill="none" stroke="${dColors[i % dColors.length]}"
              stroke-width="18" stroke-dasharray="${len} ${C - len}" stroke-dashoffset="-${offset}" />`;
    cumulative += pct;
  });
  const wrap = document.createElement('div');
  wrap.className = 'donut-item';
  wrap.innerHTML = `<svg class="donut" width="100" height="100" viewBox="0 0 100 100">${arcs}
    <text x="50" y="54" text-anchor="middle" fill="var(--text)" font-size="14" font-weight="700"
          transform="rotate(90 50 50)">${(t.accuracy*100).toFixed(1)}%</text></svg>
    <div class="donut-label">${LABELS[name] || name}</div>`;
  donutArea.appendChild(wrap);
});

// ─── Latency scatter ─────────────────────────────────────────────────
const latArea = document.getElementById('latency-area');
const latColors = { gpqa: '#58a6ff', math500: '#3fb950', livecodebench: '#d29922' };
Object.entries(DATA).forEach(([task, samples]) => {
  const chart = document.createElement('div');
  chart.className = 'latency-chart';
  chart.style.marginBottom = '8px';
  const maxLat = Math.max(...samples.map(s => s.latency_ms), 1);
  samples.forEach((s, i) => {
    const dot = document.createElement('div');
    dot.className = 'lat-dot';
    dot.style.background = latColors[task];
    dot.style.left = (i / samples.length * 100) + '%';
    dot.style.bottom = Math.min(s.latency_ms / maxLat * 100, 99) + '%';
    dot.title = `#${s.sample_id} — ${s.latency_ms.toFixed(0)}ms`;
    chart.appendChild(dot);
  });
  const label = document.createElement('div');
  label.style.cssText = 'font-size:11px;color:var(--muted);margin-top:2px;';
  label.textContent = `${LABELS[task]}  —  mean: ${SUMMARY.tasks[task].latency_ms.mean.toFixed(0)}ms  p50: ${SUMMARY.tasks[task].latency_ms.p50.toFixed(0)}ms  p95: ${SUMMARY.tasks[task].latency_ms.p95.toFixed(0)}ms`;
  latArea.appendChild(chart);
  latArea.appendChild(label);
});

// ─── Sample browser ──────────────────────────────────────────────────
let currentTab = 'gpqa';
let currentFilter = 'all';
let currentIdx = 0;
let filtered = [];

const tabBtns = document.querySelectorAll('.tab-btn');
const sampleArea = document.getElementById('sample-area');

function applyFilter(samples, filter) {
  if (filter === 'all') return samples;
  if (filter === 'correct') return samples.filter(s => s.score > 0);
  if (filter === 'wrong') return samples.filter(s => s.score === 0 && s.error_category === 'none');
  return samples.filter(s => s.error_category === filter);
}

function escapeHtml(s) {
  if (!s) return '<span style="color:var(--muted)">(empty)</span>';
  const d = document.createElement('div'); d.textContent = s; return d.innerHTML;
}

function renderSample() {
  if (currentTab === 'errors') { renderErrors(); return; }
  const samples = DATA[currentTab];
  filtered = applyFilter(samples, currentFilter);
  if (currentIdx >= filtered.length) currentIdx = 0;
  const s = filtered[currentIdx];

  // Build filter options
  const errCats = [...new Set(samples.map(r => r.error_category))].sort();
  const filterOpts = ['all','correct','wrong'].concat(errCats.filter(e => e !== 'none'))
    .map(f => `<option value="${f}" ${f===currentFilter?'selected':''}>${f}</option>`).join('');

  let html = `
  <div class="sample-controls">
    <label style="font-size:13px;color:var(--muted)">Filter:</label>
    <select id="filter-sel">${filterOpts}</select>
    <input type="text" id="search-box" placeholder="Search in response…">
    <div class="sample-nav">
      <button id="prev-btn">← Prev</button>
      <span id="pos-label">${filtered.length ? currentIdx+1 : 0} / ${filtered.length}</span>
      <button id="next-btn">Next →</button>
    </div>
  </div>`;

  if (!s) {
    html += '<div style="color:var(--muted);padding:20px;text-align:center;">No samples match this filter.</div>';
    sampleArea.innerHTML = html;
    bindControls();
    return;
  }

  // Meta tags
  const scoreCls = s.score > 0 ? 'correct' : (s.error_category !== 'none' ? 'error' : 'wrong');
  const scoreLabel = s.score > 0 ? '✓ Correct' : (s.error_category !== 'none' ? '✗ ' + s.error_category : '✗ Wrong');
  html += `<div class="detail-meta">
    <span class="meta-tag ${scoreCls}">${scoreLabel}</span>
    <span class="meta-tag info">ID: ${s.sample_id}</span>
    <span class="meta-tag info">${s.latency_ms.toFixed(0)} ms</span>`;
  if (s.match_method) html += `<span class="meta-tag info">match: ${s.match_method}</span>`;
  if (s.pass_at_1 !== undefined) html += `<span class="meta-tag ${s.pass_at_1?'correct':'wrong'}">pass@1: ${s.pass_at_1}</span>`;
  if (s.difficulty) html += `<span class="meta-tag info">${s.difficulty}</span>`;
  html += '</div>';

  // Comparison boxes
  if (currentTab === 'livecodebench') {
    html += `<div class="sample-detail full">
      <div class="detail-box"><h4>Prompt (truncated)</h4><pre>${escapeHtml(s.prompt)}</pre></div>
    </div>
    <div class="sample-detail" style="margin-top:12px">
      <div class="detail-box"><h4>Generated Code</h4><pre>${escapeHtml(s.generated_code || s.parsed_answer)}</pre></div>
      <div class="detail-box"><h4>Raw Response</h4><pre>${escapeHtml(s.raw_response)}</pre></div>
    </div>`;
    if (s.test_results && s.test_results.length) {
      html += `<div style="margin-top:12px"><h4 style="font-size:12px;text-transform:uppercase;color:var(--muted);margin-bottom:6px;">
        Test Cases (${s.test_results.filter(t=>t.passed).length}/${s.test_results.length} passed)</h4>
        <div class="test-results">`;
      s.test_results.forEach((t, i) => {
        html += `<div class="test-pip ${t.passed?'pass':'fail'}" title="Test ${i}: ${t.error_type} (${t.execution_time_ms.toFixed(0)}ms)">${t.passed?'✓':'✗'}</div>`;
      });
      html += '</div></div>';
    }
  } else {
    html += `<div class="sample-detail">
      <div class="detail-box"><h4>Predicted Answer</h4>
        <pre style="font-size:18px;font-weight:600;margin-bottom:8px;color:${s.score>0?'var(--green)':'var(--red)'}">${escapeHtml(s.parsed_answer)}</pre>
        <h4>Normalized</h4><pre>${escapeHtml(s.normalized_answer)}</pre>
      </div>
      <div class="detail-box"><h4>Ground Truth</h4>
        <pre style="font-size:18px;font-weight:600;margin-bottom:8px">${escapeHtml(s.ground_truth)}</pre>
      </div>
    </div>
    <div class="sample-detail" style="margin-top:12px">
      <div class="detail-box"><h4>Model Response</h4><pre>${escapeHtml(s.raw_response)}</pre></div>
      <div class="detail-box"><h4>Prompt (truncated)</h4><pre>${escapeHtml(s.prompt)}</pre></div>
    </div>`;
  }

  sampleArea.innerHTML = html;
  bindControls();
}

function renderErrors() {
  let html = `<div class="sample-controls">
    <span style="font-size:13px;color:var(--muted)">${ERRORS.length} errors logged during evaluation</span>
  </div>
  <div class="err-table-wrap"><table class="err-table">
    <thead><tr><th>#</th><th>Timestamp</th><th>Type</th><th>Status</th><th>Retry</th><th>Message</th></tr></thead><tbody>`;
  ERRORS.forEach((e, i) => {
    const ts = new Date(e.timestamp).toLocaleTimeString();
    html += `<tr>
      <td>${i+1}</td>
      <td>${ts}</td>
      <td style="color:${errColors[e.error_type]||'var(--muted)'}">${e.error_type}</td>
      <td>${e.status_code}</td>
      <td>${e.retry_count}</td>
      <td style="font-size:12px;max-width:500px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
          title="${escapeHtml(e.message)}">${escapeHtml(e.message)}</td>
    </tr>`;
  });
  html += '</tbody></table></div>';
  sampleArea.innerHTML = html;
}

function bindControls() {
  const prev = document.getElementById('prev-btn');
  const next = document.getElementById('next-btn');
  const filterSel = document.getElementById('filter-sel');
  const searchBox = document.getElementById('search-box');

  if (prev) prev.onclick = () => { currentIdx = Math.max(0, currentIdx - 1); renderSample(); };
  if (next) next.onclick = () => { currentIdx = Math.min(filtered.length - 1, currentIdx + 1); renderSample(); };
  if (filterSel) filterSel.onchange = () => { currentFilter = filterSel.value; currentIdx = 0; renderSample(); };
  if (searchBox) searchBox.oninput = (e) => {
    const q = e.target.value.toLowerCase();
    if (!q) { filtered = applyFilter(DATA[currentTab], currentFilter); }
    else { filtered = applyFilter(DATA[currentTab], currentFilter).filter(s =>
      (s.raw_response||'').toLowerCase().includes(q) || (s.parsed_answer||'').toLowerCase().includes(q)); }
    currentIdx = 0;
    const posLabel = document.getElementById('pos-label');
    if (posLabel) posLabel.textContent = `${filtered.length ? 1 : 0} / ${filtered.length}`;
  };
}

tabBtns.forEach(btn => btn.addEventListener('click', () => {
  tabBtns.forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentTab = btn.dataset.tab;
  currentFilter = 'all';
  currentIdx = 0;
  renderSample();
}));

// Keyboard navigation
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft') { currentIdx = Math.max(0, currentIdx - 1); renderSample(); }
  if (e.key === 'ArrowRight') { currentIdx = Math.min(filtered.length - 1, currentIdx + 1); renderSample(); }
});

renderSample();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
