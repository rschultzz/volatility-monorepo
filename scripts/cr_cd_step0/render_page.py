"""Build the Step 0 Q4 render page from q4.json (data embedded; lightweight-charts from the worktree)."""
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
lib = Path(sys.argv[2]).read_text()
win = sys.argv[3] if len(sys.argv) > 3 else ""          # "07:00-08:45" or "" for the full RTH session
out = Path(sys.argv[4])

page = """<!doctype html><html><head><meta charset="utf-8"><title>CR-CD Step 0</title>
<style>
 body{margin:0;background:#020617;color:#e2e8f0;font:12px -apple-system,Segoe UI,sans-serif;width:1440px}
 .t{padding:6px 10px;font-weight:700;color:#cbd5e1;border-top:1px solid #1f2937}
 .t span{font-weight:400;color:#94a3b8}
 .h{padding:10px;font-size:15px;font-weight:800}
 .h span{display:block;font-size:12px;font-weight:400;color:#94a3b8;margin-top:3px}
 .leg{padding:4px 10px;color:#94a3b8}
 .leg b{display:inline-block;width:10px;height:10px;margin:0 4px 0 12px;vertical-align:middle}
</style></head><body>
<div class="h">CR-CD Step 0 — 2026-05-01 (ES M26, RTH, PT) — prototype of the L4/L5 v2 series, not the implementation
<span>Entry 07:39 long 7279.25, stop 7273.50 (CR-CC fixture 5). Markers: D = divergence, P = push decay, F = fading
 (green = bullish read, red = bearish read). Amber band = absorption (L4d). Price candles = ironbeam_es_1m_bars as the chart serves them.</span></div>
<div id="price"></div>
<div class="t">Flow v2 — Databento (es_flow_1s_databento) <span>— 220 px, today's pane height: histogram = delta per bar (left axis), white = CVD from 06:30 (own scale), cyan = delta share EMA3 (right axis, fixed −1…+1 per L4c)</span></div>
<div id="v2db"></div>
<div class="t">Flow v2 — Ironbeam (ironbeam_es_flow_1s) <span>— same definitions and prior-20 baseline on the Ironbeam scale; what a live session would show</span></div>
<div id="v2ib"></div>
<div class="t">Current indicator <span>— EMA(buy, 14 min) − EMA(sell, 14 min), Ironbeam, FULL / 1m / ema_len 14, as /api/ironbeam/flow serves it today</span></div>
<div id="cur"></div>
<div class="leg"><b style="background:#22c55e"></b>buy-dominant bar<b style="background:#ef4444"></b>sell-dominant bar<b style="background:#e5e7eb"></b>CVD<b style="background:#22d3ee"></b>delta share<b style="background:rgba(245,158,11,.35)"></b>absorption</div>
<script>__LIB__</script>
<script>
const D = __DATA__;
const WIN = "__WIN__";
const LW = LightweightCharts;
const T = (hhmm) => Date.UTC(2026, 4, 1, +hhmm.slice(0, 2), +hhmm.slice(3, 5)) / 1000;
const inWin = (t) => !WIN || (t >= WIN.slice(0, 5) && t <= WIN.slice(6, 11));
const base = (h) => ({ width: 1440, height: h, layout: { background: { color: '#020617' }, textColor: '#94a3b8' },
  grid: { vertLines: { color: '#0f172a' }, horzLines: { color: '#0f172a' } },
  timeScale: { timeVisible: true, secondsVisible: false, borderColor: '#1f2937' },
  rightPriceScale: { borderColor: '#1f2937' }, leftPriceScale: { borderColor: '#1f2937' },
  crosshair: { mode: 0 } });
const charts = [];

// price
const pc = LW.createChart(document.getElementById('price'), base(330)); charts.push(pc);
const cs = pc.addSeries(LW.CandlestickSeries, { upColor: '#22c55e', downColor: '#ef4444', wickUpColor: '#22c55e',
  wickDownColor: '#ef4444', borderVisible: false, priceLineVisible: false });
cs.setData(D.bars.filter(b => inWin(b.t)).map(b => ({ time: T(b.t), open: b.o, high: b.h, low: b.l, close: b.c })));
cs.createPriceLine({ price: 7279.25, color: '#60a5fa', lineWidth: 1, lineStyle: 2, title: 'entry 07:39' });
cs.createPriceLine({ price: 7273.5, color: '#f87171', lineWidth: 1, lineStyle: 2, title: 'stop' });
cs.createPriceLine({ price: 7277.0, color: '#a78bfa', lineWidth: 1, lineStyle: 0, title: 'level +157.8 bn' });
const mk = (src) => D[src].marks.filter(m => inWin(m.t)).map(m => ({ time: T(m.t),
  position: m.side === 'bull' ? 'belowBar' : 'aboveBar', shape: m.side === 'bull' ? 'arrowUp' : 'arrowDown',
  color: m.side === 'bull' ? '#22c55e' : '#ef4444', text: { divergence: 'D', push_decay: 'P', fading: 'F' }[m.kind] }));
LW.createSeriesMarkers(cs, [...mk('databento'), { time: T('07:39'), position: 'belowBar', shape: 'circle',
  color: '#60a5fa', text: 'entry' }].sort((a, b) => a.time - b.time));

function v2(el, src) {
  const c = LW.createChart(document.getElementById(el), { ...base(190),
    leftPriceScale: { visible: true, borderColor: '#1f2937' }, rightPriceScale: { visible: true, borderColor: '#1f2937' } });
  charts.push(c);
  const S = D[src].series.filter(r => inWin(r.t));
  const ab = c.addSeries(LW.HistogramSeries, { priceScaleId: 'ab', priceLineVisible: false, lastValueVisible: false });
  c.priceScale('ab').applyOptions({ visible: false, scaleMargins: { top: 0, bottom: 0 } });
  ab.setData(S.map(r => ({ time: T(r.t), value: r.absorb ? 1 : 0, color: 'rgba(245,158,11,0.35)' })));
  const h = c.addSeries(LW.HistogramSeries, { priceScaleId: 'left', priceLineVisible: false, lastValueVisible: false });
  h.setData(S.map(r => ({ time: T(r.t), value: r.delta, color: r.delta >= 0 ? 'rgba(34,197,94,0.75)' : 'rgba(239,68,68,0.75)' })));
  const cv = c.addSeries(LW.LineSeries, { priceScaleId: 'cvd', color: '#e5e7eb', lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
  c.priceScale('cvd').applyOptions({ visible: false });
  cv.setData(S.map(r => ({ time: T(r.t), value: r.cvd })));
  const sh = c.addSeries(LW.LineSeries, { priceScaleId: 'right', color: '#22d3ee', lineWidth: 1, priceLineVisible: false, lastValueVisible: false,
    autoscaleInfoProvider: () => ({ priceRange: { minValue: -1, maxValue: 1 } }) });
  sh.setData(S.filter(r => r.share !== null).map(r => ({ time: T(r.t), value: r.share })));
  LW.createSeriesMarkers(cv, mk(src).map(m => ({ ...m, position: m.position })).sort((a, b) => a.time - b.time));
}
v2('v2db', 'databento');
v2('v2ib', 'ironbeam');

const cc = LW.createChart(document.getElementById('cur'), base(190)); charts.push(cc);
const bs = cc.addSeries(LW.BaselineSeries, { baseValue: { type: 'price', price: 0 }, topLineColor: 'rgba(96,165,250,.95)',
  topFillColor1: 'rgba(96,165,250,.3)', topFillColor2: 'rgba(96,165,250,.2)', bottomLineColor: 'rgba(229,231,235,.95)',
  bottomFillColor1: 'rgba(229,231,235,.3)', bottomFillColor2: 'rgba(229,231,235,.2)', lineWidth: 2,
  priceLineVisible: false, lastValueVisible: false });
bs.setData(D.current.filter(r => inWin(r.t)).map(r => ({ time: T(r.t), value: r.v })));
charts.forEach(c => c.timeScale().fitContent());
window.__ready = true;
</script></body></html>"""
out.write_text(page.replace("__LIB__", lib).replace("__DATA__", json.dumps(data)).replace("__WIN__", win))
