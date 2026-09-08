import { pts } from '../format';

// ── Post-touch bars (analogue behaviour only; CR-AL: no predictive value) ──────

const TFS = ['t1', 't5', 't15'];
const TF_LABEL = { t1: '+1', t5: '+5', t15: '+15' };

export function PostTouchBars({ sp }) {
  const pt = sp?.post_touch;
  if (!pt || !pt.fractions) {
    return <p className="muted" data-testid="post-touch-none">Post-touch distribution not available for this day.</p>;
  }
  const adv = pt.advisory || {};
  const n = adv.n ?? pt.same_bucket_n ?? pt.total_touchers;
  return (
    <div data-testid="post-touch-bars">
      {TFS.map(tf => {
        const f = pt.fractions[tf] || {};
        const b = f.below ?? 0, a = f.at ?? 0, ab = f.above ?? 0;
        return (
          <div className="pt-row" key={tf}>
            <span className="pt-legend">{TF_LABEL[tf]}</span>
            <div className="pt-bar" title={`below ${Math.round(b * 100)}% · at ${Math.round(a * 100)}% · above ${Math.round(ab * 100)}%`}>
              <div className="below" style={{ width: `${b * 100}%` }} />
              <div className="at" style={{ width: `${a * 100}%` }} />
              <div className="above" style={{ width: `${ab * 100}%` }} />
            </div>
            <span className="pt-legend">{Math.round(b * 100)} / {Math.round(a * 100)} / {Math.round(ab * 100)}</span>
          </div>
        );
      })}
      <p className="muted">
        below / at / above the wall after the touch · pattern {pt.pattern_label || 'unlabeled'}{n != null ? ` · n ${n}` : ''}.
        Analogue behaviour only — the label does not predict P&amp;L (CR-AL) and does not change the numbers above.
      </p>
    </div>
  );
}

// ── Quote by minute vs the max price ──────────────────────────────────────────

export function QuoteStrip({ quote, maxPrice }) {
  const rows = (quote?.by_minute || []);
  const priced = rows.filter(r => r.net_debit != null);
  if (!priced.length) {
    return <p className="muted" data-testid="quote-strip-none">No cached minute quotes for the entry window {quote?.window_pt ? `${quote.window_pt[0]}–${quote.window_pt[1]} PT` : ''}.</p>;
  }
  const W = 420, H = 110, padL = 34, padR = 10, padT = 10, padB = 18;
  const toMin = m => { const [h, mm] = m.split(':').map(Number); return h * 60 + mm; };
  const xs = rows.map(r => toMin(r.minute));
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
  const ys = priced.map(r => r.net_debit).concat(maxPrice != null ? [maxPrice] : []);
  const y0 = Math.min(...ys) * 0.9, y1 = Math.max(...ys) * 1.1 || 1;
  const X = m => padL + ((toMin(m) - x0) / (x1 - x0)) * (W - padL - padR);
  const Y = v => padT + (1 - (v - y0) / (y1 - y0)) * (H - padT - padB);
  const path = priced.map((r, i) => `${i ? 'L' : 'M'}${X(r.minute).toFixed(1)},${Y(r.net_debit).toFixed(1)}`).join(' ');
  return (
    <div className="strip" data-testid="quote-strip">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Spread quote by minute against the max price">
        {maxPrice != null && (
          <>
            <line x1={padL} x2={W - padR} y1={Y(maxPrice)} y2={Y(maxPrice)} stroke="var(--ok)" strokeDasharray="4 3" />
            <text className="axis" x={W - padR} y={Y(maxPrice) - 3} textAnchor="end" fill="var(--ok)">max {pts(maxPrice, 1)}</text>
          </>
        )}
        <path d={path} fill="none" stroke="#fff" strokeWidth="1.5" />
        {priced.map(r => <circle key={r.minute} cx={X(r.minute)} cy={Y(r.net_debit)} r="2.5" fill="#fff" />)}
        <text className="axis" x={padL} y={H - 4}>{rows[0].minute}</text>
        <text className="axis" x={W - padR} y={H - 4} textAnchor="end">{rows[rows.length - 1].minute}</text>
        <text className="axis" x={2} y={Y(y1) + 4}>{pts(y1, 1)}</text>
        <text className="axis" x={2} y={Y(y0) + 4}>{pts(y0, 1)}</text>
      </svg>
      <p className="muted">Spread price at each cached minute of the entry window (PT). Above the dashed line the quote is over the max.</p>
    </div>
  );
}

export default function DetailPanel({ card }) {
  return (
    <div className="detail-body" data-testid="detail-panel">
      <div>
        <h4>Post-touch behaviour</h4>
        <PostTouchBars sp={card.structural_probability} />
      </div>
      <div>
        <h4>Quote by minute</h4>
        <QuoteStrip quote={card.quote} maxPrice={card.fair_value?.max_price} />
      </div>
    </div>
  );
}
