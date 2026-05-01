import React, { useState, useMemo } from 'react';

// ─────────────────────────────────────────────────────────────────────
//  aggregateForwardOutcomes
//  Re-aggregates per-row forward_outcomes into per-horizon stats.
//  When `flipped` is true, swaps MFE↔MAE and negates close, so the
//  numbers reflect what the same set of moves would have produced if
//  every trade had been taken in the opposite direction.
//  Mirrors the backend's rounding (2dp for points, 3dp for win rate)
//  so flipped/unflipped views are numerically consistent.
// ─────────────────────────────────────────────────────────────────────
function aggregateForwardOutcomes(rows, horizons, flipped) {
  const round2 = (v) => Math.round(v * 100) / 100;
  const round3 = (v) => Math.round(v * 1000) / 1000;
  const median = (xs) => {
    if (!xs.length) return 0;
    const s = [...xs].sort((a, b) => a - b);
    const n = s.length;
    return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
  };

  const out = {};
  for (const h of horizons) {
    const mfes = [];
    const maes = [];
    const closes = [];
    for (const r of rows || []) {
      const fo = r.forward_outcomes?.[h];
      if (!fo) continue;
      const origMfe = fo.mfe_pts;
      const origMae = fo.mae_pts;
      const origCls = fo.close_pts;

      // Flip transform:
      //   mfe (favorable magnitude) ↔ mae (adverse magnitude)
      //   close (signed pnl) → −close
      if (origMfe !== null && origMfe !== undefined) {
        mfes.push(flipped ? Number(origMae) : Number(origMfe));
      }
      if (origMae !== null && origMae !== undefined) {
        maes.push(flipped ? Number(origMfe) : Number(origMae));
      }
      if (origCls !== null && origCls !== undefined) {
        closes.push(flipped ? -Number(origCls) : Number(origCls));
      }
    }

    if (!closes.length) {
      out[h] = { count: 0 };
      continue;
    }
    const sum = (xs) => xs.reduce((a, b) => a + b, 0);
    out[h] = {
      count: closes.length,
      mfe_mean:          mfes.length   ? round2(sum(mfes)   / mfes.length)   : 0,
      mfe_median:        round2(median(mfes)),
      mae_mean:          maes.length   ? round2(sum(maes)   / maes.length)   : 0,
      mae_median:        round2(median(maes)),
      close_mean:        round2(sum(closes) / closes.length),
      close_median:      round2(median(closes)),
      win_rate_at_close: round3(closes.filter((c) => c > 0).length / closes.length),
    };
  }
  return out;
}

function fmt(value, digits = 2) {
  if (value === null || value === undefined || value === '') return '—';
  const num = Number(value);
  if (Number.isNaN(num)) return String(value);
  return num.toFixed(digits);
}

function StatCell({ label, value }) {
  return (
    <div className="diag-stat-card">
      <div className="diag-stat-label">{label}</div>
      <div className="diag-stat-value">{value}</div>
    </div>
  );
}

function sortRowsForEquity(rows) {
  return [...(rows || [])].sort((a, b) => {
    const aKey = `${a.trade_date || ''} ${a.trade_entry_ts_pt || a.short_signal_ts_pt || a.target_ts_pt || a.start_ts_pt || ''}`;
    const bKey = `${b.trade_date || ''} ${b.trade_entry_ts_pt || b.short_signal_ts_pt || b.target_ts_pt || b.start_ts_pt || ''}`;
    return aKey.localeCompare(bKey);
  });
}

function computePerformance(rows) {
  const executedTradeRows = sortRowsForEquity(rows).filter(
    (row) => row.trade_entry_found && row.trade_realized_points !== null && row.trade_realized_points !== undefined
  );

  const ordered = executedTradeRows
    .map((row) => Number(row.trade_realized_points))
    .filter((v) => Number.isFinite(v));

  if (!ordered.length) {
    return {
      totalPnLPts: null,
      evPtsPerTrade: null,
      maxDrawdownPts: null,
    };
  }

  const totalPnLPts = ordered.reduce((sum, v) => sum + v, 0);
  const evPtsPerTrade = totalPnLPts / ordered.length;

  let equity = 0;
  let peak = 0;
  let maxDrawdown = 0;

  for (const pnl of ordered) {
    equity += pnl;
    peak = Math.max(peak, equity);
    maxDrawdown = Math.max(maxDrawdown, peak - equity);
  }

  return {
    totalPnLPts,
    evPtsPerTrade,
    maxDrawdownPts: maxDrawdown,
  };
}

function FunnelStage({ stage, index }) {
  const { label, kind, bypassed, candidates_in, kept, dropped, drop_reasons } = stage;
  const pct = candidates_in > 0 ? (kept / candidates_in) * 100 : 0;

  const kindColor = kind === 'construction' ? '#3b82f6' : kind === 'filter' ? '#10b981' : '#64748b';
  const barColor = bypassed ? '#475569' : kindColor;

  return (
    <div style={{
      background: '#0f172a',
      border: '1px solid #1e293b',
      borderRadius: '10px',
      padding: '12px 14px',
      marginBottom: '8px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{
            fontSize: '11px',
            fontWeight: '700',
            color: '#64748b',
            minWidth: '20px',
          }}>
            {index + 1}.
          </span>
          <span style={{
            fontSize: '13px',
            fontWeight: '600',
            color: bypassed ? '#64748b' : '#e2e8f0',
            opacity: bypassed ? 0.6 : 1,
          }}>
            {label}
          </span>
          {bypassed && (
            <span style={{
              fontSize: '9px',
              fontWeight: '700',
              color: '#64748b',
              background: '#1e293b',
              padding: '2px 6px',
              borderRadius: '4px',
              letterSpacing: '0.05em',
            }}>
              BYPASSED
            </span>
          )}
        </div>
        <div style={{ fontSize: '12px', color: '#94a3b8', fontWeight: '500' }}>
          {kept} / {candidates_in}
          {dropped > 0 && !bypassed && (
            <span style={{ color: '#f87171', marginLeft: '8px' }}>
              (−{dropped})
            </span>
          )}
        </div>
      </div>

      <div style={{
        height: '6px',
        background: '#1e293b',
        borderRadius: '3px',
        overflow: 'hidden',
      }}>
        <div style={{
          height: '100%',
          width: `${pct}%`,
          background: barColor,
          transition: 'width 0.3s ease',
        }} />
      </div>

      {!bypassed && dropped > 0 && drop_reasons && Object.keys(drop_reasons).length > 0 && (
        <div style={{ marginTop: '8px', fontSize: '11px', color: '#64748b' }}>
          {Object.entries(drop_reasons).map(([reason, count]) => (
            <div key={reason} style={{ marginTop: '2px' }}>
              • {reason.replace(/_/g, ' ')}: {count}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function DiagnosticsPanel({
  diagnostics,
  rows = [],
  funnel = [],
  executionMode = 'managed',
  // Optional controlled toggle for the Long/Short direction view on the
  // "Forward Outcomes by Horizon" card. When provided, the aggregate card
  // mirrors this state and the parent can use it to drive other surfaces
  // (e.g. flipping fwd_* columns in a sibling instances table). When not
  // provided, the card falls back to managing its own state.
  viewDirection,                // 'long' | 'short' | undefined
  onViewDirectionChange,        // (newView) => void
  originalTrade,                // 'long' | 'short' | undefined — explicit override
}) {
  if (!diagnostics) return null;

  const isStudy = executionMode === 'study_target_hits';
  const perf = isStudy
    ? { totalPnLPts: null, evPtsPerTrade: null, maxDrawdownPts: null }
    : computePerformance(rows);

  const aggregate = diagnostics.forward_outcomes_aggregate || null;
  const horizonOrder = ['30m', '60m', '90m', '120m', '180m', 'eod'];
  const horizonsInAgg = aggregate
    ? horizonOrder.filter(h => aggregate[h] && aggregate[h].count > 0)
    : [];

  return (
    <div className="diag-card">
      <div className="results-header">
        <div>
          <h2>Diagnostics</h2>
          <p>
            {isStudy
              ? 'Study mode — forward-price outcomes from target-touch to close, across all surviving rows.'
              : 'This section emphasizes strategy performance first, with a smaller set of useful trade counts.'}
          </p>
        </div>
      </div>

      {funnel && funnel.length > 0 && (
        <div style={{ marginBottom: '20px' }}>
          <h3 style={{
            fontSize: '13px',
            fontWeight: '700',
            color: '#93c5fd',
            marginBottom: '12px',
            letterSpacing: '0.05em',
            textTransform: 'uppercase',
          }}>
            Pipeline Funnel
          </h3>
          {funnel.map((stage, idx) => (
            <FunnelStage key={stage.key} stage={stage} index={idx} />
          ))}
        </div>
      )}

      {isStudy && aggregate && horizonsInAgg.length > 0 && (() => {
        // Compute the effective flip flag at this level so the aggregate
        // table AND the histograms can both apply the same transform.
        // Mirrors the derivation inside ForwardOutcomesAggregate so the
        // two panels stay in lockstep when "reversed from setup" is on.
        const rowDir = rows?.[0]?.direction;
        const derivedOriginalTrade = rowDir === 'up' ? 'short'
                                    : rowDir === 'down' ? 'long'
                                    : 'short';
        const effectiveOriginalTrade = originalTrade || derivedOriginalTrade;
        const effectiveView = viewDirection !== undefined ? viewDirection : effectiveOriginalTrade;
        const flipped = effectiveView !== effectiveOriginalTrade;
        return (
          <>
            <ForwardOutcomesAggregate
              rows={rows}
              horizons={horizonsInAgg}
              viewDirection={viewDirection}
              onViewDirectionChange={onViewDirectionChange}
              originalTrade={originalTrade}
            />
            <ForwardOutcomesHistograms
              rows={rows}
              horizons={horizonsInAgg}
              flipped={flipped}
            />
          </>
        );
      })()}

      {isStudy && diagnostics.realized_vs_implied_aggregate && (
        <RealizedVsImpliedAggregate
          aggregate={diagnostics.realized_vs_implied_aggregate}
          ivSummary={diagnostics.iv_at_entry_summary}
          rows={rows}
        />
      )}
    </div>
  );
}


// ─────────────────────────────────────────────────────────────────────
//  Forward outcomes aggregate table (study mode)
// ─────────────────────────────────────────────────────────────────────

function ForwardOutcomesAggregate({
  rows,
  horizons,
  viewDirection,           // controlled value (optional)
  onViewDirectionChange,   // controlled setter (optional)
  originalTrade: originalTradeProp,
}) {
  // Each saved scan is homogeneous (all "up" or all "down"), so the
  // first row's direction tells us the scan's original trade setup:
  //   direction "up"   → up move faded with a SHORT
  //   direction "down" → down move faded with a LONG
  const rowDir = rows?.[0]?.direction;
  const derivedOriginalTrade = rowDir === 'up' ? 'short'
                              : rowDir === 'down' ? 'long'
                              : 'short'; // safe fallback if direction is missing
  const originalTrade = originalTradeProp || derivedOriginalTrade;

  // Controlled vs. uncontrolled: if a parent passes viewDirection,
  // mirror that; otherwise fall back to local state so the panel still
  // works standalone. The toggle UI itself lives in the SavedScans
  // header now — this component only renders the (possibly flipped)
  // numbers.
  const [internalView, setInternalView] = useState(originalTrade);
  React.useEffect(() => {
    if (viewDirection === undefined) setInternalView(originalTrade);
  }, [originalTrade, viewDirection]);

  const view = viewDirection !== undefined ? viewDirection : internalView;
  const flipped = view !== originalTrade;

  const aggregate = useMemo(
    () => aggregateForwardOutcomes(rows, horizons, flipped),
    [rows, horizons, flipped]
  );

  const cellStyle = {
    padding: '8px 12px',
    fontSize: '12px',
    color: '#e2e8f0',
    borderBottom: '1px solid #1e293b',
    textAlign: 'right',
  };
  const labelCellStyle = {
    ...cellStyle,
    textAlign: 'left',
    fontWeight: 600,
    color: '#cbd5e1',
  };
  const headerStyle = {
    ...cellStyle,
    fontSize: '11px',
    color: '#64748b',
    fontWeight: 700,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    borderBottom: '1px solid #334155',
  };

  const colorForValue = (v) => {
    if (v === null || v === undefined) return '#64748b';
    if (v > 0) return '#86efac';
    if (v < 0) return '#fca5a5';
    return '#e2e8f0';
  };

  return (
    <div style={{ marginBottom: '24px' }}>
      <div style={{
        marginBottom: '12px',
      }}>
        <h3 style={{
          fontSize: '13px',
          fontWeight: '700',
          color: '#93c5fd',
          margin: 0,
          letterSpacing: '0.05em',
          textTransform: 'uppercase',
        }}>
          Forward Outcomes by Horizon
          {flipped && (
            <span style={{
              marginLeft: '8px',
              color: '#fcd34d',
              fontSize: '11px',
              fontWeight: 600,
              letterSpacing: '0.04em',
            }}>
              · reversed from setup
            </span>
          )}
        </h3>
      </div>
      <div style={{
        background: '#0f172a',
        border: '1px solid #1e293b',
        borderRadius: '10px',
        overflow: 'hidden',
      }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{ ...headerStyle, textAlign: 'left' }}>Horizon</th>
              <th style={headerStyle}>Count</th>
              <th style={headerStyle}>MFE median</th>
              <th style={headerStyle}>MFE mean</th>
              <th style={headerStyle}>MAE median</th>
              <th style={headerStyle}>MAE mean</th>
              <th style={headerStyle}>Close median</th>
              <th style={headerStyle}>Close mean</th>
              <th style={headerStyle}>Win rate @ close</th>
            </tr>
          </thead>
          <tbody>
            {horizons.map(h => {
              const a = aggregate[h] || {};
              return (
                <tr key={h}>
                  <td style={labelCellStyle}>{h}</td>
                  <td style={cellStyle}>{a.count ?? '—'}</td>
                  <td style={{ ...cellStyle, color: '#86efac' }}>{a.mfe_median ?? '—'}</td>
                  <td style={{ ...cellStyle, color: '#86efac' }}>{a.mfe_mean ?? '—'}</td>
                  <td style={{ ...cellStyle, color: '#fca5a5' }}>{a.mae_median ?? '—'}</td>
                  <td style={{ ...cellStyle, color: '#fca5a5' }}>{a.mae_mean ?? '—'}</td>
                  <td style={{ ...cellStyle, color: colorForValue(a.close_median), fontWeight: 700 }}>{a.close_median ?? '—'}</td>
                  <td style={{ ...cellStyle, color: colorForValue(a.close_mean), fontWeight: 700 }}>{a.close_mean ?? '—'}</td>
                  <td style={cellStyle}>{a.win_rate_at_close !== undefined && a.win_rate_at_close !== null
                    ? `${(a.win_rate_at_close * 100).toFixed(0)}%`
                    : '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}


// ─────────────────────────────────────────────────────────────────────
//  Forward outcomes histograms (study mode)
// ─────────────────────────────────────────────────────────────────────

function ForwardOutcomesHistograms({ rows, horizons, flipped = false }) {
  return (
    <div style={{ marginBottom: '24px' }}>
      <h3 style={{
        fontSize: '13px',
        fontWeight: '700',
        color: '#93c5fd',
        marginBottom: '12px',
        letterSpacing: '0.05em',
        textTransform: 'uppercase',
      }}>
        Close P&amp;L Distribution
        {flipped && (
          <span style={{
            marginLeft: '8px',
            color: '#fcd34d',
            fontSize: '11px',
            fontWeight: 600,
            letterSpacing: '0.04em',
          }}>
            · reversed from setup
          </span>
        )}
      </h3>
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
        gap: '12px',
      }}>
        {horizons.map(h => (
          <Histogram
            key={h}
            title={h}
            // Flip transform: close_pts → -close_pts when viewing the
            // reversed direction. Matches what the aggregate table is
            // doing one row up — keeps both panels in sync so a "win
            // rate" of 85% in the table doesn't sit next to a sea of
            // red bars below.
            values={rows
              .map(r => {
                const v = r.forward_outcomes?.[h]?.close_pts;
                if (v === null || v === undefined) return v;
                return flipped ? -Number(v) : Number(v);
              })
              .filter(v => v !== null && v !== undefined && Number.isFinite(v))}
          />
        ))}
      </div>
    </div>
  );
}

function Histogram({ title, values }) {
  if (!values.length) {
    return (
      <div style={{
        background: '#0f172a',
        border: '1px solid #1e293b',
        borderRadius: '10px',
        padding: '12px',
        minHeight: '140px',
      }}>
        <div style={{ fontSize: '12px', fontWeight: 700, color: '#cbd5e1', marginBottom: '8px' }}>{title}</div>
        <div style={{ fontSize: '11px', color: '#64748b' }}>No data</div>
      </div>
    );
  }

  // Compute bins: anchor at zero so positive/negative are visually symmetric.
  // Use fixed bin width of 2 points for readability on typical SPX/ES moves.
  const binWidth = 2;
  const minV = Math.min(...values);
  const maxV = Math.max(...values);

  // Extend bin range outward to nearest bin edge around zero
  const firstBinStart = Math.floor(minV / binWidth) * binWidth;
  const lastBinStart = Math.floor(maxV / binWidth) * binWidth;
  const nBins = Math.max(1, Math.round((lastBinStart - firstBinStart) / binWidth) + 1);

  const bins = new Array(nBins).fill(0);
  for (const v of values) {
    let idx = Math.floor((v - firstBinStart) / binWidth);
    if (idx < 0) idx = 0;
    if (idx >= nBins) idx = nBins - 1;
    bins[idx] += 1;
  }
  const maxBin = Math.max(...bins);

  // Width of SVG
  const width = 240;
  const height = 100;
  const barGap = 1;
  const barWidth = Math.max(1, (width - (nBins - 1) * barGap) / nBins);

  // Find bin that contains zero, for the axis reference
  const zeroIdx = Math.floor((0 - firstBinStart) / binWidth);

  // Stats
  const sorted = [...values].sort((a, b) => a - b);
  const median = sorted.length % 2
    ? sorted[Math.floor(sorted.length / 2)]
    : 0.5 * (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]);
  const winRate = values.filter(v => v > 0).length / values.length;

  return (
    <div style={{
      background: '#0f172a',
      border: '1px solid #1e293b',
      borderRadius: '10px',
      padding: '12px',
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'baseline',
        marginBottom: '8px',
      }}>
        <div style={{ fontSize: '12px', fontWeight: 700, color: '#cbd5e1' }}>{title}</div>
        <div style={{ fontSize: '10px', color: '#64748b' }}>
          n={values.length} &middot; med {median.toFixed(1)} &middot; win {(winRate * 100).toFixed(0)}%
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        {/* Baseline */}
        <line x1="0" y1={height - 1} x2={width} y2={height - 1} stroke="#1e293b" strokeWidth="1" />
        {/* Zero line */}
        {zeroIdx >= 0 && zeroIdx <= nBins && (
          <line
            x1={zeroIdx * (barWidth + barGap)}
            y1="0"
            x2={zeroIdx * (barWidth + barGap)}
            y2={height}
            stroke="#334155"
            strokeWidth="1"
            strokeDasharray="2,2"
          />
        )}
        {/* Bars */}
        {bins.map((count, i) => {
          const binStart = firstBinStart + i * binWidth;
          const barHeight = maxBin > 0 ? (count / maxBin) * (height - 4) : 0;
          const color = binStart + binWidth / 2 > 0 ? '#22c55e' : binStart + binWidth / 2 < 0 ? '#ef4444' : '#64748b';
          return (
            <rect
              key={i}
              x={i * (barWidth + barGap)}
              y={height - barHeight - 1}
              width={barWidth}
              height={barHeight}
              fill={color}
              opacity="0.8"
            >
              <title>{`[${binStart.toFixed(1)}, ${(binStart + binWidth).toFixed(1)}): ${count}`}</title>
            </rect>
          );
        })}
      </svg>
      {/* Axis labels: min on the left, max on the right, and "0" anchored at
          the actual zero position (which usually isn't the geometric middle).
          We use absolute positioning relative to a wrapper so the "0" lands
          on top of the dashed zero line in the SVG above. */}
      {(() => {
        const totalBinSpan = (lastBinStart + binWidth) - firstBinStart;
        // Fractional x-position of zero within the chart [0..1]
        const zeroFrac = totalBinSpan > 0
          ? Math.max(0, Math.min(1, (0 - firstBinStart) / totalBinSpan))
          : 0.5;
        return (
          <div style={{ position: 'relative', height: '14px', marginTop: '4px', fontSize: '10px', color: '#64748b' }}>
            <span style={{ position: 'absolute', left: 0 }}>{firstBinStart.toFixed(0)}</span>
            <span style={{
              position: 'absolute',
              left: `${zeroFrac * 100}%`,
              transform: 'translateX(-50%)',
            }}>
              0
            </span>
            <span style={{ position: 'absolute', right: 0 }}>{(lastBinStart + binWidth).toFixed(0)}</span>
          </div>
        );
      })()}
    </div>
  );
}


// ─────────────────────────────────────────────────────────────────────
//  Realized vs Implied aggregate table (short-vol lens)
// ─────────────────────────────────────────────────────────────────────

// Helper functions for normal distribution probability
function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const t = 1.0 / (1.0 + p * ax);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return sign * y;
}

const normalProbInside = (x) => {
  if (!Number.isFinite(x) || x <= 0) return 0;
  return erf(x / Math.SQRT2);
};

function RealizedVsImpliedAggregate({ aggregate, ivSummary, rows = [] }) {
  const [sigma1, setSigma1] = React.useState(1.0);
  const [sigma2, setSigma2] = React.useState(2.0);
  // Time-convention toggle. Calendar = 60×24×365 min/year (matches service.py's
  // implied-sigma math). Trading = 252×390 min/year (intraday convention used
  // by most desks for 0DTE). Trading-time σ is √(525600/98280) ≈ 2.31× larger
  // than calendar-time σ for the same horizon. We don't recompute per-row; we
  // just rescale the existing calendar-time ratios on the way out.
  const [useTradingTime, setUseTradingTime] = React.useState(false);
  const TT_SCALE = Math.sqrt((60 * 24 * 365) / (252 * 390)); // ≈ 2.31
  const sigmaScale = useTradingTime ? TT_SCALE : 1.0;

  const fixedHorizons = ['30m', '60m', '90m', '120m', '180m'];
  const horizonsToShow = [
    ...fixedHorizons.filter(h => aggregate[h] && aggregate[h].count > 0),
    'eod',
  ];

  // Live computation of stats from rows
  const liveStats = React.useMemo(() => {
    const median = (arr) => {
      if (!arr.length) return null;
      const sorted = [...arr].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    };

    const stats = {};

    // Process fixed horizons
    for (const h of fixedHorizons) {
      const contributors = [];
      const ivValues = [];
      const impliedValues = [];
      const rangeRatios = [];
      const closeRatios = [];
      const maxExcursionRatios = [];   // max(mfe, mae) / impl_σ — for "Range inside" stat
      let insideSigma1 = 0;
      let insideSigma2 = 0;
      let rangeInsideSigma1 = 0;       // count of trades where max-excursion < sigma1
      let rangeInsideSigma2 = 0;       // count of trades where max-excursion < sigma2

      for (const row of rows) {
        const rvi = row.realized_vs_implied?.[h];
        if (!rvi) continue;

        const closeOverSigma = rvi.close_over_1sigma;
        if (closeOverSigma === null || closeOverSigma === undefined) continue;

        contributors.push(row);

        if (row.iv?.atm_0dte_pct !== null && row.iv?.atm_0dte_pct !== undefined) {
          ivValues.push(row.iv.atm_0dte_pct);
        }

        if (rvi.implied_1sigma_pts !== null && rvi.implied_1sigma_pts !== undefined) {
          impliedValues.push(rvi.implied_1sigma_pts);
        }

        if (rvi.range_over_1sigma !== null && rvi.range_over_1sigma !== undefined) {
          rangeRatios.push(rvi.range_over_1sigma);
        }

        closeRatios.push(closeOverSigma);

        // Max-excursion ratio: did EITHER side breach the implied band?
        // For an iron condor sized at ±1σ, this is the "did the short strike
        // get tested" question — much more relevant than terminal close.
        // Pulls mfe/mae from forward_outcomes since realized_vs_implied only
        // stores the sum (range), not the individual sides.
        const fo = row.forward_outcomes?.[h];
        const impl = rvi.implied_1sigma_pts;
        if (
          fo && impl !== null && impl !== undefined && impl > 0 &&
          fo.mfe_pts !== null && fo.mfe_pts !== undefined &&
          fo.mae_pts !== null && fo.mae_pts !== undefined
        ) {
          const maxExc = Math.max(Math.abs(fo.mfe_pts), Math.abs(fo.mae_pts));
          const maxExcRatio = maxExc / impl;
          maxExcursionRatios.push(maxExcRatio);
          // Threshold check uses time-convention-aware threshold.
          // Trading-time σ is bigger, so threshold is sigma1 * TT_SCALE.
          const effectiveThreshold1 = sigma1 * sigmaScale;
          const effectiveThreshold2 = sigma2 * sigmaScale;
          if (maxExcRatio < effectiveThreshold1) rangeInsideSigma1++;
          if (maxExcRatio < effectiveThreshold2) rangeInsideSigma2++;
        }

        // Inside-σ check uses the SAME time-convention scaling
        const effectiveThreshold1 = sigma1 * sigmaScale;
        const effectiveThreshold2 = sigma2 * sigmaScale;
        if (closeOverSigma < effectiveThreshold1) insideSigma1++;
        if (closeOverSigma < effectiveThreshold2) insideSigma2++;
      }

      if (contributors.length > 0) {
        // Apply time-convention scaling to displayed medians.
        // - Implied σ scales UP under trading-time (×TT_SCALE)
        // - Ratios (range/σ, close/σ) scale DOWN under trading-time (÷TT_SCALE)
        const rawImpl     = median(impliedValues);
        const rawRangeRat = median(rangeRatios);
        const rawCloseRat = median(closeRatios);
        const rawMaxRat   = median(maxExcursionRatios);
        stats[h] = {
          count: contributors.length,
          iv_median: median(ivValues),
          implied_1sigma_median: rawImpl !== null ? rawImpl * sigmaScale : null,
          range_over_1sigma_median: rawRangeRat !== null ? rawRangeRat / sigmaScale : null,
          close_over_1sigma_median: rawCloseRat !== null ? rawCloseRat / sigmaScale : null,
          max_excursion_over_1sigma_median: rawMaxRat !== null ? rawMaxRat / sigmaScale : null,
          pct_inside_sigma1: insideSigma1 / contributors.length,
          pct_inside_sigma2: insideSigma2 / contributors.length,
          pct_range_inside_sigma1: maxExcursionRatios.length > 0 ? rangeInsideSigma1 / maxExcursionRatios.length : null,
          pct_range_inside_sigma2: maxExcursionRatios.length > 0 ? rangeInsideSigma2 / maxExcursionRatios.length : null,
        };
      }
    }

    // Process EOD
    const eodContributors = [];
    const eodIvValues = [];
    const eodImpliedValues = [];
    const eodRangeRatios = [];
    const eodCloseRatios = [];
    const eodMaxExcursionRatios = [];
    let eodInsideSigma1 = 0;
    let eodInsideSigma2 = 0;
    let eodRangeInsideSigma1 = 0;
    let eodRangeInsideSigma2 = 0;

    for (const row of rows) {
      const ivAtm = row.iv?.atm_0dte_pct;
      const spx = row.target_spx_price;
      const mtc = row.minutes_to_close;
      const fo = row.forward_outcomes?.eod;
      const closePts = fo?.close_pts;
      const mfe = fo?.mfe_pts;
      const mae = fo?.mae_pts;

      if (
        ivAtm === null || ivAtm === undefined ||
        spx === null || spx === undefined ||
        mtc === null || mtc === undefined || mtc <= 0 ||
        closePts === null || closePts === undefined
      ) continue;

      const impl = spx * (ivAtm / 100) * Math.sqrt(mtc / (60 * 24 * 365));
      if (impl <= 0) continue;

      const closeRatio = Math.abs(closePts) / impl;

      eodContributors.push(row);
      eodIvValues.push(ivAtm);
      eodImpliedValues.push(impl);
      eodCloseRatios.push(closeRatio);

      if (mfe !== null && mfe !== undefined && mae !== null && mae !== undefined) {
        const rangeRatio = (mfe + mae) / impl;
        eodRangeRatios.push(rangeRatio);
        // Max-excursion: did either side touch ±1σ during the session?
        const maxExc = Math.max(Math.abs(mfe), Math.abs(mae));
        const maxExcRatio = maxExc / impl;
        eodMaxExcursionRatios.push(maxExcRatio);
        const eodEffThreshold1 = sigma1 * sigmaScale;
        const eodEffThreshold2 = sigma2 * sigmaScale;
        if (maxExcRatio < eodEffThreshold1) eodRangeInsideSigma1++;
        if (maxExcRatio < eodEffThreshold2) eodRangeInsideSigma2++;
      }

      const eodEffThreshold1 = sigma1 * sigmaScale;
      const eodEffThreshold2 = sigma2 * sigmaScale;
      if (closeRatio < eodEffThreshold1) eodInsideSigma1++;
      if (closeRatio < eodEffThreshold2) eodInsideSigma2++;
    }

    if (eodContributors.length > 0) {
      const rawImpl     = median(eodImpliedValues);
      const rawRangeRat = median(eodRangeRatios);
      const rawCloseRat = median(eodCloseRatios);
      const rawMaxRat   = median(eodMaxExcursionRatios);
      stats.eod = {
        count: eodContributors.length,
        iv_median: median(eodIvValues),
        implied_1sigma_median: rawImpl !== null ? rawImpl * sigmaScale : null,
        range_over_1sigma_median: rawRangeRat !== null ? rawRangeRat / sigmaScale : null,
        close_over_1sigma_median: rawCloseRat !== null ? rawCloseRat / sigmaScale : null,
        max_excursion_over_1sigma_median: rawMaxRat !== null ? rawMaxRat / sigmaScale : null,
        pct_inside_sigma1: eodInsideSigma1 / eodContributors.length,
        pct_inside_sigma2: eodInsideSigma2 / eodContributors.length,
        pct_range_inside_sigma1: eodMaxExcursionRatios.length > 0 ? eodRangeInsideSigma1 / eodMaxExcursionRatios.length : null,
        pct_range_inside_sigma2: eodMaxExcursionRatios.length > 0 ? eodRangeInsideSigma2 / eodMaxExcursionRatios.length : null,
      };
    }

    return stats;
  }, [rows, sigma1, sigma2, sigmaScale]);

  const cellStyle = {
    padding: '8px 12px',
    fontSize: '12px',
    color: '#e2e8f0',
    borderBottom: '1px solid #1e293b',
    textAlign: 'right',
  };
  const labelCellStyle = {
    ...cellStyle,
    textAlign: 'left',
    fontWeight: 600,
    color: '#cbd5e1',
  };
  const headerStyle = {
    ...cellStyle,
    fontSize: '11px',
    color: '#64748b',
    fontWeight: 700,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    borderBottom: '1px solid #334155',
  };

  // Range-over-1sigma: Brownian expects ~1.60
  const rangeRatioColor = (v) => {
    if (v === null || v === undefined) return '#64748b';
    if (v < 1.40) return '#86efac';   // meaningfully below random walk
    if (v < 1.60) return '#fcd34d';   // slightly below
    return '#fca5a5';                 // at or above random walk
  };

  // Hit rate vs normality baseline - now uses dynamic sigma threshold
  const hitRateColor = (observed, sigmaThreshold) => {
    if (observed === null || observed === undefined) return '#64748b';
    const baseline = normalProbInside(sigmaThreshold);
    const edge = observed - baseline;
    if (edge >= 0.10) return '#86efac';   // 10+ points above normality → green
    if (edge >= 0.00) return '#fcd34d';   // at or slightly above → yellow
    return '#fca5a5';                     // below → red
  };

  const pct = (v) => (v !== null && v !== undefined) ? `${(v * 100).toFixed(0)}%` : '—';

  const prob1Pct = Math.round(normalProbInside(sigma1) * 100);
  const prob2Pct = Math.round(normalProbInside(sigma2) * 100);

  return (
    <div style={{ marginBottom: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
        <h3 style={{
          fontSize: '13px',
          fontWeight: '700',
          color: '#93c5fd',
          margin: 0,
          letterSpacing: '0.05em',
          textTransform: 'uppercase',
        }}>
          Realized vs Implied Movement
        </h3>

        {/* Time-convention toggle (segmented control).
            Calendar = 365 cal-days; Trading = 252 days × 6.5 hrs.
            Trading-time σ ≈ 2.31× larger for the same horizon — so
            "Inside ±σ" rates jump dramatically and condor strikes
            (anchor ± σ) move further out. Default = calendar to match service.py. */}
        <div style={{
          display: 'inline-flex',
          background: '#0f172a',
          border: '1px solid #334155',
          borderRadius: 6,
          overflow: 'hidden',
          fontSize: 10,
          fontWeight: 700,
          letterSpacing: '0.04em',
          textTransform: 'uppercase',
        }}>
          <button
            type="button"
            onClick={() => setUseTradingTime(false)}
            style={{
              background: !useTradingTime ? '#2563eb' : 'transparent',
              color: !useTradingTime ? '#fff' : '#94a3b8',
              border: 'none',
              padding: '4px 10px',
              cursor: 'pointer',
              fontFamily: 'inherit',
              fontSize: 'inherit',
              fontWeight: 'inherit',
              letterSpacing: 'inherit',
              textTransform: 'inherit',
            }}
            title="Annualize using calendar minutes (60 × 24 × 365). Matches service.py."
          >
            Calendar σ
          </button>
          <button
            type="button"
            onClick={() => setUseTradingTime(true)}
            style={{
              background: useTradingTime ? '#2563eb' : 'transparent',
              color: useTradingTime ? '#fff' : '#94a3b8',
              border: 'none',
              padding: '4px 10px',
              cursor: 'pointer',
              fontFamily: 'inherit',
              fontSize: 'inherit',
              fontWeight: 'inherit',
              letterSpacing: 'inherit',
              textTransform: 'inherit',
            }}
            title={`Annualize using trading minutes (252 × 390). σ scales by ${TT_SCALE.toFixed(2)}×.`}
          >
            Trading σ ({TT_SCALE.toFixed(2)}×)
          </button>
        </div>
      </div>

      {ivSummary && (
        <div style={{
          fontSize: '11px',
          color: '#94a3b8',
          marginBottom: '10px',
          fontStyle: 'italic',
        }}>
          IV at entry (n={ivSummary.count}): min {ivSummary.min}%, p25 {ivSummary.p25}%,
          median <strong style={{ color: '#cbd5e1' }}>{ivSummary.median}%</strong>,
          p75 {ivSummary.p75}%, max {ivSummary.max}%
        </div>
      )}

      <div style={{
        fontSize: '10px',
        color: '#64748b',
        marginBottom: '12px',
      }}>
        Baselines: <strong>Range/1σ ≈ 1.60</strong> for random walk. Green cells beat normality baseline.
        Implied σ is capped at time-to-session-close per trade, so ratios are honest across setups
        with different amounts of time remaining. Use the sliders below to adjust sigma thresholds.
      </div>

      {/* Sigma sliders */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '16px',
        marginBottom: '12px',
        padding: '12px',
        background: '#0f172a',
        border: '1px solid #1e293b',
        borderRadius: '8px',
      }}>
        <div>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '6px',
          }}>
            <label style={{ fontSize: '11px', fontWeight: 600, color: '#cbd5e1' }}>
              Inner σ: {sigma1.toFixed(1)}
            </label>
            <span style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 500 }}>
              ≈ {prob1Pct}% normal
            </span>
          </div>
          <input
            type="range"
            min="0.3"
            max="3.0"
            step="0.1"
            value={sigma1}
            onChange={(e) => setSigma1(parseFloat(e.target.value))}
            style={{
              width: '100%',
              accentColor: '#3b82f6',
            }}
          />
        </div>
        <div>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '6px',
          }}>
            <label style={{ fontSize: '11px', fontWeight: 600, color: '#cbd5e1' }}>
              Outer σ: {sigma2.toFixed(1)}
            </label>
            <span style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 500 }}>
              ≈ {prob2Pct}% normal
            </span>
          </div>
          <input
            type="range"
            min="0.3"
            max="3.0"
            step="0.1"
            value={sigma2}
            onChange={(e) => setSigma2(parseFloat(e.target.value))}
            style={{
              width: '100%',
              accentColor: '#3b82f6',
            }}
          />
        </div>
      </div>

      <div style={{
        background: '#0f172a',
        border: '1px solid #1e293b',
        borderRadius: '10px',
        overflow: 'hidden',
      }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{ ...headerStyle, textAlign: 'left' }}>Horizon</th>
              <th style={headerStyle}>n</th>
              <th style={headerStyle}>IV median</th>
              <th style={headerStyle}>Impl 1σ</th>
              <th style={headerStyle}>Range / 1σ</th>
              <th style={headerStyle}>Max / 1σ</th>
              <th style={headerStyle}>|Close| / 1σ</th>
              <th style={headerStyle}>
                Close Inside ±{sigma1.toFixed(1)}σ{' '}
                <span style={{ color: '#94a3b8', fontWeight: 500 }}>({prob1Pct}%)</span>
              </th>
              <th style={headerStyle}>
                Close Inside ±{sigma2.toFixed(1)}σ{' '}
                <span style={{ color: '#94a3b8', fontWeight: 500 }}>({prob2Pct}%)</span>
              </th>
              <th style={headerStyle}>
                Range Inside ±{sigma1.toFixed(1)}σ
              </th>
              <th style={headerStyle}>
                Range Inside ±{sigma2.toFixed(1)}σ
              </th>
            </tr>
          </thead>
          <tbody>
            {horizonsToShow.map(h => {
              const a = liveStats[h] || {};
              return (
                <tr key={h}>
                  <td style={labelCellStyle}>{h}</td>
                  <td style={cellStyle}>{a.count ?? '—'}</td>
                  <td style={cellStyle}>{a.iv_median !== null && a.iv_median !== undefined ? `${a.iv_median.toFixed(1)}%` : '—'}</td>
                  <td style={cellStyle}>{a.implied_1sigma_median !== null && a.implied_1sigma_median !== undefined ? a.implied_1sigma_median.toFixed(2) : '—'}</td>
                  <td style={{ ...cellStyle, color: rangeRatioColor(a.range_over_1sigma_median), fontWeight: 700 }}>
                    {a.range_over_1sigma_median !== null && a.range_over_1sigma_median !== undefined
                      ? a.range_over_1sigma_median.toFixed(2)
                      : '—'}
                  </td>
                  <td style={cellStyle}>
                    {a.max_excursion_over_1sigma_median !== null && a.max_excursion_over_1sigma_median !== undefined
                      ? a.max_excursion_over_1sigma_median.toFixed(2)
                      : '—'}
                  </td>
                  <td style={cellStyle}>
                    {a.close_over_1sigma_median !== null && a.close_over_1sigma_median !== undefined
                      ? a.close_over_1sigma_median.toFixed(2)
                      : '—'}
                  </td>
                  <td style={{ ...cellStyle, color: hitRateColor(a.pct_inside_sigma1, sigma1), fontWeight: 700 }}>
                    {pct(a.pct_inside_sigma1)}
                  </td>
                  <td style={{ ...cellStyle, color: hitRateColor(a.pct_inside_sigma2, sigma2), fontWeight: 700 }}>
                    {pct(a.pct_inside_sigma2)}
                  </td>
                  <td style={{ ...cellStyle, color: hitRateColor(a.pct_range_inside_sigma1, sigma1), fontWeight: 700 }}>
                    {pct(a.pct_range_inside_sigma1)}
                  </td>
                  <td style={{ ...cellStyle, color: hitRateColor(a.pct_range_inside_sigma2, sigma2), fontWeight: 700 }}>
                    {pct(a.pct_range_inside_sigma2)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
