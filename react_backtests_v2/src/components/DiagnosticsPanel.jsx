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

export default function DiagnosticsPanel({ diagnostics, rows = [], funnel = [], executionMode = 'managed' }) {
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

      {isStudy && aggregate && horizonsInAgg.length > 0 && (
        <>
          <ForwardOutcomesAggregate aggregate={aggregate} horizons={horizonsInAgg} />
          <ForwardOutcomesHistograms rows={rows} horizons={horizonsInAgg} />
        </>
      )}

      {isStudy && diagnostics.realized_vs_implied_aggregate && (
        <RealizedVsImpliedAggregate
          aggregate={diagnostics.realized_vs_implied_aggregate}
          ivSummary={diagnostics.iv_at_entry_summary}
        />
      )}

      <div className="diag-stat-grid diag-stat-grid-compact">
        <StatCell label="Days total" value={diagnostics.days_total ?? '—'} />
        <StatCell label="Valid instances" value={diagnostics.valid_instances ?? '—'} />
        {isStudy
          ? <StatCell label="Study target hits" value={diagnostics.entries_found ?? diagnostics.valid_instances ?? '—'} />
          : <StatCell label="Up short setups" value={diagnostics.up_short_setups_found ?? '—'} />}
        <StatCell label="Actual trades" value={isStudy ? '—' : (diagnostics.actual_trades_found ?? '—')} />
        <StatCell label="Winning trades" value={isStudy ? '—' : (diagnostics.winning_trades ?? '—')} />
        <StatCell label="Total P/L (pts)" value={isStudy ? '—' : fmt(perf.totalPnLPts)} />
        <StatCell label="EV (pts/trade)" value={isStudy ? '—' : fmt(perf.evPtsPerTrade)} />
        <StatCell label="Max drawdown (pts)" value={isStudy ? '—' : fmt(perf.maxDrawdownPts)} />
      </div>
    </div>
  );
}


// ─────────────────────────────────────────────────────────────────────
//  Forward outcomes aggregate table (study mode)
// ─────────────────────────────────────────────────────────────────────

function ForwardOutcomesAggregate({ aggregate, horizons }) {
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
      <h3 style={{
        fontSize: '13px',
        fontWeight: '700',
        color: '#93c5fd',
        marginBottom: '12px',
        letterSpacing: '0.05em',
        textTransform: 'uppercase',
      }}>
        Forward Outcomes by Horizon
      </h3>
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

function ForwardOutcomesHistograms({ rows, horizons }) {
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
            values={rows
              .map(r => r.forward_outcomes?.[h]?.close_pts)
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
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: '10px',
        color: '#64748b',
        marginTop: '4px',
      }}>
        <span>{firstBinStart.toFixed(0)}</span>
        <span>0</span>
        <span>{(lastBinStart + binWidth).toFixed(0)}</span>
      </div>
    </div>
  );
}


// ─────────────────────────────────────────────────────────────────────
//  Realized vs Implied aggregate table (short-vol lens)
// ─────────────────────────────────────────────────────────────────────

function RealizedVsImpliedAggregate({ aggregate, ivSummary }) {
  const horizonOrder = ['30m', '60m', '90m', '120m', '180m'];
  const horizons = horizonOrder.filter(h => aggregate[h] && aggregate[h].count > 0);

  if (!horizons.length) return null;

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
  // Hit rate vs normality baseline (68% for 1σ, 95% for 2σ)
  const hitRateColor = (v, baseline) => {
    if (v === null || v === undefined) return '#64748b';
    const edge = v - baseline;
    if (edge >= 0.10) return '#86efac';   // >10pts above baseline
    if (edge >= 0.00) return '#fcd34d';   // at or slightly above
    return '#fca5a5';                     // below baseline
  };

  const pct = (v) => (v !== null && v !== undefined) ? `${(v * 100).toFixed(0)}%` : '—';

  return (
    <div style={{ marginBottom: '24px' }}>
      <h3 style={{
        fontSize: '13px',
        fontWeight: '700',
        color: '#93c5fd',
        marginBottom: '8px',
        letterSpacing: '0.05em',
        textTransform: 'uppercase',
      }}>
        Realized vs Implied Movement
      </h3>

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
        marginBottom: '8px',
      }}>
        Baselines: <strong>Range/1σ ≈ 1.60</strong> for random walk, <strong>68%</strong> inside ±1σ,
        <strong> 95%</strong> inside ±2σ under normality. Green cells beat baseline.
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
              <th style={headerStyle}>|Close| / 1σ</th>
              <th style={headerStyle}>Inside ±1σ</th>
              <th style={headerStyle}>Inside ±2σ</th>
            </tr>
          </thead>
          <tbody>
            {horizons.map(h => {
              const a = aggregate[h] || {};
              return (
                <tr key={h}>
                  <td style={labelCellStyle}>{h}</td>
                  <td style={cellStyle}>{a.count ?? '—'}</td>
                  <td style={cellStyle}>{a.iv_median !== null && a.iv_median !== undefined ? `${a.iv_median}%` : '—'}</td>
                  <td style={cellStyle}>{a.implied_1sigma_median ?? '—'}</td>
                  <td style={{ ...cellStyle, color: rangeRatioColor(a.range_over_1sigma_median), fontWeight: 700 }}>
                    {a.range_over_1sigma_median !== null && a.range_over_1sigma_median !== undefined
                      ? a.range_over_1sigma_median.toFixed(2)
                      : '—'}
                  </td>
                  <td style={cellStyle}>
                    {a.close_over_1sigma_median !== null && a.close_over_1sigma_median !== undefined
                      ? a.close_over_1sigma_median.toFixed(2)
                      : '—'}
                  </td>
                  <td style={{ ...cellStyle, color: hitRateColor(a.pct_inside_1sigma, 0.68), fontWeight: 700 }}>
                    {pct(a.pct_inside_1sigma)}
                  </td>
                  <td style={{ ...cellStyle, color: hitRateColor(a.pct_inside_2sigma, 0.95), fontWeight: 700 }}>
                    {pct(a.pct_inside_2sigma)}
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
