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

export default function DiagnosticsPanel({ diagnostics, rows = [], funnel = [] }) {
  if (!diagnostics) return null;

  const perf = computePerformance(rows);

  return (
    <div className="diag-card">
      <div className="results-header">
        <div>
          <h2>Diagnostics</h2>
          <p>
            This section emphasizes strategy performance first, with a smaller set of useful trade counts.
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

      <div className="diag-stat-grid diag-stat-grid-compact">
        <StatCell label="Days total" value={diagnostics.days_total ?? '—'} />
        <StatCell label="Valid instances" value={diagnostics.valid_instances ?? '—'} />
        <StatCell label="Up short setups" value={diagnostics.up_short_setups_found ?? '—'} />
        <StatCell label="Actual trades" value={diagnostics.actual_trades_found ?? '—'} />
        <StatCell label="Winning trades" value={diagnostics.winning_trades ?? '—'} />
        <StatCell label="Total P/L (pts)" value={fmt(perf.totalPnLPts)} />
        <StatCell label="EV (pts/trade)" value={fmt(perf.evPtsPerTrade)} />
        <StatCell label="Max drawdown (pts)" value={fmt(perf.maxDrawdownPts)} />
      </div>
    </div>
  );
}
