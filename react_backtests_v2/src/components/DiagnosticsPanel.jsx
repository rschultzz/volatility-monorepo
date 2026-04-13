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

export default function DiagnosticsPanel({ diagnostics, rows = [] }) {
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