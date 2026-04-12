function fmt(value, digits = 2) {
  if (value === null || value === undefined || value === '') return '—';
  const num = Number(value);
  if (Number.isNaN(num)) return String(value);
  return num.toFixed(digits);
}

function rowKey(row, idx) {
  return `${row.trade_date}-${row.start_ts_utc}-${row.target_ts_utc}-${idx}`;
}

function setupLabel(row) {
  if (row.direction !== 'up') return 'N/A';
  return row.short_setup_found ? 'Short setup' : 'No setup';
}

function tradeLabel(row) {
  if (row.direction !== 'up') return 'N/A';
  return row.trade_entry_found ? 'Trade entered' : 'No trade';
}

export default function ResultsTable({ rows, selectedRowKey, onSelectRow }) {
  if (!rows.length) {
    return (
      <div className="empty-state">
        No instances yet. Open Settings, tune the scan, and run it.
      </div>
    );
  }

  return (
    <div className="table-wrap">
      <table className="results-table">
        <thead>
          <tr>
            <th>Select</th>
            <th>Date</th>
            <th>Dir</th>
            <th>Source Zone</th>
            <th>Zone Levels</th>
            <th>Start Time (PT)</th>
            <th>Start Open</th>
            <th>Pivot Px</th>
            <th>Target Time (PT)</th>
            <th>Target Open</th>
            <th>Target Level</th>
            <th>Clean Space</th>
            <th>Move Pts</th>
            <th>Bars</th>
            <th>Consol. Mins</th>
            <th>Setup</th>
            <th>Signal Time (PT)</th>
            <th>Signal Px</th>
            <th>Δ Put Skew %</th>
            <th>Δ Call Skew %</th>
            <th>Trade</th>
            <th>Range High</th>
            <th>Range Low</th>
            <th>Entry Band Floor</th>
            <th>Entry Time (PT)</th>
            <th>Entry Px</th>
            <th>Init Stop</th>
            <th>Take Profit</th>
            <th>Trailing Stop</th>
            <th>Exit Time (PT)</th>
            <th>Exit Px</th>
            <th>Exit Reason</th>
            <th>Realized Pts</th>
            <th>MFE</th>
            <th>MAE</th>
            <th>Outcome</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => {
            const key = rowKey(row, idx);
            return (
              <tr key={key} className={selectedRowKey === key ? 'selected-row' : ''}>
                <td>
                  <input
                    type="radio"
                    name="bt2-selected-trade"
                    checked={selectedRowKey === key}
                    onChange={() => onSelectRow(row, idx)}
                  />
                </td>
                <td>{row.trade_date}</td>
                <td>
                  <span className={`direction-chip ${row.direction === 'up' ? 'up' : 'down'}`}>
                    {row.direction}
                  </span>
                </td>
                <td>
                  <div>{fmt(row.source_zone_low)} – {fmt(row.source_zone_high)}</div>
                  <div className="subcell">width {fmt(row.source_zone_width)}</div>
                </td>
                <td className="wrap-cell">
                  <div>{row.source_zone_levels}</div>
                </td>
                <td>
                  <div>{row.start_ts_pt}</div>
                  <div className="subcell">{row.start_context}</div>
                </td>
                <td>{fmt(row.start_open)}</td>
                <td>{fmt(row.start_pivot_price)}</td>
                <td>{row.target_ts_pt}</td>
                <td>{fmt(row.target_open)}</td>
                <td>
                  <div>{fmt(row.target_level)}</div>
                  <div className="subcell">{row.target_zone_range}</div>
                </td>
                <td>{fmt(row.clean_space_points)}</td>
                <td>{fmt(row.move_points)}</td>
                <td>{row.elapsed_bars}</td>
                <td>
                  <div>{row.consolidation_minutes_observed ?? '—'}</div>
                  <div className="subcell">{row.consolidation_end_ts_pt || '—'}</div>
                </td>
                <td>
                  <span className={`setup-chip ${row.short_setup_found ? 'hit' : 'miss'}`}>
                    {setupLabel(row)}
                  </span>
                </td>
                <td>{row.short_signal_ts_pt || '—'}</td>
                <td>{fmt(row.short_signal_price)}</td>
                <td>{fmt(row.short_signal_delta_put_skew_pct)}</td>
                <td>{fmt(row.short_signal_delta_call_skew_pct)}</td>
                <td>
                  <span className={`trade-chip ${row.trade_entry_found ? 'hit' : 'miss'}`}>
                    {tradeLabel(row)}
                  </span>
                </td>
                <td>{fmt(row.trade_range_high_at_entry)}</td>
                <td>{fmt(row.trade_range_low_at_entry)}</td>
                <td>{fmt(row.trade_entry_band_floor)}</td>
                <td>{row.trade_entry_ts_pt || '—'}</td>
                <td>{fmt(row.trade_entry_price)}</td>
                <td>{fmt(row.trade_initial_stop_price)}</td>
                <td>{fmt(row.trade_take_profit_price)}</td>
                <td>{fmt(row.trade_trailing_stop_price)}</td>
                <td>{row.trade_exit_ts_pt || '—'}</td>
                <td>{fmt(row.trade_exit_price)}</td>
                <td>{row.trade_exit_reason || '—'}</td>
                <td>{fmt(row.trade_realized_points)}</td>
                <td>{fmt(row.trade_mfe_points)}</td>
                <td>{fmt(row.trade_mae_points)}</td>
                <td>{row.trade_outcome || '—'}</td>
                <td className="wrap-cell">{row.short_setup_reason || '—'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}