import React, { useMemo, useState, forwardRef, useImperativeHandle } from 'react';

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
  if (row.direction === 'up') return row.short_setup_found ? 'Short setup' : 'No setup';
  if (row.direction === 'down') return row.long_setup_found ? 'Long setup' : 'No setup';
  return 'N/A';
}

function tradeLabel(row) {
  return row.trade_entry_found ? 'Trade entered' : 'No trade';
}

const COLUMN_DATA_MAP = {
  date: 'trade_date',
  direction: 'direction',
  source_zone: 'source_zone_low',
  zone_levels: 'source_zone_levels',
  start_time: 'start_ts_pt',
  start_open: 'start_open',
  pivot_px: 'start_pivot_price',
  target_time: 'target_ts_pt',
  target_open: 'target_open',
  target_level: 'target_level',
  clean_space: 'clean_space_points',
  move_pts: 'move_points',
  bars: 'elapsed_bars',
  consol_mins: 'consolidation_minutes_observed',
  setup: (row) => row.direction === 'down' ? row.long_setup_found : row.short_setup_found,
  signal_time: (row) => row.direction === 'down' ? row.long_signal_ts_pt : row.short_signal_ts_pt,
  signal_px: (row) => row.direction === 'down' ? row.long_signal_price : row.short_signal_price,
  put_skew: (row) => row.direction === 'down' ? row.long_signal_delta_put_skew_pct : row.short_signal_delta_put_skew_pct,
  call_skew: (row) => row.direction === 'down' ? row.long_signal_delta_call_skew_pct : row.short_signal_delta_call_skew_pct,
  trade: 'trade_entry_found',
  range_high: 'trade_range_high_at_entry',
  range_low: 'trade_range_low_at_entry',
  entry_band: 'trade_entry_band_floor',
  entry_time: 'trade_entry_ts_pt',
  entry_px: 'trade_entry_price',
  init_stop: 'trade_initial_stop_price',
  take_profit: 'trade_take_profit_price',
  trailing_stop: 'trade_trailing_stop_price',
  exit_time: 'trade_exit_ts_pt',
  exit_px: 'trade_exit_price',
  exit_reason: 'trade_exit_reason',
  realized_pts: 'trade_realized_points',
  mfe: 'trade_mfe_points',
  mae: 'trade_mae_points',
  outcome: 'trade_outcome',
  reason: (row) => row.direction === 'down' ? row.long_setup_reason : row.short_setup_reason,
  prior_down_pts: 'prior_session_down_pts',
  prior_down_ratio: 'prior_down_vs_up_ratio',
  start_pct_range: 'start_pct_of_session_range',

  // ── Study mode columns ──
  skew_passed: 'skew_threshold_passed',
  target_price: 'target_price',

  fwd_30m_mfe:   (row) => row.forward_outcomes?.['30m']?.mfe_pts,
  fwd_30m_mae:   (row) => row.forward_outcomes?.['30m']?.mae_pts,
  fwd_30m_close: (row) => row.forward_outcomes?.['30m']?.close_pts,

  fwd_60m_mfe:   (row) => row.forward_outcomes?.['60m']?.mfe_pts,
  fwd_60m_mae:   (row) => row.forward_outcomes?.['60m']?.mae_pts,
  fwd_60m_close: (row) => row.forward_outcomes?.['60m']?.close_pts,

  fwd_90m_mfe:   (row) => row.forward_outcomes?.['90m']?.mfe_pts,
  fwd_90m_mae:   (row) => row.forward_outcomes?.['90m']?.mae_pts,
  fwd_90m_close: (row) => row.forward_outcomes?.['90m']?.close_pts,

  fwd_120m_mfe:   (row) => row.forward_outcomes?.['120m']?.mfe_pts,
  fwd_120m_mae:   (row) => row.forward_outcomes?.['120m']?.mae_pts,
  fwd_120m_close: (row) => row.forward_outcomes?.['120m']?.close_pts,

  fwd_180m_mfe:   (row) => row.forward_outcomes?.['180m']?.mfe_pts,
  fwd_180m_mae:   (row) => row.forward_outcomes?.['180m']?.mae_pts,
  fwd_180m_close: (row) => row.forward_outcomes?.['180m']?.close_pts,

  fwd_eod_mfe:   (row) => row.forward_outcomes?.['eod']?.mfe_pts,
  fwd_eod_mae:   (row) => row.forward_outcomes?.['eod']?.mae_pts,
  fwd_eod_close: (row) => row.forward_outcomes?.['eod']?.close_pts,

  // IV snapshot at entry (study mode)
  iv_atm_0dte: (row) => row.iv?.atm_0dte_pct,

  // Realized vs implied at 120m (short-vol lens)
  rvi_ratio_120m: (row) => row.realized_vs_implied?.['120m']?.close_over_1sigma,
  rvi_inside_1s_120m: (row) => row.realized_vs_implied?.['120m']?.inside_1sigma,

  // Hypothetical 120m iron condor strikes
  condor_short_put:  (row) => row.hypothetical_condor_120m?.short_put_strike,
  condor_long_put:   (row) => row.hypothetical_condor_120m?.long_put_strike,
  condor_short_call: (row) => row.hypothetical_condor_120m?.short_call_strike,
  condor_long_call:  (row) => row.hypothetical_condor_120m?.long_call_strike,
};

const ResultsTable = forwardRef(({ rows, selectedRowKey, onSelectRow, columns }, ref) => {
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });

  const handleSort = (colId) => {
    let direction = 'asc';
    if (sortConfig.key === colId && sortConfig.direction === 'asc') {
      direction = 'desc';
    } else if (sortConfig.key === colId && sortConfig.direction === 'desc') {
      // Optional: cycle back to no sort
      setSortConfig({ key: null, direction: 'asc' });
      return;
    }
    setSortConfig({ key: colId, direction });
  };

  const processedRows = useMemo(() => {
    let result = [...rows];

    if (sortConfig.key) {
      const dataKey = COLUMN_DATA_MAP[sortConfig.key];
      if (dataKey) {
        result.sort((a, b) => {
          let aVal, bVal;
          if (typeof dataKey === 'function') {
            aVal = dataKey(a);
            bVal = dataKey(b);
          } else {
            aVal = a[dataKey];
            bVal = b[dataKey];
          }

          if (aVal === bVal) return 0;
          if (aVal === null || aVal === undefined || aVal === '') return 1;
          if (bVal === null || bVal === undefined || bVal === '') return -1;

          const comparison = aVal < bVal ? -1 : 1;
          return sortConfig.direction === 'asc' ? comparison : -comparison;
        });
      }
    }

    return result;
  }, [rows, sortConfig]);

  useImperativeHandle(ref, () => ({
    downloadCSV: () => {
      if (!processedRows.length) return;

      const visibleCols = columns.filter(c => c.visible && c.id !== 'select');
      const headers = visibleCols.map(c => c.label);
      
      const csvContent = [
        headers.join(','),
        ...processedRows.map(row => {
          return visibleCols.map(col => {
            let val;
            const dataKey = COLUMN_DATA_MAP[col.id];
            
            switch (col.id) {
              case 'source_zone':
                val = `${fmt(row.source_zone_low)} - ${fmt(row.source_zone_high)}`;
                break;
              case 'start_time':
                val = `${row.start_ts_pt}${row.start_context ? ' (' + row.start_context + ')' : ''}`;
                break;
              case 'target_level':
                val = `${fmt(row.target_level)}${row.target_zone_range ? ' (' + row.target_zone_range + ')' : ''}`;
                break;
              case 'consol_mins':
                val = `${row.consolidation_minutes_observed ?? ''}${row.consolidation_end_ts_pt ? ' ' + row.consolidation_end_ts_pt : ''}`;
                break;
              case 'setup':
                val = setupLabel(row);
                break;
              case 'trade':
                val = tradeLabel(row);
                break;
              case 'signal_time':
                val = (row.direction === 'down' ? row.long_signal_ts_pt : row.short_signal_ts_pt) || '';
                break;
              case 'put_skew':
                val = fmt(row.direction === 'down' ? row.long_signal_delta_put_skew_pct : row.short_signal_delta_put_skew_pct);
                break;
              case 'call_skew':
                val = fmt(row.direction === 'down' ? row.long_signal_delta_call_skew_pct : row.short_signal_delta_call_skew_pct);
                break;
              case 'reason':
                val = (row.direction === 'down' ? row.long_setup_reason : row.short_setup_reason) || '';
                break;
              case 'prior_down_ratio':
                val = fmt(row.prior_down_vs_up_ratio, 2);
                break;
              case 'start_pct_range':
                val = row.start_pct_of_session_range != null ? (row.start_pct_of_session_range * 100).toFixed(1) + '%' : '';
                break;
              default:
                val = typeof dataKey === 'function' ? dataKey(row) : row[dataKey];
            }

            if (val === null || val === undefined) return '';
            const s = String(val).replace(/"/g, '""');
            return s.includes(',') || s.includes('"') || s.includes('\n') ? `"${s}"` : s;
          }).join(',');
        })
      ].join('\n');

      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', url);
      link.setAttribute('download', `backtest_results_${new Date().toISOString().slice(0, 10)}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }
  }));

  if (!rows.length) {
    return (
      <div className="empty-state">
        No instances yet. Open Settings, tune the scan, and run it.
      </div>
    );
  }

  const renderCell = (col, row, idx) => {
    switch (col.id) {
      case 'select':
        return (
          <input
            type="radio"
            name="bt2-selected-trade"
            checked={selectedRowKey === rowKey(row, idx)}
            onChange={() => onSelectRow(row, idx)}
          />
        );
      case 'date': return row.trade_date;
      case 'direction':
        return (
          <span className={`direction-chip ${row.direction === 'up' ? 'up' : 'down'}`}>
            {row.direction}
          </span>
        );
      case 'source_zone':
        return (
          <>
            <div>{fmt(row.source_zone_low)} – {fmt(row.source_zone_high)}</div>
            <div className="subcell">width {fmt(row.source_zone_width)}</div>
          </>
        );
      case 'zone_levels':
        return <div>{row.source_zone_levels}</div>;
      case 'start_time':
        return (
          <>
            <div>{row.start_ts_pt}</div>
            <div className="subcell">{row.start_context}</div>
          </>
        );
      case 'start_open': return fmt(row.start_open);
      case 'pivot_px': return fmt(row.start_pivot_price);
      case 'target_time': return row.target_ts_pt;
      case 'target_open': return fmt(row.target_open);
      case 'target_level':
        return (
          <>
            <div>{fmt(row.target_level)}</div>
            <div className="subcell">{row.target_zone_range}</div>
          </>
        );
      case 'clean_space': return fmt(row.clean_space_points);
      case 'move_pts': return fmt(row.move_points);
      case 'bars': return row.elapsed_bars;
      case 'consol_mins':
        return (
          <>
            <div>{row.consolidation_minutes_observed ?? '—'}</div>
            <div className="subcell">{row.consolidation_end_ts_pt || '—'}</div>
          </>
        );
      case 'setup':
        return (
          <span className={`setup-chip ${(row.short_setup_found || row.long_setup_found) ? 'hit' : 'miss'}`}>
            {setupLabel(row)}
          </span>
        );
      case 'signal_time':
        return (row.direction === 'down' ? row.long_signal_ts_pt : row.short_signal_ts_pt) || '—';
      case 'signal_px':
        return fmt(row.direction === 'down' ? row.long_signal_price : row.short_signal_price);
      case 'put_skew':
        return fmt(row.direction === 'down' ? row.long_signal_delta_put_skew_pct : row.short_signal_delta_put_skew_pct);
      case 'call_skew':
        return fmt(row.direction === 'down' ? row.long_signal_delta_call_skew_pct : row.short_signal_delta_call_skew_pct);
      case 'trade':
        return (
          <span className={`trade-chip ${row.trade_entry_found ? 'hit' : 'miss'}`}>
            {tradeLabel(row)}
          </span>
        );
      case 'range_high': return fmt(row.trade_range_high_at_entry);
      case 'range_low': return fmt(row.trade_range_low_at_entry);
      case 'entry_band': return fmt(row.trade_entry_band_floor);
      case 'entry_time': return row.trade_entry_ts_pt || '—';
      case 'entry_px': return fmt(row.trade_entry_price);
      case 'init_stop': return fmt(row.trade_initial_stop_price);
      case 'take_profit': return fmt(row.trade_take_profit_price);
      case 'trailing_stop': return fmt(row.trade_trailing_stop_price);
      case 'exit_time': return row.trade_exit_ts_pt || '—';
      case 'exit_px': return fmt(row.trade_exit_price);
      case 'exit_reason': return row.trade_exit_reason || '—';
      case 'realized_pts': return fmt(row.trade_realized_points);
      case 'mfe': return fmt(row.trade_mfe_points);
      case 'mae': return fmt(row.trade_mae_points);
      case 'outcome': return row.trade_outcome || '—';
      case 'reason':
        return (row.direction === 'down' ? row.long_setup_reason : row.short_setup_reason) || '—';
      case 'prior_down_pts': return fmt(row.prior_session_down_pts);
      case 'prior_down_ratio': {
        const ratio = row.prior_down_vs_up_ratio;
        if (ratio === null || ratio === undefined) return '—';
        const color = ratio > 1.5 ? '#fca5a5' : ratio > 1.0 ? '#fcd34d' : '#86efac';
        return <span style={{ color, fontWeight: 700 }}>{fmt(ratio, 2)}</span>;
      }
      case 'start_pct_range': {
        const pct = row.start_pct_of_session_range;
        if (pct === null || pct === undefined) return '—';
        const color = pct < 0.25 ? '#fca5a5' : pct < 0.45 ? '#fcd34d' : '#86efac';
        return <span style={{ color }}>{(pct * 100).toFixed(1)}%</span>;
      }

      // ── Study mode columns ──
      case 'skew_passed': {
        if (row.skew_threshold_passed === null || row.skew_threshold_passed === undefined) return '—';
        return row.skew_threshold_passed
          ? <span style={{ color: '#86efac', fontWeight: 700 }}>✓</span>
          : <span style={{ color: '#fca5a5', fontWeight: 700 }}>✗</span>;
      }
      case 'target_price': return fmt(row.target_price);

      // Close columns: color green if > 0, red if < 0
      case 'fwd_30m_close':
      case 'fwd_60m_close':
      case 'fwd_90m_close':
      case 'fwd_120m_close':
      case 'fwd_180m_close':
      case 'fwd_eod_close': {
        const horizonKey = col.id === 'fwd_eod_close' ? 'eod' : col.id.replace('fwd_', '').replace('_close', '');
        const v = row.forward_outcomes?.[horizonKey]?.close_pts;
        if (v === null || v === undefined) return '—';
        const color = v > 0 ? '#86efac' : v < 0 ? '#fca5a5' : '#e5e7eb';
        return <span style={{ color, fontWeight: 600 }}>{fmt(v)}</span>;
      }

      // MFE columns: always positive favorable, show in green if > 0
      case 'fwd_30m_mfe':
      case 'fwd_60m_mfe':
      case 'fwd_90m_mfe':
      case 'fwd_120m_mfe':
      case 'fwd_180m_mfe':
      case 'fwd_eod_mfe': {
        const horizonKey = col.id === 'fwd_eod_mfe' ? 'eod' : col.id.replace('fwd_', '').replace('_mfe', '');
        const v = row.forward_outcomes?.[horizonKey]?.mfe_pts;
        if (v === null || v === undefined) return '—';
        const color = v > 0 ? '#86efac' : '#94a3b8';
        return <span style={{ color }}>{fmt(v)}</span>;
      }

      // MAE columns: adverse magnitude (positive number), show in amber/red proportional
      case 'fwd_30m_mae':
      case 'fwd_60m_mae':
      case 'fwd_90m_mae':
      case 'fwd_120m_mae':
      case 'fwd_180m_mae':
      case 'fwd_eod_mae': {
        const horizonKey = col.id === 'fwd_eod_mae' ? 'eod' : col.id.replace('fwd_', '').replace('_mae', '');
        const v = row.forward_outcomes?.[horizonKey]?.mae_pts;
        if (v === null || v === undefined) return '—';
        const color = v > 0 ? '#fca5a5' : '#94a3b8';
        return <span style={{ color }}>{fmt(v)}</span>;
      }

      // IV at entry (0DTE ATM) — plain numeric, no color prejudice
      case 'iv_atm_0dte': {
        const v = row.iv?.atm_0dte_pct;
        if (v === null || v === undefined) return '—';
        return <span>{fmt(v, 2)}</span>;
      }

      // Realized vs implied at 120m: |close| / implied_1sigma
      // Color green if <1 (realized tighter than priced), red if >=1
      case 'rvi_ratio_120m': {
        const v = row.realized_vs_implied?.['120m']?.close_over_1sigma;
        if (v === null || v === undefined) return '—';
        const color = v < 1.0 ? '#86efac' : v < 2.0 ? '#fcd34d' : '#fca5a5';
        return <span style={{ color, fontWeight: 600 }}>{fmt(v, 2)}</span>;
      }

      // Inside ±1σ at 120m: ✓ if price stayed within the implied band
      case 'rvi_inside_1s_120m': {
        const v = row.realized_vs_implied?.['120m']?.inside_1sigma;
        if (v === null || v === undefined) return '—';
        return v
          ? <span style={{ color: '#86efac', fontWeight: 700 }}>✓</span>
          : <span style={{ color: '#fca5a5', fontWeight: 700 }}>✗</span>;
      }

      // Hypothetical 120m iron condor strikes.
      // Short strikes colored (amber) — they're the ones the trade "stays below/above"
      // Long strikes faded — they're the defensive wings.
      case 'condor_short_put': {
        const v = row.hypothetical_condor_120m?.short_put_strike;
        if (v === null || v === undefined) return '—';
        return <span style={{ color: '#fcd34d', fontWeight: 600 }}>{fmt(v, 0)}</span>;
      }
      case 'condor_short_call': {
        const v = row.hypothetical_condor_120m?.short_call_strike;
        if (v === null || v === undefined) return '—';
        return <span style={{ color: '#fcd34d', fontWeight: 600 }}>{fmt(v, 0)}</span>;
      }
      case 'condor_long_put': {
        const v = row.hypothetical_condor_120m?.long_put_strike;
        if (v === null || v === undefined) return '—';
        return <span style={{ color: '#94a3b8' }}>{fmt(v, 0)}</span>;
      }
      case 'condor_long_call': {
        const v = row.hypothetical_condor_120m?.long_call_strike;
        if (v === null || v === undefined) return '—';
        return <span style={{ color: '#94a3b8' }}>{fmt(v, 0)}</span>;
      }

      default: return null;
    }
  };

  const visibleColumns = columns.filter(c => c.visible);

  return (
    <div className="table-wrap">
      <table className="results-table">
        <thead>
          <tr>
            {visibleColumns.map(col => {
              const isSortable = col.id !== 'select';
              const isSorted = sortConfig.key === col.id;
              
              return (
                <th 
                  key={col.id}
                  className={isSortable ? 'sortable-header' : ''}
                  onClick={() => isSortable && handleSort(col.id)}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    {col.label}
                    {isSorted && (
                      <span className="sort-indicator">
                        {sortConfig.direction === 'asc' ? ' ↑' : ' ↓'}
                      </span>
                    )}
                  </div>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {processedRows.map((row, idx) => {
            const key = rowKey(row, idx);
            return (
              <tr key={key} className={selectedRowKey === key ? 'selected-row' : ''}>
                {visibleColumns.map(col => (
                  <td key={col.id} className={col.className || ''}>
                    {renderCell(col, row, idx)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
});

export default ResultsTable;
