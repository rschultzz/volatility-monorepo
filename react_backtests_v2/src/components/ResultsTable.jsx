import React, { useMemo, useState } from 'react';

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
  setup: 'short_setup_found',
  signal_time: 'short_signal_ts_pt',
  signal_px: 'short_signal_price',
  put_skew: 'short_signal_delta_put_skew_pct',
  call_skew: 'short_signal_delta_call_skew_pct',
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
  reason: 'short_setup_reason'
};

export default function ResultsTable({ rows, selectedRowKey, onSelectRow, columns }) {
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
          const aVal = a[dataKey];
          const bVal = b[dataKey];

          if (aVal === bVal) return 0;
          if (aVal === null || aVal === undefined) return 1;
          if (bVal === null || bVal === undefined) return -1;

          const comparison = aVal < bVal ? -1 : 1;
          return sortConfig.direction === 'asc' ? comparison : -comparison;
        });
      }
    }

    return result;
  }, [rows, sortConfig]);

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
          <span className={`setup-chip ${row.short_setup_found ? 'hit' : 'miss'}`}>
            {setupLabel(row)}
          </span>
        );
      case 'signal_time': return row.short_signal_ts_pt || '—';
      case 'signal_px': return fmt(row.short_signal_price);
      case 'put_skew': return fmt(row.short_signal_delta_put_skew_pct);
      case 'call_skew': return fmt(row.short_signal_delta_call_skew_pct);
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
      case 'reason': return row.short_setup_reason || '—';
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
}
