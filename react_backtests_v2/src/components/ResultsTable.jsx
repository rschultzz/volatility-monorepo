function fmt(value, digits = 2) {
  if (value === null || value === undefined || value === '') return '—';
  const num = Number(value);
  if (Number.isNaN(num)) return String(value);
  return num.toFixed(digits);
}

function rowKey(row, idx) {
  return `${row.trade_date}-${row.start_ts_utc}-${row.target_ts_utc}-${idx}`;
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
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
