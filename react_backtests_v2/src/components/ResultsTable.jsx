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
            <th>Trade Date</th>
            <th>Start Time (PT)</th>
            <th>Start Open</th>
            <th>Start Level</th>
            <th>Start GEX BN</th>
            <th>Target Time (PT)</th>
            <th>Target Open</th>
            <th>Target Level</th>
            <th>Target GEX BN</th>
            <th>Move Pts</th>
            <th>Bars</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => {
            const key = rowKey(row, idx);
            return (
              <tr key={key}>
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
                  <div>{row.start_ts_pt}</div>
                  <div className="subcell">{row.start_level_type}</div>
                </td>
                <td>{fmt(row.start_open)}</td>
                <td>{fmt(row.start_level)}</td>
                <td>{fmt(row.start_level_gex_bn)}</td>
                <td>
                  <div>{row.target_ts_pt}</div>
                  <div className="subcell">{row.target_level_type}</div>
                </td>
                <td>{fmt(row.target_open)}</td>
                <td>{fmt(row.target_level)}</td>
                <td>{fmt(row.target_level_gex_bn)}</td>
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
