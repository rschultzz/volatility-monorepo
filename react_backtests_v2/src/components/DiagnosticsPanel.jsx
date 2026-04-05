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

export default function DiagnosticsPanel({ diagnostics }) {
  if (!diagnostics) return null;

  const columnStats = diagnostics.column_stats || {};
  const columnRows = Object.entries(columnStats);
  const sampleRows = diagnostics.sample_qualifying_rows || [];

  return (
    <div className="diag-card">
      <div className="results-header">
        <div>
          <h2>Diagnostics</h2>
          <p>
            This tells us whether the issue is missing wall data, GEX threshold units, or touch logic being too strict.
          </p>
        </div>
      </div>

      <div className="diag-stat-grid">
        <StatCell label="Rows with any wall" value={diagnostics.rows_with_any_level ?? '—'} />
        <StatCell label="Rows meeting GEX threshold" value={diagnostics.rows_with_any_qualifying_level ?? '—'} />
        <StatCell label="Bars touching qualifying wall" value={diagnostics.bars_touching_qualifying_level ?? '—'} />
        <StatCell label="Start candidates" value={diagnostics.start_candidates ?? '—'} />
        <StatCell label="Within 1 pt of a qualifying wall" value={diagnostics.bars_within_1_pt ?? '—'} />
        <StatCell label="Within 5 pts of a qualifying wall" value={diagnostics.bars_within_5_pts ?? '—'} />
      </div>

      <div className="diag-two-col">
        <div className="diag-panel">
          <div className="diag-panel-title">Wall Column Stats</div>
          <div className="table-wrap diag-table-wrap">
            <table className="results-table diag-table">
              <thead>
                <tr>
                  <th>Column</th>
                  <th>Rows With Level</th>
                  <th>Rows ≥ Threshold</th>
                  <th>Max Abs GEX BN</th>
                </tr>
              </thead>
              <tbody>
                {columnRows.length ? (
                  columnRows.map(([label, stats]) => (
                    <tr key={label}>
                      <td>{label}</td>
                      <td>{stats.rows_with_level ?? '—'}</td>
                      <td>{stats.rows_meeting_gex_threshold ?? '—'}</td>
                      <td>{fmt(stats.max_abs_gex_bn)}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={4}>No diagnostic column stats yet.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="diag-panel">
          <div className="diag-panel-title">Sample Qualifying Rows</div>
          <div className="table-wrap diag-table-wrap">
            <table className="results-table diag-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Time</th>
                  <th>Close</th>
                  <th>Nearest Dist</th>
                  <th>Levels</th>
                </tr>
              </thead>
              <tbody>
                {sampleRows.length ? (
                  sampleRows.map((row, idx) => (
                    <tr key={`${row.trade_date}-${row.ts_pt}-${idx}`}>
                      <td>{row.trade_date}</td>
                      <td>{row.ts_pt}</td>
                      <td>{fmt(row.close)}</td>
                      <td>{fmt(row.nearest_distance)}</td>
                      <td>
                        <div className="diag-level-list">
                          {(row.levels || []).map((levelText, levelIdx) => (
                            <div key={levelIdx}>{levelText}</div>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={5}>No qualifying rows found for the current threshold.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
