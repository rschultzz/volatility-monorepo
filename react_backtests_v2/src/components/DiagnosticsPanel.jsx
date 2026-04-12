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

  const sampleZones = diagnostics.sample_zones || [];
  const sampleResults = diagnostics.sample_results || [];
  const sampleShortSetups = diagnostics.sample_short_setups || [];

  return (
    <div className="diag-card">
      <div className="results-header">
        <div>
          <h2>Diagnostics</h2>
          <p>
            This version diagnoses both the zone scan and the new up-move short-setup logic near the target level.
          </p>
        </div>
      </div>

      <div className="diag-stat-grid">
        <StatCell label="Bars total" value={diagnostics.bars_total ?? '—'} />
        <StatCell label="Days total" value={diagnostics.days_total ?? '—'} />
        <StatCell label="Qualifying levels seen" value={diagnostics.qualifying_levels_seen ?? '—'} />
        <StatCell label="Zones built" value={diagnostics.zones_total ?? '—'} />
        <StatCell label="Clean-target zones" value={diagnostics.source_zones_with_clean_targets ?? '—'} />
        <StatCell label="Zone episodes considered" value={diagnostics.zone_episodes_considered ?? '—'} />
        <StatCell label="Valid instances" value={diagnostics.valid_instances ?? '—'} />
        <StatCell label="Up short setups" value={diagnostics.up_short_setups_found ?? '—'} />
      </div>

      <div className="diag-two-col">
        <div className="diag-panel">
          <div className="diag-panel-title">Sample Zones</div>
          <div className="table-wrap diag-table-wrap">
            <table className="results-table diag-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Range</th>
                  <th>Levels</th>
                  <th>Width</th>
                  <th>Count</th>
                  <th>Max Abs GEX BN</th>
                </tr>
              </thead>
              <tbody>
                {sampleZones.length ? (
                  sampleZones.map((row, idx) => (
                    <tr key={`${row.trade_date}-${row.range}-${idx}`}>
                      <td>{row.trade_date}</td>
                      <td>{row.range}</td>
                      <td className="wrap-cell">{row.levels}</td>
                      <td>{fmt(row.width)}</td>
                      <td>{row.count}</td>
                      <td>{fmt(row.max_abs_gex_bn)}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={6}>No zones found.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="diag-panel">
          <div className="diag-panel-title">Sample Results</div>
          <div className="table-wrap diag-table-wrap">
            <table className="results-table diag-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Dir</th>
                  <th>Source Zone</th>
                  <th>Target</th>
                  <th>Start</th>
                  <th>Target Time</th>
                  <th>Clean Space</th>
                </tr>
              </thead>
              <tbody>
                {sampleResults.length ? (
                  sampleResults.map((row, idx) => (
                    <tr key={`${row.trade_date}-${row.start_ts_pt}-${idx}`}>
                      <td>{row.trade_date}</td>
                      <td>{row.direction}</td>
                      <td>{row.source_zone}</td>
                      <td>{fmt(row.target_level)}</td>
                      <td>{row.start_ts_pt}</td>
                      <td>{row.target_ts_pt}</td>
                      <td>{fmt(row.clean_space_points)}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={7}>No valid instances found yet.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="diag-panel">
        <div className="diag-panel-title">Sample Up Short Setups</div>
        <div className="table-wrap diag-table-wrap">
          <table className="results-table diag-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Start</th>
                <th>Target</th>
                <th>Signal</th>
                <th>Target Level</th>
                <th>Δ Put Skew %</th>
                <th>Δ Call Skew %</th>
              </tr>
            </thead>
            <tbody>
              {sampleShortSetups.length ? (
                sampleShortSetups.map((row, idx) => (
                  <tr key={`${row.trade_date}-${row.signal_ts_pt}-${idx}`}>
                    <td>{row.trade_date}</td>
                    <td>{row.start_ts_pt}</td>
                    <td>{row.target_ts_pt}</td>
                    <td>{row.signal_ts_pt}</td>
                    <td>{fmt(row.target_level)}</td>
                    <td>{fmt(row.delta_put_skew_pct)}</td>
                    <td>{fmt(row.delta_call_skew_pct)}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7}>No up short setups found yet.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}