import { useMemo, useState } from 'react';
import SettingsModal from './components/SettingsModal';
import ResultsTable from './components/ResultsTable';
import DiagnosticsPanel from './components/DiagnosticsPanel';

function isoDateOffset(days) {
  const d = new Date();
  d.setDate(d.getDate() + days);
  return d.toISOString().slice(0, 10);
}

const DEFAULT_SETTINGS = {
  startDate: isoDateOffset(-14),
  endDate: isoDateOffset(0),
  minLevelGexBn: 50,
  zoneMergeDistancePts: 10,
  minCleanMovePoints: 20,
  targetProximityPts: 5,
  maxZoneBreachPts: 5,
  pivotStrengthBars: 3,
  levelFamily: 'primary',
  maxResults: 2500,
  consolidationWindowMinutes: 15,
  shortPutSkewIncreasePct: 80,
  shortCallSkewMaxPct: 30,
  entryRangeTopPts: 2,
  initialStopPts: 6,
  trailActivationProfitPts: 10,
  trailingStopPts: 6,
  takeProfitPts: 20,
};

function summaryText(settings) {
  return [
    `${settings.startDate} → ${settings.endDate}`,
    `min GEX ${settings.minLevelGexBn} BN`,
    `merge ${settings.zoneMergeDistancePts} pts`,
    `clean move ${settings.minCleanMovePoints} pts`,
    `target ±${settings.targetProximityPts}`,
    `consolidation ${settings.consolidationWindowMinutes}m`,
    `put ≥ ${settings.shortPutSkewIncreasePct}%`,
    `call ≤ ${settings.shortCallSkewMaxPct}%`,
    `entry top ${settings.entryRangeTopPts} pts`,
    `stop ${settings.initialStopPts}`,
    `trail on +${settings.trailActivationProfitPts}`,
    `trail ${settings.trailingStopPts}`,
    `tp ${settings.takeProfitPts}`,
  ].join(' | ');
}

function rowKey(row, idx) {
  return `${row.trade_date}-${row.start_ts_utc}-${row.target_ts_utc}-${idx}`;
}

export default function App() {
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [settingsDraft, setSettingsDraft] = useState(DEFAULT_SETTINGS);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [rows, setRows] = useState([]);
  const [summary, setSummary] = useState(null);
  const [diagnostics, setDiagnostics] = useState(null);
  const [sourceView, setSourceView] = useState('es_minutes_with_features_bt');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedRowKey, setSelectedRowKey] = useState(null);

  const cards = useMemo(() => {
    return [
      { label: 'Source View', value: sourceView || '—' },
      { label: 'Instances Found', value: summary?.instances_found ?? rows.length },
      { label: 'Bars Scanned', value: summary?.bars_scanned ?? '—' },
      { label: 'Zones Built', value: summary?.zones_total ?? '—' },
      { label: 'Up Short Setups', value: summary?.up_short_setups_found ?? '—' },
      { label: 'Executed Shorts', value: summary?.executed_short_trades ?? '—' },
    ];
  }, [rows.length, sourceView, summary]);

  function updateDraft(name, value) {
    setSettingsDraft((prev) => ({ ...prev, [name]: value }));
  }

  async function runScan(nextSettings = settingsDraft) {
    setLoading(true);
    setError('');

    try {
      const payload = {
        ...nextSettings,
        minLevelGexBn: Number(nextSettings.minLevelGexBn),
        zoneMergeDistancePts: Number(nextSettings.zoneMergeDistancePts),
        minCleanMovePoints: Number(nextSettings.minCleanMovePoints),
        targetProximityPts: Number(nextSettings.targetProximityPts),
        maxZoneBreachPts: Number(nextSettings.maxZoneBreachPts),
        pivotStrengthBars: Number(nextSettings.pivotStrengthBars),
        maxResults: Number(nextSettings.maxResults),
        consolidationWindowMinutes: Number(nextSettings.consolidationWindowMinutes),
        shortPutSkewIncreasePct: Number(nextSettings.shortPutSkewIncreasePct),
        shortCallSkewMaxPct: Number(nextSettings.shortCallSkewMaxPct),
        entryRangeTopPts: Number(nextSettings.entryRangeTopPts),
        initialStopPts: Number(nextSettings.initialStopPts),
        trailActivationProfitPts: Number(nextSettings.trailActivationProfitPts),
        trailingStopPts: Number(nextSettings.trailingStopPts),
        takeProfitPts: Number(nextSettings.takeProfitPts),
      };

      const res = await fetch('/api/backtests-v2/gex-moves', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok || !data.ok) {
        throw new Error(data.error || 'Scan failed');
      }

      setSettings(payload);
      setSettingsDraft(payload);
      setRows(data.rows || []);
      setSummary(data.summary || null);
      setDiagnostics(data.diagnostics || null);
      setSourceView(data.sourceView || 'es_minutes_with_features_bt');
      setSelectedRowKey(null);
      setIsSettingsOpen(false);
    } catch (err) {
      setError(err.message || 'Scan failed');
    } finally {
      setLoading(false);
    }
  }

  async function handleSelectRow(row, idx) {
    const key = rowKey(row, idx);
    setSelectedRowKey(key);
    setError('');

    try {
      const res = await fetch('/api/backtests-v2/select-trade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          trade_date: row.trade_date,
          start_ts_pt: row.start_ts_pt,
          target_ts_pt: row.target_ts_pt,
          signal_ts_pt: row.short_signal_ts_pt || '',
        }),
      });

      const data = await res.json();
      if (!res.ok || !data.ok) {
        throw new Error(data.error || 'Could not select trade');
      }
    } catch (err) {
      setError(err.message || 'Could not select trade');
    }
  }

  return (
    <div className="page">
      <div className="workspace-card">
        <div className="workspace-header">
          <div>
            <div className="eyebrow">Surface Dynamics</div>
            <h1>Backtests</h1>
            <p className="lead">
              Zone-based scan plus up-move short setup and trade execution: source zone → last pivot → target arrival → consolidation → skew trigger → entry/stop/target simulation.
            </p>
          </div>

          <div className="header-actions">
            <button className="ghost-button" onClick={() => setIsSettingsOpen(true)}>
              Settings
            </button>
            <button className="primary-button" onClick={() => runScan(settings)} disabled={loading}>
              {loading ? 'Running…' : 'Run Scan'}
            </button>
          </div>
        </div>

        <div className="toolbar-row">
          <div className="pill">RTH only</div>
          <div className="pill">Same day only</div>
          <div className="pill">Transitive GEX zones</div>
          <div className="pill">Last pivot inside source zone</div>
          <div className="pill">Up short setup near target</div>
          <div className="pill">Simulated short entry + exits</div>
          <div className="pill pill-wide">{summaryText(settings)}</div>
        </div>

        {error ? <div className="error-banner">{error}</div> : null}

        <div className="status-grid">
          {cards.map((card) => (
            <div className="status-card" key={card.label}>
              <div className="status-label">{card.label}</div>
              <div className="status-value">{card.value}</div>
            </div>
          ))}
        </div>

        <DiagnosticsPanel diagnostics={diagnostics} />

        <div className="results-card">
          <div className="results-header">
            <div>
              <h2>Instances</h2>
              <p>
                Up moves now include short setup detection plus simulated execution, stop management, trailing stop activation, and take-profit handling.
              </p>
            </div>
          </div>

          <ResultsTable
            rows={rows}
            selectedRowKey={selectedRowKey}
            onSelectRow={handleSelectRow}
          />
        </div>
      </div>

      <SettingsModal
        isOpen={isSettingsOpen}
        settingsDraft={settingsDraft}
        onChange={updateDraft}
        onClose={() => setIsSettingsOpen(false)}
        onRun={() => runScan(settingsDraft)}
      />
    </div>
  );
}