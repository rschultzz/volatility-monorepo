import { useEffect, useMemo, useState } from 'react';
import SettingsModal from './components/SettingsModal';
import ResultsTable from './components/ResultsTable';
import DiagnosticsPanel from './components/DiagnosticsPanel';

function isoDateOffset(days) {
  const d = new Date();
  d.setDate(d.getDate() + days);
  return d.toISOString().slice(0, 10);
}

const FALLBACK_DEFAULT_SETTINGS = {
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
  entryWithinTopPts: 2,
  entrySearchWindowMinutes: 30,
  initialStopPts: 6,
  trailActivateProfitPts: 10,
  trailingStopPts: 6,
  takeProfitPts: 20,
};

function rowKey(row, idx) {
  return `${row.trade_date}-${row.start_ts_utc}-${row.target_ts_utc}-${idx}`;
}

function normalizeNumericSettings(nextSettings) {
  return {
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
    entryWithinTopPts: Number(nextSettings.entryWithinTopPts),
    entrySearchWindowMinutes: Number(nextSettings.entrySearchWindowMinutes),
    initialStopPts: Number(nextSettings.initialStopPts),
    trailActivateProfitPts: Number(nextSettings.trailActivateProfitPts),
    trailingStopPts: Number(nextSettings.trailingStopPts),
    takeProfitPts: Number(nextSettings.takeProfitPts),
  };
}

function summaryText(settings) {
  if (!settings) return 'No settings loaded';

  return [
    `${settings.startDate} → ${settings.endDate}`,
    `min GEX ${settings.minLevelGexBn} BN`,
    `merge ${settings.zoneMergeDistancePts} pts`,
    `clean move ${settings.minCleanMovePoints} pts`,
    `target ±${settings.targetProximityPts}`,
    `consolidation ${settings.consolidationWindowMinutes}m`,
    `put ≥ ${settings.shortPutSkewIncreasePct}%`,
    `call ≤ ${settings.shortCallSkewMaxPct}%`,
    `entry top ${settings.entryWithinTopPts} pts`,
    `entry window ${settings.entrySearchWindowMinutes}m`,
    `stop ${settings.initialStopPts}`,
    `trail on +${settings.trailActivateProfitPts}`,
    `trail ${settings.trailingStopPts}`,
    `tp ${settings.takeProfitPts}`,
  ].join(' | ');
}

export default function App() {
  const [strategies, setStrategies] = useState([]);
  const [selectedStrategyKey, setSelectedStrategyKey] = useState('up_move_short');
  const [settings, setSettings] = useState(FALLBACK_DEFAULT_SETTINGS);
  const [settingsDraft, setSettingsDraft] = useState(FALLBACK_DEFAULT_SETTINGS);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [rows, setRows] = useState([]);
  const [summary, setSummary] = useState(null);
  const [diagnostics, setDiagnostics] = useState(null);
  const [sourceView, setSourceView] = useState('es_minutes_with_features_bt');
  const [loading, setLoading] = useState(false);
  const [savingDefaults, setSavingDefaults] = useState(false);
  const [bootstrapping, setBootstrapping] = useState(true);
  const [error, setError] = useState('');
  const [selectedRowKey, setSelectedRowKey] = useState(null);
  const [activeStrategyMeta, setActiveStrategyMeta] = useState(null);

  const selectedStrategy = useMemo(() => {
    return strategies.find((item) => item.key === selectedStrategyKey) || activeStrategyMeta || null;
  }, [activeStrategyMeta, selectedStrategyKey, strategies]);

  useEffect(() => {
    let isMounted = true;

    async function loadStrategies() {
      setBootstrapping(true);
      setError('');

      try {
        const res = await fetch('/api/backtests-v2/strategies');
        const data = await res.json();

        if (!res.ok || !data.ok) {
          throw new Error(data.error || 'Could not load strategies');
        }

        const nextStrategies = Array.isArray(data.strategies) ? data.strategies : [];
        const defaultKey = data.defaultStrategyKey || nextStrategies[0]?.key || 'up_move_short';
        const defaultStrategy = nextStrategies.find((item) => item.key === defaultKey) || nextStrategies[0] || null;
        const defaults = defaultStrategy?.defaults || FALLBACK_DEFAULT_SETTINGS;

        if (!isMounted) return;

        setStrategies(nextStrategies);
        setSelectedStrategyKey(defaultKey);
        setActiveStrategyMeta(defaultStrategy);
        setSettings(defaults);
        setSettingsDraft(defaults);
      } catch (err) {
        if (!isMounted) return;
        setError(err.message || 'Could not load strategies');
      } finally {
        if (isMounted) {
          setBootstrapping(false);
        }
      }
    }

    loadStrategies();
    return () => {
      isMounted = false;
    };
  }, []);

  const cards = useMemo(() => {
    return [
      { label: 'Strategy', value: selectedStrategy?.label || '—' },
      { label: 'Strategy ID', value: selectedStrategy?.strategyId ?? '—' },
      { label: 'Source View', value: sourceView || '—' },
      { label: 'Instances Found', value: summary?.instances_found ?? rows.length },
      { label: 'Bars Scanned', value: summary?.bars_scanned ?? '—' },
      { label: 'Executed Trades', value: summary?.executed_short_trades ?? '—' },
    ];
  }, [rows.length, selectedStrategy, sourceView, summary]);

  function updateDraft(name, value) {
    setSettingsDraft((prev) => ({ ...prev, [name]: value }));
  }

  function applyStrategyDefaults(strategyKey) {
    const nextStrategy = strategies.find((item) => item.key === strategyKey);
    if (!nextStrategy) {
      setSelectedStrategyKey(strategyKey);
      return;
    }

    const nextDefaults = nextStrategy.defaults || FALLBACK_DEFAULT_SETTINGS;
    setSelectedStrategyKey(strategyKey);
    setActiveStrategyMeta(nextStrategy);
    setSettings(nextDefaults);
    setSettingsDraft(nextDefaults);
    setSelectedRowKey(null);
    setRows([]);
    setSummary(null);
    setDiagnostics(null);
  }

  async function runScan(nextSettings = settingsDraft) {
    setLoading(true);
    setError('');

    try {
      const payload = normalizeNumericSettings({
        ...nextSettings,
        strategyKey: selectedStrategyKey,
      });

      const res = await fetch('/api/backtests-v2/gex-moves', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok || !data.ok) {
        throw new Error(data.error || 'Scan failed');
      }

      const normalizedSettings = data.settings || payload;

      setSettings(normalizedSettings);
      setSettingsDraft(normalizedSettings);
      setRows(data.rows || []);
      setSummary(data.summary || null);
      setDiagnostics(data.diagnostics || null);
      setSourceView(data.sourceView || 'es_minutes_with_features_bt');
      setSelectedRowKey(null);
      setIsSettingsOpen(false);
      setActiveStrategyMeta(data.strategy || selectedStrategy || null);
      setStrategies((prev) => prev.map((item) => (item.key === data.strategy?.key ? data.strategy : item)));
    } catch (err) {
      setError(err.message || 'Scan failed');
    } finally {
      setLoading(false);
    }
  }

  async function handleSaveDefaults() {
    setSavingDefaults(true);
    setError('');

    try {
      const params = normalizeNumericSettings(settingsDraft);

      const res = await fetch('/api/backtests-v2/strategy-defaults', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategyKey: selectedStrategyKey,
          params,
        }),
      });

      const data = await res.json();
      if (!res.ok || !data.ok) {
        throw new Error(data.error || 'Could not save strategy defaults');
      }

      const savedStrategy = data.strategy;
      setActiveStrategyMeta(savedStrategy);
      setStrategies((prev) => prev.map((item) => (item.key === savedStrategy.key ? savedStrategy : item)));
      setSettings(savedStrategy.defaults || settingsDraft);
      setSettingsDraft(savedStrategy.defaults || settingsDraft);
    } catch (err) {
      setError(err.message || 'Could not save strategy defaults');
    } finally {
      setSavingDefaults(false);
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
          trade_entry_ts_pt: row.trade_entry_ts_pt || '',
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
              {selectedStrategy?.description || 'Loading strategy library…'}
            </p>
          </div>

          <div className="header-actions" style={{ alignItems: 'stretch' }}>
            <div style={{ minWidth: 260, display: 'flex', flexDirection: 'column', gap: 6 }}>
              <div style={{ color: '#9ca3af', fontSize: 12 }}>Strategy</div>
              <select
                value={selectedStrategyKey}
                onChange={(e) => applyStrategyDefaults(e.target.value)}
                disabled={bootstrapping || loading || savingDefaults}
                style={{
                  minWidth: 260,
                  backgroundColor: '#111827',
                  border: '1px solid #374151',
                  color: 'white',
                  borderRadius: '8px',
                  padding: '10px 12px',
                }}
              >
                {strategies.map((strategy) => (
                  <option key={strategy.key} value={strategy.key}>
                    {strategy.label}
                  </option>
                ))}
              </select>
            </div>

            <button className="ghost-button" onClick={() => setIsSettingsOpen(true)} disabled={bootstrapping}>
              Settings
            </button>
            <button className="ghost-button" onClick={handleSaveDefaults} disabled={bootstrapping || savingDefaults}>
              {savingDefaults ? 'Saving…' : 'Save settings to strategy'}
            </button>
            <button className="primary-button" onClick={() => runScan(settings)} disabled={loading || bootstrapping}>
              {loading ? 'Running…' : 'Run Scan'}
            </button>
          </div>
        </div>

        <div className="toolbar-row">
          {(selectedStrategy?.badges || []).map((badge) => (
            <div className="pill" key={badge}>{badge}</div>
          ))}
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
                Iteration 3 loads saved parameter defaults from bt_strategies and lets you explicitly persist the current settings back to the selected strategy.
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
