import { useEffect, useMemo, useState, useRef } from 'react';
import SettingsModal from './components/SettingsModal';
import ResultsTable from './components/ResultsTable';
import DiagnosticsPanel from './components/DiagnosticsPanel';
import ColumnSettingsModal from './components/ColumnSettingsModal';
import TradeLog from './components/TradeLog';   // ← NEW

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
  maxPriorDownUpRatio: 2.0,
  maxStartPctOfRange: 0.20,
  minRangeProofMinutes: 20,
  maxMoveLossPct: 0.75,
  minMinutesAfterOpen: 15,
  longPutSkewMinDecreasePct: 80.0,
  longCallSkewMinIncreasePct: 30.0,
  maxMinutesBeforeClose: 45,
  longInitialStopPts: 10.0,
  longTrailActivateProfitPts: 20.0,
  longTrailingStopPts: 10.0,
  longTakeProfitPts: 35.0,
  bypassFilters: [],
  executionMode: 'managed',
  forwardHorizonsMinutes: [30, 60, 90, 120, 180],
  condorWingWidthPts: 10.0,
};

const DEFAULT_COLUMNS = [
  { id: 'select', label: 'Select', visible: true, alwaysVisible: true },
  { id: 'date', label: 'Date', visible: true },
  { id: 'direction', label: 'Dir', visible: true },
  { id: 'source_zone', label: 'Source Zone', visible: true },
  { id: 'zone_levels', label: 'Zone Levels', visible: true, className: 'wrap-cell' },
  { id: 'start_time', label: 'Start Time (PT)', visible: true },
  { id: 'start_open', label: 'Start Open', visible: true },
  { id: 'pivot_px', label: 'Pivot Px', visible: true },
  { id: 'target_time', label: 'Target Time (PT)', visible: true },
  { id: 'target_open', label: 'Target Open', visible: true },
  { id: 'target_level', label: 'Target Level', visible: true },
  { id: 'clean_space', label: 'Clean Space', visible: true },
  { id: 'move_pts', label: 'Move Pts', visible: true },
  { id: 'bars', label: 'Bars', visible: true },
  { id: 'consol_mins', label: 'Consol. Mins', visible: true },
  { id: 'setup', label: 'Setup', visible: true },
  { id: 'signal_time', label: 'Signal Time (PT)', visible: true },
  { id: 'signal_px', label: 'Signal Px', visible: true },
  { id: 'put_skew', label: 'Δ Put Skew %', visible: true },
  { id: 'call_skew', label: 'Δ Call Skew %', visible: true },
  { id: 'trade', label: 'Trade', visible: true },
  { id: 'range_high', label: 'Range High', visible: true },
  { id: 'range_low', label: 'Range Low', visible: true },
  { id: 'entry_band', label: 'Entry Band Floor', visible: true },
  { id: 'entry_time', label: 'Entry Time (PT)', visible: true },
  { id: 'entry_px', label: 'Entry Px', visible: true },
  { id: 'init_stop', label: 'Init Stop', visible: true },
  { id: 'take_profit', label: 'Take Profit', visible: true },
  { id: 'trailing_stop', label: 'Trailing Stop', visible: true },
  { id: 'exit_time', label: 'Exit Time (PT)', visible: true },
  { id: 'exit_px', label: 'Exit Px', visible: true },
  { id: 'exit_reason', label: 'Exit Reason', visible: true },
  { id: 'realized_pts', label: 'Realized Pts', visible: true },
  { id: 'mfe', label: 'MFE', visible: true },
  { id: 'mae', label: 'MAE', visible: true },
  { id: 'outcome', label: 'Outcome', visible: true },
  { id: 'reason', label: 'Reason', visible: true, className: 'wrap-cell' },
  { id: 'prior_down_pts', label: 'Prior Down (pts)', visible: true },
  { id: 'prior_down_ratio', label: 'Down/Up Ratio', visible: true },
  { id: 'start_pct_range', label: 'Start % of Range', visible: true },

  { id: 'skew_passed', label: 'Skew Passed', visible: false },
  { id: 'target_price', label: 'Target Px', visible: false },

  { id: 'fwd_30m_mfe',   label: 'MFE 30m',   visible: false, className: 'study-col' },
  { id: 'fwd_30m_mae',   label: 'MAE 30m',   visible: false, className: 'study-col' },
  { id: 'fwd_30m_close', label: 'Close 30m', visible: false, className: 'study-col' },

  { id: 'fwd_60m_mfe',   label: 'MFE 60m',   visible: false, className: 'study-col' },
  { id: 'fwd_60m_mae',   label: 'MAE 60m',   visible: false, className: 'study-col' },
  { id: 'fwd_60m_close', label: 'Close 60m', visible: false, className: 'study-col' },

  { id: 'fwd_90m_mfe',   label: 'MFE 90m',   visible: false, className: 'study-col' },
  { id: 'fwd_90m_mae',   label: 'MAE 90m',   visible: false, className: 'study-col' },
  { id: 'fwd_90m_close', label: 'Close 90m', visible: false, className: 'study-col' },

  { id: 'fwd_120m_mfe',   label: 'MFE 120m',   visible: false, className: 'study-col' },
  { id: 'fwd_120m_mae',   label: 'MAE 120m',   visible: false, className: 'study-col' },
  { id: 'fwd_120m_close', label: 'Close 120m', visible: false, className: 'study-col' },

  { id: 'fwd_180m_mfe',   label: 'MFE 180m',   visible: false, className: 'study-col' },
  { id: 'fwd_180m_mae',   label: 'MAE 180m',   visible: false, className: 'study-col' },
  { id: 'fwd_180m_close', label: 'Close 180m', visible: false, className: 'study-col' },

  { id: 'fwd_eod_mfe',   label: 'MFE EOD',   visible: false, className: 'study-col' },
  { id: 'fwd_eod_mae',   label: 'MAE EOD',   visible: false, className: 'study-col' },
  { id: 'fwd_eod_close', label: 'Close EOD', visible: false, className: 'study-col' },

  { id: 'iv_atm_0dte', label: 'IV ATM 0DTE', visible: false, className: 'study-col' },

  { id: 'target_spx_price', label: 'SPX @ Target', visible: false, className: 'study-col' },
  { id: 'minutes_to_close', label: 'Min Remaining', visible: false, className: 'study-col' },
  { id: 'skew_delta_put',   label: 'ΔPut Skew %',  visible: false, className: 'study-col' },
  { id: 'skew_delta_call',  label: 'ΔCall Skew %', visible: false, className: 'study-col' },

  { id: 'rvi_ratio_120m',     label: '|Close|/1σ 120m', visible: false, className: 'study-col' },
  { id: 'rvi_inside_1s_120m', label: 'Inside ±1σ 120m', visible: false, className: 'study-col' },

  { id: 'condor_short_put',  label: 'Short Put',  visible: false, className: 'study-col' },
  { id: 'condor_long_put',   label: 'Long Put',   visible: false, className: 'study-col' },
  { id: 'condor_short_call', label: 'Short Call', visible: false, className: 'study-col' },
  { id: 'condor_long_call',  label: 'Long Call',  visible: false, className: 'study-col' },
];

const MANAGED_ONLY_COLUMNS = new Set([
  'signal_time', 'signal_px', 'put_skew', 'call_skew',
  'trade', 'range_high', 'range_low', 'entry_band',
  'entry_time', 'entry_px',
  'init_stop', 'take_profit', 'trailing_stop',
  'exit_time', 'exit_px', 'exit_reason',
  'realized_pts', 'mfe', 'mae', 'outcome',
  'reason', 'consol_mins', 'setup',
]);

const STUDY_ONLY_COLUMNS = new Set([
  'skew_passed', 'target_price',
  'fwd_30m_mfe',  'fwd_30m_mae',  'fwd_30m_close',
  'fwd_60m_mfe',  'fwd_60m_mae',  'fwd_60m_close',
  'fwd_90m_mfe',  'fwd_90m_mae',  'fwd_90m_close',
  'fwd_120m_mfe', 'fwd_120m_mae', 'fwd_120m_close',
  'fwd_180m_mfe', 'fwd_180m_mae', 'fwd_180m_close',
  'fwd_eod_mfe',  'fwd_eod_mae',  'fwd_eod_close',
  'iv_atm_0dte',
  'target_spx_price',
  'minutes_to_close',
  'skew_delta_put', 'skew_delta_call',
  'rvi_ratio_120m', 'rvi_inside_1s_120m',
  'condor_short_put', 'condor_long_put', 'condor_short_call', 'condor_long_call',
]);

const COLUMNS_STORAGE_KEY = 'bt2-results-table-columns';

function slugify(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '');
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
    maxPriorDownUpRatio: Number(nextSettings.maxPriorDownUpRatio),
    maxStartPctOfRange: Number(nextSettings.maxStartPctOfRange),
    minRangeProofMinutes: Number(nextSettings.minRangeProofMinutes),
    maxMoveLossPct: Number(nextSettings.maxMoveLossPct),
    minMinutesAfterOpen: Number(nextSettings.minMinutesAfterOpen),
    longPutSkewMinDecreasePct: Number(nextSettings.longPutSkewMinDecreasePct),
    longCallSkewMinIncreasePct: Number(nextSettings.longCallSkewMinIncreasePct),
    maxMinutesBeforeClose: Number(nextSettings.maxMinutesBeforeClose),
    longInitialStopPts: Number(nextSettings.longInitialStopPts),
    longTrailActivateProfitPts: Number(nextSettings.longTrailActivateProfitPts),
    longTrailingStopPts: Number(nextSettings.longTrailingStopPts),
    longTakeProfitPts: Number(nextSettings.longTakeProfitPts),
    bypassFilters: nextSettings.bypassFilters || [],
    executionMode: nextSettings.executionMode || 'managed',
    forwardHorizonsMinutes: Array.isArray(nextSettings.forwardHorizonsMinutes)
      ? nextSettings.forwardHorizonsMinutes.map(Number).filter(n => Number.isFinite(n) && n > 0)
      : [30, 60, 90, 120, 180],
    condorWingWidthPts: Number.isFinite(Number(nextSettings.condorWingWidthPts)) && Number(nextSettings.condorWingWidthPts) > 0
      ? Number(nextSettings.condorWingWidthPts)
      : 10.0,
  };
}

export default function App() {
  const tableRef = useRef();
  const [strategies, setStrategies] = useState([]);
  const [selectedStrategyKey, setSelectedStrategyKey] = useState('up_move_short');
  const [settings, setSettings] = useState(FALLBACK_DEFAULT_SETTINGS);
  const [settingsDraft, setSettingsDraft] = useState(FALLBACK_DEFAULT_SETTINGS);
  const [strategyMetaDraft, setStrategyMetaDraft] = useState({
    displayName: '',
    strategyKey: '',
    notes: '',
  });
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isColumnsOpen, setIsColumnsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('diagnostics');

  const [columns, setColumns] = useState(() => {
    const saved = localStorage.getItem(COLUMNS_STORAGE_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        const merged = DEFAULT_COLUMNS.map(def => {
          const found = parsed.find(p => p.id === def.id);
          return found ? { ...def, ...found } : def;
        });
        const sorted = parsed.map(p => merged.find(m => m.id === p.id)).filter(Boolean);
        const missing = merged.filter(m => !sorted.find(s => s.id === m.id));
        return [...sorted, ...missing];
      } catch (e) {
        return DEFAULT_COLUMNS;
      }
    }
    return DEFAULT_COLUMNS;
  });

  const [rows, setRows] = useState([]);
  const [summary, setSummary] = useState(null);
  const [diagnostics, setDiagnostics] = useState(null);
  const [funnel, setFunnel] = useState([]);
  const [loading, setLoading] = useState(false);
  const [savingDefaults, setSavingDefaults] = useState(false);
  const [creatingStrategy, setCreatingStrategy] = useState(false);
  const [bootstrapping, setBootstrapping] = useState(true);
  const [error, setError] = useState('');
  const [selectedRowKey, setSelectedRowKey] = useState(null);

  useEffect(() => {
    localStorage.setItem(COLUMNS_STORAGE_KEY, JSON.stringify(columns));
  }, [columns]);

  const effectiveExecutionMode = settings?.executionMode || 'managed';
  const effectiveColumns = useMemo(() => {
    const isStudy = effectiveExecutionMode === 'study_target_hits';
    return columns.map(col => {
      if (isStudy && STUDY_ONLY_COLUMNS.has(col.id))    return { ...col, visible: true };
      if (isStudy && MANAGED_ONLY_COLUMNS.has(col.id))  return { ...col, visible: false };
      if (!isStudy && STUDY_ONLY_COLUMNS.has(col.id))   return { ...col, visible: false };
      return col;
    });
  }, [columns, effectiveExecutionMode]);

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
        setSettings(defaults);
        setSettingsDraft(defaults);
        setStrategyMetaDraft({
          displayName: defaultStrategy?.displayName || defaultStrategy?.label || '',
          strategyKey: defaultStrategy?.strategyKey || defaultKey,
          notes: defaultStrategy?.notes || '',
        });
      } catch (err) {
        if (!isMounted) return;
        setError(err.message || 'Could not load strategies');
      } finally {
        if (isMounted) setBootstrapping(false);
      }
    }

    loadStrategies();
    return () => { isMounted = false; };
  }, []);

  function updateDraft(name, value) {
    setSettingsDraft((prev) => ({ ...prev, [name]: value }));
  }

  function updateStrategyMeta(name, value) {
    setStrategyMetaDraft((prev) => {
      const next = { ...prev, [name]: value };
      if (name === 'displayName' && (!prev.strategyKey || prev.strategyKey === slugify(prev.displayName))) {
        next.strategyKey = slugify(value);
      }
      return next;
    });
  }

  function applyStrategyDefaults(strategyKey) {
    const nextStrategy = strategies.find((item) => item.key === strategyKey);
    setSelectedStrategyKey(strategyKey);
    if (nextStrategy) {
      const nextDefaults = nextStrategy?.defaults || FALLBACK_DEFAULT_SETTINGS;
      setSettings(nextDefaults);
      setSettingsDraft(nextDefaults);
      setStrategyMetaDraft({
        displayName: nextStrategy?.displayName || nextStrategy?.label || '',
        strategyKey: nextStrategy?.strategyKey || nextStrategy?.key || '',
        notes: nextStrategy?.notes || '',
      });
      setSelectedRowKey(null);
      setRows([]);
      setSummary(null);
      setDiagnostics(null);
    }
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

      setSettings(data.settings || payload);
      setSettingsDraft(data.settings || payload);
      setRows(data.rows || []);
      setSummary(data.summary || null);
      setDiagnostics(data.diagnostics || null);
      setFunnel(data.funnel || []);
      setSelectedRowKey(null);
      setIsSettingsOpen(false);
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
          name: strategyMetaDraft.displayName,
          notes: strategyMetaDraft.notes,
          params,
        }),
      });

      const data = await res.json();
      if (!res.ok || !data.ok) {
        throw new Error(data.error || 'Could not save strategy');
      }

      const savedStrategy = data.strategy;
      setStrategies((prev) => prev.map((item) => (item.key === savedStrategy.key ? savedStrategy : item)));
    } catch (err) {
      setError(err.message || 'Could not save strategy');
    } finally {
      setSavingDefaults(false);
    }
  }

  async function handleCreateStrategy() {
    setCreatingStrategy(true);
    setError('');

    try {
      const params = normalizeNumericSettings(settingsDraft);

      const res = await fetch('/api/backtests-v2/strategy-create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          baseStrategyKey: selectedStrategyKey,
          name: strategyMetaDraft.displayName,
          strategyKey: strategyMetaDraft.strategyKey,
          notes: strategyMetaDraft.notes,
          params,
        }),
      });

      const data = await res.json();
      if (!res.ok || !data.ok) {
        throw new Error(data.error || 'Could not create strategy');
      }

      setStrategies((prev) => [...prev, data.strategy]);
      setSelectedStrategyKey(data.strategy.key);
      setIsSettingsOpen(false);
    } catch (err) {
      setError(err.message || 'Could not create strategy');
    } finally {
      setCreatingStrategy(false);
    }
  }

  async function handleSelectRow(row, idx) {
    const key = `${row.trade_date}-${row.start_ts_utc}-${row.target_ts_utc}-${idx}`;
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
          signal_ts_pt: (row.direction === 'down' ? row.long_signal_ts_pt : row.short_signal_ts_pt) || '',
          trade_entry_ts_pt: row.trade_entry_ts_pt || '',
          trade_exit_ts_pt: row.trade_exit_ts_pt || '',
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

  // Trade Log tab doesn't use the strategy toolbar, so hide it when active
  const showStrategyToolbar = activeTab !== 'trade_log';

  return (
    <div className="page">
      <div className="workspace-card">
        {error ? <div className="error-banner">{error}</div> : null}

        {/* Strategy selector + action buttons — hidden on Trade Log tab */}
        {showStrategyToolbar && (
          <div className="results-header" style={{ alignItems: 'center', marginBottom: 0, padding: '4px 0' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, flex: 1 }}>
              <select
                value={selectedStrategyKey}
                onChange={(e) => applyStrategyDefaults(e.target.value)}
                disabled={bootstrapping || loading}
                style={{
                  backgroundColor: '#111827',
                  border: '1px solid #374151',
                  color: 'white',
                  borderRadius: '6px',
                  padding: '6px 10px',
                  fontSize: '13px'
                }}
              >
                {strategies.map((s) => (
                  <option key={s.key} value={s.key}>{s.displayName || s.label}</option>
                ))}
              </select>
            </div>

            <div style={{ display: 'flex', gap: 8 }}>
              <button className="ghost-button" style={{ padding: '6px 10px', fontSize: '12px' }} onClick={() => setIsColumnsOpen(true)}>
                ⚙️ Columns
              </button>
              <button className="ghost-button" style={{ padding: '6px 10px', fontSize: '12px' }} onClick={() => setIsSettingsOpen(true)}>
                Settings
              </button>
              <button className="ghost-button" style={{ padding: '6px 10px', fontSize: '12px' }} onClick={handleSaveDefaults} disabled={savingDefaults}>
                {savingDefaults ? 'Saving…' : 'Save'}
              </button>
              <button className="primary-button" style={{ padding: '6px 10px', fontSize: '12px' }} onClick={() => runScan(settings)} disabled={loading}>
                {loading ? 'Running…' : 'Run Scan'}
              </button>
            </div>
          </div>
        )}

        {/* Tab bar */}
        <div className="tab-bar">
          <button
            className={`tab-button ${activeTab === 'diagnostics' ? 'active' : ''}`}
            onClick={() => setActiveTab('diagnostics')}
          >
            Diagnostics
          </button>
          <button
            className={`tab-button ${activeTab === 'instances' ? 'active' : ''}`}
            onClick={() => setActiveTab('instances')}
          >
            Instances
            {rows.length > 0 && (
              <span className="tab-badge">{rows.length}</span>
            )}
          </button>
          <button
            className={`tab-button ${activeTab === 'trade_log' ? 'active' : ''}`}
            onClick={() => setActiveTab('trade_log')}
          >
            Trade Log
          </button>
        </div>

        {activeTab === 'diagnostics' && (
          <DiagnosticsPanel
            diagnostics={diagnostics}
            rows={rows}
            funnel={funnel}
            executionMode={effectiveExecutionMode}
          />
        )}

        {activeTab === 'instances' && (
          <div className="results-card" style={{ flex: 1 }}>
            <div className="results-header">
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
                <h2 style={{ fontSize: 16 }}>Instances</h2>
                <span style={{ color: '#64748b', fontSize: 12 }}>{rows.length} trades found</span>
                <button
                  className="ghost-button"
                  style={{ padding: '4px 10px', fontSize: '12px', marginLeft: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}
                  onClick={() => tableRef.current?.downloadCSV()}
                  disabled={!rows.length}
                >
                  📥 CSV
                </button>
              </div>
            </div>

            <ResultsTable
              ref={tableRef}
              rows={rows}
              selectedRowKey={selectedRowKey}
              onSelectRow={handleSelectRow}
              columns={effectiveColumns}
            />
          </div>
        )}

        {/* Trade Log tab — TradeLog is fully self-contained */}
        {activeTab === 'trade_log' && (
          <TradeLog />
        )}
      </div>

      <SettingsModal
        isOpen={isSettingsOpen}
        settingsDraft={settingsDraft}
        strategyMetaDraft={strategyMetaDraft}
        creatingStrategy={creatingStrategy}
        onStrategyMetaChange={updateStrategyMeta}
        onCreateStrategy={handleCreateStrategy}
        onChange={updateDraft}
        onClose={() => setIsSettingsOpen(false)}
        onRun={() => runScan(settingsDraft)}
        bypassFilters={settingsDraft.bypassFilters || []}
        onToggleBypass={(stageKey) => {
          const current = settingsDraft.bypassFilters || [];
          const next = current.includes(stageKey)
            ? current.filter(k => k !== stageKey)
            : [...current, stageKey];
          updateDraft('bypassFilters', next);
        }}
      />

      <ColumnSettingsModal
        isOpen={isColumnsOpen}
        onClose={() => setIsColumnsOpen(false)}
        columns={columns}
        onUpdateColumns={setColumns}
      />
    </div>
  );
}
