// Shared column definitions for the Backtests V2 results table.
// Used by both the live "Instances" tab in App.jsx and the
// "Saved Scans" tab.

// ──────────────────────────────────────────────────────────────────────
// Column filter types
// ──────────────────────────────────────────────────────────────────────
// Each filterable column declares one of: 'numeric', 'date', 'categorical'.
// Columns omitted from this map are not filterable (e.g. composite cells
// like 'select' or 'source_zone' that mix multiple values).
//
// Exposed as a const map rather than per-column property so adding a new
// column to DEFAULT_COLUMNS doesn't require touching this file unless the
// column should be filterable.

export const COLUMN_FILTER_TYPES = {
  // Date
  date: 'date',

  // Categorical (text/enum)
  direction: 'categorical',
  zone_levels: 'categorical',
  setup: 'categorical',
  trade: 'categorical',
  exit_reason: 'categorical',
  outcome: 'categorical',
  reason: 'categorical',
  skew_passed: 'categorical',
  target_gamma_regime: 'categorical',
  target_gamma_regime_all_exp: 'categorical',
  source_zone_gamma_regime: 'categorical',
  source_zone_gamma_regime_all_exp: 'categorical',
  rvi_inside_1s_120m: 'categorical',
  rvi_inside_1s_to_close: 'categorical',

  // Numeric — measurements
  start_open: 'numeric',
  pivot_px: 'numeric',
  target_open: 'numeric',
  target_level: 'numeric',
  target_level_gex: 'numeric',
  target_level_gex_all_exp: 'numeric',
  source_zone_signed_gex: 'numeric',
  source_zone_signed_gex_all_exp: 'numeric',
  clean_space: 'numeric',
  move_pts: 'numeric',
  bars: 'numeric',
  consol_mins: 'numeric',
  signal_px: 'numeric',
  put_skew: 'numeric',
  call_skew: 'numeric',
  range_high: 'numeric',
  range_low: 'numeric',
  entry_band: 'numeric',
  entry_px: 'numeric',
  init_stop: 'numeric',
  take_profit: 'numeric',
  trailing_stop: 'numeric',
  exit_px: 'numeric',
  realized_pts: 'numeric',
  mfe: 'numeric',
  mae: 'numeric',
  start_pct_range: 'numeric',
  target_price: 'numeric',
  iv_atm_0dte: 'numeric',
  target_spx_price: 'numeric',
  minutes_to_close: 'numeric',
  skew_delta_put: 'numeric',
  skew_delta_call: 'numeric',
  rvi_ratio_120m: 'numeric',
  rvi_ratio_to_close: 'numeric',
  condor_short_put: 'numeric',
  condor_long_put: 'numeric',
  condor_short_call: 'numeric',
  condor_long_call: 'numeric',
  condor_to_close_short_put: 'numeric',
  condor_to_close_long_put: 'numeric',
  condor_to_close_short_call: 'numeric',
  condor_to_close_long_call: 'numeric',
  // Forward outcomes (all numeric)
  fwd_30m_mfe: 'numeric', fwd_30m_mae: 'numeric', fwd_30m_close: 'numeric',
  fwd_60m_mfe: 'numeric', fwd_60m_mae: 'numeric', fwd_60m_close: 'numeric',
  fwd_90m_mfe: 'numeric', fwd_90m_mae: 'numeric', fwd_90m_close: 'numeric',
  fwd_120m_mfe: 'numeric', fwd_120m_mae: 'numeric', fwd_120m_close: 'numeric',
  fwd_180m_mfe: 'numeric', fwd_180m_mae: 'numeric', fwd_180m_close: 'numeric',
  fwd_eod_mfe: 'numeric', fwd_eod_mae: 'numeric', fwd_eod_close: 'numeric',
}

// Columns intentionally NOT filterable (composite cells, etc.)
// Listed here so it's visible at a glance.
export const COLUMNS_NOT_FILTERABLE = new Set([
  'select',         // checkbox cell
  'source_zone',    // composite "low - high"
  'start_time',     // includes context suffix
  'target_time',    // includes target_zone_range
  'signal_time',    // direction-dependent
  'entry_time',     // managed-mode timestamp
  'exit_time',      // managed-mode timestamp
])

export function filterTypeFor(columnId) {
  if (COLUMNS_NOT_FILTERABLE.has(columnId)) return null
  return COLUMN_FILTER_TYPES[columnId] || null
}

export const DEFAULT_COLUMNS = [
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
  { id: 'target_level_gex', label: 'Target GEX 0DTE (BN)', visible: true },
  { id: 'target_gamma_regime', label: 'Target Regime 0DTE', visible: true },
  { id: 'target_level_gex_all_exp', label: 'Target GEX All Exp (BN)', visible: false, className: 'study-col' },
  { id: 'target_gamma_regime_all_exp', label: 'Target Regime All Exp', visible: false, className: 'study-col' },
  { id: 'source_zone_signed_gex', label: 'Source GEX 0DTE (BN)', visible: false, className: 'study-col' },
  { id: 'source_zone_gamma_regime', label: 'Source Regime 0DTE', visible: false, className: 'study-col' },
  { id: 'source_zone_signed_gex_all_exp', label: 'Source GEX All Exp (BN)', visible: false, className: 'study-col' },
  { id: 'source_zone_gamma_regime_all_exp', label: 'Source Regime All Exp', visible: false, className: 'study-col' },
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
  { id: 'start_pct_range', label: 'Target % of Range', visible: true },

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
  { id: 'rvi_ratio_to_close',     label: '|Close|/1σ to Close', visible: false, className: 'study-col' },
  { id: 'rvi_inside_1s_to_close', label: 'Inside ±1σ to Close', visible: false, className: 'study-col' },

  { id: 'condor_short_put',  label: 'Short Put',  visible: false, className: 'study-col' },
  { id: 'condor_long_put',   label: 'Long Put',   visible: false, className: 'study-col' },
  { id: 'condor_short_call', label: 'Short Call', visible: false, className: 'study-col' },
  { id: 'condor_long_call',  label: 'Long Call',  visible: false, className: 'study-col' },
  { id: 'condor_to_close_short_put',  label: 'Short Put (to Close)',  visible: false, className: 'study-col' },
  { id: 'condor_to_close_long_put',   label: 'Long Put (to Close)',   visible: false, className: 'study-col' },
  { id: 'condor_to_close_short_call', label: 'Short Call (to Close)', visible: false, className: 'study-col' },
  { id: 'condor_to_close_long_call',  label: 'Long Call (to Close)',  visible: false, className: 'study-col' },
];

export const MANAGED_ONLY_COLUMNS = new Set([
  'signal_time', 'signal_px', 'put_skew', 'call_skew',
  'trade', 'range_high', 'range_low', 'entry_band',
  'entry_time', 'entry_px',
  'init_stop', 'take_profit', 'trailing_stop',
  'exit_time', 'exit_px', 'exit_reason',
  'realized_pts', 'mfe', 'mae', 'outcome',
  'reason', 'consol_mins', 'setup',
]);

export const STUDY_ONLY_COLUMNS = new Set([
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
  'rvi_ratio_to_close', 'rvi_inside_1s_to_close',
  'condor_short_put', 'condor_long_put', 'condor_short_call', 'condor_long_call',
  'condor_to_close_short_put', 'condor_to_close_long_put',
  'condor_to_close_short_call', 'condor_to_close_long_call',
]);

// Compute which columns should be visible given the executionMode.
// We force-HIDE columns that have no data in the current mode (managed-only
// columns in study mode, study-only columns in managed mode), but we do NOT
// force-show anything — the user's visibility preference always wins for
// columns that DO have data. This way unchecking a column in the modal
// actually removes it from the table.
export function computeEffectiveColumns(columns, executionMode) {
  const isStudy = executionMode === 'study_target_hits';
  return columns.map(col => {
    if (isStudy && MANAGED_ONLY_COLUMNS.has(col.id))  return { ...col, visible: false };
    if (!isStudy && STUDY_ONLY_COLUMNS.has(col.id))   return { ...col, visible: false };
    return col;
  });
}

// Merge a saved column preferences array (from localStorage) with the current
// DEFAULT_COLUMNS. Preserves visibility/order from the saved version, picks up
// any new columns we've added since.
export function mergeColumnsWithDefaults(saved) {
  if (!Array.isArray(saved) || saved.length === 0) return DEFAULT_COLUMNS;
  const savedById = new Map(saved.map(c => [c.id, c]));
  // Saved-order columns first, then any new columns from defaults
  const ordered = saved
    .map(s => {
      const def = DEFAULT_COLUMNS.find(d => d.id === s.id);
      if (!def) return null; // dropped from defaults
      return { ...def, visible: s.visible !== false };
    })
    .filter(Boolean);
  const seenIds = new Set(ordered.map(c => c.id));
  for (const def of DEFAULT_COLUMNS) {
    if (!seenIds.has(def.id)) ordered.push(def);
  }
  return ordered;
}
