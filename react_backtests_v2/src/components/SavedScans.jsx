import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import DiagnosticsPanel from './DiagnosticsPanel';
import ResultsTable from './ResultsTable';
import ColumnSettingsModal from './ColumnSettingsModal';
import {
  DEFAULT_COLUMNS,
  computeEffectiveColumns,
  mergeColumnsWithDefaults,
} from './columnsConfig';

// ──────────────────────────────────────────────────────────────────────
// Saved Scans tab
//
// Persists scan results in DB, lets user filter and explore without re-running.
//
// Feature parity with the live Instances tab:
//   - Full DEFAULT_COLUMNS list
//   - Column visibility/order via ColumnSettingsModal
//   - Separate localStorage key for persistence (don't share with live tab)
//   - CSV export
//   - Row selection drives the price chart + smile snapshots via the
//     parent's handleSelectRow (same handler as the live Instances tab)
// ──────────────────────────────────────────────────────────────────────

function isoDateOffset(days) {
  const d = new Date();
  d.setDate(d.getDate() + days);
  return d.toISOString().slice(0, 10);
}

// Saved-scan-side defaults that match the backend SAVED_SCAN_DEFAULTS.
// Used as a fallback if /api/backtests-v2/saved-scans/defaults can't be reached.
// The backend fetch overrides these on mount so they stay in sync.
const FALLBACK_SCAN_DEFAULTS = {
  // Basic tier
  minLevelGexBn:           50,
  levelFamily:             'primary',
  minCleanMovePoints:      20,
  pivotStrengthBars:       3,
  // Advanced tier
  zoneMergeDistancePts:    10,
  targetProximityPts:      5,
  maxZoneBreachPts:        5,
  minMinutesAfterOpen:     15,
  maxMinutesBeforeClose:   45,
  maxPriorDownUpRatio:     2.0,
  maxStartPctOfRange:      0.20,
  maxMoveLossPct:          0.75,
  forwardHorizonsMinutes:  [30, 60, 90, 120, 180],
  condorWingWidthPts:      10.0,
  maxResults:              10000,
}

const DEFAULT_NEW_SCAN = {
  direction: 'down',
  startDate: isoDateOffset(-90),
  endDate: isoDateOffset(0),
  label: '',
  notes: '',
  params: { ...FALLBACK_SCAN_DEFAULTS },
};

const DEFAULT_FILTERS = {
  gammaRegime: 'all',
  skewPassed: 'all',
  ivMin: '',
  ivMax: '',
  minMinutesRemaining: '',
  startDate: '',
  endDate: '',
};

// Separate localStorage key — saved-scan column prefs don't sync with live.
const COLUMNS_STORAGE_KEY = 'bt2-savedscans-table-columns';


export default function SavedScans({ onSelectRow }) {
  const [scans, setScans] = useState([])
  const [scansLoading, setScansLoading] = useState(false)
  const [error, setError] = useState(null)

  const [selectedScanId, setSelectedScanId] = useState(null)
  const [loadedScan, setLoadedScan] = useState(null)
  const [scanLoading, setScanLoading] = useState(false)

  const [showRunDialog, setShowRunDialog] = useState(false)
  const [newScan, setNewScan] = useState(DEFAULT_NEW_SCAN)
  const [running, setRunning] = useState(false)

  // Backend's authoritative defaults for the scan params. Fetched on mount.
  // Used to seed RunScanDialog so it pre-populates with whatever the backend
  // currently considers default. Falls back to FALLBACK_SCAN_DEFAULTS.
  const [serverDefaults, setServerDefaults] = useState(FALLBACK_SCAN_DEFAULTS)

  const [innerTab, setInnerTab] = useState('diagnostics')
  const [filters, setFilters] = useState(DEFAULT_FILTERS)

  // Selected row key for visual highlighting in the table
  const [selectedRowKey, setSelectedRowKey] = useState(null)
  const tableRef = useRef(null)

  // ── Column state with localStorage persistence ──
  const [columns, setColumns] = useState(() => {
    try {
      const saved = localStorage.getItem(COLUMNS_STORAGE_KEY)
      if (saved) return mergeColumnsWithDefaults(JSON.parse(saved))
    } catch (e) {
      // fall through to default
    }
    return DEFAULT_COLUMNS
  })

  useEffect(() => {
    try {
      localStorage.setItem(COLUMNS_STORAGE_KEY, JSON.stringify(columns))
    } catch (e) {
      // localStorage might be full or disabled; fail silently
    }
  }, [columns])

  const [isColumnsOpen, setIsColumnsOpen] = useState(false)

  // Edit-scan dialog state
  const [editingScan, setEditingScan] = useState(null) // the scan object being edited, or null
  const [editing, setEditing] = useState(false)        // submission in progress

  // ── Server fetches ──
  const refreshScans = useCallback(async () => {
    setScansLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/backtests-v2/saved-scans')
      const data = await res.json()
      if (!data.ok) throw new Error(data.error || 'Failed to list saved scans')
      setScans(Array.isArray(data.scans) ? data.scans : [])
    } catch (err) {
      setError(String(err.message || err))
    } finally {
      setScansLoading(false)
    }
  }, [])

  useEffect(() => { refreshScans() }, [refreshScans])

  // Fetch the backend's saved-scan defaults so the run dialog pre-populates
  // with current values rather than stale frontend constants.
  useEffect(() => {
    let cancelled = false
    fetch('/api/backtests-v2/saved-scans/defaults')
      .then(r => r.json())
      .then(data => {
        if (cancelled) return
        if (data?.ok && data?.defaults) {
          // Pick out only the fields we actually expose in the form
          const known = Object.keys(FALLBACK_SCAN_DEFAULTS)
          const next = { ...FALLBACK_SCAN_DEFAULTS }
          for (const k of known) {
            if (data.defaults[k] !== undefined) next[k] = data.defaults[k]
          }
          setServerDefaults(next)
          // Also seed newScan.params if it's still untouched (matches fallback)
          setNewScan(prev => {
            const isUntouched = JSON.stringify(prev.params) === JSON.stringify(FALLBACK_SCAN_DEFAULTS)
            if (isUntouched) return { ...prev, params: next }
            return prev
          })
        }
      })
      .catch(() => {
        // Stay on FALLBACK_SCAN_DEFAULTS — no UI error since this is a soft enhancement
      })
    return () => { cancelled = true }
  }, [])

  const loadScan = useCallback(async (scanId) => {
    if (!scanId) return
    setScanLoading(true)
    setError(null)
    try {
      const res = await fetch(`/api/backtests-v2/saved-scans/${scanId}`)
      const data = await res.json()
      if (!data.ok) throw new Error(data.error || 'Failed to load saved scan')
      setLoadedScan(data)
      setSelectedScanId(scanId)
      setInnerTab('diagnostics')
      setFilters(DEFAULT_FILTERS)
      setSelectedRowKey(null)
    } catch (err) {
      setError(String(err.message || err))
    } finally {
      setScanLoading(false)
    }
  }, [])

  const runAndSave = useCallback(async () => {
    setRunning(true)
    setError(null)
    try {
      const res = await fetch('/api/backtests-v2/saved-scans/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          direction: newScan.direction,
          startDate: newScan.startDate,
          endDate: newScan.endDate,
          label: newScan.label || null,
          notes: newScan.notes || null,
          params: newScan.params || {},
        }),
      })
      const data = await res.json()
      if (!data.ok) throw new Error(data.error || 'Scan failed')
      setShowRunDialog(false)
      // Reset to defaults but seed with serverDefaults so next-run is sane
      setNewScan({ ...DEFAULT_NEW_SCAN, params: { ...serverDefaults } })
      await refreshScans()
      if (data.scan_id) await loadScan(data.scan_id)
    } catch (err) {
      setError(String(err.message || err))
    } finally {
      setRunning(false)
    }
  }, [newScan, refreshScans, loadScan, serverDefaults])

  const deleteScan = useCallback(async (scanId) => {
    if (!scanId) return
    if (!window.confirm('Delete this saved scan?')) return
    setError(null)
    try {
      const res = await fetch(`/api/backtests-v2/saved-scans/${scanId}`, { method: 'DELETE' })
      const data = await res.json()
      if (!data.ok) throw new Error(data.error || 'Delete failed')
      if (selectedScanId === scanId) {
        setSelectedScanId(null)
        setLoadedScan(null)
        setSelectedRowKey(null)
      }
      await refreshScans()
    } catch (err) {
      setError(String(err.message || err))
    }
  }, [refreshScans, selectedScanId])

  // ── Edit handlers ──
  const openEditDialog = useCallback((scan) => {
    setEditingScan(scan)
  }, [])

  const closeEditDialog = useCallback(() => {
    if (editing) return // don't allow close mid-submit
    setEditingScan(null)
  }, [editing])

  const submitEdit = useCallback(async (form) => {
    if (!editingScan) return
    setEditing(true)
    setError(null)
    try {
      const res = await fetch(
        `/api/backtests-v2/saved-scans/${editingScan.scan_id}/update`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            label: form.label || null,
            notes: form.notes || null,
            startDate: form.startDate,
            endDate: form.endDate,
            params: form.params || {},
          }),
        }
      )
      const data = await res.json()
      if (!data.ok) throw new Error(data.error || 'Update failed')
      setEditingScan(null)
      await refreshScans()
      if (selectedScanId === editingScan.scan_id) {
        await loadScan(editingScan.scan_id)
      }
    } catch (err) {
      setError(String(err.message || err))
    } finally {
      setEditing(false)
    }
  }, [editingScan, refreshScans, selectedScanId, loadScan])

  // Duplicate a saved scan: pre-populate the run dialog with this scan's
  // params (and label/dates as a starting point) and open it.
  const duplicateScan = useCallback(async (scan) => {
    if (!scan?.scan_id) return
    setError(null)
    try {
      // Need full row to access params (the list endpoint omits params for size)
      const res = await fetch(`/api/backtests-v2/saved-scans/${scan.scan_id}`)
      const data = await res.json()
      if (!data.ok) throw new Error(data.error || 'Failed to load scan for duplication')
      const sourceParams = data.params || {}
      // Project sourceParams onto the form's known keys; anything else is dropped
      const formParams = { ...serverDefaults }
      for (const k of Object.keys(FALLBACK_SCAN_DEFAULTS)) {
        if (sourceParams[k] !== undefined) formParams[k] = sourceParams[k]
      }
      setNewScan({
        direction: data.direction || 'down',
        startDate: data.start_date || isoDateOffset(-90),
        endDate:   data.end_date   || isoDateOffset(0),
        label:    `${data.label || `Scan #${scan.scan_id}`} (copy)`,
        notes:    data.notes || '',
        params:   formParams,
      })
      setShowRunDialog(true)
    } catch (err) {
      setError(String(err.message || err))
    }
  }, [serverDefaults])

  // ── Row selection (drives price chart + smile via parent handler) ──
  const handleSelectRowLocal = useCallback(async (row, idx) => {
    const key = `${row.trade_date}-${row.start_ts_utc}-${row.target_ts_utc}-${idx}`
    setSelectedRowKey(key)
    if (typeof onSelectRow === 'function') {
      // The parent handler (App.handleSelectRow) posts to /select-trade
      // which drives the price chart navigation and smile snapshots.
      // It also sets its OWN selectedRowKey state which is wired to the
      // live Instances tab; that's harmless since our table is a separate
      // ResultsTable instance.
      try {
        await onSelectRow(row, idx)
      } catch (err) {
        setError(String(err.message || err))
      }
    }
  }, [onSelectRow])

  // ── Filtered rows ──
  const allRows = loadedScan?.rows || []
  const filteredRows = useMemo(
    () => applyFilters(allRows, filters),
    [allRows, filters]
  )

  const filteredDiagnostics = useMemo(() => {
    if (!loadedScan) return null
    return computeFilteredDiagnostics(loadedScan, filteredRows, filters)
  }, [loadedScan, filteredRows, filters])

  const effectiveExecutionMode =
    loadedScan?.params?.executionMode || 'study_target_hits'

  // Apply study/managed visibility rules on top of user's column prefs
  const effectiveColumns = useMemo(
    () => computeEffectiveColumns(columns, effectiveExecutionMode),
    [columns, effectiveExecutionMode]
  )

  const filtersAreActive = useMemo(
    () => filtersAreNonDefault(filters),
    [filters]
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16, flex: 1, minHeight: 0 }}>
      {/* Saved scans browser */}
      <div className="diag-card" style={{ flex: '0 0 auto' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <div>
            <div className="eyebrow">Saved Scans</div>
            <h2 style={{ fontSize: 18 }}>Cached scan results</h2>
            <p style={{ color: '#94a3b8', fontSize: 12, margin: '4px 0 0' }}>
              Run a permissive scan once, then explore the result without re-scanning.
            </p>
          </div>
          <button
            className="primary-button"
            onClick={() => setShowRunDialog(true)}
            disabled={running}
          >
            + Run &amp; Save Scan
          </button>
        </div>

        {error && <div className="error-banner" style={{ marginBottom: 12 }}>{error}</div>}

        <SavedScansList
          scans={scans}
          loading={scansLoading}
          selectedScanId={selectedScanId}
          onSelect={loadScan}
          onDelete={deleteScan}
          onEdit={openEditDialog}
          onDuplicate={duplicateScan}
        />
      </div>

      {showRunDialog && (
        <RunScanDialog
          newScan={newScan}
          setNewScan={setNewScan}
          onSubmit={runAndSave}
          onCancel={() => { setShowRunDialog(false); setNewScan(DEFAULT_NEW_SCAN) }}
          running={running}
        />
      )}

      {editingScan && (
        <EditScanDialog
          scan={editingScan}
          onSubmit={submitEdit}
          onCancel={closeEditDialog}
          submitting={editing}
        />
      )}

      {/* Loaded scan view */}
      {scanLoading && (
        <div className="diag-card" style={{ flex: 1 }}>
          <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>
            Loading saved scan…
          </div>
        </div>
      )}
      {!scanLoading && loadedScan && (
        <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, gap: 8 }}>
          <LoadedScanHeader
            scan={loadedScan}
            allCount={allRows.length}
            filteredCount={filteredRows.length}
            filtersActive={filtersAreActive}
          />

          <FiltersPanel
            filters={filters}
            setFilters={setFilters}
            scan={loadedScan}
            filteredCount={filteredRows.length}
            allCount={allRows.length}
          />

          {/* Inner tab bar */}
          <div className="tab-bar">
            <button
              className={`tab-button ${innerTab === 'diagnostics' ? 'active' : ''}`}
              onClick={() => setInnerTab('diagnostics')}
            >
              Diagnostics
            </button>
            <button
              className={`tab-button ${innerTab === 'instances' ? 'active' : ''}`}
              onClick={() => setInnerTab('instances')}
            >
              Instances
              {filteredRows.length > 0 && <span className="tab-badge">{filteredRows.length}</span>}
            </button>
          </div>

          {innerTab === 'diagnostics' && (
            <DiagnosticsPanel
              diagnostics={filteredDiagnostics}
              rows={filteredRows}
              funnel={loadedScan?.funnel || []}
              executionMode={effectiveExecutionMode}
            />
          )}

          {innerTab === 'instances' && (
            <div className="results-card" style={{ flex: 1 }}>
              <div className="results-header">
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
                  <h2 style={{ fontSize: 16 }}>Instances</h2>
                  <span style={{ color: '#64748b', fontSize: 12 }}>
                    {filteredRows.length}{filtersAreActive ? ` of ${allRows.length}` : ''} trades
                  </span>
                  <button
                    className="ghost-button"
                    style={{ padding: '4px 10px', fontSize: 12, marginLeft: 4, display: 'flex', alignItems: 'center', gap: 4 }}
                    onClick={() => setIsColumnsOpen(true)}
                  >
                    ⚙ Columns
                  </button>
                  <button
                    className="ghost-button"
                    style={{ padding: '4px 10px', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4 }}
                    onClick={() => tableRef.current?.downloadCSV()}
                    disabled={!filteredRows.length}
                  >
                    📥 CSV
                  </button>
                </div>
              </div>

              <ResultsTable
                ref={tableRef}
                rows={filteredRows}
                selectedRowKey={selectedRowKey}
                onSelectRow={handleSelectRowLocal}
                columns={effectiveColumns}
              />
            </div>
          )}
        </div>
      )}

      {!scanLoading && !loadedScan && !scansLoading && scans.length === 0 && (
        <div className="empty-state" style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          No saved scans yet. Click &quot;Run &amp; Save Scan&quot; to create your first.
        </div>
      )}
      {!scanLoading && !loadedScan && scans.length > 0 && (
        <div className="empty-state" style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          Select a saved scan above to view its results.
        </div>
      )}

      <ColumnSettingsModal
        isOpen={isColumnsOpen}
        onClose={() => setIsColumnsOpen(false)}
        columns={columns}
        onUpdateColumns={setColumns}
      />
    </div>
  )
}


// ──────────────────────────────────────────────────────────────────────
// Filtering logic
// ──────────────────────────────────────────────────────────────────────

function applyFilters(rows, f) {
  if (!Array.isArray(rows) || rows.length === 0) return []

  const ivMin  = f.ivMin === '' ? null : Number(f.ivMin)
  const ivMax  = f.ivMax === '' ? null : Number(f.ivMax)
  const minMtc = f.minMinutesRemaining === '' ? null : Number(f.minMinutesRemaining)
  const dateStart = (f.startDate || '').trim()
  const dateEnd   = (f.endDate || '').trim()

  return rows.filter(row => {
    if (f.gammaRegime !== 'all') {
      const r = row.target_gamma_regime || 'unknown'
      if (f.gammaRegime === 'non_neutral') {
        if (r !== 'positive' && r !== 'negative') return false
      } else if (r !== f.gammaRegime) {
        return false
      }
    }

    if (f.skewPassed !== 'all') {
      const passed = row.skew_threshold_passed === true
      if (f.skewPassed === 'yes' && !passed) return false
      if (f.skewPassed === 'no' && passed) return false
    }

    const iv = row.iv?.atm_0dte_pct
    if (ivMin != null && Number.isFinite(ivMin)) {
      if (iv == null || iv < ivMin) return false
    }
    if (ivMax != null && Number.isFinite(ivMax)) {
      if (iv == null || iv > ivMax) return false
    }

    if (minMtc != null && Number.isFinite(minMtc)) {
      const mtc = row.minutes_to_close
      if (mtc == null || mtc < minMtc) return false
    }

    if (dateStart && row.trade_date && String(row.trade_date) < dateStart) return false
    if (dateEnd && row.trade_date && String(row.trade_date) > dateEnd) return false

    return true
  })
}

function filtersAreNonDefault(f) {
  return (
    f.gammaRegime !== 'all' ||
    f.skewPassed !== 'all' ||
    f.ivMin !== '' ||
    f.ivMax !== '' ||
    f.minMinutesRemaining !== '' ||
    f.startDate !== '' ||
    f.endDate !== ''
  )
}


// ──────────────────────────────────────────────────────────────────────
// Diagnostics recomputation from filtered rows
// ──────────────────────────────────────────────────────────────────────

function computeFilteredDiagnostics(loadedScan, filteredRows, filters) {
  const base = { ...(loadedScan?.diagnostics || {}) }

  if (!filtersAreNonDefault(filters)) {
    return base
  }

  base.entries_found = filteredRows.length
  base.valid_instances = filteredRows.length

  const ivs = []
  for (const r of filteredRows) {
    const v = r?.iv?.atm_0dte_pct
    if (v != null && Number.isFinite(v)) ivs.push(Number(v))
  }
  if (ivs.length) {
    const sorted = [...ivs].sort((a, b) => a - b)
    base.iv_at_entry_summary = {
      count: sorted.length,
      min: round2(sorted[0]),
      p25: round2(sorted[Math.floor(sorted.length * 0.25)]),
      median: round2(median(sorted)),
      p75: round2(sorted[Math.floor(sorted.length * 0.75)]),
      max: round2(sorted[sorted.length - 1]),
    }
  } else {
    base.iv_at_entry_summary = null
  }

  base.forward_outcomes_aggregate = computeForwardOutcomesAggregate(filteredRows)
  base.realized_vs_implied_aggregate = computeRvIAggregate(filteredRows)

  return base
}

function computeForwardOutcomesAggregate(rows) {
  const horizons = ['30m', '60m', '90m', '120m', '180m', 'eod']
  const out = {}
  for (const h of horizons) {
    const mfes = [], maes = [], closes = []
    for (const r of rows) {
      const fo = r?.forward_outcomes?.[h]
      if (!fo) continue
      if (fo.mfe_pts   != null) mfes.push(Number(fo.mfe_pts))
      if (fo.mae_pts   != null) maes.push(Number(fo.mae_pts))
      if (fo.close_pts != null) closes.push(Number(fo.close_pts))
    }
    if (closes.length === 0) {
      out[h] = { count: 0 }
      continue
    }
    out[h] = {
      count: closes.length,
      mfe_mean:           round2(mean(mfes)),
      mfe_median:         round2(median(mfes)),
      mae_mean:           round2(mean(maes)),
      mae_median:         round2(median(maes)),
      close_mean:         round2(mean(closes)),
      close_median:       round2(median(closes)),
      win_rate_at_close:  round3(closes.filter(c => c > 0).length / closes.length),
    }
  }
  return out
}

function computeRvIAggregate(rows) {
  const horizons = ['30m', '60m', '90m', '120m', '180m']
  const out = {}
  for (const h of horizons) {
    const ivs = [], implPts = [], rangeRatios = [], closeRatios = []
    let inside1 = 0, inside2 = 0, total = 0
    for (const r of rows) {
      const rvi = r?.realized_vs_implied?.[h]
      if (!rvi || rvi.implied_1sigma_pts == null) continue
      const iv = r?.iv?.atm_0dte_pct
      if (iv != null && Number.isFinite(iv)) ivs.push(Number(iv))
      implPts.push(Number(rvi.implied_1sigma_pts))
      if (rvi.range_over_1sigma != null) rangeRatios.push(Number(rvi.range_over_1sigma))
      if (rvi.close_over_1sigma != null) closeRatios.push(Number(rvi.close_over_1sigma))
      total++
      if (rvi.inside_1sigma === true) inside1++
      if (rvi.inside_2sigma === true) inside2++
    }
    if (total === 0) {
      out[h] = { count: 0 }
      continue
    }
    out[h] = {
      count: total,
      iv_median:                ivs.length ? round2(median(ivs)) : null,
      implied_1sigma_median:    round2(median(implPts)),
      range_over_1sigma_median: rangeRatios.length ? round3(median(rangeRatios)) : null,
      close_over_1sigma_median: closeRatios.length ? round3(median(closeRatios)) : null,
      pct_inside_1sigma:        round3(inside1 / total),
      pct_inside_2sigma:        round3(inside2 / total),
    }
  }
  return out
}

function median(arr) {
  if (!arr.length) return 0
  const s = [...arr].sort((a, b) => a - b)
  const n = s.length
  return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2])
}
function mean(arr) {
  if (!arr.length) return 0
  return arr.reduce((a, b) => a + b, 0) / arr.length
}
function round2(v) { return Number.isFinite(v) ? Math.round(v * 100) / 100 : null }
function round3(v) { return Number.isFinite(v) ? Math.round(v * 1000) / 1000 : null }


// ──────────────────────────────────────────────────────────────────────
// UI components
// ──────────────────────────────────────────────────────────────────────

function FiltersPanel({ filters, setFilters, scan, filteredCount, allCount }) {
  const update = (key, value) => setFilters(prev => ({ ...prev, [key]: value }))
  const reset = () => setFilters(DEFAULT_FILTERS)
  const active = filtersAreNonDefault(filters)

  const regimeCounts = useMemo(() => {
    const counts = { positive: 0, negative: 0, neutral: 0, unknown: 0 }
    for (const r of scan?.rows || []) {
      const k = r?.target_gamma_regime || 'unknown'
      counts[k] = (counts[k] || 0) + 1
    }
    return counts
  }, [scan])

  const cardStyle = {
    background: '#0f172a',
    border: active ? '1px solid #2563eb' : '1px solid #1f2937',
    borderRadius: 14,
    padding: 12,
    transition: 'border-color 0.15s',
  }

  const labelStyle = {
    fontSize: 11,
    color: '#94a3b8',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.04em',
    marginBottom: 4,
    display: 'block',
  }

  const inputStyle = {
    background: '#020617',
    border: '1px solid #334155',
    color: '#e5e7eb',
    borderRadius: 8,
    padding: '6px 10px',
    fontSize: 12,
    width: '100%',
  }

  return (
    <div style={cardStyle}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 13, fontWeight: 700, color: active ? '#93c5fd' : '#cbd5e1' }}>
            Filters
          </span>
          {active ? (
            <span style={{ fontSize: 11, color: '#fcd34d' }}>
              {filteredCount} of {allCount} rows match
            </span>
          ) : (
            <span style={{ fontSize: 11, color: '#64748b' }}>
              All {allCount} rows
            </span>
          )}
        </div>
        {active && (
          <button
            className="ghost-button"
            style={{ padding: '4px 12px', fontSize: 11 }}
            onClick={reset}
          >
            Reset filters
          </button>
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 10 }}>
        <div>
          <span style={labelStyle}>Gamma regime</span>
          <select style={inputStyle} value={filters.gammaRegime} onChange={(e) => update('gammaRegime', e.target.value)}>
            <option value="all">All ({allCount})</option>
            <option value="positive">Positive ({regimeCounts.positive})</option>
            <option value="negative">Negative ({regimeCounts.negative})</option>
            <option value="neutral">Neutral ({regimeCounts.neutral})</option>
            <option value="non_neutral">Non-neutral ({regimeCounts.positive + regimeCounts.negative})</option>
          </select>
        </div>

        <div>
          <span style={labelStyle}>Skew passed</span>
          <select style={inputStyle} value={filters.skewPassed} onChange={(e) => update('skewPassed', e.target.value)}>
            <option value="all">All</option>
            <option value="yes">Yes</option>
            <option value="no">No</option>
          </select>
        </div>

        <div>
          <span style={labelStyle}>IV min %</span>
          <input
            type="number" step="0.5" min="0" max="100"
            style={inputStyle}
            value={filters.ivMin}
            onChange={(e) => update('ivMin', e.target.value)}
            placeholder="—"
          />
        </div>

        <div>
          <span style={labelStyle}>IV max %</span>
          <input
            type="number" step="0.5" min="0" max="100"
            style={inputStyle}
            value={filters.ivMax}
            onChange={(e) => update('ivMax', e.target.value)}
            placeholder="—"
          />
        </div>

        <div>
          <span style={labelStyle}>Min remaining (min)</span>
          <input
            type="number" step="5" min="0"
            style={inputStyle}
            value={filters.minMinutesRemaining}
            onChange={(e) => update('minMinutesRemaining', e.target.value)}
            placeholder="—"
          />
        </div>

        <div>
          <span style={labelStyle}>Date start</span>
          <input
            type="date" style={inputStyle}
            value={filters.startDate}
            onChange={(e) => update('startDate', e.target.value)}
            min={scan?.start_date}
            max={scan?.end_date}
          />
        </div>

        <div>
          <span style={labelStyle}>Date end</span>
          <input
            type="date" style={inputStyle}
            value={filters.endDate}
            onChange={(e) => update('endDate', e.target.value)}
            min={scan?.start_date}
            max={scan?.end_date}
          />
        </div>
      </div>
    </div>
  )
}


function SavedScansList({ scans, loading, selectedScanId, onSelect, onDelete, onEdit, onDuplicate }) {
  if (loading && scans.length === 0) {
    return <div style={{ color: '#94a3b8' }}>Loading…</div>
  }
  if (!scans.length) {
    return <div style={{ color: '#94a3b8', fontSize: 13 }}>No saved scans yet.</div>
  }
  return (
    <div style={{ overflow: 'auto', maxHeight: 320 }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ color: '#94a3b8', textAlign: 'left', borderBottom: '1px solid #1f2937' }}>
            <th style={{ padding: '6px 10px' }}>Created</th>
            <th style={{ padding: '6px 10px' }}>Label</th>
            <th style={{ padding: '6px 10px' }}>Direction</th>
            <th style={{ padding: '6px 10px' }}>Range</th>
            <th style={{ padding: '6px 10px', textAlign: 'right' }}>Rows</th>
            <th style={{ padding: '6px 10px' }}></th>
          </tr>
        </thead>
        <tbody>
          {scans.map((scan) => {
            const isSelected = scan.scan_id === selectedScanId
            return (
              <tr
                key={scan.scan_id}
                style={{
                  cursor: 'pointer',
                  background: isSelected ? 'rgba(37,99,235,0.16)' : 'transparent',
                  borderBottom: '1px solid #1f2937',
                }}
                onClick={() => onSelect(scan.scan_id)}
              >
                <td style={{ padding: '8px 10px', color: '#cbd5e1' }}>
                  {scan.created_at ? new Date(scan.created_at).toLocaleString() : '—'}
                </td>
                <td style={{ padding: '8px 10px', color: '#e5e7eb', fontWeight: 600 }}>
                  {scan.label || `Scan #${scan.scan_id}`}
                </td>
                <td style={{ padding: '8px 10px' }}>
                  <span className={`direction-chip ${scan.direction}`}>{scan.direction}</span>
                </td>
                <td style={{ padding: '8px 10px', color: '#cbd5e1', fontFamily: 'monospace' }}>
                  {scan.start_date} → {scan.end_date}
                </td>
                <td style={{ padding: '8px 10px', textAlign: 'right', color: '#fcd34d', fontWeight: 700 }}>
                  {scan.row_count}
                </td>
                <td style={{ padding: '8px 10px', textAlign: 'right' }}>
                  <button
                    className="ghost-button"
                    style={{ padding: '4px 10px', fontSize: 11, marginRight: 6 }}
                    onClick={(e) => { e.stopPropagation(); onDuplicate && onDuplicate(scan) }}
                    title="Open the run dialog pre-filled with this scan's params"
                  >
                    Duplicate
                  </button>
                  <button
                    className="ghost-button"
                    style={{ padding: '4px 10px', fontSize: 11, marginRight: 6 }}
                    onClick={(e) => { e.stopPropagation(); onEdit && onEdit(scan) }}
                  >
                    Edit
                  </button>
                  <button
                    className="ghost-button"
                    style={{ padding: '4px 10px', fontSize: 11 }}
                    onClick={(e) => { e.stopPropagation(); onDelete(scan.scan_id) }}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}


function RunScanDialog({ newScan, setNewScan, onSubmit, onCancel, running }) {
  return (
    <div className="modal-backdrop" onClick={running ? undefined : onCancel}>
      <div className="modal-card" style={{ width: 720, maxHeight: '90vh', overflow: 'auto' }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <div className="eyebrow">Saved Scans</div>
            <h2 style={{ fontSize: 18 }}>Run &amp; Save New Scan</h2>
            <p style={{ color: '#94a3b8', fontSize: 12, margin: '4px 0 0' }}>
              Runs a permissive scan (skew bypassed, study mode on) and persists the full result.
            </p>
          </div>
          <button className="ghost-button" onClick={onCancel} disabled={running}>×</button>
        </div>

        <div className="form-grid">
          <label className="field">
            <span>Direction</span>
            <select value={newScan.direction} onChange={(e) => setNewScan({ ...newScan, direction: e.target.value })} disabled={running}>
              <option value="down">Down → Long bounce</option>
              <option value="up">Up → Short rejection</option>
            </select>
          </label>
          <label className="field">
            <span>Label (optional)</span>
            <input
              type="text"
              value={newScan.label}
              onChange={(e) => setNewScan({ ...newScan, label: e.target.value })}
              placeholder="e.g. Q1 2025 down-move sample"
              disabled={running}
            />
          </label>
          <label className="field">
            <span>Start date</span>
            <input
              type="date"
              value={newScan.startDate}
              onChange={(e) => setNewScan({ ...newScan, startDate: e.target.value })}
              disabled={running}
            />
          </label>
          <label className="field">
            <span>End date</span>
            <input
              type="date"
              value={newScan.endDate}
              onChange={(e) => setNewScan({ ...newScan, endDate: e.target.value })}
              disabled={running}
            />
          </label>
          <label className="field field-wide">
            <span>Notes (optional)</span>
            <textarea
              value={newScan.notes}
              onChange={(e) => setNewScan({ ...newScan, notes: e.target.value })}
              placeholder="Anything you want to remember about this scan..."
              disabled={running}
            />
          </label>
        </div>

        <ScanParamsForm
          params={newScan.params || {}}
          onChange={(nextParams) => setNewScan({ ...newScan, params: nextParams })}
          disabled={running}
        />

        {running && (
          <div className="helper-note" style={{ color: '#fcd34d' }}>
            Running scan… this may take a few minutes for long date ranges. Don't close the page.
          </div>
        )}

        <div className="modal-actions">
          <button className="ghost-button" onClick={onCancel} disabled={running}>Cancel</button>
          <button className="primary-button" onClick={onSubmit} disabled={running || !newScan.startDate || !newScan.endDate}>
            {running ? 'Running…' : 'Run & Save'}
          </button>
        </div>
      </div>
    </div>
  )
}


function LoadedScanHeader({ scan, allCount, filteredCount, filtersActive }) {
  if (!scan) return null
  return (
    <div className="diag-card" style={{ flex: '0 0 auto', padding: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <h2 style={{ fontSize: 16 }}>{scan.label || `Scan #${scan.scan_id}`}</h2>
          <div style={{ color: '#94a3b8', fontSize: 12, marginTop: 4 }}>
            <span className={`direction-chip ${scan.direction}`} style={{ marginRight: 8 }}>
              {scan.direction}
            </span>
            <span style={{ fontFamily: 'monospace' }}>{scan.start_date} → {scan.end_date}</span>
            <span style={{ marginLeft: 12 }}>
              ·  {filtersActive ? `${filteredCount} of ${allCount}` : allCount} rows
            </span>
            {scan.created_at && (
              <span style={{ marginLeft: 12 }}>
                · saved {new Date(scan.created_at).toLocaleString()}
              </span>
            )}
          </div>
          {scan.notes && (
            <div style={{ color: '#cbd5e1', fontSize: 12, marginTop: 6, fontStyle: 'italic' }}>
              {scan.notes}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}


function EditScanDialog({ scan, onSubmit, onCancel, submitting }) {
  // Local form state seeded from the scan being edited.
  const [form, setForm] = React.useState({
    label: scan.label || '',
    notes: scan.notes || '',
    startDate: scan.start_date || '',
    endDate: scan.end_date || '',
    params: { ...FALLBACK_SCAN_DEFAULTS },
  })

  // We don't get scan.params from the list endpoint — fetch the full scan
  // on mount so the params form starts with the actual current values.
  const [paramsLoaded, setParamsLoaded] = React.useState(false)
  React.useEffect(() => {
    let cancelled = false
    if (!scan?.scan_id) return
    fetch(`/api/backtests-v2/saved-scans/${scan.scan_id}`)
      .then(r => r.json())
      .then(data => {
        if (cancelled) return
        if (data?.ok) {
          const sourceParams = data.params || {}
          const formParams = { ...FALLBACK_SCAN_DEFAULTS }
          for (const k of Object.keys(FALLBACK_SCAN_DEFAULTS)) {
            if (sourceParams[k] !== undefined) formParams[k] = sourceParams[k]
          }
          setForm(prev => ({ ...prev, params: formParams }))
        }
        setParamsLoaded(true)
      })
      .catch(() => setParamsLoaded(true))
    return () => { cancelled = true }
  }, [scan?.scan_id])

  // Detect whether anything that requires re-scanning has changed.
  // Date range changes always re-scan. Param changes also re-scan.
  const datesChanged = (
    form.startDate !== (scan.start_date || '') ||
    form.endDate   !== (scan.end_date || '')
  )
  const paramsChanged = paramsLoaded && (() => {
    // Compare against currently-loaded params
    // We don't have the original here without re-fetching; we approximate
    // by saying "if datesChanged, treat as re-scan; otherwise check params
    // against FALLBACK_SCAN_DEFAULTS would be wrong."
    // Better: just always send params; backend will re-scan if any scan-
    // affecting key changed. The user-visible "Re-run" indicator is just for
    // dates since that's the most obvious trigger.
    return false
  })()
  const willRescan = datesChanged || paramsChanged

  return (
    <div className="modal-backdrop" onClick={submitting ? undefined : onCancel}>
      <div className="modal-card" style={{ width: 720, maxHeight: '90vh', overflow: 'auto' }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <div className="eyebrow">Edit Saved Scan</div>
            <h2 style={{ fontSize: 18 }}>{scan.label || `Scan #${scan.scan_id}`}</h2>
            <p style={{ color: '#94a3b8', fontSize: 12, margin: '4px 0 0' }}>
              Editing label/notes saves instantly. Changing dates or any
              scan parameter re-runs the scan and replaces the cached results.
            </p>
          </div>
          <button className="ghost-button" onClick={onCancel} disabled={submitting}>×</button>
        </div>

        <div className="form-grid">
          <label className="field">
            <span>Direction</span>
            <input
              type="text"
              value={scan.direction}
              disabled
              readOnly
              style={{ opacity: 0.6, cursor: 'not-allowed' }}
            />
          </label>
          <label className="field">
            <span>Label</span>
            <input
              type="text"
              value={form.label}
              onChange={(e) => setForm({ ...form, label: e.target.value })}
              placeholder="Scan label"
              disabled={submitting}
            />
          </label>
          <label className="field">
            <span>Start date</span>
            <input
              type="date"
              value={form.startDate}
              onChange={(e) => setForm({ ...form, startDate: e.target.value })}
              disabled={submitting}
            />
          </label>
          <label className="field">
            <span>End date</span>
            <input
              type="date"
              value={form.endDate}
              onChange={(e) => setForm({ ...form, endDate: e.target.value })}
              disabled={submitting}
            />
          </label>
          <label className="field field-wide">
            <span>Notes</span>
            <textarea
              value={form.notes}
              onChange={(e) => setForm({ ...form, notes: e.target.value })}
              placeholder="Anything you want to remember about this scan..."
              disabled={submitting}
            />
          </label>
        </div>

        {paramsLoaded ? (
          <ScanParamsForm
            params={form.params}
            onChange={(nextParams) => setForm({ ...form, params: nextParams })}
            disabled={submitting}
          />
        ) : (
          <div className="helper-note" style={{ color: '#94a3b8' }}>Loading parameters…</div>
        )}

        {willRescan && (
          <div className="helper-note" style={{ color: '#fcd34d' }}>
            ⚠ Date range changed — clicking Save will re-run the scan with the
            new range and replace the cached rows. Changing scan parameters also
            triggers a re-run.
          </div>
        )}

        {submitting && (
          <div className="helper-note" style={{ color: '#fcd34d' }}>
            {willRescan
              ? "Re-running scan… this may take a few minutes. Don't close the page."
              : "Saving…"}
          </div>
        )}

        <div className="modal-actions">
          <button className="ghost-button" onClick={onCancel} disabled={submitting}>Cancel</button>
          <button
            className="primary-button"
            onClick={() => onSubmit(form)}
            disabled={submitting || !form.startDate || !form.endDate || !paramsLoaded}
          >
            {submitting
              ? (willRescan ? 'Re-running…' : 'Saving…')
              : 'Save'}
          </button>
        </div>
      </div>
    </div>
  )
}


// ──────────────────────────────────────────────────────────────────────
// ScanParamsForm — Basic + Advanced controls for a saved scan's params.
// Used by both RunScanDialog and EditScanDialog.
// `params` is an object containing all overridable keys.
// `onChange` receives a new params object when any field changes.
// ──────────────────────────────────────────────────────────────────────

function ScanParamsForm({ params, onChange, disabled = false }) {
  const [showAdvanced, setShowAdvanced] = React.useState(false)

  const update = (key, value) => onChange({ ...params, [key]: value })

  // Helpers for the comma-separated horizons list
  const horizonsString = Array.isArray(params.forwardHorizonsMinutes)
    ? params.forwardHorizonsMinutes.join(', ')
    : ''
  const updateHorizons = (str) => {
    const parts = String(str).split(',').map(s => s.trim()).filter(Boolean)
    const nums = parts.map(p => parseInt(p, 10)).filter(n => Number.isFinite(n) && n > 0)
    update('forwardHorizonsMinutes', nums)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div className="eyebrow" style={{ marginTop: 8 }}>Move Detection — Basic</div>
      <div className="form-grid">
        <label className="field">
          <span>Min level GEX (BN)</span>
          <input
            type="number" step="5" min="0"
            value={params.minLevelGexBn ?? ''}
            onChange={(e) => update('minLevelGexBn', Number(e.target.value))}
            disabled={disabled}
          />
        </label>
        <label className="field">
          <span>Level family</span>
          <select
            value={params.levelFamily ?? 'primary'}
            onChange={(e) => update('levelFamily', e.target.value)}
            disabled={disabled}
          >
            <option value="primary">Primary only</option>
            <option value="primary+strong">Primary + Strong</option>
          </select>
        </label>
        <label className="field">
          <span>Min clean move (pts)</span>
          <input
            type="number" step="1" min="0"
            value={params.minCleanMovePoints ?? ''}
            onChange={(e) => update('minCleanMovePoints', Number(e.target.value))}
            disabled={disabled}
          />
        </label>
        <label className="field">
          <span>Pivot strength (bars)</span>
          <input
            type="number" step="1" min="1"
            value={params.pivotStrengthBars ?? ''}
            onChange={(e) => update('pivotStrengthBars', Number(e.target.value))}
            disabled={disabled}
          />
        </label>
      </div>

      <button
        type="button"
        onClick={() => setShowAdvanced(v => !v)}
        style={{
          background: 'none',
          border: '1px solid #334155',
          color: '#93c5fd',
          padding: '6px 12px',
          borderRadius: 8,
          cursor: 'pointer',
          fontSize: 11,
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '0.04em',
          alignSelf: 'flex-start',
        }}
        disabled={disabled}
      >
        {showAdvanced ? '▼ Hide advanced' : '▶ Show advanced'}
      </button>

      {showAdvanced && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div className="eyebrow">Move Detection — Advanced</div>
          <div className="form-grid">
            <label className="field">
              <span>Zone merge dist (pts)</span>
              <input type="number" step="1" min="0"
                value={params.zoneMergeDistancePts ?? ''}
                onChange={(e) => update('zoneMergeDistancePts', Number(e.target.value))}
                disabled={disabled} />
            </label>
            <label className="field">
              <span>Target proximity (pts)</span>
              <input type="number" step="1" min="0"
                value={params.targetProximityPts ?? ''}
                onChange={(e) => update('targetProximityPts', Number(e.target.value))}
                disabled={disabled} />
            </label>
            <label className="field">
              <span>Max zone breach (pts)</span>
              <input type="number" step="1" min="0"
                value={params.maxZoneBreachPts ?? ''}
                onChange={(e) => update('maxZoneBreachPts', Number(e.target.value))}
                disabled={disabled} />
            </label>
            <label className="field">
              <span>Min minutes after open</span>
              <input type="number" step="5" min="0"
                value={params.minMinutesAfterOpen ?? ''}
                onChange={(e) => update('minMinutesAfterOpen', Number(e.target.value))}
                disabled={disabled} />
            </label>
            <label className="field">
              <span>Max minutes before close</span>
              <input type="number" step="5" min="0"
                value={params.maxMinutesBeforeClose ?? ''}
                onChange={(e) => update('maxMinutesBeforeClose', Number(e.target.value))}
                disabled={disabled} />
            </label>
            <label className="field">
              <span>Max prior down/up ratio</span>
              <input type="number" step="0.1" min="0"
                value={params.maxPriorDownUpRatio ?? ''}
                onChange={(e) => update('maxPriorDownUpRatio', Number(e.target.value))}
                disabled={disabled} />
            </label>
            <label className="field">
              <span>Max start % of range</span>
              <input type="number" step="0.05" min="0" max="1"
                value={params.maxStartPctOfRange ?? ''}
                onChange={(e) => update('maxStartPctOfRange', Number(e.target.value))}
                disabled={disabled} />
            </label>
            <label className="field">
              <span>Max move loss %</span>
              <input type="number" step="0.05" min="0" max="1"
                value={params.maxMoveLossPct ?? ''}
                onChange={(e) => update('maxMoveLossPct', Number(e.target.value))}
                disabled={disabled} />
            </label>
            <label className="field">
              <span>Max results</span>
              <input type="number" step="100" min="100"
                value={params.maxResults ?? ''}
                onChange={(e) => update('maxResults', Number(e.target.value))}
                disabled={disabled} />
            </label>
          </div>

          <div className="eyebrow">Forward Outcome Measurement</div>
          <div className="form-grid">
            <label className="field field-wide">
              <span>Forward horizons (minutes)</span>
              <input
                type="text"
                value={horizonsString}
                onChange={(e) => updateHorizons(e.target.value)}
                placeholder="30, 60, 90, 120, 180"
                disabled={disabled}
              />
            </label>
            <label className="field">
              <span>Condor wing width (pts)</span>
              <input type="number" step="1" min="0"
                value={params.condorWingWidthPts ?? ''}
                onChange={(e) => update('condorWingWidthPts', Number(e.target.value))}
                disabled={disabled} />
            </label>
          </div>
        </div>
      )}
    </div>
  )
}
