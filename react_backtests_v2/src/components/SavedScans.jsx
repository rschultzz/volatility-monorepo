import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import DiagnosticsPanel from './DiagnosticsPanel';
import ResultsTable, { COLUMN_DATA_MAP } from './ResultsTable';
import ColumnSettingsModal from './ColumnSettingsModal';
import ActiveFiltersBar from './ActiveFiltersBar';
import FilterPresetsBar from './FilterPresetsBar';
import {
  DEFAULT_COLUMNS,
  computeEffectiveColumns,
  mergeColumnsWithDefaults,
  filterTypeFor,
} from './columnsConfig';
import {
  applyColumnFilters,
  isFilterActive,
  activeFilterCount,
} from './columnFilters';

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

// Column filters are stored as a map of columnId → filter object.
// See columnFilters.js for filter shape.
// Persists across scan loads (per user request).
const DEFAULT_COLUMN_FILTERS = {};

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

  // Active background scan jobs. Each entry: { job_id, kind ('run'|'update'),
  // label, status, scan_id?, error?, started_at }. Populated when the user
  // submits the run or update dialog (which now return a job_id immediately
  // instead of blocking) and cleared when the job reaches a terminal state.
  // The jobs poll loop below keeps these in sync with the backend.
  const [activeJobs, setActiveJobs] = useState([])
  // Recently-completed jobs (last 60s) — surfaced as toasts so the user
  // sees outcomes when they're not on this tab. Auto-cleared after display.
  const [completedJobs, setCompletedJobs] = useState([])

  // Backend's authoritative defaults for the scan params. Fetched on mount.
  // Used to seed RunScanDialog so it pre-populates with whatever the backend
  // currently considers default. Falls back to FALLBACK_SCAN_DEFAULTS.
  const [serverDefaults, setServerDefaults] = useState(FALLBACK_SCAN_DEFAULTS)

  const [innerTab, setInnerTab] = useState('diagnostics')
  const [columnFilters, setColumnFilters] = useState(DEFAULT_COLUMN_FILTERS)

  // Saved filter presets — list of { id, name, notes, filters,
  // view_direction, created_at, updated_at } per scan. Hydrated from
  // loadedScan.filter_presets on load; autosaved to backend on change.
  const [filterPresets, setFilterPresets] = useState([])
  // The id of the preset most recently loaded or saved. Used to drive
  // Save vs. Save-as semantics: when set, the bar shows an Update button
  // that writes current filters back into this preset; when null, only
  // a Save-as-new button shows. Cleared on Clear-All, on delete of the
  // active preset, and on scan change.
  const [activePresetId, setActivePresetId] = useState(null)
  const lastSavedPresetsKeyRef = useRef(null)

  // Collapsible state for saved scans browser
  const [scansCollapsed, setScansCollapsed] = useState(false)

  // Selected row key for visual highlighting in the table
  const [selectedRowKey, setSelectedRowKey] = useState(null)
  const tableRef = useRef(null)

  // ── Column state ──
  // Initial value comes from a global localStorage key — used as the
  // default for any scan that has never been customized. Once a scan
  // gets its own column_prefs (via the autosave effect below), that
  // per-scan version takes precedence on load.
  const [columns, setColumns] = useState(() => {
    try {
      const saved = localStorage.getItem(COLUMNS_STORAGE_KEY)
      if (saved) return mergeColumnsWithDefaults(JSON.parse(saved))
    } catch (e) {
      // fall through to default
    }
    return DEFAULT_COLUMNS
  })

  // Keep the global localStorage in sync with the most-recent column
  // shape, so brand-new scans (no per-scan prefs yet) feel familiar.
  useEffect(() => {
    try {
      localStorage.setItem(COLUMNS_STORAGE_KEY, JSON.stringify(columns))
    } catch (e) {
      // localStorage might be full or disabled; fail silently
    }
  }, [columns])

  // Tracks the (scan_id, columns) shape we last wrote to the server, so
  // the autosave effect can skip no-op writes — including the very first
  // setColumns that fires when a scan loads (we don't want to immediately
  // POST the scan's own column_prefs back to itself).
  const lastSavedColumnsKeyRef = useRef(null)

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

  // ── Background scan-job polling ──
  // While there are queued/running scan jobs, poll their status every
  // 2 seconds. On completion, refresh the scan list and (for a 'run'
  // job) load the new scan; (for an 'update' job) reload the affected
  // scan if it's currently selected. The job is then moved to the
  // completedJobs list for a brief toast, then dropped.
  useEffect(() => {
    if (activeJobs.length === 0) return

    let cancelled = false
    const tick = async () => {
      // Pull current state once per tick so we batch updates cleanly.
      const jobsToCheck = activeJobs.filter(j => j.status === 'queued' || j.status === 'running')
      if (jobsToCheck.length === 0) return

      const updates = await Promise.all(jobsToCheck.map(async (j) => {
        try {
          const res = await fetch(`/api/backtests-v2/saved-scans/jobs/${j.job_id}`)
          const data = await res.json()
          if (!data.ok) return { ...j, status: 'failed', error: data.error || 'Job lookup failed' }
          return { ...j, ...data }
        } catch (err) {
          // Network blip — keep the job alive, try again next tick.
          return j
        }
      }))

      if (cancelled) return

      const stillActive = []
      const justCompleted = []
      for (const j of updates) {
        if (j.status === 'complete' || j.status === 'failed') {
          justCompleted.push(j)
        } else {
          stillActive.push(j)
        }
      }

      // Merge in any jobs that arrived between the snapshot and now.
      setActiveJobs(prev => {
        const updatedIds = new Set(updates.map(u => u.job_id))
        const newcomers = prev.filter(p => !updatedIds.has(p.job_id))
        return [...newcomers, ...stillActive]
      })

      if (justCompleted.length) {
        // Show as toasts for ~5s, then drop.
        setCompletedJobs(prev => [...prev, ...justCompleted])
        setTimeout(() => {
          if (cancelled) return
          setCompletedJobs(prev => prev.filter(c => !justCompleted.find(j => j.job_id === c.job_id)))
        }, 5000)

        // Side effects for completed jobs: reload data so the UI catches
        // up. Errors are surfaced via the toast and don't trigger reloads.
        const anySuccess = justCompleted.some(j => j.status === 'complete')
        if (anySuccess) {
          await refreshScans()
          for (const j of justCompleted) {
            if (j.status !== 'complete') continue
            if (j.kind === 'run' && j.scan_id) {
              // Load the newly-created scan so the user sees results.
              await loadScan(j.scan_id)
            } else if (j.kind === 'update' && j.scan_id && selectedScanId === j.scan_id) {
              // Re-load if the user is still looking at the updated scan.
              await loadScan(j.scan_id)
            }
          }
        }
      }
    }

    const interval = setInterval(tick, 2000)
    // Run once immediately so quick scans don't sit at "queued" for 2s.
    tick()
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [activeJobs, refreshScans, loadScan, selectedScanId])

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
      // Column filters PERSIST across scan loads — user explicitly chose this
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
          // Optional: present only via the duplicate flow, which seeds
          // them from the source scan so the copy is a true clone.
          filter_presets: newScan.filterPresets || undefined,
          column_prefs:   newScan.columnPrefs   || undefined,
        }),
      })
      const data = await res.json()
      if (!data.ok) throw new Error(data.error || 'Scan failed')
      // Backend now returns immediately with a job_id; the actual scan
      // runs on a background thread. Push onto activeJobs and let the
      // poll effect surface progress + load the result on completion.
      if (data.job_id) {
        setActiveJobs(prev => [...prev, {
          job_id: data.job_id,
          kind: 'run',
          label: newScan.label || `${newScan.direction.toUpperCase()} ${newScan.startDate} to ${newScan.endDate}`,
          status: 'queued',
          started_at: Date.now() / 1000,
        }])
      }
      setShowRunDialog(false)
      // Reset to defaults but seed with serverDefaults so next-run is sane
      setNewScan({ ...DEFAULT_NEW_SCAN, params: { ...serverDefaults } })
    } catch (err) {
      setError(String(err.message || err))
    } finally {
      setRunning(false)
    }
  }, [newScan, serverDefaults])

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

      if (data.rescanned && data.job_id) {
        // Re-scan runs in the background. Track the job so the user
        // can navigate away while it works; the poll effect will reload
        // the active scan when it completes.
        setActiveJobs(prev => [...prev, {
          job_id: data.job_id,
          kind: 'update',
          label: form.label || `Scan #${editingScan.scan_id}`,
          scan_id: editingScan.scan_id,
          status: 'queued',
          started_at: Date.now() / 1000,
        }])
      } else {
        // Metadata-only update completed synchronously — refresh now.
        await refreshScans()
        if (selectedScanId === editingScan.scan_id) {
          await loadScan(editingScan.scan_id)
        }
      }

      setEditingScan(null)
    } catch (err) {
      setError(String(err.message || err))
    } finally {
      setEditing(false)
    }
  }, [editingScan, refreshScans, selectedScanId, loadScan])

  // Duplicate a saved scan: pre-populate the run dialog with this scan's
  // params (and label/dates as a starting point) and open it. Filter
  // presets and column prefs are carried along so the duplicate is a
  // true clone — when it runs, the new scan ID will land with the same
  // saved presets and column layout as the source.
  const duplicateScan = useCallback(async (scan) => {
    if (!scan?.scan_id) return
    setError(null)
    try {
      // Need full row to access params + presets + column prefs
      // (the list endpoint omits all of these for size).
      const res = await fetch(`/api/backtests-v2/saved-scans/${scan.scan_id}`)
      const data = await res.json()
      if (!data.ok) throw new Error(data.error || 'Failed to load scan for duplication')
      const sourceParams = data.params || {}
      // Project sourceParams onto the form's known keys; anything else is dropped
      const formParams = { ...serverDefaults }
      for (const k of Object.keys(FALLBACK_SCAN_DEFAULTS)) {
        if (sourceParams[k] !== undefined) formParams[k] = sourceParams[k]
      }
      // Re-stamp preset IDs so the duplicate has its own identifiers
      // even though the content is identical to the source's presets.
      const sourcePresets = Array.isArray(data.filter_presets) ? data.filter_presets : []
      const duplicatedPresets = sourcePresets.map(p => ({
        ...p,
        id: `preset_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
        // Preserve original timestamps so notes about "saved on X" stay
        // semantically meaningful — the duplicate inherits its history.
      }))
      setNewScan({
        direction: data.direction || 'down',
        startDate: data.start_date || isoDateOffset(-90),
        endDate:   data.end_date   || isoDateOffset(0),
        label:    `${data.label || `Scan #${scan.scan_id}`} (copy)`,
        notes:    data.notes || '',
        params:   formParams,
        filterPresets: duplicatedPresets,
        columnPrefs:   Array.isArray(data.column_prefs) ? data.column_prefs : null,
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
    () => applyColumnFilters(allRows, columnFilters, COLUMN_DATA_MAP),
    [allRows, columnFilters]
  )

  const filtersAreActive = useMemo(
    () => activeFilterCount(columnFilters) > 0,
    [columnFilters]
  )

  const filteredDiagnostics = useMemo(() => {
    if (!loadedScan) return null
    return computeFilteredDiagnostics(loadedScan, filteredRows, filtersAreActive)
  }, [loadedScan, filteredRows, filtersAreActive])

  const effectiveExecutionMode =
    loadedScan?.params?.executionMode || 'study_target_hits'

  // ── Forward-outcomes direction toggle ──
  // Lives at the SavedScans level so a single toggle (rendered inside the
  // Diagnostics panel's "Forward Outcomes by Horizon" card) drives both
  // that card AND the fwd_* columns in the Instances tab. Defaults to the
  // scan's original setup; resets when a different scan loads.
  const originalTradeDirection =
    loadedScan?.direction === 'up' ? 'short'
    : loadedScan?.direction === 'down' ? 'long'
    : 'short'
  const [forwardOutcomesView, setForwardOutcomesView] = useState(originalTradeDirection)
  useEffect(() => {
    setForwardOutcomesView(originalTradeDirection)
  }, [originalTradeDirection, loadedScan?.scan_id])
  const forwardOutcomesFlipped = forwardOutcomesView !== originalTradeDirection

  // ── Per-scan column prefs: hydrate on load, autosave on change ──
  // When a scan loads with its own column_prefs, snap `columns` to that
  // version and remember the snapshot so the autosave effect knows to
  // skip the freshly-loaded state. When a scan loads WITHOUT prefs, we
  // keep whatever columns are currently in memory (the global default
  // from localStorage) and let the user's first edit save them per-scan.
  useEffect(() => {
    if (!loadedScan?.scan_id) return
    const incoming = Array.isArray(loadedScan.column_prefs) && loadedScan.column_prefs.length > 0
      ? mergeColumnsWithDefaults(loadedScan.column_prefs)
      : null
    if (incoming) {
      setColumns(incoming)
      lastSavedColumnsKeyRef.current = `${loadedScan.scan_id}:${JSON.stringify(incoming.map(c => ({ id: c.id, visible: c.visible })))}`
    } else {
      // No saved prefs yet for this scan. Mark current columns as the
      // "baseline" so we don't immediately POST them — the first real
      // user edit will be what gets saved.
      lastSavedColumnsKeyRef.current = `${loadedScan.scan_id}:${JSON.stringify(columns.map(c => ({ id: c.id, visible: c.visible })))}`
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadedScan?.scan_id])

  // Autosave column changes back to the scan, debounced. Skips writes
  // that match the last-saved snapshot (including the post-hydration
  // baseline above) so we don't echo the load back as a save.
  useEffect(() => {
    if (!loadedScan?.scan_id) return
    const minimal = columns.map(c => ({ id: c.id, visible: c.visible }))
    const key = `${loadedScan.scan_id}:${JSON.stringify(minimal)}`
    if (lastSavedColumnsKeyRef.current === key) return

    const t = setTimeout(() => {
      fetch(`/api/backtests-v2/saved-scans/${loadedScan.scan_id}/column-prefs`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ column_prefs: minimal }),
      })
        .then(r => r.json())
        .then(data => {
          if (data?.ok) lastSavedColumnsKeyRef.current = key
        })
        .catch(() => {
          // Network failure shouldn't disrupt the UI; the user will
          // see their column changes locally. Next change will retry.
        })
    }, 500)
    return () => clearTimeout(t)
  }, [columns, loadedScan?.scan_id])

  // ── Per-scan filter presets: hydrate on load, autosave on change ──
  // Same pattern as column_prefs above. Presets are stored as a list
  // on the scan; mutations replace the whole list via PATCH.
  useEffect(() => {
    if (!loadedScan?.scan_id) {
      setFilterPresets([])
      setActivePresetId(null)
      lastSavedPresetsKeyRef.current = null
      return
    }
    const incoming = Array.isArray(loadedScan.filter_presets)
      ? loadedScan.filter_presets
      : []
    setFilterPresets(incoming)
    setActivePresetId(null)
    lastSavedPresetsKeyRef.current = `${loadedScan.scan_id}:${JSON.stringify(incoming)}`
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadedScan?.scan_id])

  useEffect(() => {
    if (!loadedScan?.scan_id) return
    const key = `${loadedScan.scan_id}:${JSON.stringify(filterPresets)}`
    if (lastSavedPresetsKeyRef.current === key) return

    const t = setTimeout(() => {
      fetch(`/api/backtests-v2/saved-scans/${loadedScan.scan_id}/filter-presets`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filter_presets: filterPresets }),
      })
        .then(r => r.json())
        .then(data => {
          if (data?.ok) lastSavedPresetsKeyRef.current = key
        })
        .catch(() => {
          // Same defensive posture as column autosave.
        })
    }, 400)
    return () => clearTimeout(t)
  }, [filterPresets, loadedScan?.scan_id])

  // Filter-preset action handlers
  const handleCreatePreset = useCallback(({ name, notes, filters, view_direction }) => {
    const now = new Date().toISOString()
    const id = `preset_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
    setFilterPresets(prev => [
      ...prev,
      { id, name, notes: notes || '', filters: filters || {}, view_direction: view_direction || null, created_at: now, updated_at: now },
    ])
    // Newly-created preset becomes the active one — subsequent edits
    // will Update it (rather than create yet another).
    setActivePresetId(id)
  }, [])

  const handleUpdatePreset = useCallback((id, { name, notes }) => {
    const now = new Date().toISOString()
    setFilterPresets(prev => prev.map(p =>
      p.id === id ? { ...p, name, notes: notes || '', updated_at: now } : p
    ))
  }, [])

  // Update an existing preset's stored filters + direction with whatever
  // the user has currently active. The "Save" half of Save vs Save-As.
  const handleUpdatePresetFilters = useCallback((id) => {
    if (!id) return
    const now = new Date().toISOString()
    setFilterPresets(prev => prev.map(p =>
      p.id === id
        ? {
            ...p,
            filters: { ...columnFilters },
            view_direction: forwardOutcomesView,
            updated_at: now,
          }
        : p
    ))
  }, [columnFilters, forwardOutcomesView])

  const handleDeletePreset = useCallback((id) => {
    setFilterPresets(prev => prev.filter(p => p.id !== id))
    // If the user just deleted what was active, drop the active marker.
    setActivePresetId(prev => (prev === id ? null : prev))
  }, [])

  const handleApplyPreset = useCallback((preset) => {
    if (!preset) return
    // Replace current filters wholesale with the preset's snapshot.
    setColumnFilters(preset.filters || {})
    if (preset.view_direction === 'long' || preset.view_direction === 'short') {
      setForwardOutcomesView(preset.view_direction)
    }
    setActivePresetId(preset.id)
  }, [])

  // Apply study/managed visibility rules on top of user's column prefs
  const effectiveColumns = useMemo(
    () => computeEffectiveColumns(columns, effectiveExecutionMode),
    [columns, effectiveExecutionMode]
  )

  // Column-filter handlers
  const handleColumnFilterChange = useCallback((columnId, filterObject) => {
    setColumnFilters(prev => {
      const next = { ...prev }
      if (filterObject == null) {
        delete next[columnId]
      } else {
        next[columnId] = filterObject
      }
      return next
    })
  }, [])

  const clearAllFilters = useCallback(() => {
    setColumnFilters({})
    // Clearing filters means "fresh start" — no preset is active anymore.
    setActivePresetId(null)
  }, [])

  // Map of columnId → label for the active-filters chip bar
  const columnLabelMap = useMemo(() => {
    const out = {}
    for (const c of DEFAULT_COLUMNS) out[c.id] = c.label
    return out
  }, [])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16, flex: 1, minHeight: 0 }}>
      {/* Background-job indicator. Visible regardless of card collapsed
          state so the user knows scans are running even with the
          saved-scan list closed. Also visible on other tabs because
          this component is always mounted. */}
      <BackgroundJobsBar activeJobs={activeJobs} completedJobs={completedJobs} />

      {/* Saved scans browser */}
      <div className="diag-card" style={{ flex: '0 0 auto' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: scansCollapsed ? 0 : 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <button
              onClick={() => setScansCollapsed(!scansCollapsed)}
              style={{
                background: 'none',
                border: 'none',
                color: '#93c5fd',
                cursor: 'pointer',
                fontSize: 16,
                padding: 4,
                display: 'flex',
                alignItems: 'center',
              }}
              title={scansCollapsed ? 'Expand' : 'Collapse'}
            >
              {scansCollapsed ? '▶' : '▼'}
            </button>
            <div>
              <div className="eyebrow">Saved Scans</div>
              <h2 style={{ fontSize: 18 }}>Cached scan results</h2>
              {!scansCollapsed && (
                <p style={{ color: '#94a3b8', fontSize: 12, margin: '4px 0 0' }}>
                  Run a permissive scan once, then explore the result without re-scanning.
                </p>
              )}
            </div>
          </div>
          <button
            className="primary-button"
            onClick={() => setShowRunDialog(true)}
            disabled={running}
          >
            + Run &amp; Save Scan
          </button>
        </div>

        {!scansCollapsed && (
          <>
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
          </>
        )}
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
            view={forwardOutcomesView}
            onViewChange={setForwardOutcomesView}
            originalTrade={originalTradeDirection}
          />

          <FilterPresetsBar
            presets={filterPresets}
            activeFilters={columnFilters}
            activeView={forwardOutcomesView}
            originalTrade={originalTradeDirection}
            activePresetId={activePresetId}
            onApplyPreset={handleApplyPreset}
            onCreatePreset={handleCreatePreset}
            onUpdatePreset={handleUpdatePreset}
            onUpdatePresetFilters={handleUpdatePresetFilters}
            onDeletePreset={handleDeletePreset}
          />

          <div style={{ display: 'flex', gap: 8, alignItems: 'stretch' }}>
            <button
              className="ghost-button"
              onClick={() => setIsPresetsModalOpen(true)}
              title="Save and recall named filter combinations for this scan"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                padding: '0 14px',
                fontSize: 12,
                fontWeight: 600,
                whiteSpace: 'nowrap',
                borderRadius: 14,
                background: filterPresets.length > 0 ? 'rgba(37, 99, 235, 0.10)' : 'transparent',
                borderColor: filterPresets.length > 0 ? 'rgba(37, 99, 235, 0.40)' : undefined,
              }}
            >
              📋 Presets
              {filterPresets.length > 0 && (
                <span style={{
                  fontSize: 11,
                  color: '#bfdbfe',
                  background: 'rgba(37, 99, 235, 0.25)',
                  padding: '1px 7px',
                  borderRadius: 999,
                  fontWeight: 700,
                }}>
                  {filterPresets.length}
                </span>
              )}
            </button>
            <div style={{ flex: 1, minWidth: 0 }}>
              <ActiveFiltersBar
                filters={columnFilters}
                columnLabels={columnLabelMap}
                filteredCount={filteredRows.length}
                totalCount={allRows.length}
                onClearOne={(colId) => handleColumnFilterChange(colId, null)}
                onClearAll={clearAllFilters}
              />
            </div>
          </div>

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
            <button
              className={`tab-button ${innerTab === 'pipeline' ? 'active' : ''}`}
              onClick={() => setInnerTab('pipeline')}
            >
              Pipeline
            </button>
          </div>

          {innerTab === 'diagnostics' && (
            <DiagnosticsPanel
              diagnostics={filteredDiagnostics}
              rows={filteredRows}
              funnel={[]}
              executionMode={effectiveExecutionMode}
              viewDirection={forwardOutcomesView}
              onViewDirectionChange={setForwardOutcomesView}
              originalTrade={originalTradeDirection}
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
                columnFilters={columnFilters}
                onColumnFilterChange={handleColumnFilterChange}
                filterTypeForColumn={filterTypeFor}
                flippedForwardOutcomes={forwardOutcomesFlipped}
              />
            </div>
          )}

          {innerTab === 'pipeline' && (
            <PipelinePanel funnel={loadedScan?.funnel || []} />
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
// Diagnostics recomputation from filtered rows
// ──────────────────────────────────────────────────────────────────────

function computeFilteredDiagnostics(loadedScan, filteredRows, filtersAreActive) {
  const base = { ...(loadedScan?.diagnostics || {}) }

  if (!filtersAreActive) {
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
// Pipeline Panel — Displays the funnel stages
// ──────────────────────────────────────────────────────────────────────

function FunnelStage({ stage, index }) {
  const { label, kind, bypassed, candidates_in, kept, dropped, drop_reasons } = stage;
  const pct = candidates_in > 0 ? (kept / candidates_in) * 100 : 0;

  const kindColor = kind === 'construction' ? '#3b82f6' : kind === 'filter' ? '#10b981' : '#64748b';
  const barColor = bypassed ? '#475569' : kindColor;

  return (
    <div style={{
      background: '#0f172a',
      border: '1px solid #1e293b',
      borderRadius: '10px',
      padding: '12px 14px',
      marginBottom: '8px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{
            fontSize: '11px',
            fontWeight: '700',
            color: '#64748b',
            minWidth: '20px',
          }}>
            {index + 1}.
          </span>
          <span style={{
            fontSize: '13px',
            fontWeight: '600',
            color: bypassed ? '#64748b' : '#e2e8f0',
            opacity: bypassed ? 0.6 : 1,
          }}>
            {label}
          </span>
          {bypassed && (
            <span style={{
              fontSize: '9px',
              fontWeight: '700',
              color: '#64748b',
              background: '#1e293b',
              padding: '2px 6px',
              borderRadius: '4px',
              letterSpacing: '0.05em',
            }}>
              BYPASSED
            </span>
          )}
        </div>
        <div style={{ fontSize: '12px', color: '#94a3b8', fontWeight: '500' }}>
          {kept} / {candidates_in}
          {dropped > 0 && !bypassed && (
            <span style={{ color: '#f87171', marginLeft: '8px' }}>
              (−{dropped})
            </span>
          )}
        </div>
      </div>

      <div style={{
        height: '6px',
        background: '#1e293b',
        borderRadius: '3px',
        overflow: 'hidden',
      }}>
        <div style={{
          height: '100%',
          width: `${pct}%`,
          background: barColor,
          transition: 'width 0.3s ease',
        }} />
      </div>

      {!bypassed && dropped > 0 && drop_reasons && Object.keys(drop_reasons).length > 0 && (
        <div style={{ marginTop: '8px', fontSize: '11px', color: '#64748b' }}>
          {Object.entries(drop_reasons).map(([reason, count]) => (
            <div key={reason} style={{ marginTop: '2px' }}>
              • {reason.replace(/_/g, ' ')}: {count}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function PipelinePanel({ funnel }) {
  if (!funnel || funnel.length === 0) {
    return (
      <div className="diag-card" style={{ flex: 1 }}>
        <div className="results-header">
          <h2>Pipeline Funnel</h2>
          <p>No pipeline data available for this scan.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="diag-card" style={{ flex: 1, overflow: 'auto' }}>
      <div className="results-header">
        <h2>Pipeline Funnel</h2>
        <p>Shows how candidates flow through each stage of the scan pipeline.</p>
      </div>
      <div style={{ marginTop: '16px' }}>
        {funnel.map((stage, idx) => (
          <FunnelStage key={stage.key} stage={stage} index={idx} />
        ))}
      </div>
    </div>
  );
}


// ──────────────────────────────────────────────────────────────────────
// UI components
// ──────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────
// BackgroundJobsBar
//   Compact strip showing in-progress scan jobs and recently-completed
//   ones. Renders nothing when there's no activity, so it's invisible
//   in the steady state. Active jobs show an animated "running"
//   indicator + elapsed seconds; completed jobs show a brief toast
//   announcing success or failure.
// ─────────────────────────────────────────────────────────────────────
function BackgroundJobsBar({ activeJobs, completedJobs }) {
  if (!activeJobs.length && !completedJobs.length) return null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {activeJobs.map(job => (
        <ActiveJobRow key={job.job_id} job={job} />
      ))}
      {completedJobs.map(job => (
        <CompletedJobToast key={job.job_id} job={job} />
      ))}
    </div>
  )
}

function ActiveJobRow({ job }) {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    const interval = setInterval(() => {
      setElapsed(Math.max(0, Math.floor(Date.now() / 1000 - (job.started_at || 0))))
    }, 1000)
    return () => clearInterval(interval)
  }, [job.started_at])

  const statusLabel = job.status === 'queued' ? 'Queued' : 'Running'
  const verb = job.kind === 'update' ? 'Re-scanning' : 'Scanning'

  return (
    <div style={{
      background: '#0b1220',
      border: '1px solid rgba(96, 165, 250, 0.40)',
      borderRadius: 10,
      padding: '8px 12px',
      display: 'flex',
      alignItems: 'center',
      gap: 10,
      fontSize: 12,
      color: '#bfdbfe',
    }}>
      <span style={{
        display: 'inline-block',
        width: 8,
        height: 8,
        borderRadius: '50%',
        background: job.status === 'queued' ? '#fcd34d' : '#60a5fa',
        animation: 'pulse 1.4s ease-in-out infinite',
        boxShadow: '0 0 8px currentColor',
      }} />
      <span style={{
        fontSize: 11,
        fontWeight: 700,
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
        color: job.status === 'queued' ? '#fcd34d' : '#60a5fa',
      }}>
        {statusLabel}
      </span>
      <span style={{ flex: 1 }}>
        {verb} <strong style={{ color: '#e2e8f0' }}>{job.label}</strong>
        <span style={{ color: '#64748b', marginLeft: 8 }}>· {elapsed}s elapsed</span>
      </span>
      <span style={{ fontSize: 11, color: '#64748b' }}>
        Continues in background — feel free to navigate
      </span>
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.35; }
        }
      `}</style>
    </div>
  )
}

function CompletedJobToast({ job }) {
  const isSuccess = job.status === 'complete'
  return (
    <div style={{
      background: isSuccess ? 'rgba(34, 197, 94, 0.10)' : 'rgba(239, 68, 68, 0.10)',
      border: `1px solid ${isSuccess ? '#22c55e' : '#ef4444'}`,
      borderRadius: 10,
      padding: '8px 12px',
      display: 'flex',
      alignItems: 'center',
      gap: 10,
      fontSize: 12,
      color: isSuccess ? '#86efac' : '#fca5a5',
    }}>
      <span style={{ fontSize: 14 }}>{isSuccess ? '✓' : '⚠'}</span>
      <span style={{ flex: 1 }}>
        {isSuccess ? (
          <>
            <strong>{job.label}</strong>
            {' '}— finished with {typeof job.row_count === 'number' ? job.row_count : '?'} rows
          </>
        ) : (
          <>
            <strong>{job.label}</strong>
            {' '}— failed: {job.error || 'unknown error'}
          </>
        )}
      </span>
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


function LoadedScanHeader({
  scan,
  allCount,
  filteredCount,
  filtersActive,
  view,                  // 'long' | 'short'
  onViewChange,          // (next) => void
  originalTrade,         // 'long' | 'short' — the scan's natural setup
}) {
  if (!scan) return null

  // Segmented pill button styles for the Long/Short toggle.
  const segBtn = (side) => {
    const active = view === side
    return {
      background: active
        ? (side === 'long' ? 'rgba(34, 197, 94, 0.18)' : 'rgba(239, 68, 68, 0.18)')
        : 'transparent',
      border: `1px solid ${active
        ? (side === 'long' ? '#22c55e' : '#ef4444')
        : '#334155'}`,
      color: active
        ? (side === 'long' ? '#86efac' : '#fca5a5')
        : '#94a3b8',
      padding: '4px 14px',
      cursor: active ? 'default' : 'pointer',
      fontSize: '11px',
      fontWeight: 700,
      letterSpacing: '0.05em',
      textTransform: 'uppercase',
    }
  }
  const flipped = view && originalTrade && view !== originalTrade

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

        {/* Long/Short view toggle — drives both Diagnostics aggregate and Instances fwd_* columns. */}
        {view && onViewChange && originalTrade && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4 }}>
            <span style={{
              fontSize: 10,
              color: flipped ? '#fcd34d' : '#64748b',
              fontWeight: 700,
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
            }}>
              Forward Outcomes View{flipped ? ' · Reversed' : ''}
            </span>
            <div
              role="group"
              aria-label="Forward outcomes direction view"
              title={`Original setup: ${originalTrade.toUpperCase()}. Click the other side to view forward outcomes (Diagnostics aggregate + fwd_* columns) as if every trade were taken in the opposite direction.`}
              style={{
                display: 'inline-flex',
                border: '1px solid #1e293b',
                borderRadius: '8px',
                overflow: 'hidden',
                background: '#0b1220',
              }}
            >
              <button
                type="button"
                onClick={() => onViewChange('long')}
                disabled={view === 'long'}
                style={{
                  ...segBtn('long'),
                  borderTopLeftRadius: 7,
                  borderBottomLeftRadius: 7,
                  borderRight: 'none',
                }}
              >
                Long{originalTrade === 'long' ? ' ●' : ''}
              </button>
              <button
                type="button"
                onClick={() => onViewChange('short')}
                disabled={view === 'short'}
                style={{
                  ...segBtn('short'),
                  borderTopRightRadius: 7,
                  borderBottomRightRadius: 7,
                }}
              >
                Short{originalTrade === 'short' ? ' ●' : ''}
              </button>
            </div>
          </div>
        )}
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
            <option value="both">Primary + Strong</option>
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
