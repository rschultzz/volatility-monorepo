// Column filter engine for the saved-scans Instances table.
//
// State shape (persisted across scan loads in SavedScans):
//   columnFilters: { [columnId]: { type, ...config } }
//
// Filter shapes:
//   numeric:     { type: 'numeric', op: '<' | '>' | '=' | 'between', value: number | [number, number] }
//   date:        { type: 'date', from?: 'YYYY-MM-DD', to?: 'YYYY-MM-DD' }
//   categorical: { type: 'categorical', selected: string[] }
//
// "Active" rules:
//   numeric:     value is a finite number (or [num, num] for between)
//   date:        from or to is non-empty
//   categorical: selected is non-empty
//
// Predicate engine resolves each filter against each row using the dataMap
// (COLUMN_DATA_MAP from ResultsTable).

// ── Active-filter detection ────────────────────────────────────────────

export function isFilterActive(filter) {
  if (!filter || typeof filter !== 'object') return false
  if (filter.type === 'numeric') {
    if (filter.op === 'between') {
      const [a, b] = filter.value || []
      return Number.isFinite(Number(a)) && Number.isFinite(Number(b))
    }
    return Number.isFinite(Number(filter.value))
  }
  if (filter.type === 'date') {
    return Boolean((filter.from && String(filter.from).trim()) || (filter.to && String(filter.to).trim()))
  }
  if (filter.type === 'categorical') {
    return Array.isArray(filter.selected) && filter.selected.length > 0
  }
  return false
}

export function activeFilterCount(filters) {
  if (!filters) return 0
  return Object.values(filters).filter(isFilterActive).length
}

// ── Value extraction ───────────────────────────────────────────────────
// dataMap is COLUMN_DATA_MAP — values are either string (key into row) or
// (row) => value. Mirror that resolution here.

export function getColumnValue(row, columnId, dataMap) {
  if (!row) return undefined
  const accessor = dataMap?.[columnId]
  if (typeof accessor === 'function') {
    try { return accessor(row) } catch { return undefined }
  }
  if (typeof accessor === 'string') {
    return row[accessor]
  }
  // Fallback — try direct key
  return row[columnId]
}

// ── Predicate per filter type ──────────────────────────────────────────

function passesNumericFilter(value, filter) {
  const v = Number(value)
  if (!Number.isFinite(v)) return false  // null/undefined/non-numeric fails
  if (filter.op === 'between') {
    const [a, b] = filter.value || []
    const lo = Number(a), hi = Number(b)
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return true
    const [actualLo, actualHi] = lo <= hi ? [lo, hi] : [hi, lo]
    return v >= actualLo && v <= actualHi
  }
  const t = Number(filter.value)
  if (!Number.isFinite(t)) return true
  if (filter.op === '<') return v < t
  if (filter.op === '>') return v > t
  if (filter.op === '=') return v === t
  if (filter.op === '<=') return v <= t
  if (filter.op === '>=') return v >= t
  return true
}

function passesDateFilter(value, filter) {
  const s = value == null ? '' : String(value).trim()
  if (!s) return false
  // Compare on ISO date prefix (YYYY-MM-DD) — works for trade_date strings,
  // also for full timestamps that start with the date.
  const datePart = s.slice(0, 10)
  if (filter.from && String(filter.from).trim()) {
    if (datePart < String(filter.from).trim()) return false
  }
  if (filter.to && String(filter.to).trim()) {
    if (datePart > String(filter.to).trim()) return false
  }
  return true
}

function passesCategoricalFilter(value, filter) {
  const selected = Array.isArray(filter.selected) ? filter.selected : []
  if (!selected.length) return true
  // Coerce value to string for comparison; treat null/undefined as the
  // literal string '(empty)' so user can include or exclude them as a
  // distinct bucket.
  let key
  if (value === null || value === undefined || value === '') {
    key = '(empty)'
  } else if (typeof value === 'boolean') {
    key = value ? 'true' : 'false'
  } else {
    key = String(value)
  }
  return selected.includes(key)
}

// ── Apply all filters ──────────────────────────────────────────────────

export function applyColumnFilters(rows, filters, dataMap) {
  if (!Array.isArray(rows) || !rows.length) return rows || []
  const active = Object.entries(filters || {}).filter(([_, f]) => isFilterActive(f))
  if (!active.length) return rows

  return rows.filter(row => {
    for (const [columnId, filter] of active) {
      const value = getColumnValue(row, columnId, dataMap)
      if (filter.type === 'numeric'     && !passesNumericFilter(value, filter))     return false
      if (filter.type === 'date'        && !passesDateFilter(value, filter))        return false
      if (filter.type === 'categorical' && !passesCategoricalFilter(value, filter)) return false
    }
    return true
  })
}

// ── Categorical: enumerate distinct values ─────────────────────────────

export function distinctValues(rows, columnId, dataMap, limit = 50) {
  if (!Array.isArray(rows)) return []
  const counts = new Map()
  for (const row of rows) {
    const v = getColumnValue(row, columnId, dataMap)
    let key
    if (v === null || v === undefined || v === '') key = '(empty)'
    else if (typeof v === 'boolean') key = v ? 'true' : 'false'
    else key = String(v)
    counts.set(key, (counts.get(key) || 0) + 1)
  }
  // Sort: most frequent first, ties alphabetical
  const entries = [...counts.entries()].sort((a, b) => {
    if (b[1] !== a[1]) return b[1] - a[1]
    return a[0].localeCompare(b[0])
  })
  return entries.slice(0, limit).map(([value, count]) => ({ value, count }))
}

// ── Filter chip summary ────────────────────────────────────────────────
// Build a human-readable description of an active filter for the chip UI.

export function describeFilter(columnId, filter, columnLabel) {
  if (!isFilterActive(filter)) return null
  const label = columnLabel || columnId
  if (filter.type === 'numeric') {
    if (filter.op === 'between') {
      const [a, b] = filter.value
      return `${label}: ${a} – ${b}`
    }
    return `${label} ${filter.op} ${filter.value}`
  }
  if (filter.type === 'date') {
    const parts = []
    if (filter.from) parts.push(`from ${filter.from}`)
    if (filter.to)   parts.push(`to ${filter.to}`)
    return `${label}: ${parts.join(' ')}`
  }
  if (filter.type === 'categorical') {
    const sel = filter.selected || []
    if (sel.length === 1) return `${label}: ${sel[0]}`
    if (sel.length <= 3)  return `${label}: ${sel.join(', ')}`
    return `${label}: ${sel.length} values`
  }
  return null
}

// ── Default filter for a given filterType ──────────────────────────────
// Returns a "blank" filter object. Used when a user opens the popover for
// a column that doesn't yet have a filter set.

export function defaultFilterFor(filterType) {
  if (filterType === 'numeric') {
    return { type: 'numeric', op: '>', value: '' }
  }
  if (filterType === 'date') {
    return { type: 'date', from: '', to: '' }
  }
  if (filterType === 'categorical') {
    return { type: 'categorical', selected: [] }
  }
  return null
}
