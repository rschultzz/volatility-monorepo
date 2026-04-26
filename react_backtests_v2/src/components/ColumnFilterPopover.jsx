import React, { useEffect, useRef, useState } from 'react';
import { defaultFilterFor, distinctValues } from './columnFilters';

// ColumnFilterPopover — opens below a column-header filter icon.
//
// Props:
//   filterType: 'numeric' | 'date' | 'categorical'
//   filter: current filter object (or null/undefined if none active)
//   anchorRect: DOMRect of the icon that opened it (for positioning)
//   columnLabel: string for the popover title
//   columnId: column id (for distinct-value enumeration)
//   rows: full row set (used for categorical distinct values)
//   dataMap: COLUMN_DATA_MAP (used for distinct values)
//   onChange: (filterObject | null) => void   — called when user updates
//   onClear: () => void  — called by Clear button
//   onClose: () => void  — close popover

export default function ColumnFilterPopover({
  filterType,
  filter,
  anchorRect,
  columnLabel,
  columnId,
  rows,
  dataMap,
  onChange,
  onClear,
  onClose,
}) {
  const popoverRef = useRef(null)

  // Local draft state — applied to onChange on every change so filtering is live
  const [draft, setDraft] = useState(() => filter || defaultFilterFor(filterType))

  // Sync draft when filter prop changes externally (e.g. Clear All)
  useEffect(() => {
    setDraft(filter || defaultFilterFor(filterType))
  }, [filter, filterType])

  // Close on click outside or Escape
  useEffect(() => {
    function handleClick(e) {
      if (popoverRef.current && !popoverRef.current.contains(e.target)) {
        onClose && onClose()
      }
    }
    function handleKey(e) {
      if (e.key === 'Escape') onClose && onClose()
    }
    document.addEventListener('mousedown', handleClick)
    document.addEventListener('keydown', handleKey)
    return () => {
      document.removeEventListener('mousedown', handleClick)
      document.removeEventListener('keydown', handleKey)
    }
  }, [onClose])

  // Position: below anchor by default; flip if it would clip the bottom
  const POPOVER_W = 280
  const POPOVER_MARGIN = 12
  const top = anchorRect ? anchorRect.bottom + 4 : 100
  let left = anchorRect ? anchorRect.left : 100
  if (typeof window !== 'undefined') {
    const maxLeft = window.innerWidth - POPOVER_W - POPOVER_MARGIN
    if (left > maxLeft) left = Math.max(POPOVER_MARGIN, maxLeft)
  }

  // Update draft AND propagate to parent in one shot
  const updateDraft = (next) => {
    setDraft(next)
    onChange && onChange(next)
  }

  return (
    <div
      ref={popoverRef}
      style={{
        position: 'fixed',
        top, left,
        width: POPOVER_W,
        background: '#0f172a',
        border: '1px solid #334155',
        borderRadius: 10,
        boxShadow: '0 12px 32px rgba(0,0,0,0.6)',
        padding: 12,
        zIndex: 3000,
        fontSize: 12,
        color: '#e5e7eb',
      }}
      onClick={(e) => e.stopPropagation()}
    >
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 8,
      }}>
        <div style={{ fontSize: 11, color: '#94a3b8', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          Filter: {columnLabel}
        </div>
        <button
          onClick={onClose}
          style={{
            background: 'none', border: 'none', color: '#94a3b8',
            cursor: 'pointer', fontSize: 14, padding: 0, width: 20, height: 20,
          }}
          title="Close"
        >×</button>
      </div>

      {filterType === 'numeric' && (
        <NumericFilterControls draft={draft} onChange={updateDraft} />
      )}
      {filterType === 'date' && (
        <DateFilterControls draft={draft} onChange={updateDraft} />
      )}
      {filterType === 'categorical' && (
        <CategoricalFilterControls
          draft={draft}
          onChange={updateDraft}
          rows={rows}
          columnId={columnId}
          dataMap={dataMap}
        />
      )}

      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        marginTop: 12,
        paddingTop: 8,
        borderTop: '1px solid #1e293b',
      }}>
        <button
          onClick={() => { onClear && onClear(); onClose && onClose() }}
          style={{
            background: 'rgba(239,68,68,0.10)',
            border: '1px solid #475569',
            color: '#fca5a5',
            padding: '4px 10px',
            borderRadius: 6,
            cursor: 'pointer',
            fontSize: 11,
            fontWeight: 600,
          }}
        >
          Clear filter
        </button>
        <button
          onClick={onClose}
          style={{
            background: '#2563eb',
            border: '1px solid #3b82f6',
            color: 'white',
            padding: '4px 12px',
            borderRadius: 6,
            cursor: 'pointer',
            fontSize: 11,
            fontWeight: 700,
          }}
        >
          Done
        </button>
      </div>
    </div>
  )
}

// ── Numeric controls ───────────────────────────────────────────────────

function NumericFilterControls({ draft, onChange }) {
  const inputStyle = {
    background: '#020617',
    border: '1px solid #334155',
    color: '#e5e7eb',
    borderRadius: 6,
    padding: '6px 8px',
    fontSize: 12,
    width: '100%',
    fontFamily: 'inherit',
  }
  const op = draft?.op || '>'
  const isBetween = op === 'between'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <select
        style={inputStyle}
        value={op}
        onChange={(e) => {
          const newOp = e.target.value
          if (newOp === 'between') {
            const v = Array.isArray(draft.value) ? draft.value : ['', '']
            onChange({ ...draft, op: newOp, value: v })
          } else {
            const single = Array.isArray(draft.value) ? draft.value[0] : draft.value
            onChange({ ...draft, op: newOp, value: single ?? '' })
          }
        }}
      >
        <option value=">">Greater than (&gt;)</option>
        <option value="<">Less than (&lt;)</option>
        <option value=">=">Greater or equal (&gt;=)</option>
        <option value="<=">Less or equal (&lt;=)</option>
        <option value="=">Equals (=)</option>
        <option value="between">Between</option>
      </select>

      {isBetween ? (
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <input
            type="number"
            step="any"
            style={inputStyle}
            value={(draft.value && draft.value[0]) ?? ''}
            onChange={(e) => onChange({ ...draft, value: [e.target.value, (draft.value || [])[1]] })}
            placeholder="min"
          />
          <span style={{ color: '#64748b', fontSize: 11 }}>and</span>
          <input
            type="number"
            step="any"
            style={inputStyle}
            value={(draft.value && draft.value[1]) ?? ''}
            onChange={(e) => onChange({ ...draft, value: [(draft.value || [])[0], e.target.value] })}
            placeholder="max"
          />
        </div>
      ) : (
        <input
          type="number"
          step="any"
          style={inputStyle}
          value={draft.value ?? ''}
          onChange={(e) => onChange({ ...draft, value: e.target.value })}
          placeholder="value"
          autoFocus
        />
      )}
    </div>
  )
}

// ── Date controls ──────────────────────────────────────────────────────

function DateFilterControls({ draft, onChange }) {
  const inputStyle = {
    background: '#020617',
    border: '1px solid #334155',
    color: '#e5e7eb',
    borderRadius: 6,
    padding: '6px 8px',
    fontSize: 12,
    width: '100%',
    fontFamily: 'inherit',
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <span style={{ fontSize: 10, color: '#94a3b8' }}>From</span>
        <input
          type="date"
          style={inputStyle}
          value={draft.from || ''}
          onChange={(e) => onChange({ ...draft, from: e.target.value })}
        />
      </label>
      <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <span style={{ fontSize: 10, color: '#94a3b8' }}>To</span>
        <input
          type="date"
          style={inputStyle}
          value={draft.to || ''}
          onChange={(e) => onChange({ ...draft, to: e.target.value })}
        />
      </label>
    </div>
  )
}

// ── Categorical controls ───────────────────────────────────────────────

function CategoricalFilterControls({ draft, onChange, rows, columnId, dataMap }) {
  // Enumerate distinct values from the row set
  const distinct = React.useMemo(
    () => distinctValues(rows, columnId, dataMap, 100),
    [rows, columnId, dataMap]
  )

  const selected = new Set(Array.isArray(draft.selected) ? draft.selected : [])

  const toggle = (value) => {
    const next = new Set(selected)
    if (next.has(value)) next.delete(value)
    else next.add(value)
    onChange({ ...draft, selected: [...next] })
  }

  const selectAll = () => onChange({ ...draft, selected: distinct.map(d => d.value) })
  const selectNone = () => onChange({ ...draft, selected: [] })

  if (distinct.length === 0) {
    return <div style={{ color: '#64748b', fontSize: 11 }}>No values to filter on.</div>
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', gap: 6 }}>
        <button
          onClick={selectAll}
          style={{
            background: '#1e293b', border: '1px solid #334155', color: '#cbd5e1',
            padding: '2px 8px', borderRadius: 4, fontSize: 10, cursor: 'pointer',
          }}
        >All</button>
        <button
          onClick={selectNone}
          style={{
            background: '#1e293b', border: '1px solid #334155', color: '#cbd5e1',
            padding: '2px 8px', borderRadius: 4, fontSize: 10, cursor: 'pointer',
          }}
        >None</button>
        <span style={{ color: '#64748b', fontSize: 10, marginLeft: 'auto', alignSelf: 'center' }}>
          {selected.size} of {distinct.length}
        </span>
      </div>
      <div style={{
        maxHeight: 200,
        overflow: 'auto',
        background: '#020617',
        border: '1px solid #1e293b',
        borderRadius: 6,
        padding: 4,
      }}>
        {distinct.map(({ value, count }) => (
          <label
            key={value}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '4px 6px',
              cursor: 'pointer',
              borderRadius: 4,
              userSelect: 'none',
            }}
            onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.04)'}
            onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
          >
            <input
              type="checkbox"
              checked={selected.has(value)}
              onChange={() => toggle(value)}
            />
            <span style={{ flex: 1, fontFamily: 'monospace', fontSize: 11 }}>{value}</span>
            <span style={{ color: '#64748b', fontSize: 10 }}>{count}</span>
          </label>
        ))}
      </div>
    </div>
  )
}
