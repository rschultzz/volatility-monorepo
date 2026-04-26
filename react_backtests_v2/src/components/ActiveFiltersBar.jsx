import React from 'react';
import { isFilterActive, describeFilter } from './columnFilters';

// ActiveFiltersBar — shows a row of chips for currently-active column filters.
// Click chip to clear that column's filter. "Clear all" button on the right.
//
// Props:
//   filters: { [columnId]: filterObject }
//   columnLabels: { [columnId]: string }  — for human-readable chip text
//   filteredCount: number
//   totalCount: number
//   onClearOne: (columnId) => void
//   onClearAll: () => void

export default function ActiveFiltersBar({
  filters,
  columnLabels,
  filteredCount,
  totalCount,
  onClearOne,
  onClearAll,
}) {
  const activeEntries = Object.entries(filters || {})
    .filter(([_, f]) => isFilterActive(f))

  if (!activeEntries.length) {
    return (
      <div style={{
        background: '#0f172a',
        border: '1px solid #1f2937',
        borderRadius: 14,
        padding: '8px 12px',
        color: '#64748b',
        fontSize: 12,
      }}>
        No filters active — showing all {totalCount} rows.
      </div>
    )
  }

  return (
    <div style={{
      background: '#0f172a',
      border: '1px solid #2563eb',
      borderRadius: 14,
      padding: '8px 12px',
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      flexWrap: 'wrap',
    }}>
      <span style={{
        fontSize: 11,
        color: '#93c5fd',
        fontWeight: 700,
        textTransform: 'uppercase',
        letterSpacing: '0.04em',
      }}>
        {activeEntries.length} {activeEntries.length === 1 ? 'filter' : 'filters'}
      </span>
      <span style={{ color: '#fcd34d', fontSize: 11 }}>
        {filteredCount} of {totalCount} rows
      </span>

      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', flex: 1 }}>
        {activeEntries.map(([colId, filter]) => {
          const text = describeFilter(colId, filter, columnLabels[colId])
          if (!text) return null
          return (
            <span
              key={colId}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                background: 'rgba(37, 99, 235, 0.20)',
                border: '1px solid rgba(37, 99, 235, 0.40)',
                color: '#bfdbfe',
                padding: '2px 8px',
                borderRadius: 999,
                fontSize: 11,
                fontFamily: 'monospace',
              }}
            >
              {text}
              <button
                onClick={() => onClearOne && onClearOne(colId)}
                title={`Clear ${columnLabels[colId] || colId} filter`}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#bfdbfe',
                  padding: 0,
                  cursor: 'pointer',
                  fontSize: 12,
                  lineHeight: 1,
                  opacity: 0.7,
                }}
              >×</button>
            </span>
          )
        })}
      </div>

      <button
        onClick={onClearAll}
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
        Clear all
      </button>
    </div>
  )
}
