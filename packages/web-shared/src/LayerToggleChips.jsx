// LayerToggleChips — reusable row of toggle-pill buttons for chart layer control.
//
// Props:
//   layers:   [{id: string, label: string, defaultVisible?: bool}]
//   onChange: (visibilityState: {[id]: bool}) => void
//   value:    {[id]: bool}  optional — controlled state; if omitted the component
//             manages internal state (uncontrolled mode).
//   size:     'sm' | 'md'  default 'sm'
//
// Mobile: chips wrap to multiple rows; tap targets are ≥ 44px height in 'md' mode.
// Uncontrolled → calls onChange on every toggle (for parent to track).
// Controlled   → ignores internal state; pure controlled by value prop.
//
// Used by ProposalEdgeChart (Step 5) and re-used for GEX toggles in Step 6b.

import { useState, useCallback } from 'react'

const SIZES = {
  sm: { fontSize: 10, padding: '3px 10px', minHeight: 28, borderRadius: 14, gap: 5 },
  md: { fontSize: 12, padding: '5px 14px', minHeight: 44, borderRadius: 22, gap: 6 },
}

export default function LayerToggleChips({ layers = [], onChange, value, size = 'sm' }) {
  // Initialise internal state from layers[].defaultVisible
  const [internal, setInternal] = useState(() =>
    Object.fromEntries(layers.map(l => [l.id, l.defaultVisible !== false]))
  )

  const controlled = value !== undefined && value !== null
  const vis = controlled ? value : internal

  const toggle = useCallback((id) => {
    const next = { ...vis, [id]: !vis[id] }
    if (!controlled) setInternal(next)
    onChange?.(next)
  }, [vis, controlled, onChange])

  const sz = SIZES[size] || SIZES.sm

  return (
    <div
      role="group"
      aria-label="Chart layer toggles"
      style={{ display: 'flex', flexWrap: 'wrap', gap: sz.gap, alignItems: 'center' }}
    >
      {layers.map(layer => {
        const active = vis[layer.id] !== false
        return (
          <button
            key={layer.id}
            type="button"
            data-layer-id={layer.id}
            onClick={() => toggle(layer.id)}
            aria-pressed={active}
            aria-label={`${active ? 'Hide' : 'Show'} ${layer.label}`}
            style={{
              cursor: 'pointer',
              border: `1.5px solid ${active ? '#3b82f6' : '#334155'}`,
              borderRadius: sz.borderRadius,
              padding: sz.padding,
              fontSize: sz.fontSize,
              fontWeight: 700,
              letterSpacing: '0.04em',
              background: active ? '#1e3a8a' : 'transparent',
              color: active ? '#93c5fd' : '#64748b',
              minHeight: sz.minHeight,
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontFamily: 'inherit',
              userSelect: 'none',
              lineHeight: 1,
              transition: 'background 0.1s, color 0.1s, border-color 0.1s',
            }}
          >
            {layer.label}
          </button>
        )
      })}
    </div>
  )
}
