// Right-docked panel rendering the GEX landscape from /api/gex-landscape.
// Curves are rotated 90 degrees vs the Phase 0 matplotlib reference: price
// runs down the Y axis, GEX runs along X. Pure presentation — App.jsx owns
// the fetch lifecycle and the LIVE/OPEN spot mode.
//
// Visual reference: outputs/landscape_2026-05-20_stacked.png.

import { useEffect, useRef, useState } from 'react'

export const PANEL_WIDTH = 300

// DTE buckets — landscape column -> label + curve color (matches the Phase 0
// script's DTE_BUCKETS / plot_stacked palette).
const BUCKETS = [
  { col: 'gex_0dte', label: '0DTE', color: '#ef4444' },
  { col: 'gex_near', label: '1-7 DTE', color: '#f59e0b' },
  { col: 'gex_med', label: '8-30 DTE', color: '#10b981' },
  { col: 'gex_struct', label: '30+ DTE', color: '#3b82f6' },
]

const REGIME_COLORS = {
  'magnetic-pin': '#10b981',
  'magnet-above': '#fbbf24',
  'magnet-below': '#fbbf24',
  bounded: '#10b981',
  amplification: '#06b6d4',
  'broken-magnet': '#a78bfa',
  untethered: '#94a3b8',
}

// Confluence line color by bucket count; line style by quality grade.
const CONFLUENCE_COLORS = { 2: '#fbbf24', 3: '#fb923c', 4: '#10b981' }
const QUALITY_DASH = { 'pin-grade': null, 'drift-grade': '6 4', waypoint: '2 4' }
const QUALITY_SHORT = { 'pin-grade': 'PIN', 'drift-grade': 'DRIFT', waypoint: 'soft' }

const SPOT_COLOR = '#fbbf24'
const NEG_COLOR = '#06b6d4'
const PAD = { top: 12, right: 14, bottom: 26, left: 46 }

function fmtB(raw) {
  const b = Number(raw) / 1e9
  if (!Number.isFinite(b)) return '—'
  return `${b >= 0 ? '+' : ''}${b.toFixed(0)}B`
}

function SpotModeSwitch({ mode, onChange }) {
  return (
    <div style={{ display: 'inline-flex', borderRadius: 6, overflow: 'hidden', border: '1px solid #334155' }}>
      {['LIVE', 'OPEN'].map((m) => {
        const active = mode === m
        return (
          <button
            key={m}
            type="button"
            onClick={() => onChange(m)}
            style={{
              padding: '2px 8px',
              fontSize: 9,
              fontWeight: 700,
              letterSpacing: '0.05em',
              border: 'none',
              cursor: 'pointer',
              background: active ? '#1d4ed8' : '#0f172a',
              color: active ? '#dbeafe' : '#64748b',
              fontFamily: 'inherit',
            }}
          >
            {m}
          </button>
        )
      })}
    </div>
  )
}

export default function GexLandscapePanel({ data, spotMode = 'LIVE', onSpotModeChange, onClose }) {
  const bodyRef = useRef(null)
  const [size, setSize] = useState({ width: PANEL_WIDTH, height: 360 })

  useEffect(() => {
    const el = bodyRef.current
    if (!el || typeof ResizeObserver === 'undefined') return undefined
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        const { width, height } = e.contentRect
        setSize({ width: Math.max(120, width), height: Math.max(120, height) })
      }
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const regimeTag = data?.regime?.regime || null
  const regimeColor = REGIME_COLORS[regimeTag] || '#94a3b8'

  // ─── Geometry ──────────────────────────────────────────────────────────
  const landscape = Array.isArray(data?.landscape) ? data.landscape : []
  const plotW = Math.max(10, size.width - PAD.left - PAD.right)
  const plotH = Math.max(10, size.height - PAD.top - PAD.bottom)

  let geom = null
  if (landscape.length > 1) {
    let pMin = Infinity
    let pMax = -Infinity
    let gMin = 0
    let gMax = 0
    for (const pt of landscape) {
      const price = Number(pt.price)
      if (price < pMin) pMin = price
      if (price > pMax) pMax = price
      for (const b of BUCKETS) {
        const v = Number(pt[b.col]) / 1e9
        if (v < gMin) gMin = v
        if (v > gMax) gMax = v
      }
      const tot = Number(pt.gex_total) / 1e9
      if (tot < gMin) gMin = tot
      if (tot > gMax) gMax = tot
    }
    const gPad = (gMax - gMin) * 0.06 || 1
    gMin -= gPad
    gMax += gPad
    const priceSpan = pMax - pMin || 1
    const gexSpan = gMax - gMin || 1
    geom = {
      pMin,
      pMax,
      yOf: (price) => PAD.top + ((pMax - price) / priceSpan) * plotH,
      xOf: (gexB) => PAD.left + ((gexB - gMin) / gexSpan) * plotW,
    }
  }

  const dominantBucket = data?.bucket_summary?.primary_bucket || null
  const confluences = Array.isArray(data?.confluences) ? data.confluences : []
  const negWalls = Array.isArray(data?.walls)
    ? data.walls.filter((w) => Number(w.sign) < 0)
    : []

  const buildPolyline = (col) =>
    landscape
      .map((pt) => `${geom.xOf(Number(pt[col]) / 1e9).toFixed(1)},${geom.yOf(Number(pt.price)).toFixed(1)}`)
      .join(' ')

  // Price-axis ticks — 5 evenly spaced reference prices.
  const priceTicks = []
  if (geom) {
    for (let i = 0; i <= 4; i += 1) {
      priceTicks.push(geom.pMin + ((geom.pMax - geom.pMin) * i) / 4)
    }
  }

  return (
    <div
      style={{
        position: 'absolute',
        top: 0,
        right: 0,
        height: '100%',
        width: PANEL_WIDTH,
        zIndex: 8,
        background: 'rgba(11, 18, 32, 0.96)',
        borderLeft: '1px solid #1f2937',
        boxShadow: '-12px 0 28px rgba(0,0,0,0.4)',
        display: 'flex',
        flexDirection: 'column',
        pointerEvents: 'auto',
        fontFamily: 'inherit',
      }}
      onMouseDown={(e) => e.stopPropagation()}
      onWheel={(e) => e.stopPropagation()}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 6,
          padding: '8px 10px',
          borderBottom: '1px solid #1f2937',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 0 }}>
          <span
            style={{
              fontSize: 10,
              fontWeight: 800,
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              padding: '2px 6px',
              borderRadius: 5,
              color: regimeColor,
              background: `${regimeColor}22`,
              border: `1px solid ${regimeColor}55`,
              whiteSpace: 'nowrap',
            }}
            title={data?.regime?.notes?.join('\n') || ''}
          >
            {regimeTag ? regimeTag.replace(/-/g, ' ') : 'landscape'}
          </span>
          {data?.regime?.drift_target != null && (
            <span style={{ fontSize: 10, color: '#94a3b8', whiteSpace: 'nowrap' }}>
              → {Number(data.regime.drift_target).toFixed(0)}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <SpotModeSwitch mode={spotMode} onChange={onSpotModeChange || (() => {})} />
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              title="Close GEX landscape panel"
              style={{
                background: 'transparent',
                border: 'none',
                color: '#64748b',
                cursor: 'pointer',
                fontSize: 14,
                lineHeight: 1,
                padding: 0,
              }}
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Plot body */}
      <div ref={bodyRef} style={{ flex: 1, minHeight: 0, position: 'relative' }}>
        {!data ? (
          <div style={{ color: '#64748b', fontSize: 12, textAlign: 'center', padding: '24px 0' }}>
            Loading landscape…
          </div>
        ) : !geom ? (
          <div style={{ color: '#64748b', fontSize: 12, textAlign: 'center', padding: '24px 0' }}>
            No landscape data for this session.
          </div>
        ) : (
          <svg width={size.width} height={size.height} style={{ display: 'block' }}>
            {/* zero-GEX axis */}
            <line
              x1={geom.xOf(0)}
              x2={geom.xOf(0)}
              y1={PAD.top}
              y2={PAD.top + plotH}
              stroke="#475569"
              strokeWidth="0.8"
            />

            {/* price-axis ticks */}
            {priceTicks.map((p, i) => (
              <g key={`tick-${i}`}>
                <line
                  x1={PAD.left}
                  x2={size.width - PAD.right}
                  y1={geom.yOf(p)}
                  y2={geom.yOf(p)}
                  stroke="rgba(148,163,184,0.08)"
                  strokeWidth="1"
                />
                <text x={PAD.left - 5} y={geom.yOf(p) + 3} textAnchor="end" fontSize="9" fill="#64748b">
                  {p.toFixed(0)}
                </text>
              </g>
            ))}

            {/* negative-wall tick markers on the left edge */}
            {negWalls.map((w, i) => (
              <line
                key={`neg-${i}`}
                x1={PAD.left}
                x2={PAD.left + 10}
                y1={geom.yOf(Number(w.price))}
                y2={geom.yOf(Number(w.price))}
                stroke={NEG_COLOR}
                strokeWidth="2.5"
              />
            ))}

            {/* total landscape — dashed overlay */}
            <polyline
              points={buildPolyline('gex_total')}
              fill="none"
              stroke="#f1f5f9"
              strokeWidth="1"
              strokeDasharray="3 3"
              opacity="0.6"
            />

            {/* 4 DTE bucket curves — dominant bucket bolder */}
            {BUCKETS.map((b) => {
              const isDominant = b.label === dominantBucket
              return (
                <polyline
                  key={b.col}
                  points={buildPolyline(b.col)}
                  fill="none"
                  stroke={b.color}
                  strokeWidth={isDominant ? 2.4 : 1.2}
                  opacity={isDominant ? 1 : 0.65}
                />
              )
            })}

            {/* confluence horizontal lines + right-edge labels */}
            {confluences.map((c, i) => {
              const n = c.n_buckets
              const color = CONFLUENCE_COLORS[n] || '#10b981'
              const quality = c.quality || 'waypoint'
              const y = geom.yOf(Number(c.center_price))
              return (
                <g key={`conf-${i}`}>
                  <line
                    x1={PAD.left}
                    x2={size.width - PAD.right}
                    y1={y}
                    y2={y}
                    stroke={color}
                    strokeWidth={quality === 'pin-grade' ? 2 : 1.3}
                    strokeDasharray={QUALITY_DASH[quality] || undefined}
                    opacity="0.85"
                  />
                  <text
                    x={size.width - PAD.right - 2}
                    y={y - 3}
                    textAnchor="end"
                    fontSize="9"
                    fontWeight="700"
                    fill={color}
                  >
                    {Number(c.center_price).toFixed(0)} {'★'.repeat(n)} {QUALITY_SHORT[quality] || ''}
                  </text>
                </g>
              )
            })}

            {/* spot line — dashed amber, horizontal */}
            {data.spot != null && (
              <g>
                <line
                  x1={PAD.left}
                  x2={size.width - PAD.right}
                  y1={geom.yOf(Number(data.spot))}
                  y2={geom.yOf(Number(data.spot))}
                  stroke={SPOT_COLOR}
                  strokeWidth="1.6"
                  strokeDasharray="5 3"
                />
                <text
                  x={PAD.left + 3}
                  y={geom.yOf(Number(data.spot)) - 3}
                  fontSize="9"
                  fontWeight="700"
                  fill={SPOT_COLOR}
                >
                  spot {Number(data.spot).toFixed(0)}
                </text>
              </g>
            )}

            {/* GEX axis label */}
            <text
              x={PAD.left + plotW / 2}
              y={size.height - 8}
              textAnchor="middle"
              fontSize="9"
              fill="#64748b"
            >
              GEX ($B) — by DTE bucket
            </text>
          </svg>
        )}
      </div>

      {/* Footer — per-bucket dominance legend */}
      {data?.per_bucket && (
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '4px 10px',
            padding: '6px 10px',
            borderTop: '1px solid #1f2937',
            fontSize: 9,
            fontFamily: 'ui-monospace, monospace',
          }}
        >
          {BUCKETS.map((b) => {
            const bucket = data.per_bucket[b.label]
            const dom = bucket ? Number(bucket.dominance_pct) : 0
            const isDominant = b.label === dominantBucket
            return (
              <span key={b.col} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                <span
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: 2,
                    background: b.color,
                    opacity: isDominant ? 1 : 0.6,
                  }}
                />
                <span style={{ color: isDominant ? '#e2e8f0' : '#94a3b8', fontWeight: isDominant ? 700 : 400 }}>
                  {b.label} {dom.toFixed(0)}%
                </span>
              </span>
            )
          })}
        </div>
      )}
    </div>
  )
}
