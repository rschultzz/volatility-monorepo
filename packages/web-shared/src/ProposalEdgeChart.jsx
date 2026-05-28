// ProposalEdgeChart — bespoke SVG chart visualising the Step 4 P/L response.
//
// Props:
//   data:          response from POST /api/proposals/pl-data (or null for skeleton)
//   width:         number — optional; component is responsive via ResizeObserver when omitted
//   height:        number — default 400
//   onLayerToggle: ({[id]: bool}) => void — optional; called when layer visibility changes
//
// Layers (toggleable via LayerToggleChips):
//   pl_curve         ON  — blue P/L polyline
//   structural_range ON  — amber trade-thesis shaded band (one-sided aware)
//   edge_zones       ON  — per-bin classification colour bands behind everything
//   current_spot     ON  — amber dashed vertical line at spot
//   iv_curve         OFF — purple dashed line on secondary Y axis
//   breakeven        OFF — dotted vertical lines at breakeven crossings
//   pl_markers       OFF — horizontal dashed lines at max-profit / max-loss
//
// Rendering order (back → front): edge zones → structural range → grid/zero →
//   P/L curve → IV curve → reference lines → axes → crosshair → tooltip/popover.
//
// SVG convention: same as GexLandscape.jsx — bespoke SVG, no charting library.

import { useState, useRef, useEffect, useCallback } from 'react'
import LayerToggleChips from './LayerToggleChips.jsx'

// ── Edge zone fill colours (classification → CSS colour) ───────────────────────
const ZONE_FILL = {
  'strong-positive':   '#22c55e',
  'moderate-positive': '#86efac',
  'neutral':           null,          // no shading; transparent
  'moderate-negative': '#fca5a5',
  'strong-negative':   '#ef4444',
  'unknown':           '#94a3b8',
}
const ZONE_OPACITY      = 0.18   // normal fill alpha
const THIN_N_THRESHOLD  = 5      // min_structural_n below which zone is "thin"
const THIN_N_OPACITY    = 0.09   // reduced alpha for thin-n zones

// ── Layer definitions (id must be stable — used as key in state + chips) ──────
const LAYERS = [
  { id: 'pl_curve',         label: 'P/L Curve',    defaultVisible: true  },
  { id: 'structural_range', label: 'Thesis Range',  defaultVisible: true  },
  { id: 'edge_zones',       label: 'Edge Zones',   defaultVisible: true  },
  { id: 'current_spot',     label: 'Spot',         defaultVisible: true  },
  { id: 'iv_curve',         label: 'IV',           defaultVisible: false },
  { id: 'breakeven',        label: 'Breakevens',   defaultVisible: false },
  { id: 'pl_markers',       label: 'P/L Markers',  defaultVisible: false },
]

// ── Chart layout padding ───────────────────────────────────────────────────────
const PAD = { top: 28, right: 52, bottom: 48, left: 68 }

// ── Helpers ───────────────────────────────────────────────────────────────────

function lerp(x, x0, x1, y0, y1) {
  if (x1 === x0) return y0
  return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))
}

function interpolateAt(xs, ys, x) {
  if (!xs || xs.length === 0) return null
  const i = xs.findIndex(p => p >= x)
  if (i < 0) return ys[ys.length - 1]
  if (i === 0) return ys[0]
  return lerp(x, xs[i - 1], xs[i], ys[i - 1], ys[i])
}

function findZone(zones, price) {
  if (!zones || zones.length === 0) return null
  return zones.find(z => price >= z.lower && price <= z.upper) || null
}

function edgeRatioColor(r) {
  if (r == null) return '#94a3b8'
  if (r > 2.0)  return '#22c55e'
  if (r > 1.3)  return '#86efac'
  if (r >= 0.7) return '#94a3b8'
  if (r >= 0.5) return '#fca5a5'
  return '#ef4444'
}

const fmtPct  = n => n != null ? (n * 100).toFixed(1) + '%' : '—'
const fmtIv   = n => n != null ? (n * 100).toFixed(1) + '%' : '—'
const fmt2    = n => n != null ? Number(n).toFixed(2) : '—'

// ── Skeleton shown when data === null ─────────────────────────────────────────
function SkeletonChart({ width, height }) {
  const bars = [0.25, 0.55, 0.75, 0.45, 0.68, 0.88, 0.38, 0.60]
  const bW = width / bars.length
  return (
    <svg
      width={width}
      height={height}
      style={{ display: 'block' }}
      aria-label="Loading chart…"
      role="img"
    >
      <rect width={width} height={height} fill="#0f172a" />
      {bars.map((h, i) => (
        <rect
          key={i}
          x={i * bW + 4} y={height * (1 - h) * 0.6 + 20}
          width={bW - 8} height={height * h * 0.6}
          rx={3} fill="#1e293b" opacity={0.5 + i * 0.05}
        />
      ))}
      <text x={width / 2} y={height - 10} textAnchor="middle" fontSize={11} fill="#475569">
        Loading…
      </text>
    </svg>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────
export default function ProposalEdgeChart({
  data,
  width: widthProp,
  height = 400,
  onLayerToggle,
}) {
  const containerRef = useRef(null)
  const svgRef       = useRef(null)

  // Responsive width — driven by ResizeObserver when widthProp is omitted
  const [width, setWidth] = useState(widthProp || 640)
  useEffect(() => {
    if (widthProp) { setWidth(widthProp); return }
    const el = containerRef.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(entries => {
      const w = entries[0]?.contentRect?.width
      if (w > 0) setWidth(Math.floor(w))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [widthProp])

  // Layer visibility — controlled outward via onLayerToggle
  const [layerVis, setLayerVis] = useState(() =>
    Object.fromEntries(LAYERS.map(l => [l.id, l.defaultVisible !== false]))
  )
  const handleLayerChange = useCallback(vis => {
    setLayerVis(vis)
    onLayerToggle?.(vis)
  }, [onLayerToggle])

  // Hover crosshair / tooltip state
  const [hover, setHover] = useState(null)
  // Click-popover state (one per clicked edge zone)
  const [popover, setPopover] = useState(null)

  // ── Null / error states ───────────────────────────────────────────────────
  if (!data) {
    return (
      <div ref={containerRef} style={{ width: '100%' }}>
        <SkeletonChart width={width} height={height} />
      </div>
    )
  }

  if (!data.ok) {
    return (
      <div style={{
        padding: '16px 20px', color: '#f87171', fontSize: 13,
        background: '#1f2937', borderRadius: 8, fontFamily: 'inherit',
      }}>
        Error: {data.error || 'Unable to load proposal data'}
      </div>
    )
  }

  // ── Unpack ────────────────────────────────────────────────────────────────
  const {
    pl_curve, iv_curve,
    trade_thesis, edge_zones = [],
    current_spot, key_levels,
  } = data

  const prices    = pl_curve?.prices || []
  const pnls      = pl_curve?.pnl    || []
  const ivPrices  = iv_curve?.prices || []
  const ivVals    = iv_curve?.iv     || []

  if (prices.length === 0) {
    return (
      <div ref={containerRef} style={{ color: '#64748b', fontSize: 12, padding: 16 }}>
        No P/L data
      </div>
    )
  }

  // ── Geometry ──────────────────────────────────────────────────────────────
  const plotW = Math.max(10, width  - PAD.left - PAD.right)
  const plotH = Math.max(10, height - PAD.top  - PAD.bottom)

  const xMin  = prices[0]
  const xMax  = prices[prices.length - 1]
  const xSpan = xMax - xMin || 1

  const pnlMinRaw = Math.min(...pnls)
  const pnlMaxRaw = Math.max(...pnls)
  const pnlPad    = (pnlMaxRaw - pnlMinRaw) * 0.12 || 1
  const pnlMin    = pnlMinRaw - pnlPad
  const pnlMax    = pnlMaxRaw + pnlPad
  const pnlSpan   = pnlMax - pnlMin || 1

  const xOf = p  => PAD.left + ((p   - xMin)   / xSpan)   * plotW
  const yOf = v  => PAD.top  + plotH * (1 - (v - pnlMin)  / pnlSpan)
  const priceAt = svgX => xMin + ((svgX - PAD.left) / plotW) * xSpan

  // Secondary Y for IV curve
  const showIv = layerVis.iv_curve && ivVals.length > 0
  let ivMin, ivMax, ivSpan, yIv
  if (showIv) {
    ivMin  = Math.min(...ivVals) * 0.94
    ivMax  = Math.max(...ivVals) * 1.06
    ivSpan = ivMax - ivMin || 0.01
    yIv    = v => PAD.top + plotH * (1 - (v - ivMin) / ivSpan)
  }

  // ── Axis ticks ────────────────────────────────────────────────────────────
  const xTicks = Array.from({ length: 7 }, (_, i) => xMin + xSpan * i / 6)
  const yTicks = Array.from({ length: 5 }, (_, i) => pnlMin + pnlSpan * i / 4)

  // ── Polyline point strings ─────────────────────────────────────────────────
  const plPoints = prices
    .map((p, i) => `${xOf(p).toFixed(1)},${yOf(pnls[i]).toFixed(1)}`)
    .join(' ')

  const ivPoints = showIv
    ? ivPrices.map((p, i) => `${xOf(p).toFixed(1)},${yIv(ivVals[i]).toFixed(1)}`).join(' ')
    : ''

  // ── Structural range geometry ──────────────────────────────────────────────
  const ttLower = trade_thesis?.lower   // null for open-ended lower
  const ttUpper = trade_thesis?.upper   // null for open-ended upper
  const hasRange = ttLower != null || ttUpper != null

  let rangeX = PAD.left, rangeW = 0
  if (hasRange) {
    const rL = ttLower != null
      ? Math.max(xOf(ttLower), PAD.left)
      : PAD.left
    const rR = ttUpper != null
      ? Math.min(xOf(ttUpper), PAD.left + plotW)
      : PAD.left + plotW
    rangeX = rL
    rangeW = Math.max(0, rR - rL)
  }

  // ── Interaction handlers ──────────────────────────────────────────────────
  function handleMouseMove(e) {
    const svg = svgRef.current
    if (!svg) return
    const rect = svg.getBoundingClientRect()
    const svgX = (e.clientX - rect.left) * (width / rect.width)
    if (svgX < PAD.left || svgX > PAD.left + plotW) { setHover(null); return }
    const price = priceAt(svgX)
    setHover({
      svgX,
      price,
      pnlV:  interpolateAt(prices, pnls, price),
      ivV:   showIv ? interpolateAt(ivPrices, ivVals, price) : null,
      zone:  findZone(edge_zones, price),
    })
  }

  function handleMouseLeave() { setHover(null) }

  function handleClick(e) {
    const svg = svgRef.current
    if (!svg) return
    const rect  = svg.getBoundingClientRect()
    const svgX  = (e.clientX - rect.left) * (width / rect.width)
    const svgY  = (e.clientY - rect.top)  * (height / rect.height)
    if (svgX < PAD.left || svgX > PAD.left + plotW) { setPopover(null); return }
    const zone = findZone(edge_zones, priceAt(svgX))
    setPopover(zone ? { svgX, svgY, zone } : null)
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div
      ref={containerRef}
      style={{ width: '100%', fontFamily: 'inherit', position: 'relative' }}
      data-testid="proposal-edge-chart"
    >
      {/* Layer toggles */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 0 10px' }}>
        <span style={{
          fontSize: 9, fontWeight: 800, letterSpacing: '0.06em',
          textTransform: 'uppercase', color: '#475569',
        }}>
          Layers
        </span>
        <LayerToggleChips layers={LAYERS} value={layerVis} onChange={handleLayerChange} size="sm" />
      </div>

      {/* Chart area */}
      <div style={{ position: 'relative' }}>
        <svg
          ref={svgRef}
          width={width}
          height={height}
          style={{ display: 'block', cursor: 'crosshair' }}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          onClick={handleClick}
          aria-label="Proposal edge chart"
          role="img"
        >
          <defs>
            <clipPath id="prec-clip">
              <rect x={PAD.left} y={PAD.top} width={plotW} height={plotH} />
            </clipPath>
          </defs>

          {/* Background */}
          <rect width={width} height={height} fill="#0f172a" />
          <rect x={PAD.left} y={PAD.top} width={plotW} height={plotH} fill="#111827" />

          {/* Grid */}
          {yTicks.map((v, i) => (
            <line key={`yg-${i}`}
              x1={PAD.left} x2={PAD.left + plotW} y1={yOf(v)} y2={yOf(v)}
              stroke="rgba(148,163,184,0.07)" strokeWidth={1}
            />
          ))}
          {xTicks.map((p, i) => (
            <line key={`xg-${i}`}
              x1={xOf(p)} x2={xOf(p)} y1={PAD.top} y2={PAD.top + plotH}
              stroke="rgba(148,163,184,0.07)" strokeWidth={1}
            />
          ))}

          {/* ── Layer 1: Edge zone bands ─────────────────────────────────── */}
          {layerVis.edge_zones && edge_zones.map((z, i) => {
            const fill = ZONE_FILL[z.classification]
            if (!fill) return null
            const thin = z.min_structural_n != null && z.min_structural_n < THIN_N_THRESHOLD
            const op   = thin ? THIN_N_OPACITY : ZONE_OPACITY
            const zX   = Math.max(PAD.left, xOf(z.lower))
            const zW   = Math.min(PAD.left + plotW, xOf(z.upper)) - zX
            if (zW <= 0) return null
            return (
              <g key={`zone-${i}`} data-testid={`edge-zone-${z.classification}`}>
                <rect
                  x={zX} y={PAD.top} width={zW} height={plotH}
                  fill={fill} opacity={op} style={{ cursor: 'pointer' }}
                />
                {thin && (
                  <text x={zX + zW / 2} y={PAD.top + 14}
                    textAnchor="middle" fontSize={8} fill={fill} opacity={0.55}
                  >
                    ⚠
                  </text>
                )}
              </g>
            )
          })}

          {/* ── Layer 2: Structural range shaded band ────────────────────── */}
          {layerVis.structural_range && hasRange && rangeW > 0 && (
            <rect
              x={rangeX} y={PAD.top} width={rangeW} height={plotH}
              fill="#fbbf24" opacity={0.06}
              stroke="#fbbf24" strokeWidth={0.8} strokeOpacity={0.35}
              strokeDasharray="4 3"
              data-testid="structural-range-band"
            />
          )}

          {/* Zero P/L reference */}
          {pnlMin < 0 && (
            <line x1={PAD.left} x2={PAD.left + plotW} y1={yOf(0)} y2={yOf(0)}
              stroke="rgba(148,163,184,0.22)" strokeWidth={0.8} strokeDasharray="3 3"
            />
          )}

          {/* ── Layer 3: P/L polyline ─────────────────────────────────────── */}
          {layerVis.pl_curve && (
            <polyline
              points={plPoints}
              fill="none" stroke="#60a5fa" strokeWidth={2}
              clipPath="url(#prec-clip)"
              data-testid="pl-curve"
            />
          )}

          {/* ── Layer 4: IV curve ─────────────────────────────────────────── */}
          {showIv && ivPoints && (
            <polyline
              points={ivPoints}
              fill="none" stroke="#a78bfa" strokeWidth={1.5}
              strokeDasharray="5 3" opacity={0.85}
              clipPath="url(#prec-clip)"
              data-testid="iv-curve"
            />
          )}

          {/* ── Layer 5: Reference lines ──────────────────────────────────── */}
          {layerVis.current_spot && current_spot != null && (() => {
            const cx = xOf(current_spot)
            return (
              <g data-testid="spot-line">
                <line x1={cx} x2={cx} y1={PAD.top} y2={PAD.top + plotH}
                  stroke="#fbbf24" strokeWidth={1.5} strokeDasharray="5 3" opacity={0.9}
                />
                <text x={cx + 3} y={PAD.top + 11} fontSize={9} fontWeight={700} fill="#fbbf24">
                  {current_spot.toFixed(0)}
                </text>
              </g>
            )
          })()}

          {layerVis.breakeven && key_levels?.breakevens?.map((be, i) => (
            <g key={`be-${i}`} data-testid={`breakeven-${i}`}>
              <line x1={xOf(be)} x2={xOf(be)} y1={PAD.top} y2={PAD.top + plotH}
                stroke="#94a3b8" strokeWidth={1} strokeDasharray="2 4" opacity={0.65}
              />
              <text x={xOf(be) + 2} y={PAD.top + plotH - 6} fontSize={8} fill="#94a3b8">
                BE
              </text>
            </g>
          ))}

          {layerVis.pl_markers && (
            <>
              {key_levels?.max_profit != null && (
                <line x1={PAD.left} x2={PAD.left + plotW}
                  y1={yOf(key_levels.max_profit)} y2={yOf(key_levels.max_profit)}
                  stroke="#22c55e" strokeWidth={1} strokeDasharray="3 3" opacity={0.55}
                  data-testid="max-profit-line"
                />
              )}
              {key_levels?.max_loss != null && (
                <line x1={PAD.left} x2={PAD.left + plotW}
                  y1={yOf(key_levels.max_loss)} y2={yOf(key_levels.max_loss)}
                  stroke="#ef4444" strokeWidth={1} strokeDasharray="3 3" opacity={0.55}
                  data-testid="max-loss-line"
                />
              )}
            </>
          )}

          {/* ── X axis ─────────────────────────────────────────────────────── */}
          <line x1={PAD.left} x2={PAD.left + plotW}
            y1={PAD.top + plotH} y2={PAD.top + plotH}
            stroke="#334155" strokeWidth={1}
          />
          {xTicks.map((p, i) => (
            <g key={`xt-${i}`}>
              <line x1={xOf(p)} x2={xOf(p)}
                y1={PAD.top + plotH} y2={PAD.top + plotH + 4} stroke="#334155"
              />
              <text x={xOf(p)} y={PAD.top + plotH + 14}
                textAnchor="middle" fontSize={9} fill="#64748b"
              >
                {p.toFixed(0)}
              </text>
            </g>
          ))}
          <text x={PAD.left + plotW / 2} y={height - 6}
            textAnchor="middle" fontSize={9} fill="#475569"
          >
            Price
          </text>

          {/* ── Y axis (P/L) ───────────────────────────────────────────────── */}
          <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={PAD.top + plotH}
            stroke="#334155" strokeWidth={1}
          />
          {yTicks.map((v, i) => (
            <g key={`yt-${i}`}>
              <line x1={PAD.left - 4} x2={PAD.left} y1={yOf(v)} y2={yOf(v)} stroke="#334155" />
              <text x={PAD.left - 6} y={yOf(v) + 3} textAnchor="end" fontSize={9} fill="#64748b">
                {v.toFixed(1)}
              </text>
            </g>
          ))}
          <text x={14} y={PAD.top + plotH / 2}
            textAnchor="middle" fontSize={9} fill="#475569"
            transform={`rotate(-90, 14, ${PAD.top + plotH / 2})`}
          >
            P/L ($)
          </text>

          {/* ── Secondary Y axis (IV) ──────────────────────────────────────── */}
          {showIv && (
            <>
              <line x1={PAD.left + plotW} x2={PAD.left + plotW}
                y1={PAD.top} y2={PAD.top + plotH} stroke="#334155" strokeWidth={1}
              />
              {[0, 0.25, 0.5, 0.75, 1].map((frac, i) => {
                const v = ivMin + (ivMax - ivMin) * frac
                return (
                  <text key={i} x={PAD.left + plotW + 4} y={yIv(v) + 3}
                    textAnchor="start" fontSize={8} fill="#a78bfa"
                  >
                    {(v * 100).toFixed(0)}%
                  </text>
                )
              })}
              <text x={width - 8} y={PAD.top + plotH / 2}
                textAnchor="middle" fontSize={8} fill="#a78bfa"
                transform={`rotate(90, ${width - 8}, ${PAD.top + plotH / 2})`}
              >
                IV
              </text>
            </>
          )}

          {/* ── Hover crosshair ────────────────────────────────────────────── */}
          {hover && (
            <line x1={hover.svgX} x2={hover.svgX} y1={PAD.top} y2={PAD.top + plotH}
              stroke="rgba(255,255,255,0.2)" strokeWidth={1} strokeDasharray="3 3"
              style={{ pointerEvents: 'none' }}
            />
          )}

          {/* ── Trade thesis summary label ──────────────────────────────────── */}
          {trade_thesis && (
            <text x={PAD.left + 4} y={PAD.top - 8} fontSize={9} fill="#64748b">
              {trade_thesis.regime_kind}
              {trade_thesis.structural_prob != null
                ? `  struct ${fmtPct(trade_thesis.structural_prob)}`
                : ''}
              {trade_thesis.implied_prob != null
                ? `  mkt ${fmtPct(trade_thesis.implied_prob)}`
                : ''}
              {trade_thesis.edge_ratio != null
                ? `  edge ×${trade_thesis.edge_ratio.toFixed(2)}`
                : ''}
            </text>
          )}
        </svg>

        {/* ── Hover tooltip ───────────────────────────────────────────────── */}
        {hover && (
          <div
            data-testid="hover-tooltip"
            style={{
              position: 'absolute',
              left: Math.min(hover.svgX + 12, width - 185),
              top: PAD.top + 10,
              background: 'rgba(15,23,42,0.96)',
              border: '1px solid #1e293b',
              borderRadius: 6,
              padding: '8px 10px',
              fontSize: 10,
              color: '#cbd5e1',
              pointerEvents: 'none',
              minWidth: 165,
              zIndex: 20,
              boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
              lineHeight: 1.65,
            }}
          >
            <div style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 3 }}>
              {hover.price.toFixed(2)}
            </div>
            <div>P/L: <span style={{ color: '#60a5fa', fontWeight: 700 }}>{fmt2(hover.pnlV)}</span></div>
            {showIv && hover.ivV != null && (
              <div>IV: <span style={{ color: '#a78bfa' }}>{fmtIv(hover.ivV)}</span></div>
            )}
            {hover.zone && (() => {
              const rep = hover.zone.representative
              return (
                <>
                  <div style={{ borderTop: '1px solid #1e293b', margin: '4px 0' }} />
                  <div>Struct: <span style={{ color: '#10b981' }}>{fmtPct(rep?.structural_prob)}</span></div>
                  <div>Mkt: <span style={{ color: '#818cf8' }}>{fmtPct(rep?.implied_prob)}</span></div>
                  <div>Edge ×: <span style={{ color: edgeRatioColor(rep?.edge_ratio) }}>
                    {rep?.edge_ratio != null ? rep.edge_ratio.toFixed(2) : '—'}
                  </span></div>
                  <div style={{ color: ZONE_FILL[hover.zone.classification] || '#94a3b8', fontSize: 9, marginTop: 2 }}>
                    {hover.zone.classification}
                  </div>
                </>
              )
            })()}
          </div>
        )}

        {/* ── Click popover (edge zone detail) ───────────────────────────── */}
        {popover?.zone && (() => {
          const z   = popover.zone
          const rep = z.representative
          const thinN = z.min_structural_n != null && z.min_structural_n < THIN_N_THRESHOLD
          const borderColor = ZONE_FILL[z.classification] || '#334155'
          return (
            <div
              data-testid="zone-popover"
              style={{
                position: 'absolute',
                left: Math.min(popover.svgX + 14, width - 225),
                top: Math.max(PAD.top, Math.min(popover.svgY - 20, height - 180)),
                background: 'rgba(15,23,42,0.97)',
                border: `1px solid ${borderColor}55`,
                borderRadius: 8,
                padding: '10px 14px',
                fontSize: 10,
                color: '#cbd5e1',
                minWidth: 205,
                zIndex: 30,
                boxShadow: '0 8px 28px rgba(0,0,0,0.55)',
                lineHeight: 1.7,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <span style={{ fontSize: 11, fontWeight: 700, color: borderColor }}>
                  {z.classification}
                </span>
                <button
                  type="button"
                  onClick={() => setPopover(null)}
                  aria-label="Close"
                  style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer', fontSize: 12, padding: 0, lineHeight: 1 }}
                >
                  ✕
                </button>
              </div>
              <div style={{ color: '#475569', fontSize: 9, marginBottom: 5 }}>
                [{z.lower.toFixed(2)}, {z.upper.toFixed(2)}]
              </div>
              <div>
                Struct: <strong style={{ color: '#10b981' }}>{fmtPct(rep?.structural_prob)}</strong>
                {rep?.structural_ci && (
                  <span style={{ color: '#475569', fontSize: 9 }}>
                    {' '}[{(rep.structural_ci[0] * 100).toFixed(0)}–{(rep.structural_ci[1] * 100).toFixed(0)}%]
                  </span>
                )}
              </div>
              <div>Mkt implied: <strong style={{ color: '#818cf8' }}>{fmtPct(rep?.implied_prob)}</strong></div>
              <div>Edge ×: <strong style={{ color: edgeRatioColor(rep?.edge_ratio) }}>
                {rep?.edge_ratio != null ? rep.edge_ratio.toFixed(2) : '—'}
              </strong></div>
              <div style={{ color: '#64748b', fontSize: 9 }}>
                n = {z.min_structural_n ?? '—'}{thinN ? ' ⚠ thin sample' : ''}
              </div>
            </div>
          )
        })()}
      </div>
    </div>
  )
}
