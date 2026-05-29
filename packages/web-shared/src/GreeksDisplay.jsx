// GreeksDisplay — compact row of net position greeks.
//
// Props:
//   greeks:          {delta, gamma, theta, vega, rho} or null
//   evaluationTime:  ISO datetime string — when greeks were evaluated.
//                    For MVP this is always the shortest leg's expiration, so
//                    theta and vega are ≈ 0 by definition.
//
// Theta unit:   $/calendar day  (engine: theta_yr / 365.25)
// Vega unit:    $ per 1.0 sigma (NOT per 1% vol)
// Rho unit:     $ per 1.0 rate  (NOT per 1% rate)
// Delta, gamma: standard dimensionless BSM greeks.
//
// Zero display: uses "≈ 0" rather than "0.0000" to signal "genuinely zero at
// expiration, not missing data" — matches the spec's framing.

import { useState } from 'react'

// ── Formatters ────────────────────────────────────────────────────────────────

const EPSILON_4DP = 0.00005   // rounds to 0.0000
const EPSILON_THETA = 0.005   // rounds to $0.00

function fmtDelta(v)  {
  if (v == null) return '—'
  return Math.abs(v) < EPSILON_4DP ? '≈ 0' : Number(v).toFixed(4)
}

function fmtGamma(v)  {
  if (v == null) return '—'
  return Math.abs(v) < EPSILON_4DP ? '≈ 0' : Number(v).toFixed(4)
}

function fmtTheta(v)  {
  if (v == null) return '—'
  if (Math.abs(v) < EPSILON_THETA) return '≈ 0'
  const sign = v < 0 ? '-' : '+'
  return `${sign}$${Math.abs(v).toFixed(2)}`
}

function fmtVega(v)   {
  if (v == null) return '—'
  return Math.abs(v) < EPSILON_4DP ? '≈ 0' : Number(v).toFixed(4)
}

function fmtRho(v)    {
  if (v == null) return '—'
  return Math.abs(v) < EPSILON_4DP ? '≈ 0' : Number(v).toFixed(4)
}

// ── Greek definitions ─────────────────────────────────────────────────────────

const GREEK_DEFS = [
  { key: 'delta', symbol: 'Δ', label: 'Delta', fmt: fmtDelta,
    title: 'Price change per $1 underlying move' },
  { key: 'gamma', symbol: 'Γ', label: 'Gamma', fmt: fmtGamma,
    title: 'Delta change per $1 underlying move' },
  { key: 'theta', symbol: 'Θ', label: 'Theta', fmt: fmtTheta,
    title: '$/calendar day (theta decay rate)' },
  { key: 'vega',  symbol: 'V', label: 'Vega',  fmt: fmtVega,
    title: '$ per 1.0 sigma (not per 1% vol)' },
  { key: 'rho',   symbol: 'ρ', label: 'Rho',   fmt: fmtRho,
    title: '$ per 1.0 rate unit (not per 1% rate)' },
]

// ── Tooltip text ──────────────────────────────────────────────────────────────

const INFO_TOOLTIP =
  'Greeks evaluated at the shortest leg’s expiration. ' +
  'At expiration, theta and vega are ≈ 0 by definition ' +
  '(no time value remaining). Mid-life greeks land in a future update.'

// ── Component ─────────────────────────────────────────────────────────────────

export default function GreeksDisplay({ greeks, evaluationTime }) {
  const [infoVisible, setInfoVisible] = useState(false)

  if (!greeks) return null

  return (
    <div
      data-testid="greeks-display"
      style={{
        display:    'flex',
        alignItems: 'center',
        gap:        '14px 20px',
        flexWrap:   'wrap',
        padding:    '8px 2px 4px',
        borderTop:  '1px solid #1e293b',
      }}
    >
      {/* Greek value chips */}
      {GREEK_DEFS.map(({ key, symbol, label, fmt }) => (
        <div
          key={key}
          style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}
          data-testid={`greek-${key}`}
        >
          <span
            style={{
              fontSize:      10,
              fontWeight:    700,
              color:         '#64748b',
              letterSpacing: '0.02em',
              fontFamily:    'inherit',
            }}
            aria-label={label}
          >
            {symbol}
          </span>
          <span
            style={{
              fontSize:    11,
              fontWeight:  600,
              color:       '#cbd5e1',
              fontFamily:  'monospace',
            }}
          >
            {fmt(greeks[key])}
          </span>
        </div>
      ))}

      {/* Info ⓘ with tooltip */}
      <div style={{ position: 'relative', marginLeft: 'auto', flexShrink: 0 }}>
        <button
          type="button"
          aria-label="Greeks info"
          data-testid="greeks-info-btn"
          onMouseEnter={() => setInfoVisible(true)}
          onMouseLeave={() => setInfoVisible(false)}
          onFocus={()     => setInfoVisible(true)}
          onBlur={()      => setInfoVisible(false)}
          style={{
            background:   'none',
            border:       'none',
            color:        '#475569',
            cursor:       'help',
            fontSize:     11,
            lineHeight:   1,
            padding:      '2px 4px',
            fontFamily:   'inherit',
          }}
        >
          ⓘ
        </button>

        {infoVisible && (
          <div
            role="tooltip"
            data-testid="greeks-info-tooltip"
            style={{
              position:    'absolute',
              right:       0,
              bottom:      'calc(100% + 6px)',
              width:       240,
              background:  'rgba(15,23,42,0.97)',
              border:      '1px solid #1e293b',
              borderRadius: 6,
              padding:     '8px 10px',
              fontSize:    10,
              color:       '#94a3b8',
              lineHeight:  1.6,
              zIndex:      40,
              boxShadow:   '0 4px 16px rgba(0,0,0,0.5)',
              pointerEvents: 'none',
            }}
          >
            {INFO_TOOLTIP}
            {evaluationTime && (
              <div style={{ marginTop: 4, color: '#475569', fontSize: 9 }}>
                Evaluated: {evaluationTime.slice(0, 10)}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
