// AnaloguesPanel — right-docked panel listing the K nearest historical
// days from /api/analogues (CR-013, Day Analogue Comparison v0.5).
//
// Independent positionally from GexLandscapePanel: docks to the right of
// the chart, top-aligned, fixed width. Click a row → opens
// AnalogueDetailModal for that day; the main chart stays unchanged.

import { useState } from 'react'
import AnalogueDetailModal from './AnalogueDetailModal'

export const PANEL_WIDTH = 320

const REGIME_BADGE_BG = {
  'magnetic-pin': '#064e3b',
  'magnet-above': '#78350f',
  'magnet-below': '#78350f',
  'broken-magnet': '#4c1d95',
  bounded: '#064e3b',
  amplification: '#164e63',
  untethered: '#334155',
}
const REGIME_BADGE_FG = {
  'magnetic-pin': '#34d399',
  'magnet-above': '#fcd34d',
  'magnet-below': '#fcd34d',
  'broken-magnet': '#a78bfa',
  bounded: '#34d399',
  amplification: '#22d3ee',
  untethered: '#cbd5e1',
}

function LabelThumb({ label }) {
  if (label === 1) return <span style={{ color: '#34d399', fontSize: 14 }}>▲</span>
  if (label === -1) return <span style={{ color: '#f87171', fontSize: 14 }}>▼</span>
  return <span style={{ color: '#64748b', fontSize: 14 }}>•</span>
}

function fmtPts(v) {
  if (v == null || Number.isNaN(v)) return '—'
  const sign = v > 0 ? '+' : ''
  return `${sign}${v.toFixed(1)}`
}

function RegimeBadge({ regime }) {
  if (!regime) return <span style={{ color: '#64748b', fontSize: 10 }}>—</span>
  const bg = REGIME_BADGE_BG[regime] || '#334155'
  const fg = REGIME_BADGE_FG[regime] || '#cbd5e1'
  // Shorten to ≤12 chars
  const short = regime === 'magnetic-pin' ? 'PIN'
    : regime === 'magnet-above' ? 'MAG↑'
    : regime === 'magnet-below' ? 'MAG↓'
    : regime === 'broken-magnet' ? 'BRK'
    : regime === 'amplification' ? 'AMP'
    : regime === 'bounded' ? 'BND'
    : regime === 'untethered' ? 'UNT'
    : regime.toUpperCase()
  return (
    <span
      style={{
        padding: '1px 6px',
        borderRadius: 4,
        fontSize: 10,
        fontWeight: 700,
        letterSpacing: '0.05em',
        background: bg,
        color: fg,
      }}
    >
      {short}
    </span>
  )
}

function AnalogueRow({ row, onClick }) {
  const outcome = row.outcomes || {}
  const topSignal = (row.labeled_signals || [])[0]
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        display: 'grid',
        gridTemplateColumns: '88px 44px 56px 1fr 40px',
        gap: 8,
        alignItems: 'center',
        padding: '8px 10px',
        background: 'transparent',
        border: 'none',
        borderBottom: '1px solid #1e293b',
        cursor: 'pointer',
        textAlign: 'left',
        color: '#e2e8f0',
        fontFamily: 'inherit',
        width: '100%',
      }}
      onMouseEnter={(e) => { e.currentTarget.style.background = '#0f172a' }}
      onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent' }}
    >
      <span style={{ fontSize: 11, fontWeight: 600 }}>{row.trade_date}</span>
      <span style={{ fontSize: 10, color: '#94a3b8', fontVariantNumeric: 'tabular-nums' }}>
        {row.similarity_score != null ? row.similarity_score.toFixed(2) : '—'}
      </span>
      <RegimeBadge regime={row.regime} />
      <span style={{ fontSize: 10, color: '#cbd5e1', fontVariantNumeric: 'tabular-nums' }}>
        eod {fmtPts(outcome.eod_return_pts)} · rng {fmtPts(outcome.intraday_range_pts)}
      </span>
      <span style={{ textAlign: 'right' }}>
        <LabelThumb label={topSignal?.label ?? null} />
      </span>
    </button>
  )
}

export default function AnaloguesPanel({
  data,
  loading,
  error,
  k,
  onKChange,
  onClose,
  // px offset from the right edge — used to stack to the left of an
  // already-docked panel (e.g. GexLandscapePanel) when both are open.
  rightOffset = 0,
}) {
  const [selected, setSelected] = useState(null)

  const analogues = data?.analogues || []
  const nCandidates = data?.n_candidates ?? 0
  const anchorDate = data?.anchor?.trade_date

  return (
    <>
      <div
        style={{
          position: 'absolute',
          right: rightOffset,
          top: 0,
          bottom: 0,
          width: PANEL_WIDTH,
          background: '#020617',
          borderLeft: '1px solid #1e293b',
          color: '#e2e8f0',
          display: 'flex',
          flexDirection: 'column',
          fontFamily: 'inherit',
          zIndex: 5,
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: '10px 12px',
            borderBottom: '1px solid #1e293b',
            display: 'flex',
            alignItems: 'center',
            gap: 8,
          }}
        >
          <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.08em' }}>
            ANALOGUES
          </span>
          <span style={{ fontSize: 10, color: '#64748b', flex: 1 }}>
            {anchorDate ? `vs ${anchorDate}` : ''}
          </span>
          <select
            value={k}
            onChange={(e) => onKChange?.(Number(e.target.value))}
            style={{
              background: '#0f172a',
              color: '#cbd5e1',
              border: '1px solid #334155',
              borderRadius: 4,
              fontSize: 10,
              padding: '2px 4px',
              fontFamily: 'inherit',
            }}
          >
            <option value={5}>K=5</option>
            <option value={10}>K=10</option>
            <option value={20}>K=20</option>
          </select>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            style={{
              background: 'transparent',
              color: '#94a3b8',
              border: 'none',
              fontSize: 16,
              cursor: 'pointer',
              padding: 0,
              lineHeight: 1,
            }}
          >
            ×
          </button>
        </div>

        {/* Column header */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '88px 44px 56px 1fr 40px',
            gap: 8,
            padding: '6px 10px',
            background: '#0b1220',
            borderBottom: '1px solid #1e293b',
            color: '#64748b',
            fontSize: 9,
            fontWeight: 700,
            letterSpacing: '0.08em',
          }}
        >
          <span>DATE</span>
          <span>DIST</span>
          <span>REG</span>
          <span>OUTCOMES</span>
          <span style={{ textAlign: 'right' }}>LBL</span>
        </div>

        {/* Body */}
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {loading && (
            <div style={{ padding: 16, fontSize: 11, color: '#64748b' }}>
              Loading analogues…
            </div>
          )}
          {!loading && error && (
            <div style={{ padding: 16, fontSize: 11, color: '#f87171' }}>
              {error}
            </div>
          )}
          {!loading && !error && analogues.length === 0 && (
            <div style={{ padding: 16, fontSize: 11, color: '#64748b', lineHeight: 1.5 }}>
              {nCandidates === 0 ? (
                <>
                  No labeled days in <code style={{ color: '#94a3b8' }}>bt_signals</code> yet.
                  Label signals via the SIGNALS panel to populate the analogue corpus.
                </>
              ) : (
                <>No close analogues found.</>
              )}
            </div>
          )}
          {!loading && !error && analogues.map((row) => (
            <AnalogueRow
              key={row.trade_date}
              row={row}
              onClick={() => setSelected(row)}
            />
          ))}
        </div>

        {/* Footer */}
        <div
          style={{
            padding: '6px 12px',
            borderTop: '1px solid #1e293b',
            fontSize: 10,
            color: '#64748b',
          }}
        >
          {nCandidates} labeled candidate{nCandidates === 1 ? '' : 's'}
        </div>
      </div>

      {selected && (
        <AnalogueDetailModal
          analogue={selected}
          onClose={() => setSelected(null)}
        />
      )}
    </>
  )
}
