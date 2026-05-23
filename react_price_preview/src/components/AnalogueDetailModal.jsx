// AnalogueDetailModal — non-destructive centered overlay showing detail
// for a single historical analogue day (CR-013). Closes on backdrop click,
// Escape, or the X button. Main app state stays untouched.

import { useEffect } from 'react'

function fmtPts(v) {
  if (v == null || Number.isNaN(v)) return '—'
  const sign = v > 0 ? '+' : ''
  return `${sign}${Number(v).toFixed(1)}`
}

function fmtB(v) {
  if (v == null || Number.isNaN(v)) return '—'
  const b = Number(v) / 1e9
  return `${b >= 0 ? '+' : ''}${b.toFixed(1)}B`
}

function LabelPill({ label }) {
  const txt = label === 1 ? 'WIN' : label === -1 ? 'LOSE' : 'NEUTRAL'
  const fg = label === 1 ? '#34d399' : label === -1 ? '#f87171' : '#cbd5e1'
  const bg = label === 1 ? '#064e3b' : label === -1 ? '#7f1d1d' : '#334155'
  return (
    <span
      style={{
        padding: '2px 8px',
        borderRadius: 4,
        background: bg,
        color: fg,
        fontSize: 10,
        fontWeight: 700,
        letterSpacing: '0.08em',
      }}
    >
      {txt}
    </span>
  )
}

function StatRow({ label, value }) {
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'baseline',
        padding: '4px 0',
        borderBottom: '1px dashed #1e293b',
      }}
    >
      <span style={{ fontSize: 10, color: '#94a3b8', letterSpacing: '0.04em' }}>
        {label}
      </span>
      <span style={{ fontSize: 12, color: '#e2e8f0', fontVariantNumeric: 'tabular-nums' }}>
        {value}
      </span>
    </div>
  )
}

export default function AnalogueDetailModal({ analogue, onClose }) {
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Escape') onClose?.()
    }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [onClose])

  const outcomes = analogue.outcomes || {}
  const summary = analogue.landscape_summary || {}
  const signals = analogue.labeled_signals || []
  const fv = analogue.feature_vector || {}

  return (
    <div
      role="dialog"
      aria-modal="true"
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(2, 6, 23, 0.7)',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 100,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 'min(680px, 90vw)',
          maxHeight: '85vh',
          background: '#0b1220',
          border: '1px solid #1e293b',
          borderRadius: 8,
          color: '#e2e8f0',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          fontFamily: 'inherit',
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: '12px 16px',
            borderBottom: '1px solid #1e293b',
            display: 'flex',
            alignItems: 'center',
            gap: 12,
          }}
        >
          <span style={{ fontSize: 16, fontWeight: 700 }}>{analogue.trade_date}</span>
          <span
            style={{
              fontSize: 11, fontWeight: 700, color: '#cbd5e1',
              background: '#1e293b', borderRadius: 4, padding: '2px 6px',
            }}
          >
            {summary.regime || analogue.regime || '—'}
          </span>
          <span style={{ fontSize: 11, color: '#64748b' }}>
            distance {analogue.similarity_score != null ? analogue.similarity_score.toFixed(3) : '—'}
          </span>
          <span style={{ flex: 1 }} />
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            style={{
              background: 'transparent',
              color: '#94a3b8',
              border: 'none',
              fontSize: 20,
              lineHeight: 1,
              cursor: 'pointer',
              padding: 0,
            }}
          >
            ×
          </button>
        </div>

        {/* Body */}
        <div
          style={{
            padding: 16,
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 20,
            overflowY: 'auto',
          }}
        >
          {/* Left col: outcomes */}
          <section>
            <h3 style={{ fontSize: 11, color: '#94a3b8', margin: '0 0 8px 0',
                         letterSpacing: '0.08em', fontWeight: 700 }}>
              SESSION OUTCOMES
            </h3>
            <StatRow label="EOD return" value={`${fmtPts(outcomes.eod_return_pts)} pts`} />
            <StatRow label="Intraday range" value={`${fmtPts(outcomes.intraday_range_pts)} pts`} />
            <StatRow label="MFE above open" value={`${fmtPts(outcomes.mfe_above_open_pts)} pts`} />
            <StatRow label="MFE below open" value={`${fmtPts(outcomes.mfe_below_open_pts)} pts`} />
            <StatRow label="Session start" value={outcomes.session_start || '—'} />
            <StatRow label="Session end" value={outcomes.session_end || '—'} />
          </section>

          {/* Right col: landscape summary + top cluster */}
          <section>
            <h3 style={{ fontSize: 11, color: '#94a3b8', margin: '0 0 8px 0',
                         letterSpacing: '0.08em', fontWeight: 700 }}>
              LANDSCAPE
            </h3>
            <StatRow label="Regime" value={summary.regime || '—'} />
            <StatRow label="Dominant bucket" value={summary.dominant_bucket || '—'} />
            <StatRow label="Implied move (1σ)"
                     value={fv.implied_move_1d != null
                       ? `${Number(fv.implied_move_1d).toFixed(1)} pts` : '—'} />
            {summary.top_cluster && (
              <>
                <div style={{ height: 6 }} />
                <h4 style={{ fontSize: 10, color: '#94a3b8', margin: 0,
                             letterSpacing: '0.08em', fontWeight: 700 }}>
                  TOP CLUSTER
                </h4>
                <StatRow label="Center" value={`${Number(summary.top_cluster.center_price).toFixed(2)}`} />
                <StatRow label="Quality" value={summary.top_cluster.quality} />
                <StatRow label="max_gex" value={fmtB(summary.top_cluster.max_gex)} />
              </>
            )}
          </section>

          {/* Full-width: labeled signals */}
          <section style={{ gridColumn: '1 / -1' }}>
            <h3 style={{ fontSize: 11, color: '#94a3b8', margin: '0 0 8px 0',
                         letterSpacing: '0.08em', fontWeight: 700 }}>
              LABELED SIGNALS ({signals.length})
            </h3>
            {signals.length === 0 && (
              <div style={{ fontSize: 11, color: '#64748b' }}>No labeled signals.</div>
            )}
            {signals.map((s) => (
              <div
                key={s.signal_id}
                style={{
                  padding: '6px 0',
                  borderBottom: '1px dashed #1e293b',
                  display: 'flex',
                  gap: 8,
                  alignItems: 'center',
                  fontSize: 11,
                }}
              >
                <LabelPill label={s.label} />
                <span style={{ color: '#cbd5e1', fontWeight: 600 }}>
                  {s.strategy_key} · {s.direction}
                </span>
                <span style={{ color: '#94a3b8', fontVariantNumeric: 'tabular-nums' }}>
                  realized {fmtPts(s.realized_pts)} pts
                </span>
                <span style={{ color: '#64748b', flex: 1, overflow: 'hidden',
                               textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {s.label_note || ''}
                </span>
              </div>
            ))}
          </section>
        </div>

        {/* Footer */}
        <div
          style={{
            padding: '8px 16px',
            borderTop: '1px solid #1e293b',
            display: 'flex',
            justifyContent: 'flex-end',
            gap: 12,
          }}
        >
          <button
            type="button"
            onClick={onClose}
            style={{
              background: '#1d4ed8',
              color: '#dbeafe',
              border: 'none',
              borderRadius: 4,
              padding: '6px 16px',
              cursor: 'pointer',
              fontSize: 12,
              fontWeight: 700,
              fontFamily: 'inherit',
            }}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
