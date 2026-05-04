/**
 * TradeAnnotationPanel.jsx
 * react_price_preview/src/components/TradeAnnotationPanel.jsx
 *
 * Floating panel that appears on the Price Chart when a Trade Log
 * record is being annotated. Shows entry/exit (locked) and computes
 * setup start/target from the clicked timeslices.
 *
 * Props:
 *   annotationState  — object from /api/trade-log/annotation-state
 *   selectedTimes    — HH:MM[] from PriceChart (the current timeslice set)
 *   tradeDate        — YYYY-MM-DD string
 *   onSaved()        — called after a successful PATCH + annotation clear
 *   onCancel()       — called when user dismisses without saving
 */

import { useEffect, useMemo, useRef, useState } from 'react'

// ── helpers ───────────────────────────────────────────────────

function fmtUsd(v) {
  if (v == null) return '—'
  const n = Number(v)
  return (n >= 0 ? '+$' : '-$') + Math.abs(n).toFixed(2)
}

function fmtPts(v) {
  if (v == null) return '—'
  const n = Number(v)
  return (n > 0 ? '+' : '') + n.toFixed(2) + ' pts'
}

function pnlColor(v) {
  if (v == null) return '#94a3b8'
  return Number(v) >= 0 ? '#4ade80' : '#f87171'
}

/**
 * Given the full sorted timeslice list and the entry HH:MM,
 * return the last 2 bars that are chronologically ≤ entryHHMM.
 * These will be used as setup start and target for skew calcs.
 */
function computeSetupTimes(selectedTimes, entryHHMM) {
  if (!entryHHMM || !selectedTimes?.length) return { start: null, target: null }
  const pre = selectedTimes.filter((t) => t <= entryHHMM).sort()
  return {
    start:  pre.length >= 2 ? pre[pre.length - 2] : pre[0] || null,
    target: pre.length >= 1 ? pre[pre.length - 1] : null,
  }
}

/**
 * Combine a trade date (YYYY-MM-DD) with an HH:MM string into
 * a datetime-local ISO string (YYYY-MM-DDTHH:MM) for the PATCH body.
 */
function toDatetimeLocalString(dateStr, hhmm) {
  if (!dateStr || !hhmm) return null
  return `${dateStr}T${hhmm}`
}

// ── component ─────────────────────────────────────────────────

export default function TradeAnnotationPanel({
  annotationState,
  selectedTimes = [],
  tradeDate,
  onSaved,
  onCancel,
}) {
  const [saving, setSaving] = useState(false)
  const [error, setError]   = useState('')

  // Drag state
  const [pos, setPos] = useState({ top: 80, right: 20 })
  const dragRef = useRef(null)

  const { start: setupStart, target: setupTarget } = useMemo(
    () => computeSetupTimes(selectedTimes, annotationState?.entry_hhmm),
    [selectedTimes, annotationState?.entry_hhmm]
  )

  // Mouse drag handlers
  function handleMouseDown(e) {
    if (e.target.closest('button') || e.target.closest('textarea')) return
    e.preventDefault()
    dragRef.current = {
      startX: e.clientX, startY: e.clientY,
      startTop: pos.top, startRight: pos.right,
    }
    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
  }

  function handleMouseMove(e) {
    if (!dragRef.current) return
    const { startX, startY, startTop, startRight } = dragRef.current
    setPos({
      top:   startTop  + (e.clientY - startY),
      right: startRight - (e.clientX - startX),
    })
  }

  function handleMouseUp() {
    dragRef.current = null
    window.removeEventListener('mousemove', handleMouseMove)
    window.removeEventListener('mouseup', handleMouseUp)
  }

  useEffect(() => () => {
    window.removeEventListener('mousemove', handleMouseMove)
    window.removeEventListener('mouseup', handleMouseUp)
  }, [])

  async function handleSave() {
    setSaving(true)
    setError('')
    try {
      const patch = {
        setup_start_ts_pt:  toDatetimeLocalString(annotationState.trade_date, setupStart),
        setup_target_ts_pt: toDatetimeLocalString(annotationState.trade_date, setupTarget),
      }

      const res = await fetch(`/api/trade-log/trades/${annotationState.trade_id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      })
      const data = await res.json()
      if (!data.ok) throw new Error(data.error || 'Save failed')

      // Auto-recompute context after saving annotation times
      fetch(`/api/trade-log/trades/${annotationState.trade_id}/recompute_context`, {
        method: 'POST',
      }).catch(() => {})

      onSaved?.()
    } catch (e) {
      setError(e.message)
    } finally {
      setSaving(false)
    }
  }

  if (!annotationState) return null

  const { direction, symbol, entry_hhmm, exit_hhmm, entry_price, exit_price, realized_pts, net_pnl_usd } = annotationState
  const isLong   = direction === 'long'
  const dirColor = isLong ? '#4ade80' : '#f87171'

  const hasBothTimes = setupStart && setupTarget
  const hasPreBars   = selectedTimes.filter((t) => t <= entry_hhmm).length

  return (
    <div
      onMouseDown={handleMouseDown}
      onWheel={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
      style={{
        position: 'absolute',
        zIndex: 20,
        top:   pos.top,
        right: Math.max(0, pos.right),
        width: 280,
        background: 'rgba(11, 18, 32, 0.97)',
        border: '1px solid #334155',
        borderRadius: 14,
        padding: '14px 16px',
        boxShadow: '0 12px 32px rgba(0,0,0,0.5)',
        color: '#e2e8f0',
        fontSize: 12,
        cursor: 'grab',
        userSelect: 'none',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
            <span style={{
              padding: '2px 7px', borderRadius: 999, fontSize: 10, fontWeight: 800,
              background: isLong ? 'rgba(74,222,128,0.15)' : 'rgba(248,113,113,0.15)',
              border: `1px solid ${dirColor}40`, color: dirColor, textTransform: 'uppercase',
            }}>
              {isLong ? 'LONG' : 'SHORT'}
            </span>
            <span style={{ fontWeight: 700, color: '#e5e7eb' }}>{symbol}</span>
          </div>
          <div style={{ color: '#93c5fd', fontSize: 10, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
            Annotation Mode
          </div>
        </div>
        <button
          onClick={onCancel}
          style={{
            background: 'none', border: 'none', color: '#64748b',
            cursor: 'pointer', fontSize: 16, lineHeight: 1, padding: '0 2px',
          }}
        >
          ✕
        </button>
      </div>

      {/* Fill summary — locked */}
      <div style={{
        background: '#020617', border: '1px solid #1f2937',
        borderRadius: 8, padding: '8px 10px', marginBottom: 10,
        display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 12px',
      }}>
        <div>
          <div style={{ color: '#64748b', fontSize: 10 }}>Entry</div>
          <div style={{ fontWeight: 600 }}>{entry_hhmm} <span style={{ color: '#94a3b8' }}>{entry_price?.toFixed(2)}</span></div>
        </div>
        <div>
          <div style={{ color: '#64748b', fontSize: 10 }}>Exit</div>
          <div style={{ fontWeight: 600 }}>{exit_hhmm} <span style={{ color: '#94a3b8' }}>{exit_price?.toFixed(2)}</span></div>
        </div>
        <div style={{ gridColumn: '1 / -1', borderTop: '1px solid #1f2937', paddingTop: 4, marginTop: 2 }}>
          <span style={{ color: pnlColor(net_pnl_usd), fontWeight: 700 }}>{fmtUsd(net_pnl_usd)}</span>
          <span style={{ color: pnlColor(realized_pts), marginLeft: 8 }}>{fmtPts(realized_pts)}</span>
        </div>
      </div>

      {/* Setup times — live preview from clicked bars */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ color: '#93c5fd', fontSize: 10, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 6 }}>
          Skew Setup Times
        </div>

        {hasPreBars === 0 ? (
          <div style={{
            background: 'rgba(234,179,8,0.08)', border: '1px solid rgba(234,179,8,0.25)',
            borderRadius: 8, padding: '8px 10px', fontSize: 11, color: '#fbbf24', lineHeight: 1.4,
          }}>
            Click bars on the chart that are <strong>before entry</strong> ({entry_hhmm}).
            The last 2 you select will be used as start → target.
          </div>
        ) : (
          <div style={{
            display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6,
          }}>
            <TimeBox label="Setup Start" time={setupStart} active={!!setupStart} />
            <TimeBox label="Setup Target" time={setupTarget} active={!!setupTarget} />
          </div>
        )}

        {hasPreBars > 0 && !hasBothTimes && (
          <div style={{ color: '#f59e0b', fontSize: 10, marginTop: 5 }}>
            Click one more bar before {entry_hhmm} to set both times.
          </div>
        )}

        {hasPreBars > 2 && (
          <div style={{ color: '#64748b', fontSize: 10, marginTop: 5 }}>
            {hasPreBars} bars before entry — using last 2 ({setupStart} → {setupTarget})
          </div>
        )}
      </div>

      {error && (
        <div style={{ color: '#f87171', fontSize: 11, marginBottom: 8 }}>{error}</div>
      )}

      {/* Actions */}
      <div style={{ display: 'flex', gap: 8 }}>
        <button
          onClick={handleSave}
          disabled={saving || !hasBothTimes}
          style={{
            flex: 1, padding: '8px 0', borderRadius: 10, fontSize: 12, fontWeight: 700,
            background: hasBothTimes ? '#2563eb' : '#1e293b',
            border: `1px solid ${hasBothTimes ? '#3b82f6' : '#334155'}`,
            color: hasBothTimes ? 'white' : '#475569',
            cursor: hasBothTimes ? 'pointer' : 'not-allowed',
          }}
        >
          {saving ? 'Saving…' : 'Save to Trade'}
        </button>
        <button
          onClick={onCancel}
          style={{
            padding: '8px 12px', borderRadius: 10, fontSize: 12,
            background: 'none', border: '1px solid #334155',
            color: '#94a3b8', cursor: 'pointer',
          }}
        >
          Cancel
        </button>
      </div>

      <div style={{ color: '#334155', fontSize: 10, marginTop: 8, textAlign: 'center' }}>
        Saving will also recompute market context
      </div>
    </div>
  )
}

function TimeBox({ label, time, active }) {
  return (
    <div style={{
      background: active ? 'rgba(37,99,235,0.12)' : '#020617',
      border: `1px solid ${active ? '#2563eb' : '#1f2937'}`,
      borderRadius: 8, padding: '6px 8px',
    }}>
      <div style={{ color: '#64748b', fontSize: 10, marginBottom: 2 }}>{label}</div>
      <div style={{ fontWeight: 700, color: active ? '#93c5fd' : '#334155', fontSize: 13 }}>
        {time || '—'}
      </div>
    </div>
  )
}
