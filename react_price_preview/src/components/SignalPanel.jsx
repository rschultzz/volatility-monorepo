import { useState, useRef, useEffect, useCallback } from 'react'

const STATUS_CONFIG = {
  watching:      { color: '#94a3b8', label: 'Watching',      dot: '#475569' },
  consolidating: { color: '#60a5fa', label: 'Consolidating', dot: '#3b82f6' },
  signal_fired:  { color: '#f59e0b', label: 'Signal fired',  dot: '#f59e0b' },
  entry_zone:    { color: '#4ade80', label: 'Entry zone',     dot: '#22c55e' },
  entered:       { color: '#a78bfa', label: 'Entered',        dot: '#8b5cf6' },
  completed:     { color: '#94a3b8', label: 'Completed',      dot: '#475569' },
  expired:       { color: '#475569', label: 'Expired',        dot: '#374151' },
}

const DIR_CONFIG = {
  up:   { label: 'SHORT', color: '#f87171', bg: 'rgba(239,68,68,0.12)',  border: 'rgba(239,68,68,0.30)' },
  down: { label: 'LONG',  color: '#4ade80', bg: 'rgba(74,222,128,0.12)', border: 'rgba(74,222,128,0.30)' },
}

function fmt(v, d = 2) {
  if (v == null || v === '') return '—'
  const n = Number(v)
  return isNaN(n) ? String(v) : n.toFixed(d)
}

function StatusDot({ status }) {
  const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.watching
  return (
    <span style={{
      display: 'inline-block',
      width: '7px', height: '7px',
      borderRadius: '50%',
      background: cfg.dot,
      marginRight: '5px',
      flexShrink: 0,
      boxShadow: status === 'entry_zone' || status === 'signal_fired' ? `0 0 6px ${cfg.dot}` : 'none',
    }} />
  )
}

function SignalCard({ signal, onLabel, isToday }) {
  const dir = DIR_CONFIG[signal.direction] || DIR_CONFIG.up
  const sta = STATUS_CONFIG[signal.status] || STATUS_CONFIG.watching

  return (
    <div style={{
      border: `1px solid ${dir.border}`,
      borderRadius: '10px',
      background: dir.bg,
      padding: '8px 10px',
      marginBottom: '6px',
    }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '5px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{
            fontSize: '10px', fontWeight: 800, letterSpacing: '0.06em',
            color: dir.color, background: dir.bg,
            border: `1px solid ${dir.border}`,
            borderRadius: '5px', padding: '1px 6px',
          }}>
            {dir.label}
          </span>
          <span style={{ fontSize: '11px', color: '#94a3b8' }}>
            {signal.source_zone_low != null ? `${fmt(signal.source_zone_low, 0)}–${fmt(signal.source_zone_high, 0)}` : '—'}
          </span>
          <span style={{ fontSize: '10px', color: '#475569' }}>→</span>
          <span style={{ fontSize: '11px', color: '#e2e8f0', fontWeight: 600 }}>
            {fmt(signal.target_level, 0)}
          </span>
        </div>
        {/* Label buttons */}
        <div style={{ display: 'flex', gap: '4px' }}>
          <button
            onClick={() => onLabel(signal.signal_id, signal.label === 1 ? 0 : 1)}
            title="Good setup"
            style={{
              background: signal.label === 1 ? 'rgba(74,222,128,0.20)' : 'transparent',
              border: `1px solid ${signal.label === 1 ? '#4ade80' : '#334155'}`,
              borderRadius: '5px', color: signal.label === 1 ? '#4ade80' : '#475569',
              cursor: 'pointer', fontSize: '11px', padding: '1px 5px', lineHeight: 1.4,
            }}
          >👍</button>
          <button
            onClick={() => onLabel(signal.signal_id, signal.label === -1 ? 0 : -1)}
            title="Bad setup"
            style={{
              background: signal.label === -1 ? 'rgba(239,68,68,0.20)' : 'transparent',
              border: `1px solid ${signal.label === -1 ? '#f87171' : '#334155'}`,
              borderRadius: '5px', color: signal.label === -1 ? '#f87171' : '#475569',
              cursor: 'pointer', fontSize: '11px', padding: '1px 5px', lineHeight: 1.4,
            }}
          >👎</button>
        </div>
      </div>

      {/* Status row */}
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '5px' }}>
        <StatusDot status={signal.status} />
        <span style={{ fontSize: '10px', color: sta.color, fontWeight: 600 }}>{sta.label}</span>
        {signal.signal_ts_pt && (
          <span style={{ fontSize: '10px', color: '#475569', marginLeft: '6px' }}>
            signal {signal.signal_ts_pt}
          </span>
        )}
        {signal.entry_ts_pt && (
          <span style={{ fontSize: '10px', color: '#475569', marginLeft: '6px' }}>
            entry {signal.entry_ts_pt}
          </span>
        )}
      </div>

      {/* Price grid */}
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
        gap: '4px', fontSize: '10px',
      }}>
        {[
          ['Entry', signal.entry_price],
          ['Stop',  signal.initial_stop],
          ['Tgt',   signal.take_profit],
          ['P&L',   signal.realized_pts],
        ].map(([label, val]) => (
          <div key={label} style={{ textAlign: 'center' }}>
            <div style={{ color: '#475569', marginBottom: '1px' }}>{label}</div>
            <div style={{
              color: label === 'P&L'
                ? (val > 0 ? '#4ade80' : val < 0 ? '#f87171' : '#94a3b8')
                : '#e2e8f0',
              fontWeight: 600,
            }}>
              {val != null ? fmt(val, 2) : '—'}
            </div>
          </div>
        ))}
      </div>

      {/* Skew row (show if available) */}
      {(signal.put_skew != null || signal.call_skew != null) && (
        <div style={{ marginTop: '5px', display: 'flex', gap: '10px', fontSize: '10px' }}>
          <span style={{ color: '#475569' }}>Put Δ</span>
          <span style={{ color: signal.put_skew < 0 ? '#f87171' : '#4ade80', fontWeight: 600 }}>
            {fmt(signal.put_skew)}%
          </span>
          <span style={{ color: '#475569', marginLeft: '4px' }}>Call Δ</span>
          <span style={{ color: signal.call_skew > 0 ? '#4ade80' : '#f87171', fontWeight: 600 }}>
            {fmt(signal.call_skew)}%
          </span>
        </div>
      )}
    </div>
  )
}

export default function SignalPanel({
  tradeDate,
  strategyKey = 'up_move_short',
  signals: externalSignals = null,
  isLoading: externalLoading = false,
  lastUpdated: externalLastUpdated = null,
  isToday = false,
  onRefresh,
  onLabel,
}) {
  // If externalSignals is null, the panel manages its own fetch state
  const selfManaged = externalSignals === null
  const [internalSignals, setInternalSignals] = useState([])
  const [internalLoading, setInternalLoading] = useState(false)
  const [internalLastUpdated, setInternalLastUpdated] = useState(null)
  const [internalStrategyKey, setInternalStrategyKey] = useState(strategyKey)
  const refreshTimerRef = useRef(null)

  const signals = selfManaged ? internalSignals : externalSignals
  const isLoading = selfManaged ? internalLoading : externalLoading
  const lastUpdated = selfManaged ? internalLastUpdated : externalLastUpdated
  const activeStrategyKey = selfManaged ? internalStrategyKey : strategyKey

  const doScan = useCallback(async (sk, td) => {
    if (!td) return
    setInternalLoading(true)
    try {
      const res = await fetch('/api/backtests-v2/signals/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tradeDate: td, strategyKey: sk }),
      })
      const data = await res.json()
      if (data.ok) {
        setInternalSignals(data.signals || [])
        const now = new Date()
        setInternalLastUpdated(`${now.getHours()}:${String(now.getMinutes()).padStart(2,'0')}`)
      } else {
        console.error('[SignalPanel] scan failed:', data.error || data)
      }
    } catch (e) {
      console.error('[SignalPanel] network error:', e)
    } finally {
      setInternalLoading(false)
    }
  }, [])

  const handleRefresh = useCallback(() => {
    if (selfManaged) doScan(internalStrategyKey, tradeDate)
    else onRefresh && onRefresh()
  }, [selfManaged, doScan, internalStrategyKey, tradeDate, onRefresh])

  const handleLabel = useCallback(async (signalId, label) => {
    if (!selfManaged && onLabel) { onLabel(signalId, label); return }
    try {
      const res = await fetch('/api/backtests-v2/signals/label', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ signalId, label }),
      })
      const data = await res.json().catch(() => ({}))
      if (!data.ok) {
        console.error('[SignalPanel] label failed:', data.error || data)
        return
      }
      setInternalSignals(prev => prev.map(s =>
        s.signal_id === signalId ? { ...s, label } : s
      ))
    } catch (e) {
      console.error('[SignalPanel] label network error:', e)
    }
  }, [selfManaged, onLabel])

  // Auto-refresh today every 60s
  useEffect(() => {
    if (!selfManaged || !isToday || !tradeDate) return
    doScan(internalStrategyKey, tradeDate)
    refreshTimerRef.current = setInterval(() => doScan(internalStrategyKey, tradeDate), 60000)
    return () => clearInterval(refreshTimerRef.current)
  }, [selfManaged, isToday, tradeDate, internalStrategyKey, doScan])

  // On date change, clear and optionally scan
  useEffect(() => {
    if (!selfManaged) return
    setInternalSignals([])
    setInternalLastUpdated(null)
    clearInterval(refreshTimerRef.current)
    if (isToday && tradeDate) {
      doScan(internalStrategyKey, tradeDate)
      refreshTimerRef.current = setInterval(() => doScan(internalStrategyKey, tradeDate), 60000)
    }
    return () => clearInterval(refreshTimerRef.current)
  }, [tradeDate, selfManaged, isToday])

  const [pos, setPos] = useState(() => {
    try {
      const saved = window.localStorage.getItem('ib-signal-panel-pos')
      if (saved) return JSON.parse(saved)
    } catch (e) {}
    return { top: 12, left: 12 }
  })
  const [size, setSize] = useState(() => {
    try {
      const saved = window.localStorage.getItem('ib-signal-panel-size')
      if (saved) return JSON.parse(saved)
    } catch (e) {}
    return { width: 310, height: 420 }
  })
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return window.localStorage.getItem('ib-signal-panel-collapsed') === 'true'
    } catch (e) {}
    return false
  })

  const dragRef = useRef(null)
  const resizeRef = useRef(null)

  const handleDragMouseDown = (e) => {
    if (e.target.closest('.signal-panel-btn')) return
    e.stopPropagation()
    dragRef.current = {
      startX: e.clientX, startY: e.clientY,
      startTop: pos.top ?? NaN, startLeft: pos.left ?? NaN,
      startRight: pos.right ?? NaN, startBottom: pos.bottom ?? NaN,
    }
    window.addEventListener('mousemove', handleDragMouseMove)
    window.addEventListener('mouseup', handleDragMouseUp)
  }

  const handleDragMouseMove = useCallback((e) => {
    if (!dragRef.current) return
    const { startX, startY, startTop, startLeft, startRight, startBottom } = dragRef.current
    const dx = e.clientX - startX
    const dy = e.clientY - startY
    const next = {}
    if (Number.isFinite(startTop))    next.top    = startTop + dy
    if (Number.isFinite(startBottom)) next.bottom = startBottom - dy
    if (Number.isFinite(startLeft))   next.left   = startLeft + dx
    if (Number.isFinite(startRight))  next.right  = startRight - dx
    setPos(next)
  }, [])

  const handleDragMouseUp = useCallback(() => {
    window.removeEventListener('mousemove', handleDragMouseMove)
    window.removeEventListener('mouseup', handleDragMouseUp)
    setPos(p => {
      try { window.localStorage.setItem('ib-signal-panel-pos', JSON.stringify(p)) } catch (e) {}
      return p
    })
    dragRef.current = null
  }, [handleDragMouseMove])

  const handleResizeMouseDown = (e, corner) => {
    e.stopPropagation()
    resizeRef.current = {
      corner,
      startX: e.clientX, startY: e.clientY,
      startWidth: size.width, startHeight: size.height,
      startTop: pos.top ?? NaN, startLeft: pos.left ?? NaN,
      startRight: pos.right ?? NaN, startBottom: pos.bottom ?? NaN,
    }
    window.addEventListener('mousemove', handleResizeMouseMove)
    window.addEventListener('mouseup', handleResizeMouseUp)
  }

  const handleResizeMouseMove = useCallback((e) => {
    if (!resizeRef.current) return
    const { corner, startX, startY, startWidth, startHeight, startTop, startLeft, startRight, startBottom } = resizeRef.current
    const dx = e.clientX - startX
    const dy = e.clientY - startY
    let nw = startWidth, nh = startHeight
    const nextPos = { top: startTop, left: startLeft, right: startRight, bottom: startBottom }

    if (corner.includes('right'))  { nw = Math.max(260, startWidth + dx); if (Number.isFinite(startRight)) nextPos.right = startRight - (nw - startWidth) }
    if (corner.includes('left'))   { const adj = Math.min(startWidth - 260, dx); nw = startWidth - adj; if (Number.isFinite(startLeft)) nextPos.left = startLeft + adj }
    if (corner.includes('bottom')) { nh = Math.max(200, startHeight + dy); if (Number.isFinite(startBottom)) nextPos.bottom = startBottom - (nh - startHeight) }
    if (corner.includes('top'))    { const adj = Math.min(startHeight - 200, dy); nh = startHeight - adj; if (Number.isFinite(startTop)) nextPos.top = startTop + adj }

    setSize({ width: nw, height: nh })
    setPos(nextPos)
  }, [])

  const handleResizeMouseUp = useCallback(() => {
    window.removeEventListener('mousemove', handleResizeMouseMove)
    window.removeEventListener('mouseup', handleResizeMouseUp)
    setSize(s => { try { window.localStorage.setItem('ib-signal-panel-size', JSON.stringify(s)) } catch (e) {} return s })
    setPos(p => { try { window.localStorage.setItem('ib-signal-panel-pos', JSON.stringify(p)) } catch (e) {} return p })
    resizeRef.current = null
  }, [handleResizeMouseMove])

  const toggleCollapsed = (e) => {
    e.stopPropagation()
    setCollapsed(prev => {
      const next = !prev
      try { window.localStorage.setItem('ib-signal-panel-collapsed', String(next)) } catch (e) {}
      return next
    })
  }

  const completedSignals = signals.filter(s => s.status === 'completed' || s.status === 'expired')
  const activeSignals = signals.filter(s => s.status !== 'completed' && s.status !== 'expired')

  return (
    <div
      onMouseDown={handleDragMouseDown}
      onWheel={e => e.stopPropagation()}
      onClick={e => e.stopPropagation()}
      style={{
        position: 'absolute',
        zIndex: 11,
        cursor: collapsed ? 'pointer' : 'grab',
        background: 'rgba(15, 23, 42, 0.94)',
        border: '1px solid #1f2937',
        borderRadius: '12px',
        padding: collapsed ? '6px 14px' : '10px 12px',
        boxShadow: '0 10px 25px rgba(0,0,0,0.4)',
        width: collapsed ? 'auto' : size.width,
        height: collapsed ? 'auto' : size.height,
        color: '#e2e8f0',
        fontSize: '12px',
        pointerEvents: 'auto',
        userSelect: 'none',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        ...(collapsed ? { top: 8, left: 64 + 200 } : pos),
      }}
      onClick={collapsed ? toggleCollapsed : e => e.stopPropagation()}
    >
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: collapsed ? 0 : '8px',
        height: collapsed ? '32px' : 'auto',
        flexShrink: 0,
      }}>
        <div style={{
          fontWeight: 800, letterSpacing: '0.06em', fontSize: '11px', textTransform: 'uppercase',
          color: collapsed ? '#60a5fa' : '#94a3b8',
        }}>
          {collapsed ? 'SIGNALS' : 'Setups'}
          {!collapsed && signals.length > 0 && (
            <span style={{ marginLeft: '6px', color: '#475569', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>
              {signals.length} found
            </span>
          )}
        </div>
        {!collapsed && (
          <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
            {/* Strategy selector */}
            {selfManaged && (
              <select
                className="signal-panel-btn"
                value={internalStrategyKey}
                onChange={e => { setInternalStrategyKey(e.target.value); setInternalSignals([]); setInternalLastUpdated(null) }}
                onClick={e => e.stopPropagation()}
                style={{
                  background: '#0f172a', border: '1px solid #334155',
                  borderRadius: '6px', color: '#94a3b8', cursor: 'pointer',
                  fontSize: '10px', padding: '2px 4px',
                }}
              >
                <option value="up_move_short">Short</option>
                <option value="down_move_scan">Long</option>
              </select>
            )}
            {/* Scan / Refresh button */}
            {selfManaged && !isToday && (
              <button
                className="signal-panel-btn"
                onClick={e => { e.stopPropagation(); doScan(internalStrategyKey, tradeDate) }}
                disabled={isLoading}
                title="Scan this date"
                style={{
                  background: 'rgba(37,99,235,0.15)', border: '1px solid #3b82f6',
                  borderRadius: '6px', color: '#93c5fd', cursor: 'pointer',
                  fontSize: '10px', padding: '2px 7px', opacity: isLoading ? 0.5 : 1,
                }}
              >
                {isLoading ? '…' : 'Scan'}
              </button>
            )}
            {(isToday || !selfManaged) && (
              <button
                className="signal-panel-btn"
                onClick={e => { e.stopPropagation(); handleRefresh() }}
                disabled={isLoading}
                title="Refresh"
                style={{
                  background: 'transparent', border: '1px solid #334155',
                  borderRadius: '6px', color: '#94a3b8', cursor: 'pointer',
                  fontSize: '11px', padding: '2px 7px', opacity: isLoading ? 0.5 : 1,
                }}
              >
                {isLoading ? '…' : '↻'}
              </button>
            )}
            <button
              className="signal-panel-btn"
              onClick={toggleCollapsed}
              style={{
                background: 'transparent', border: 'none',
                color: '#94a3b8', cursor: 'pointer',
                fontSize: '14px', padding: '2px', lineHeight: 1,
                display: 'flex', alignItems: 'center',
              }}
            >
              −
            </button>
          </div>
        )}
      </div>

      {/* Body */}
      {!collapsed && (
        <div style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
          {isLoading && signals.length === 0 ? (
            <div style={{ color: '#475569', fontSize: '12px', textAlign: 'center', paddingTop: '24px' }}>
              Scanning…
            </div>
          ) : signals.length === 0 ? (
            <div style={{ color: '#475569', fontSize: '12px', textAlign: 'center', paddingTop: '24px' }}>
              {isToday ? 'No setups yet today' : 'No setups found — click scan to search'}
            </div>
          ) : (
            <>
              {/* Active / developing setups */}
              {activeSignals.length > 0 && (
                <>
                  {isToday && (
                    <div style={{ fontSize: '10px', color: '#475569', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '5px' }}>
                      Active
                    </div>
                  )}
                  {activeSignals.map(s => (
                    <SignalCard key={s.signal_id} signal={s} onLabel={handleLabel} isToday={isToday} />
                  ))}
                </>
              )}

              {/* Completed setups */}
              {completedSignals.length > 0 && (
                <>
                  <div style={{ fontSize: '10px', color: '#475569', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '5px', marginTop: activeSignals.length > 0 ? '10px' : 0 }}>
                    Completed
                  </div>
                  {completedSignals.map(s => (
                    <SignalCard key={s.signal_id} signal={s} onLabel={handleLabel} isToday={isToday} />
                  ))}
                </>
              )}
            </>
          )}

          {/* Footer */}
          {lastUpdated && (
            <div style={{ color: '#374151', fontSize: '10px', textAlign: 'center', marginTop: '8px', paddingTop: '6px', borderTop: '1px solid #1f2937' }}>
              Updated {lastUpdated}
            </div>
          )}
        </div>
      )}

      {/* Resize handles */}
      {!collapsed && (
        <>
          {['top-left','top-right','bottom-left','bottom-right'].map(corner => (
            <div
              key={corner}
              onMouseDown={e => handleResizeMouseDown(e, corner)}
              style={{
                position: 'absolute',
                ...(corner.includes('top')    ? { top: 0 }    : { bottom: 0 }),
                ...(corner.includes('left')   ? { left: 0 }   : { right: 0 }),
                width: '80px', height: '10px',
                cursor: corner === 'top-left' || corner === 'bottom-right' ? 'nwse-resize' : 'nesw-resize',
                zIndex: 12,
              }}
            />
          ))}
        </>
      )}
    </div>
  )
}
