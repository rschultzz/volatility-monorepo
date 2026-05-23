// DayListBrowse — list of corpus days for browse mode.
const REGIME_COLORS = {
  'magnetic-pin': '#10b981', 'magnet-above': '#fbbf24',
  'magnet-below': '#fbbf24', bounded: '#10b981',
  amplification: '#06b6d4', 'broken-magnet': '#a78bfa',
  untethered: '#94a3b8', pinned: '#10b981',
}

function fmtPts(v) {
  if (v == null || !Number.isFinite(Number(v))) return '—'
  const n = Number(v)
  return (n >= 0 ? '+' : '') + n.toFixed(1)
}

export default function DayListBrowse({ days, selectedDate, onSelect }) {
  if (!days || days.length === 0) {
    return <div style={{ color: '#475569', fontSize: 12, padding: '16px 0' }}>No days found.</div>
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {days.map((d) => {
        const active = d.trade_date === selectedDate
        const rc = REGIME_COLORS[d.regime] || '#94a3b8'
        return (
          <div
            key={d.trade_date}
            onClick={() => onSelect(d.trade_date)}
            style={{
              padding: '7px 10px',
              borderRadius: 6,
              cursor: 'pointer',
              background: active ? '#1e293b' : 'transparent',
              border: active ? '1px solid #334155' : '1px solid transparent',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 12, color: '#e2e8f0', fontFamily: 'monospace', minWidth: 76 }}>
                {d.trade_date}
              </span>
              <span style={{
                fontSize: 9, fontWeight: 700, padding: '1px 5px',
                borderRadius: 4, color: rc, background: `${rc}22`,
              }}>
                {(d.regime || '—').replace(/-/g, ' ')}
              </span>
              {d.outcomes && (
                <span style={{ fontSize: 9, color: Number(d.outcomes.eod_return_pts) >= 0 ? '#22c55e' : '#ef4444', marginLeft: 'auto' }}>
                  {fmtPts(d.outcomes.eod_return_pts)}pt
                </span>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
