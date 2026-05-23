// DayListAnalogue — list of KNN analogues for the analogue mode left column.
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

function FeatureDistanceRow({ fd }) {
  if (!fd) return null
  return (
    <span style={{ fontSize: 9, color: '#64748b' }}>
      {fd.feature_name.replace(/_/g, ' ')}: {fd.sigma_delta > 0 ? '+' : ''}{fd.sigma_delta.toFixed(1)}σ
    </span>
  )
}

export default function DayListAnalogue({ analogues, selectedDate, onSelect }) {
  if (!analogues || analogues.length === 0) {
    return <div style={emptyMsg}>No analogues found.</div>
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {analogues.map((a) => {
        const active = a.trade_date === selectedDate
        const rc = REGIME_COLORS[a.regime] || '#94a3b8'
        const top3 = (a.feature_distances || []).slice(0, 3)
        return (
          <div
            key={a.trade_date}
            onClick={() => onSelect(a.trade_date)}
            style={{
              padding: '8px 10px',
              borderRadius: 6,
              cursor: 'pointer',
              background: active ? '#1e293b' : 'transparent',
              border: active ? '1px solid #334155' : '1px solid transparent',
              transition: 'background 0.12s',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 12, color: '#e2e8f0', fontFamily: 'monospace', minWidth: 76 }}>
                {a.trade_date}
              </span>
              <span style={{
                fontSize: 9, fontWeight: 700, padding: '1px 5px',
                borderRadius: 4, color: rc, background: `${rc}22`,
              }}>
                {(a.regime || '—').replace(/-/g, ' ')}
              </span>
              <span style={{ fontSize: 10, color: '#94a3b8', marginLeft: 'auto' }}>
                Δ{Number(a.similarity_score).toFixed(3)}
              </span>
            </div>

            {/* Top 3 feature contributors */}
            {top3.length > 0 && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px 10px', marginTop: 3 }}>
                {top3.map(fd => <FeatureDistanceRow key={fd.feature_name} fd={fd} />)}
              </div>
            )}

            {/* Outcome strip */}
            {a.outcomes && (
              <div style={{ display: 'flex', gap: 10, marginTop: 3 }}>
                <span style={{ fontSize: 9, color: getColor(a.outcomes.eod_return_pts) }}>
                  EOD {fmtPts(a.outcomes.eod_return_pts)}pt
                </span>
                {a.outcomes.intraday_range_pts != null && (
                  <span style={{ fontSize: 9, color: '#64748b' }}>
                    rng {Number(a.outcomes.intraday_range_pts).toFixed(1)}pt
                  </span>
                )}
                {a.outcomes.mfe_above_open_pts != null && (
                  <span style={{ fontSize: 9, color: '#22c55e' }}>
                    ↑{fmtPts(a.outcomes.mfe_above_open_pts)}
                  </span>
                )}
                {a.outcomes.mfe_below_open_pts != null && (
                  <span style={{ fontSize: 9, color: '#ef4444' }}>
                    ↓{fmtPts(a.outcomes.mfe_below_open_pts)}
                  </span>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function getColor(v) {
  if (v == null) return '#64748b'
  return Number(v) >= 0 ? '#22c55e' : '#ef4444'
}

const emptyMsg = { color: '#475569', fontSize: 12, padding: '16px 0' }
