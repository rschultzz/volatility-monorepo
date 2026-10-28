// ReviewCard — the active review candidate's header and key facts (CR-DB).
//
// CardHeader: the trade direction as a large label ("SHORT the top" / "LONG the
// bottom" when the card has a range box, else "SHORT" / "LONG"), date · time ·
// cell, and the called edge with the entry limit (Q1). KeyFacts: one readable
// line per known field; a field the card lacks is simply not shown, so every
// review set uses the same layout. Everything else goes to ReviewMeta under the
// panel's "details" toggle.

export const DIRECTION_COLOR = { long: '#60a5fa', short: '#cbd5e1' }

// meta keys KeyFacts shows; ReviewMeta leaves them out of the details
export const KEY_FACT_META_KEYS = [
  'approach', 'approach_pts', 'delta_share_pct', 'progress_pts', 'cpp_up', 'cpp_down', 'cpp_ratio',
]

function fmt(v, digits = 2) {
  if (v == null || v === '') return null
  const n = Number(v)
  if (!Number.isFinite(n)) return null
  return Number.isInteger(n) ? String(n) : n.toFixed(digits)
}

function hhmm(ts) {
  const m = /T(\d{2}:\d{2})/.exec(String(ts || ''))
  return m ? m[1] : null
}

function minutesBetween(a, b) {
  const ta = Date.parse(String(a || '').slice(0, 19))
  const tb = Date.parse(String(b || '').slice(0, 19))
  if (!Number.isFinite(ta) || !Number.isFinite(tb) || tb < ta) return null
  return Math.round((tb - ta) / 60000)
}

function rangeOf(cand) {
  const r = cand?.meta?.range
  if (!r || typeof r !== 'object') return null
  const top = Number(r.top)
  const bottom = Number(r.bottom)
  return Number.isFinite(top) && Number.isFinite(bottom) ? { ...r, top, bottom } : null
}

export function CardHeader({ cand }) {
  const long = cand.direction === 'long'
  const range = rangeOf(cand)
  const dir = cand.direction ? cand.direction.toUpperCase() : '—'
  const title = range ? `${dir} the ${long ? 'bottom' : 'top'}` : dir
  const edge = range ? (long ? range.bottom : range.top) : null
  const time = hhmm(cand.entry_ts_pt)
  return (
    <div data-testid="review-card-header" style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <span
          data-testid="review-direction"
          style={{ fontSize: 17, fontWeight: 800, letterSpacing: '0.02em', color: DIRECTION_COLOR[cand.direction] || '#e2e8f0' }}
        >
          {long ? '▲' : '▼'} {title}
        </span>
        <span style={{ marginLeft: 'auto', fontSize: 9, color: '#475569' }}>{cand.id}</span>
      </div>
      <div style={{ fontSize: 12, fontWeight: 600, color: '#e2e8f0', fontVariantNumeric: 'tabular-nums' }}>
        {cand.session_date}{time ? ` · ${time} PT` : ''}
        {cand.cell ? <span style={{ color: '#94a3b8', fontWeight: 500 }}> · {cand.cell}</span> : null}
      </div>
      <div data-testid="review-edge" style={{ fontSize: 12, color: '#cbd5e1', fontVariantNumeric: 'tabular-nums' }}>
        {edge != null ? (
          <>
            edge <b style={{ color: '#f8fafc' }}>{fmt(edge)}</b>
            {fmt(cand.entry_price) ? <span style={{ color: '#94a3b8' }}> (limit {fmt(cand.entry_price)})</span> : null}
          </>
        ) : (
          <>entry <b style={{ color: '#f8fafc' }}>{fmt(cand.entry_price) ?? '—'}</b></>
        )}
      </div>
    </div>
  )
}

export function keyFacts(cand) {
  const m = cand?.meta || {}
  const facts = []
  const range = rangeOf(cand)
  if (range) {
    const height = range.height != null ? range.height : range.top - range.bottom
    facts.push([m.box_basis ? 'Box (closes)' : 'Box', `${fmt(range.bottom)} – ${fmt(range.top)} · ${fmt(height)} pts wide`])
    const start = hhmm(range.formed_ts_pt)
    if (start) {
      const dur = minutesBetween(range.formed_ts_pt, range.qualified_ts_pt || cand.entry_ts_pt)
      facts.push(['Range start', `${start}${dur != null ? ` · ${dur} min` : ''}`])
    }
  }
  if (m.approach != null || m.approach_pts != null) {
    const pts = fmt(m.approach_pts)
    facts.push(['Approach', [m.approach ? `${m.approach} leg` : 'leg', pts ? `${pts} pts` : null].filter(Boolean).join(', ')])
  }
  if (fmt(m.delta_share_pct, 1) != null) facts.push(['In-range delta share', `${fmt(m.delta_share_pct, 1)} %`])
  if (fmt(m.progress_pts) != null) facts.push(['Progress', `${fmt(m.progress_pts)} pts`])
  if (fmt(m.cpp_ratio) != null) {
    const parts = [fmt(m.cpp_up) && `up ${fmt(m.cpp_up)}`, fmt(m.cpp_down) && `down ${fmt(m.cpp_down)}`].filter(Boolean)
    facts.push(['Contracts / pt', `${fmt(m.cpp_ratio)}×${parts.length ? ` (${parts.join(' / ')})` : ''}`])
  }
  if (fmt(cand?.level_price) != null) {
    const bn = [fmt(cand.level_net_bn, 1) && `net ${fmt(cand.level_net_bn, 1)}`, fmt(cand.level_gross_bn, 1) && `gross ${fmt(cand.level_gross_bn, 1)}`]
      .filter(Boolean)
    facts.push(['Level', `${fmt(cand.level_price)}${bn.length ? ` · ${bn.join(' / ')} $bn` : ''}`])
  }
  if (fmt(cand?.stop_price) != null) facts.push(['Stop', fmt(cand.stop_price)])
  return facts
}

export function KeyFacts({ cand }) {
  const facts = keyFacts(cand)
  if (!facts.length) return null
  return (
    <div
      data-testid="review-key-facts"
      style={{ display: 'grid', gridTemplateColumns: 'max-content 1fr', gap: '3px 10px', fontSize: 11, alignItems: 'baseline' }}
    >
      {facts.map(([k, v]) => (
        <span key={k} style={{ display: 'contents' }}>
          <span style={{ color: '#94a3b8' }}>{k}</span>
          <span style={{ color: '#e2e8f0', fontVariantNumeric: 'tabular-nums' }}>{v}</span>
        </span>
      ))}
    </div>
  )
}
