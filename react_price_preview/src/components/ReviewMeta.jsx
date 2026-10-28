// ReviewMeta — the review candidate's `meta` as readable rows (CR-CC A3.1).
//
// Known nested keys get compact one-line summaries (flow, rules, skew, named,
// range, zone / zone_e4 / cluster_members). Other keys: scalars as rows,
// objects and arrays behind a per-row "show raw" toggle. Raw JSON is never
// shown by default. Keys the panel already shows (level_frame, a cell equal to
// the candidate's) are skipped, and so are `omit` keys (CR-DB: the key facts
// already shown above the details). Long text values (rules, basis notes) are
// full-width wrapped paragraphs with the label above, not a value cell.

import { useState } from 'react'

const FLOW_ORDER = ['P', 'A_rel', 'Q', 'PD', 'LE', 'RA', 'DV', 'SS']
const DIV_TEXT = { bull: 'bullish', bear: 'bearish', none: 'none', na: 'n/a' }
const RULE_COLOR = { fade: '#60a5fa', break: '#94a3b8' } // CR-DB / C1: Dash blue / slate
const SPECIAL = new Set(['flow', 'rules', 'skew', 'named', 'range', 'zone', 'zone_e4', 'cluster_members'])
const LONG_TEXT = 40

function num(v, digits = 2) {
  if (v == null || v === '') return '—'
  const n = Number(v)
  if (!Number.isFinite(n)) return String(v)
  if (Number.isInteger(n)) return String(n)
  // skew-scale values (|x| < 0.01) keep their significant digits
  return Math.abs(n) < 0.01 && n !== 0 ? n.toFixed(5) : n.toFixed(digits)
}

function clock(iso, seconds = false) {
  const m = /T(\d{2}:\d{2})(:\d{2})?/.exec(String(iso || ''))
  if (!m) return String(iso ?? '—')
  return seconds && m[2] ? m[1] + m[2] : m[1]
}

function scalar(key, v) {
  if (typeof v === 'boolean') return v ? 'yes' : 'no'
  if (typeof v === 'string' && /_ts_pt$|_ts$/.test(key)) return clock(v, true)
  if (typeof v === 'number') return num(v)
  return v == null || v === '' ? '—' : String(v)
}

function flowSummary(flow) {
  const parts = []
  const seen = new Set()
  for (const k of FLOW_ORDER) {
    if (k === 'SS' && !(k in flow)) {
      if ('SS_vwap' in flow || 'SS_share' in flow) {
        parts.push(['SS', `${num(flow.SS_vwap)} / ${num(flow.SS_share)}`])
        seen.add('SS_vwap')
        seen.add('SS_share')
      }
      continue
    }
    if (k in flow) {
      parts.push([k, num(flow[k])])
      seen.add(k)
    }
  }
  for (const [k, v] of Object.entries(flow)) {
    if (!seen.has(k) && (v == null || typeof v !== 'object')) parts.push([k, num(v)])
  }
  return parts
}

function skewSummary(skew) {
  const out = []
  if ('div_ps' in skew) out.push(['put div', DIV_TEXT[skew.div_ps] || String(skew.div_ps)])
  if ('div_cs' in skew) out.push(['call div', DIV_TEXT[skew.div_cs] || String(skew.div_cs)])
  if ('trend_ps' in skew) out.push(['put trend', num(skew.trend_ps)])
  if ('trend_chg_ps' in skew) out.push(['Δ trend', num(skew.trend_chg_ps)])
  if ('dps' in skew) out.push(['dps', num(skew.dps)])
  const shown = new Set(['div_ps', 'div_cs', 'trend_ps', 'trend_chg_ps', 'dps'])
  for (const [k, v] of Object.entries(skew)) {
    if (shown.has(k)) continue
    out.push([k, typeof v === 'string' ? (DIV_TEXT[v] || v) : num(v)])
  }
  return out
}

function namedSummary(named) {
  const fired = Object.entries(named).filter(([, v]) => v === true).map(([k]) => k.replace(/_/g, '-'))
  return fired.length ? fired.join(', ') : 'none'
}

function rangeSummary(r) {
  const top = Number(r.top)
  const bottom = Number(r.bottom)
  const height = r.height != null ? Number(r.height) : top - bottom
  let s = `${num(bottom)}–${num(top)} (${num(height)})`
  if (r.formed_ts_pt) s += `, formed ${clock(r.formed_ts_pt)}`
  if (r.held_before_edge_min != null) s += `, held ${num(r.held_before_edge_min, 1)} min`
  return s
}

const KIND_TEXT = { gex: 'GEX', swing_high: 'swing high', swing_low: 'swing low' }

function pricesSummary(prices) {
  const ps = prices.filter((p) => Number.isFinite(p))
  if (!ps.length) return '—'
  const span = Math.max(...ps) - Math.min(...ps)
  return span > 0 ? `span ${num(span)} pts` : null
}

function zoneSummary(key, v) {
  if (key === 'zone' && v && typeof v === 'object' && !Array.isArray(v)) {
    const span = Number(v.hi) - Number(v.lo)
    let s = `${num(v.lo)}–${num(v.hi)}, span ${num(span)} pts`
    if (v.core_lo != null && v.core_hi != null) s += `; core ${num(v.core_lo)}–${num(v.core_hi)}`
    return s
  }
  if (Array.isArray(v)) {
    const items = v.map((x) => (x && typeof x === 'object'
      ? { kind: KIND_TEXT[x.kind] || x.kind || '', price: Number(x.price) }
      : { kind: '', price: Number(x) }))
    const head = items.slice(0, 4).map((x) => `${x.kind ? `${x.kind} ` : ''}${num(x.price)}`).join(' + ')
    const more = items.length > 4 ? ` + ${items.length - 4} more` : ''
    const span = pricesSummary(items.map((x) => x.price))
    return `${head}${more}${span ? `, ${span}` : ''}`
  }
  return scalar(key, v)
}

function Row({ label, children, raw }) {
  const [open, setOpen] = useState(false)
  return (
    <div style={{ gridColumn: '1 / -1', display: 'grid', gridTemplateColumns: '64px 1fr auto', gap: 6, alignItems: 'baseline' }}>
      <span style={{ color: '#64748b' }}>{label}</span>
      <span style={{ color: '#e2e8f0', fontVariantNumeric: 'tabular-nums', minWidth: 0 }}>{children}</span>
      {raw !== undefined ? (
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          style={{ background: 'transparent', border: 'none', color: '#64748b', fontSize: 9, cursor: 'pointer', padding: 0 }}
        >
          {open ? 'hide raw' : 'show raw'}
        </button>
      ) : <span />}
      {open && raw !== undefined && (
        <pre
          data-testid="review-meta-raw"
          style={{ gridColumn: '1 / -1', margin: 0, padding: 4, background: '#0f172a', color: '#94a3b8', fontSize: 9, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
        >
          {JSON.stringify(raw, null, 1)}
        </pre>
      )}
    </div>
  )
}

function Pairs({ pairs }) {
  return (
    <span style={{ display: 'flex', flexWrap: 'wrap', columnGap: 8 }}>
      {pairs.map(([k, v]) => (
        <span key={k}><span style={{ color: '#64748b' }}>{k}</span> {v}</span>
      ))}
    </span>
  )
}

const isLongText = (v) => typeof v === 'string' && v.length > LONG_TEXT

export default function ReviewMeta({ meta, candidate, omit = [] }) {
  const entries = Object.entries(meta || {}).filter(([k, v]) => {
    if (k === 'level_frame') return false // shown as "frame"
    if (k === 'cell' && candidate && v === candidate.cell) return false // shown as "cell"
    if (omit.includes(k)) return false
    return true
  })
  const texts = entries.filter(([k, v]) => !SPECIAL.has(k) && isLongText(v))
  const scalars = entries.filter(([k, v]) => !SPECIAL.has(k) && !isLongText(v) && (v == null || typeof v !== 'object'))
  const objects = entries.filter(([k, v]) => !SPECIAL.has(k) && v != null && typeof v === 'object')
  const get = (k) => (meta && k in meta ? meta[k] : undefined)

  const flow = get('flow')
  const rules = get('rules')
  const skew = get('skew')
  const named = get('named')
  const range = get('range')

  return (
    <div data-testid="review-meta" style={{ display: 'contents' }}>
      {scalars.map(([k, v]) => (
        <span key={k} style={{ display: 'contents' }}>
          <span style={{ color: '#64748b' }}>{k}</span>
          <span style={{ color: '#e2e8f0', fontVariantNumeric: 'tabular-nums', overflow: 'hidden', textOverflow: 'ellipsis' }}>{scalar(k, v)}</span>
        </span>
      ))}
      {flow && typeof flow === 'object' && <Row label="flow"><Pairs pairs={flowSummary(flow)} /></Row>}
      {rules && typeof rules === 'object' && (
        <Row label="rules">
          <span data-testid="review-rules" style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
            {Object.entries(rules).map(([k, v]) => {
              const decision = v === 'fade' || v === 'break' ? v : '—'
              return (
                <span
                  key={k}
                  style={{
                    padding: '0 5px',
                    borderRadius: 8,
                    border: `1px solid ${RULE_COLOR[decision] || '#334155'}`,
                    color: RULE_COLOR[decision] || '#64748b',
                    fontSize: 9,
                  }}
                >
                  {k} {decision}
                </span>
              )
            })}
          </span>
        </Row>
      )}
      {skew && typeof skew === 'object' && <Row label="skew"><Pairs pairs={skewSummary(skew)} /></Row>}
      {named && typeof named === 'object' && <Row label="named">{namedSummary(named)}</Row>}
      {range && typeof range === 'object' && <Row label="range">{rangeSummary(range)}</Row>}
      {['zone', 'zone_e4', 'cluster_members'].map((k) => (get(k) !== undefined ? (
        <Row key={k} label={k === 'cluster_members' ? 'cluster' : k} raw={get(k)}>{zoneSummary(k, get(k))}</Row>
      ) : null))}
      {texts.map(([k, v]) => (
        <div key={k} data-testid="review-meta-text" style={{ gridColumn: '1 / -1', marginTop: 2 }}>
          <div style={{ color: '#64748b' }}>{k.replace(/_/g, ' ')}</div>
          <p style={{ margin: 0, color: '#cbd5e1', lineHeight: 1.4, whiteSpace: 'normal', overflowWrap: 'break-word' }}>{v}</p>
        </div>
      ))}
      {objects.map(([k, v]) => (
        <Row key={k} label={k} raw={v}>
          {Array.isArray(v) ? `${v.length} item${v.length === 1 ? '' : 's'}` : `${Object.keys(v).length} fields`}
        </Row>
      ))}
    </div>
  )
}
