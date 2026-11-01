// CR-DR item 3 — the trade review panel: opened by a click on a My trades marker (or "View on chart" in the Trade Log
// tab). Styled like the CR-DQ box panel (tag background, thin border, 11 px text), draggable by its header, default
// bottom-right inside the plot, place remembered per browser, double-click on the header returns it there.
//   a. facts (backend /review): side, qty, entry / exit, pts and $, time in trade, MFE / MAE, stop resting at entry;
//   b. "What the tools showed at entry" (tradeContext.loadEntryContext: the chart's own code on an until-replay);
//   c. Ryan's fields, autosaved (PATCH) with a visible saved state;
//   d. N / P (buttons or keys) step through the day's trades; the caller scrolls the chart to each.
// B4: on holdout dates no MFE / MAE and no "at entry" context ("holdout date: context not shown").
// C1: blue long / slate short accents, the chart's slate / light text; no red / green.
import { useCallback, useEffect, useRef, useState } from 'react'
import { BOX_RULES_DEFAULTS, kfmt } from '../flowV3.js'
import { TAG_BG } from './flowV3Primitives.js'
import { PctBadge } from './BoxPanel.jsx'
import { TRADE_LONG_RGB, TRADE_SHORT_RGB } from './tradePrimitives.js'
import { loadEntryContext, shiftedToPt } from '../tradeContext.js'

const TEXT = '#e2e8f0'
const MUTED = '#94a3b8'
const POS_KEY = 'ib-react-trade-panel-pos'
const SAVE_DEBOUNCE_MS = 700

export const SETUP_CHIPS = ['fade top', 'fade bottom', 'continuation pullback', 'break & retest', 'failed break', 'second touch', '0DTE pin', 'other']
export const MISTAKE_CHIPS = ['no stop', 'chased', 'one-minute flicker', 'against flow', 'not at edge', 'event day', 'late',
  'moved stop', 'exited early', 'held too long', 'target past level']
const REVIEW_KEYS = ['setup_chips', 'pre_trade_reason', 'mistake_chips', 'checklist_followed', 'lesson', 'grade', 'anchor_ts_pt']

function readPos() {
  try {
    const p = JSON.parse(window.localStorage.getItem(POS_KEY) || 'null')
    return p && Number.isFinite(p.right) && Number.isFinite(p.bottom) ? p : null
  } catch {
    return null
  }
}
function writePos(p) {
  try {
    if (p) window.localStorage.setItem(POS_KEY, JSON.stringify(p))
    else window.localStorage.removeItem(POS_KEY)
  } catch {
    /* storage off */
  }
}

const hms = (pt) => (pt ? String(pt).slice(11, 19) : '—')
const px = (v) => (v == null ? '—' : Number(v).toFixed(2))
const signed = (v, d = 2) => (v == null ? '—' : `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(d)}`)
export function durText(s) {
  if (s == null) return '—'
  const m = Math.floor(s / 60)
  const r = s % 60
  return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : m ? `${m}m ${String(r).padStart(2, '0')}s` : `${r}s`
}

function Chip({ on, children, onClick, testid, accent }) {
  return (
    <button type="button" onClick={onClick} aria-pressed={on} data-testid={testid}
      style={{ fontSize: '10.5px', lineHeight: '16px', padding: '0 6px', margin: '0 3px 3px 0', borderRadius: 8, cursor: 'pointer',
        border: `1px solid ${on ? `rgba(${accent},0.95)` : 'rgba(148,163,184,0.35)'}`,
        background: on ? `rgba(${accent},0.22)` : 'transparent', color: on ? '#ffffff' : '#cbd5e1' }}>
      {children}
    </button>
  )
}

function Section({ title, children, testid }) {
  return (
    <div data-testid={testid} style={{ marginTop: 5, paddingTop: 4, borderTop: '1px solid rgba(148,163,184,0.18)' }}>
      <div style={{ color: MUTED, fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>{title}</div>
      {children}
    </div>
  )
}

export default function TradeReviewPanel({
  apiBase, trade, dayTrades = [], onSelect, onClose, onSaved, chartAnchor = null, priceAxisWidth = null,
  flowSession = 'FULL', flowSettings = null, markerSettings = null, rules = BOX_RULES_DEFAULTS, baseline = null,
}) {
  const id = trade?.id
  const accent = trade?.direction === 'long' ? TRADE_LONG_RGB : TRADE_SHORT_RGB
  const [review, setReview] = useState(null)
  const [reviewErr, setReviewErr] = useState('')
  const [fields, setFields] = useState(null)
  const [save, setSave] = useState({ state: 'idle', at: null, error: '' })
  const [ctx, setCtx] = useState({ state: 'idle', data: null, error: '' })
  const pending = useRef(null)
  const timer = useRef(null)

  // facts + fields for the selected trade
  useEffect(() => {
    if (id == null) return undefined
    const controller = new AbortController()
    setReview(null)
    setReviewErr('')
    setSave({ state: 'idle', at: null, error: '' })
    fetch(`${apiBase}/api/trade-log/trades/${id}/review`, { credentials: 'include', signal: controller.signal, cache: 'no-store' })
      .then((r) => r.json())
      .then((j) => {
        if (!j?.ok) throw new Error(j?.error || 'review failed')
        setReview(j.review)
        setFields(Object.fromEntries(REVIEW_KEYS.map((k) => [k, k === 'anchor_ts_pt' ? j.review.anchor_pt : j.review[k]])))
      })
      .catch((e) => {
        if (e?.name !== 'AbortError') setReviewErr(e?.message || 'review failed')
      })
    return () => controller.abort()
  }, [apiBase, id])

  // flush a pending save when the trade changes or the panel closes
  const flush = useCallback(() => {
    if (timer.current) window.clearTimeout(timer.current)
    timer.current = null
    const body = pending.current
    pending.current = null
    if (!body || body.id == null) return Promise.resolve()
    setSave((s) => ({ ...s, state: 'saving' }))
    const { id: tid, ...patch } = body
    return fetch(`${apiBase}/api/trade-log/trades/${tid}`, {
      method: 'PATCH', credentials: 'include', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patch),
    })
      .then((r) => r.json())
      .then((j) => {
        if (!j?.ok) throw new Error(j?.error || 'save failed')
        setSave({ state: 'saved', at: new Date(), error: '' })
        onSaved?.(j.trade)
      })
      .catch((e) => setSave({ state: 'error', at: null, error: e?.message || 'save failed' }))
  }, [apiBase, onSaved])
  useEffect(() => () => { flush() }, [id, flush])

  const setField = (k, v) => {
    setFields((f) => ({ ...f, [k]: v }))
    pending.current = { ...(pending.current && pending.current.id === id ? pending.current : { id }), [k]: v }
    setSave((s) => ({ ...s, state: 'dirty' }))
    if (timer.current) window.clearTimeout(timer.current)
    timer.current = window.setTimeout(flush, SAVE_DEBOUNCE_MS)
  }
  const toggleChip = (k, c) => {
    const cur = Array.isArray(fields?.[k]) ? fields[k] : []
    setField(k, cur.includes(c) ? cur.filter((x) => x !== c) : [...cur, c])
  }

  // "at entry" context (not on holdout dates); recomputed when the anchor changes
  const anchorPt = fields?.anchor_ts_pt || null
  const holdout = Boolean(review?.holdout)
  useEffect(() => {
    if (!review || holdout) {
      setCtx({ state: 'idle', data: null, error: '' })
      return undefined
    }
    const controller = new AbortController()
    setCtx({ state: 'loading', data: null, error: '' })
    loadEntryContext({
      apiBase, trade: { ...review, anchor_pt: anchorPt }, session: flowSession, settings: flowSettings,
      markerSettings: markerSettings || {}, rules, baseline, signal: controller.signal,
    })
      .then((data) => setCtx({ state: 'ok', data, error: '' }))
      .catch((e) => {
        if (e?.name !== 'AbortError') setCtx({ state: 'error', data: null, error: e?.message || 'context failed' })
      })
    return () => controller.abort()
  }, [apiBase, review, holdout, anchorPt, flowSession, flowSettings, markerSettings, rules, baseline])

  // N / P: buttons and keys (not while typing)
  const idx = dayTrades.findIndex((t) => t.id === id)
  const step = useCallback((d) => {
    const next = dayTrades[idx + d]
    if (next) onSelect?.(next.id)
  }, [dayTrades, idx, onSelect])
  useEffect(() => {
    const onKey = (e) => {
      const tag = e.target?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || e.metaKey || e.ctrlKey || e.altKey) return
      if (e.key === 'n' || e.key === 'N') step(1)
      else if (e.key === 'p' || e.key === 'P') step(-1)
      else if (e.key === 'Escape') onClose?.()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [step, onClose])

  // place: px from the pane's right / bottom edges; default just inside the plot (price axis, time axis)
  const [pos, setPos] = useState(() => readPos())
  const drag = useRef(null)
  const panelRef = useRef(null)
  const dflt = { right: (Number(priceAxisWidth) || 60) + 6, bottom: 34 }
  const place = pos || dflt
  const onDown = (e) => {
    if (e.button !== 0 || e.target?.closest?.('button')) return
    const el = panelRef.current
    const parent = el?.offsetParent
    if (!el || !parent) return
    e.preventDefault()
    e.currentTarget.setPointerCapture?.(e.pointerId)
    drag.current = { x: e.clientX, y: e.clientY, start: place, pw: parent.clientWidth, ph: parent.clientHeight, w: el.offsetWidth, h: el.offsetHeight }
  }
  const onMove = (e) => {
    const d = drag.current
    if (!d) return
    const right = Math.max(0, Math.min(d.pw - d.w, d.start.right - (e.clientX - d.x)))
    const bottom = Math.max(0, Math.min(d.ph - d.h, d.start.bottom - (e.clientY - d.y)))
    setPos({ right, bottom })
  }
  const onUp = (e) => {
    if (!drag.current) return
    drag.current = null
    e.currentTarget.releasePointerCapture?.(e.pointerId)
    setPos((p) => {
      writePos(p)
      return p
    })
  }

  if (!trade) return null
  const r = review
  const long = trade.direction === 'long'
  const usd = r ? r.realized_pts * r.qty * r.multiplier : null
  const saveText = save.state === 'saving' ? 'saving…' : save.state === 'dirty' ? 'editing…'
    : save.state === 'saved' ? `saved ${save.at.toTimeString().slice(0, 8)}` : save.state === 'error' ? `not saved: ${save.error}` : ''
  const c = ctx.data
  const box = c?.box

  return (
    <div ref={panelRef} data-testid="trade-review-panel" data-trade-id={id}
      style={{ position: 'absolute', right: place.right, bottom: place.bottom, zIndex: 6, width: 360, maxHeight: '78%',
        display: 'flex', flexDirection: 'column', background: TAG_BG(0.95), border: `1px solid rgba(${accent},0.55)`, borderRadius: 4,
        fontSize: '11px', fontFamily: 'sans-serif', color: TEXT, lineHeight: 1.55 }}>
      <div data-testid="trade-review-header" title="Drag to move · double-click to return bottom-right"
        onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}
        onDoubleClick={() => { setPos(null); writePos(null) }}
        style={{ cursor: 'move', userSelect: 'none', touchAction: 'none', padding: '2px 6px 2px 8px', color: MUTED, display: 'flex',
          alignItems: 'center', gap: 6, borderBottom: `1px solid rgba(${accent},0.25)`, flex: '0 0 auto' }}>
        <span style={{ flex: '1 1 auto' }}>
          Trade {idx >= 0 ? `${idx + 1} / ${dayTrades.length}` : ''} · {String(trade.trade_date).slice(0, 10)}
        </span>
        <button type="button" data-testid="trade-review-prev" disabled={idx <= 0} onClick={() => step(-1)} title="Previous trade (P)"
          style={navBtn(idx <= 0)}>◂ P</button>
        <button type="button" data-testid="trade-review-next" disabled={idx < 0 || idx >= dayTrades.length - 1} onClick={() => step(1)}
          title="Next trade (N)" style={navBtn(idx < 0 || idx >= dayTrades.length - 1)}>N ▸</button>
        <button type="button" data-testid="trade-review-close" aria-label="Close the trade panel" title="Close (Esc)"
          onClick={() => { flush(); onClose?.() }} style={{ ...navBtn(false), fontSize: '12px' }}>×</button>
      </div>

      <div data-testid="trade-review-body" style={{ overflowY: 'auto', padding: '4px 8px 6px', flex: '1 1 auto' }}>
        {reviewErr && <div style={{ color: MUTED }}>Could not load the trade: {reviewErr}</div>}
        {!r && !reviewErr && <div style={{ color: MUTED }}>loading…</div>}
        {r && (
          <div data-testid="trade-review-facts">
            <div style={{ fontWeight: 700, color: `rgb(${accent})` }}>
              {long ? '▲ LONG' : '▼ SHORT'} {r.qty} × {r.symbol}
            </div>
            <div data-testid="trade-review-entry">entry {hms(r.entry_pt)} @ {px(r.entry_price)} <span style={{ color: MUTED }}>({r.entry_order_type})</span></div>
            <div data-testid="trade-review-exit">exit {hms(r.exit_pt)} @ {px(r.exit_price)} <span style={{ color: MUTED }}>({r.exit_order_type})</span></div>
            <div data-testid="trade-review-result">
              <b>{signed(r.realized_pts)} pts</b> · {usd == null ? '—' : `${usd < 0 ? '−' : '+'}$${Math.abs(usd).toFixed(2)}`}
              <span style={{ color: MUTED }}> (×{r.multiplier}/pt{r.qty > 1 ? ` × ${r.qty}` : ''})</span> · {durText(r.held_s)} in trade
            </div>
            {holdout ? (
              <div data-testid="trade-review-holdout" style={{ color: MUTED }}>holdout date: context not shown</div>
            ) : (
              <div data-testid="trade-review-mfe" title="ES 1-minute bars from the entry minute through the exit minute (MES vs ES ± 1 tick)">
                MFE {signed(r.mfe_pts)} · MAE {r.mae_pts == null ? '—' : `−${Number(r.mae_pts).toFixed(2)}`} pts
              </div>
            )}
            <div data-testid="trade-review-stop">
              stop at entry: {r.stop_at_entry ? 'yes' : r.stop_after_s == null ? 'no — no stop while open' : `no — first stop after ${r.stop_after_s} s`}
              {(() => {
                const st = (r.lines || []).filter((l) => l.kind === 'stop')
                const tp = (r.lines || []).filter((l) => l.kind === 'target')
                return <span style={{ color: MUTED }}>{st.length ? ` · stop ${st.map((l) => px(l.price)).join(' → ')}` : ''}{tp.length ? ` · TP ${[...new Set(tp.map((l) => px(l.price)))].join(' → ')}` : ''}</span>
              })()}
            </div>
          </div>
        )}

        {r && !holdout && (
          <Section title="What the tools showed at entry" testid="trade-review-context">
            <div style={{ color: MUTED }}>
              replay until {hms(r.entry_pt).slice(0, 5)} (bars before the entry minute){c?.cutoff ? ` · last bar ${c.cutoff}` : ''}
            </div>
            <div data-testid="trade-review-anchor">
              anchor: {anchorPt ? hms(anchorPt).slice(0, 5) : 'no anchor'}{' '}
              <button type="button" data-testid="trade-review-anchor-chart" disabled={chartAnchor == null}
                title={chartAnchor == null ? 'Set a Flow v3 anchor on the chart first' : 'Use the chart’s Flow v3 anchor as this trade’s box anchor'}
                onClick={() => setField('anchor_ts_pt', shiftedToPt(chartAnchor))} style={smallBtn(chartAnchor == null)}>use chart anchor</button>
              {anchorPt && <button type="button" data-testid="trade-review-anchor-clear" onClick={() => setField('anchor_ts_pt', null)} style={smallBtn(false)}>clear</button>}
            </div>
            {ctx.state === 'loading' && <div style={{ color: MUTED }}>loading replay…</div>}
            {ctx.state === 'error' && <div style={{ color: MUTED }}>context failed: {ctx.error}</div>}
            {c && (
              <>
                {!anchorPt && <div data-testid="trade-review-verdict" data-kind="none" style={{ color: MUTED }}>no anchor — no box verdict</div>}
                {anchorPt && !box && <div data-testid="trade-review-verdict" data-kind="none" style={{ color: MUTED }}>no box from that anchor before entry</div>}
                {box?.verdict && (
                  <div data-testid="trade-review-verdict" data-kind={box.verdict.kind} title={box.verdict.title}>
                    <span style={{ display: 'inline-block', borderRadius: 3, padding: '0 6px', border: '1px solid rgba(226,232,240,0.6)' }}>{box.verdict.label}</span>
                    <div data-testid="trade-review-evidence" style={{ color: '#cbd5e1' }}>{box.verdict.evidence}</div>
                  </div>
                )}
                {box && (
                  <div data-testid="trade-review-box">
                    box {box.from}–{box.to} · {px(box.bottom)}–{px(box.top)}
                    {box.current?.pct && <> · Δ share <PctBadge kind="share" pct={box.current.pct} rules={rules} /> · c/pt <PctBadge kind="cpp" pct={box.current.pct} rules={rules} /></>}
                    {box.current && <div style={{ color: MUTED }}>{box.current.lines?.head}</div>}
                  </div>
                )}
                <div data-testid="trade-review-levels">
                  absorption within 5 pts: {c.levels.length ? c.levels.map((lv) => `${lv.kind === 'sellers_absorbed' ? 'sellers' : 'buyers'} absorbed ${px(lv.price)} (${signed(lv.dist)}, ${kfmt(lv.contracts)}, ${lv.since})`).join(' · ') : 'none'}
                </div>
                <div data-testid="trade-review-last5">
                  last {c.last5.minutes} min{c.last5.from ? ` (${c.last5.from}–${c.last5.to})` : ''}: Δ {c.last5.delta > 0 ? '+' : ''}{Math.round(c.last5.delta).toLocaleString('en-US')} · vol {kfmt(c.last5.vol)}
                </div>
              </>
            )}
          </Section>
        )}

        {r && fields && (
          <Section title="Your review" testid="trade-review-fields">
            <div style={{ marginTop: 2 }}>
              {SETUP_CHIPS.map((ch) => (
                <Chip key={ch} on={(fields.setup_chips || []).includes(ch)} accent={accent} testid={`trade-setup-chip-${ch}`} onClick={() => toggleChip('setup_chips', ch)}>{ch}</Chip>
              ))}
            </div>
            <textarea data-testid="trade-review-reason" placeholder="pre-trade reason" value={fields.pre_trade_reason || ''}
              onChange={(e) => setField('pre_trade_reason', e.target.value)} rows={2} style={areaStyle} />
            <div style={{ color: MUTED, marginTop: 2 }}>mistakes</div>
            <div>
              {MISTAKE_CHIPS.map((ch) => (
                <Chip key={ch} on={(fields.mistake_chips || []).includes(ch)} accent={TRADE_SHORT_RGB} testid={`trade-mistake-chip-${ch}`} onClick={() => toggleChip('mistake_chips', ch)}>{ch}</Chip>
              ))}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexWrap: 'wrap' }}>
              <span style={{ color: MUTED }}>checklist followed</span>
              {[[true, 'yes'], [false, 'no']].map(([v, label]) => (
                <Chip key={label} on={fields.checklist_followed === v} accent={accent} testid={`trade-checklist-${label}`}
                  onClick={() => setField('checklist_followed', fields.checklist_followed === v ? null : v)}>{label}</Chip>
              ))}
              <span style={{ color: MUTED, marginLeft: 8 }}>grade</span>
              {[1, 2, 3, 4, 5].map((g) => (
                <Chip key={g} on={fields.grade === g} accent={accent} testid={`trade-grade-${g}`} onClick={() => setField('grade', fields.grade === g ? null : g)}>{g}</Chip>
              ))}
            </div>
            <textarea data-testid="trade-review-lesson" placeholder="lesson" value={fields.lesson || ''}
              onChange={(e) => setField('lesson', e.target.value)} rows={2} style={areaStyle} />
            <div data-testid="trade-review-save" data-state={save.state} style={{ color: save.state === 'error' ? '#fbbf24' : MUTED, minHeight: 16 }}>{saveText}</div>
          </Section>
        )}
      </div>
    </div>
  )
}

const navBtn = (disabled) => ({ border: '1px solid rgba(148,163,184,0.35)', background: 'transparent', color: disabled ? '#475569' : TEXT,
  borderRadius: 3, fontSize: '10.5px', lineHeight: '14px', padding: '0 5px', cursor: disabled ? 'default' : 'pointer' })
const smallBtn = (disabled) => ({ ...navBtn(disabled), fontSize: '10px', marginLeft: 2 })
const areaStyle = { width: '100%', boxSizing: 'border-box', marginTop: 3, background: 'rgba(15,23,42,0.7)', color: TEXT,
  border: '1px solid rgba(148,163,184,0.3)', borderRadius: 3, fontSize: '11px', fontFamily: 'sans-serif', padding: '2px 4px', resize: 'vertical' }
