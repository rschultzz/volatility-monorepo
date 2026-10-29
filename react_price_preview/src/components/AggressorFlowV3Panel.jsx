// CR-DC — Flow v3 pane: anchored CVD, rolling net delta and the v2 delta histogram, bar-aligned with the
// price chart like Flow v2 (same slot, height, time-scale sync, crosshair link, session bands). Its series
// come from the v2 payloads (/api/ironbeam/flow?mode=v2&effort=1 — the legs give "anchor = last swing");
// the header switches the price-pane layers drawn from /api/ironbeam/flow/cells: Absorption levels and
// Profile, plus the edge readout (CR-DF); Heatmap (cells) and Absorbed (marks) moved into ⚙ (CR-DF item 3, DF1).
// Palette (C1): blue = buyers, light grey = sellers, slate = neutral / marks; magenta = buyers-absorbed levels and
// edges (C1 deviation approved by Ryan, CR-DF); orange = sellers-absorbed levels (C1 deviation approved by Ryan,
// CR-DG). Every element is a visual aid, untested as a signal. CR-DM: the middle slot is volume per minute; rolling net
// delta is a ⚙ option (off by default) that takes its old slot back and pushes volume one slot down (the pane grows).
import { useEffect, useMemo, useRef, useState } from 'react'
import { createChart, CrosshairMode, ColorType, BaselineSeries, HistogramSeries, LineSeries } from 'lightweight-charts'
import {
  FlowModeToggle,
  coordinateForLinkedCrosshair,
  computeSessionBands,
  formatShiftedHHMM,
  normalizeLogicalRange,
  normalizeShiftedCandleTimes,
  rangesClose,
  utcEpochShowingZoneTime,
} from './AggressorFlowPanel'
import { alignV2ToCandles } from './AggressorFlowV2Panel'
import {
  BOX_HOLD_MIN_RANGE,
  DEFAULT_V3_SETTINGS,
  LIVE_MIN_MOVE_RANGE,
  ROLLING_CHOICES,
  anchoredCvd,
  anchorReadout,
  EDGE_STRIP_HEIGHT,
  edgeEffortBlock,
  edgeEffortSeries,
  edgeStripDrawn,
  edgeStripColumns,
  edgeText,
  edgeVisits,
  dirArrow,
  floorIndex,
  kfmt,
  lastSwingAnchor,
  makeShifter,
  minuteFlow,
  paneSlotMargins,
  rollingDelta,
  sessionOpenAnchor,
} from '../flowV3'
import { EdgeEffortPrimitive, ORANGE_RGB } from './flowV3Primitives'

const BG_COLOR = '#1f2937'
const PANE_SEPARATOR_PX = 1 // CR-DI: the strip's 64 px include the 1-px pane separator, so pane 0 keeps its height
const MIN_PANEL_HEIGHT = 140
const BUY_RGBA = (alpha) => `rgba(96,165,250,${alpha})`
const SELL_RGBA = (alpha) => `rgba(229,231,235,${alpha})`
const SLATE_RGBA = (alpha) => `rgba(148,163,184,${alpha})`
const MAGENTA_RGBA = (alpha) => `rgba(232,121,249,${alpha})`
const ORANGE_RGBA = (alpha) => `rgba(${ORANGE_RGB},${alpha})`
const SOURCE_LABEL = { databento: 'Databento', ironbeam: 'Ironbeam' }

const chipStyle = (active) => ({
  height: '24px',
  borderRadius: '8px',
  border: `1px solid ${active ? 'rgba(96,165,250,0.6)' : 'rgba(148, 163, 184, 0.22)'}`,
  background: active ? 'rgba(96,165,250,0.18)' : 'rgba(15, 23, 42, 0.96)',
  color: '#cbd5e1',
  cursor: 'pointer',
  fontSize: '12px',
  padding: '0 8px',
  whiteSpace: 'nowrap',
})

const iconStyle = {
  width: '24px', height: '24px', borderRadius: '8px', border: '1px solid rgba(148, 163, 184, 0.22)',
  background: 'rgba(15, 23, 42, 0.96)', color: '#cbd5e1', cursor: 'pointer', display: 'inline-flex',
  alignItems: 'center', justifyContent: 'center', fontSize: '13px', lineHeight: 1, padding: 0, flex: '0 0 auto',
}

const sfmt = (v) => `${v >= 0 ? '+' : '-'}${kfmt(Math.abs(v))}`
const shareFmt = (v) => (v == null ? '–' : `${v >= 0 ? '+' : ''}${v.toFixed(2)}`)

function Modal({ title, onClose, children, width = 520 }) {
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') {
        e.stopPropagation()
        onClose()
      }
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [onClose])
  return (
    <div onClick={onClose} style={{ position: 'fixed', inset: 0, background: 'rgba(2, 6, 23, 0.58)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1200, padding: '24px' }}>
      <div role="dialog" aria-label={title} onClick={(e) => e.stopPropagation()}
        style={{ width: `min(${width}px, calc(100vw - 32px))`, maxHeight: 'calc(100vh - 48px)', overflowY: 'auto', borderRadius: '16px', border: '1px solid #1f2937', background: '#0b1220', boxShadow: '0 20px 60px rgba(0, 0, 0, 0.42)', padding: '18px', color: '#e2e8f0', fontSize: '13px', lineHeight: 1.5 }}>
        <div style={{ fontSize: '16px', fontWeight: 800, marginBottom: '12px' }}>{title}</div>
        {children}
      </div>
    </div>
  )
}

function GuideBody() {
  const h = { fontWeight: 700, marginTop: '12px', color: '#f1f5f9' }
  const tag = <span style={{ color: '#94a3b8' }}> Visual aid — untested.</span>
  const sw = (bg) => ({ display: 'inline-block', width: 10, height: 10, background: bg, marginRight: 6, verticalAlign: 'middle' })
  return (
    <div>
      <div style={h}>This pane</div>
      <div><b>Anchored CVD</b> (line, top): buying minus selling added up from the anchor bar, starting at 0. Set the anchor with 📍 and a click on a price bar, or with "session open" / "last swing" (the end of the last confirmed swing, ⚙ swing size of Flow v2). Esc or "clear" removes it; without an anchor the line is the session CVD. The readout shows delta, price change, minutes and contracts per point since the anchor.{tag}</div>
      <div><span style={sw(SLATE_RGBA(0.55))} /><b>Volume per minute</b> (middle bars, slate grey): all contracts traded in each 1-minute bar (the flow's volume, every print, including the rare unknown-side ones), from the same data as the delta bars. Hover a minute for its volume, delta and delta share (delta ÷ volume).{tag}</div>
      <div><span style={sw(BUY_RGBA(0.7))} /><span style={sw(SELL_RGBA(0.6))} /><b>Rolling net delta</b> (⚙, off by default): buying minus selling over the last N minutes (5 / 10 / 15). When on it draws in the middle slot, volume moves to the slot below it and the pane grows by one slot.{tag}</div>
      <div><b>Delta per bar</b> (bottom bars): the Flow v2 histogram, so this pane stands alone.{tag}</div>
      <div style={h}>On the price chart</div>
      <div><b>Live effort</b> (with Effort on): the leg in progress — from the end of the last confirmed swing to the running extreme since — gets a dimmed label at that extreme, e.g. "↑ +6.5 pts so far · 1.2× vs last ↑": contracts per point vs the previous leg the same way, as the confirmed labels. Shown once the leg has moved at least M pts (⚙ "Live effort min move", default 4) and a previous leg the same way exists. It moves with the extreme on each refresh and becomes the normal label once the swing is confirmed; in a review replay it uses bars up to the replay time only. Hover it for the numbers. The label and M are display aids, not tested thresholds.{tag}</div>
      <div><span style={sw(ORANGE_RGBA(0.9))} /><span style={sw(MAGENTA_RGBA(0.9))} /><b>Absorption levels</b> (orange and magenta lines): a price where aggressive selling ("sellers absorbed", a floor, orange) or buying ("buyers absorbed", a ceiling, magenta) hit the Absorbed rule below. The line starts at the minute the volume built up, appears once the hold window has passed, and ends when price trades through it (else it runs to now). Thicker = more contracts absorbed; it fades from full to 25 % over 60 minutes. At most 8 are drawn, the largest after fading. The label says side, contracts and minutes held; hover for start, opposite-side contracts and where it broke. Exact data only.{tag}</div>
      <div><span style={{ ...sw('transparent'), borderTop: `2px dashed ${MAGENTA_RGBA(0.8)}`, height: 0 }} /><b>Edge readout</b> (header, with an anchor; dashed magenta lines): the box starts at the anchor minute's close. A close up to 2 pts beyond an edge is a test and the edge stays; a close more than 2 pts beyond moves the edge or breaks the box (below). For each edge: aggressive contracts traded within 1 pt of it since the box started (buyers at the top, sellers at the bottom; exact data only) and minutes since price first came within 1 pt of it.{tag}
        <div style={{ marginTop: '4px' }}><b>Restart on a break</b> (CR-DI): while an edge is younger than N minutes (⚙, default 15) a close more than 2 pts beyond it moves the edge out ("moved hh:mm") and the box grows. Once the edge has held N minutes, such a close is a break: the box freezes (drawn faded, ending at the break minute; hover it for its edges, contracts on both sides, minutes held and "broke hh:mm ↑/↓"), and a new box starts at the break bar from that bar's close. The header ("Box from hh:mm ↑") and the Edge effort strip use the current box. The last 4 frozen boxes are drawn. Clearing or moving the anchor removes all boxes. N is a display default, not a tested threshold.</div>
      </div>
      <div><span style={sw(BUY_RGBA(0.85))} /><span style={sw(SELL_RGBA(0.38))} /><span style={sw(SELL_RGBA(0.72))} /><span style={sw(BUY_RGBA(0.38))} /><b>Edge effort</b> (header chip, needs Edges and an anchor; a strip under the delta bars): per minute since the anchor, the same aggressive contracts the edge readout counts, within 1 pt of each edge, both sides. Top edge, drawn up: buyers (blue) with sellers beside them (thinner, lighter grey). Bottom edge, drawn down: sellers (grey) with buyers beside them (thinner, lighter blue). Each minute is counted against the box it belongs to (the drawn boxes: the last 4 frozen and the current one). Over the current box, the blue bars at the top and the grey bars at the bottom add up to the header's numbers; when an edge of the current box moves, its minutes are re-counted against the new edge, as the header is. A visit is a run of minutes with bars at that edge, allowing gaps of up to 2 minutes, and never crosses a break; its label says main side / other side (top buyers / sellers, bottom sellers / buyers). Hover a minute for both sides, the net and the visit's running totals. Exact data only. Visit sizes are effort, not a tested signal; Ironbeam live data understates volume.{tag}
        <div style={{ marginTop: '4px', color: '#cbd5e1' }}><i>How to read, untested</i> (bottom edge; mirror it for the top):
          <ul style={{ margin: '2px 0 0 18px', padding: 0 }}>
            <li>heavy aggressive selling at the low, CVD falls, price holds = passive defense (resting bids absorb);</li>
            <li>heavy aggressive selling at the low, CVD flat, price holds = active defense (buyers lift offers as hard as sellers hit bids);</li>
            <li>heavy selling, CVD falls, price falls = sellers winning;</li>
            <li>balanced flow, CVD flat, price falls = bids pulled / thin book.</li>
          </ul>
        </div>
      </div>
      <div><span style={sw(SELL_RGBA(0.6))} /><span style={sw(BUY_RGBA(0.6))} /><b>Heatmap (cells)</b> (⚙, behind the candles): for each 1-minute bar and 0.25 pt (0.5 pt when zoomed out), aggressive selling (market sells hitting the bid, grey, left half) and aggressive buying (market buys lifting the offer, blue, right half). Darker = a bigger cell for that session. Either layer can be switched off in ⚙.{tag}</div>
      <div><b>Profile</b> (right edge): the same volume summed per price from the anchor (or the session open) to now, selling grey and buying blue.{tag}</div>
      <div><b>Absorbed (marks)</b> (⚙, slate triangles): ▲ "sellers absorbed" — aggressive selling at one price at least K× the session's typical cell (median so far), and nothing traded more than 1 tick below it for that minute and the next M minutes. ▼ "buyers absorbed" is the mirror. The slate line runs until price trades through. Hover for the numbers. K and M (default 3 and 3) are display defaults, not tested thresholds.{tag}</div>
      <div style={h}>Data</div>
      <div><b>exact: trades</b> / <b>exact: es_flow_price_1m</b> — each trade at its own price and side. Ironbeam keeps raw trades for about 2 days, so this is usually only the live and previous session.</div>
      <div><b>approximate: 1-s data</b> — older sessions: each second's selling is placed at that second's low and its buying at its high. About 61 % of volume trades in seconds wider than 1 tick, so cells are off by a tick or two and selling leans toward lows, buying toward highs. <b>Absorption levels, Absorbed marks, edge contracts and the Edge effort strip are not drawn on approximate data</b>, because that lean manufactures them.</div>
      <div>Sessions through 2026-06-05 use Databento for RTH; later sessions use Ironbeam, which misses about 6 % of volume and understates buy − sell by about 30 %.</div>
    </div>
  )
}

function Settings({ value, layers, onApply, onClose }) {
  const [draft, setDraft] = useState({ ...value, heat: Boolean(layers?.heat), marks: Boolean(layers?.marks) })
  const input = { width: '100%', background: '#020617', color: '#e2e8f0', border: '1px solid #334155', borderRadius: '10px', padding: '8px 10px', fontSize: '14px' }
  const apply = (e) => {
    e.preventDefault()
    const n = (k, lo, hi, d) => {
      const v = Number(draft[k])
      return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d
    }
    onApply({ heat: Boolean(draft.heat), marks: Boolean(draft.marks) }, {
      ...value,
      rollingMin: ROLLING_CHOICES.includes(Number(draft.rollingMin)) ? Number(draft.rollingMin) : DEFAULT_V3_SETTINGS.rollingMin,
      k: n('k', 0.5, 50, DEFAULT_V3_SETTINGS.k),
      m: Math.round(n('m', 1, 30, DEFAULT_V3_SETTINGS.m)),
      sellLayer: Boolean(draft.sellLayer),
      buyLayer: Boolean(draft.buyLayer),
      boxHoldMin: Math.round(n('boxHoldMin', BOX_HOLD_MIN_RANGE[0], BOX_HOLD_MIN_RANGE[1], DEFAULT_V3_SETTINGS.boxHoldMin)),
      rollingOn: Boolean(draft.rollingOn),
      liveMinMove: n('liveMinMove', LIVE_MIN_MOVE_RANGE[0], LIVE_MIN_MOVE_RANGE[1], DEFAULT_V3_SETTINGS.liveMinMove),
    })
  }
  const label = (text) => <div style={{ fontSize: '12px', color: '#cbd5e1', marginBottom: '4px' }}>{text}</div>
  return (
    <Modal title="Flow v3 settings" onClose={onClose} width={380}>
      <form onSubmit={apply}>
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
          <input type="checkbox" aria-label="Rolling net delta" data-testid="flow-v3-rolling-toggle" checked={Boolean(draft.rollingOn)} onChange={(e) => setDraft((d) => ({ ...d, rollingOn: e.target.checked }))} />
          Rolling net delta ({draft.rollingMin} min) — middle bars; volume moves one slot down
        </label>
        <label style={{ display: 'block', marginBottom: '10px' }}>
          {label('Rolling net delta window (minutes)')}
          <select aria-label="Rolling net delta window" value={draft.rollingMin} style={input}
            onChange={(e) => setDraft((d) => ({ ...d, rollingMin: Number(e.target.value) }))}>
            {ROLLING_CHOICES.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label style={{ display: 'block', marginBottom: '10px' }}>
          {label('Absorbed: K × the session median cell')}
          <input type="number" step={0.5} min={0.5} aria-label="Absorbed K" value={draft.k} style={input} onChange={(e) => setDraft((d) => ({ ...d, k: e.target.value }))} />
        </label>
        <label style={{ display: 'block', marginBottom: '10px' }}>
          {label('Absorbed: M minutes held')}
          <input type="number" step={1} min={1} aria-label="Absorbed M" value={draft.m} style={input} onChange={(e) => setDraft((d) => ({ ...d, m: e.target.value }))} />
          <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>Display defaults (3 and 3), not tested thresholds.</div>
        </label>
        <label style={{ display: 'block', marginBottom: '10px' }}>
          {label('Box edge must hold N min before a break freezes it')}
          <input type="number" step={1} min={BOX_HOLD_MIN_RANGE[0]} max={BOX_HOLD_MIN_RANGE[1]} aria-label="Box hold minutes" data-testid="flow-v3-box-hold"
            value={draft.boxHoldMin ?? DEFAULT_V3_SETTINGS.boxHoldMin} style={input} onChange={(e) => setDraft((d) => ({ ...d, boxHoldMin: e.target.value }))} />
          <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>Edges box, 5–60 (default 15). A display default, not a tested threshold: ranges are measured against the prior move, not a fixed time (A2); this only decides when the drawn box restarts.</div>
        </label>
        <label style={{ display: 'block', marginBottom: '10px' }}>
          {label('Live effort min move (pts)')}
          <input type="number" step={0.5} min={LIVE_MIN_MOVE_RANGE[0]} max={LIVE_MIN_MOVE_RANGE[1]} aria-label="Live effort min move" data-testid="flow-v3-live-min-move"
            value={draft.liveMinMove ?? DEFAULT_V3_SETTINGS.liveMinMove} style={input} onChange={(e) => setDraft((d) => ({ ...d, liveMinMove: e.target.value }))} />
          <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>With Effort on, the leg in progress is labelled once it has moved this far (2–12, default 4). A display default, not a tested threshold.</div>
        </label>
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
          <input type="checkbox" aria-label="Heatmap (cells)" data-testid="flow-v3-heatmap-toggle" checked={Boolean(draft.heat)} onChange={(e) => setDraft((d) => ({ ...d, heat: e.target.checked }))} />
          Heatmap (cells) — behind the candles
        </label>
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
          <input type="checkbox" aria-label="Absorbed (marks)" data-testid="flow-v3-absorbed-toggle" checked={Boolean(draft.marks)} onChange={(e) => setDraft((d) => ({ ...d, marks: e.target.checked }))} />
          Absorbed (marks) — slate triangles
        </label>
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
          <input type="checkbox" aria-label="Sell heat layer" checked={Boolean(draft.sellLayer)} onChange={(e) => setDraft((d) => ({ ...d, sellLayer: e.target.checked }))} />
          Heatmap: sell layer (grey)
        </label>
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
          <input type="checkbox" aria-label="Buy heat layer" checked={Boolean(draft.buyLayer)} onChange={(e) => setDraft((d) => ({ ...d, buyLayer: e.target.checked }))} />
          Heatmap: buy layer (blue)
        </label>
        <div style={{ display: 'flex', gap: '8px', justifyContent: 'space-between', marginTop: '14px' }}>
          <button type="button" style={chipStyle(false)} onClick={() => setDraft({ ...DEFAULT_V3_SETTINGS, heat: false, marks: false })}>Defaults</button>
          <div style={{ display: 'flex', gap: '6px' }}>
            <button type="button" style={chipStyle(false)} onClick={onClose}>Cancel</button>
            <button type="submit" style={{ ...chipStyle(true), fontWeight: 700 }}>Apply</button>
          </div>
        </div>
      </form>
    </Modal>
  )
}

export default function AggressorFlowV3Panel({
  candles,
  payloads,
  sessionPayload = null, // the selected session's v2 payload: its segments and legs give the anchors
  cellsPayload = null,
  cellsLoading = false,
  cellsError = '',
  visibleLogicalRange,
  linkedCrosshair,
  height = 220,
  rollingExtra = 0, // CR-DM: px App adds to the pane while rolling net delta is on (flowV3.rollingExtraHeight)
  loading = false,
  error = '',
  flowMode = 'v3',
  onFlowModeChange,
  settings = DEFAULT_V3_SETTINGS,
  onSettingsChange,
  layers = { heat: false, profile: false, marks: false, levels: false, edges: false },
  onLayersChange,
  edges = null, // CR-DF: flowV3.edgeReadout (App), null without an anchor or with the readout off
  anchor = null,
  onAnchorChange,
  anchorArmed = false,
  onAnchorArmedChange,
  effortOn = false, // CR-DG item 4: the Flow v2 Effort switch (same state), labels on the price chart
  onEffortChange,
  edgeEffortOn = false, // CR-DI: the Edge effort strip (stored per viewer); draws only with Edges on, an anchor and exact data
  onEdgeEffortChange,
}) {
  const stageRef = useRef(null)
  const hostRef = useRef(null)
  const chartRef = useRef(null)
  const seriesRef = useRef({})
  const lastAppliedLogicalRef = useRef(null)
  const candleTimesRef = useRef([])
  const linkedCrosshairRef = useRef(null)
  const anchorRef = useRef(null)
  const refreshDecorationsRef = useRef(() => {})
  const slotsRef = useRef({ rollingOn: false, extra: 0 })

  const [sessionBands, setSessionBands] = useState([])
  const [linkedCrosshairX, setLinkedCrosshairX] = useState(null)
  const [anchorX, setAnchorX] = useState(null)
  const [guideOpen, setGuideOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [hover, setHover] = useState(null)

  const shift = useMemo(() => makeShifter((t) => utcEpochShowingZoneTime(t, 'America/Los_Angeles')), [])
  const candleTimes = useMemo(() => normalizeShiftedCandleTimes(candles), [candles])
  const allBars = useMemo(() => (payloads || []).flatMap((p) => (Array.isArray(p?.bars) ? p.bars : [])), [payloads])
  const aligned = useMemo(() => alignV2ToCandles(allBars, candles), [allBars, candles])
  const closes = useMemo(() => {
    const m = new Map()
    for (const c of Array.isArray(candles) ? candles : []) m.set(shift(c.time), Number(c.close))
    return aligned.map((r) => m.get(r.time))
  }, [candles, aligned, shift])
  const cvd = useMemo(() => anchoredCvd(aligned, anchor), [aligned, anchor])
  const rollingOn = Boolean(settings.rollingOn)
  const rolling = useMemo(() => (rollingOn ? rollingDelta(aligned, settings.rollingMin) : null), [aligned, rollingOn, settings.rollingMin])
  const readout = useMemo(() => anchorReadout(aligned, closes, anchor), [aligned, closes, anchor])
  // An anchor in ETH sums Ironbeam ETH delta with the RTH segment's delta; say so when the sources differ.
  const mixedNote = useMemo(() => {
    if (!readout || readout.fromSeg === readout.toSeg) return ''
    const src = Object.fromEntries((sessionPayload?.segments || []).map((g) => [g.name, g.source]))
    const a = src[readout.fromSeg]
    const b = src[readout.toSeg]
    return a && b && a !== b ? ` · mixed sources (${SOURCE_LABEL[a] || a} + ${SOURCE_LABEL[b] || b})` : ''
  }, [readout, sessionPayload])
  const sources = useMemo(() => {
    const s = new Set()
    for (const p of payloads || []) for (const seg of p?.segments || []) if (seg.name === 'rth') s.add(seg.source)
    return [...s]
  }, [payloads])
  const cellBadges = useMemo(() => {
    const out = []
    for (const seg of cellsPayload?.segments || []) {
      const label = `${seg.name.toUpperCase()} ${seg.badge}`
      if (!out.some((b) => b.label === label)) out.push({ label, exact: seg.exact, source: seg.source })
    }
    return out
  }, [cellsPayload])
  const marksNote = useMemo(() => {
    const rth = (cellsPayload?.segments || []).find((s) => s.name === 'rth')
    return rth && !Array.isArray(rth.marks) ? rth.marks_note : null
  }, [cellsPayload])
  const edgeLine = useMemo(() => (edges ? edgeText(edges, (t) => formatShiftedHHMM(shift(t))) : ''), [edges, shift])
  const notes = useMemo(() => [...new Set((payloads || []).flatMap((p) => p?.notes || []))], [payloads])
  const hasValues = aligned.some((r) => Number.isFinite(r.delta))
  const edgeEffortWhy = edgeEffortBlock(layers.edges, anchor, edges)
  const stripOn = edgeStripDrawn(edgeEffortOn, layers.edges, anchor, edges)
  // Amendment 9: the strip draws the drawn boxes (last 4 frozen + current), each minute against its own box.
  const series = useMemo(() => (stripOn ? edgeEffortSeries(edges) : null), [stripOn, edges])
  const strip = useMemo(() => (series ? edgeStripColumns(series, candleTimes, shift) : null), [series, candleTimes, shift])
  // Visit labels (main first) placed on candle indexes; the primitive drops the smaller of two colliding labels.
  const visits = useMemo(() => {
    if (!strip) return null
    const v = edgeVisits(series)
    const place = (list) => list.map((x) => ({
      ...x, i0: floorIndex(candleTimes, shift(x.start)), i1: floorIndex(candleTimes, shift(x.end)),
      text: `${kfmt(x.main)} / ${kfmt(x.opp)}`,
    })).filter((x) => x.i0 >= 0)
    return { ...v, labels: { top: place(v.top), bottom: place(v.bottom) } }
  }, [strip, series, candleTimes, shift])
  const anyLayer = layers.heat || layers.profile || layers.marks || layers.levels || layers.edges

  useEffect(() => { candleTimesRef.current = candleTimes }, [candleTimes])
  useEffect(() => { linkedCrosshairRef.current = linkedCrosshair || null }, [linkedCrosshair])
  useEffect(() => {
    anchorRef.current = anchor
    requestAnimationFrame(() => refreshDecorationsRef.current())
  }, [anchor])

  // DC5: Esc disarms a pending pick, else clears the anchor.
  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== 'Escape' || e.defaultPrevented) return
      const tag = (e.target?.tagName || '').toLowerCase()
      if (tag === 'input' || tag === 'select' || tag === 'textarea') return
      if (anchorArmed) onAnchorArmedChange?.(false)
      else if (anchor != null) onAnchorChange?.(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [anchor, anchorArmed, onAnchorChange, onAnchorArmedChange])

  useEffect(() => {
    if (!hostRef.current || !stageRef.current) return undefined
    const stage = stageRef.current
    const chart = createChart(hostRef.current, {
      width: stage.clientWidth || 900,
      height: Math.max(stage.clientHeight || MIN_PANEL_HEIGHT, MIN_PANEL_HEIGHT),
      layout: { background: { type: ColorType.Solid, color: BG_COLOR }, textColor: '#cbd5e1', attributionLogo: false },
      localization: { locale: 'en-US', priceFormatter: (v) => kfmt(Number(v)) },
      grid: { vertLines: { color: 'rgba(148, 163, 184, 0.08)' }, horzLines: { color: 'rgba(148, 163, 184, 0.08)' } },
      rightPriceScale: { visible: true, borderColor: 'rgba(148, 163, 184, 0.18)', scaleMargins: { top: 0.42, bottom: 0.30 } },
      timeScale: { borderColor: 'rgba(148, 163, 184, 0.18)', timeVisible: true, secondsVisible: false, tickMarkFormatter: (t) => formatShiftedHHMM(t) },
      crosshair: { mode: CrosshairMode.Normal, vertLine: { visible: false, labelVisible: false }, horzLine: { visible: false, labelVisible: false } },
      handleScroll: { mouseWheel: false, pressedMouseMove: false, horzTouchDrag: false, vertTouchDrag: false },
      handleScale: { axisPressedMouseMove: false, mouseWheel: false, pinch: false },
    })
    const common = { priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false }
    const delta = chart.addSeries(HistogramSeries, { ...common, priceScaleId: 'delta', base: 0 })
    chart.priceScale('delta').applyOptions({ visible: false, scaleMargins: { top: 0.74, bottom: 0.02 } })
    // The middle slot carries the visible right axis (the pane keeps a right scale like Flow v2, so its plot width and
    // bar positions follow the price chart's logical range the same way): volume per minute, or rolling net delta when
    // it is on (CR-DM; volume then sits on its own hidden scale in the slot below). applySlots moves them.
    const rollingS = chart.addSeries(HistogramSeries, { ...common, priceScaleId: 'rolling', base: 0 })
    const volS = chart.addSeries(HistogramSeries, { ...common, priceScaleId: 'right', base: 0, color: SLATE_RGBA(0.55) })
    const cvdS = chart.addSeries(BaselineSeries, {
      ...common,
      priceScaleId: 'cvd',
      baseValue: { type: 'price', price: 0 },
      topLineColor: BUY_RGBA(0.95),
      bottomLineColor: SELL_RGBA(0.95),
      topFillColor1: 'rgba(0,0,0,0)', topFillColor2: 'rgba(0,0,0,0)',
      bottomFillColor1: 'rgba(0,0,0,0)', bottomFillColor2: 'rgba(0,0,0,0)',
      lineWidth: 2,
    })
    chart.priceScale('cvd').applyOptions({ visible: false, scaleMargins: { top: 0.04, bottom: 0.62 } })
    chartRef.current = chart
    seriesRef.current = { delta, rolling: rollingS, cvd: cvdS, vol: volS }

    const refreshDecorations = () => {
      const w = stageRef.current?.clientWidth || 0
      setSessionBands(computeSessionBands(chart, candleTimesRef.current, w))
      const x = coordinateForLinkedCrosshair(chart, linkedCrosshairRef.current)
      setLinkedCrosshairX(!Number.isFinite(x) || x < 0 || x > w ? null : x)
      const a = anchorRef.current
      const ax = a == null ? null : chart.timeScale().timeToCoordinate(a)
      setAnchorX(ax == null || !Number.isFinite(ax) || ax < 0 || ax > w ? null : ax)
    }
    refreshDecorationsRef.current = refreshDecorations
    const handleResize = () => {
      if (!stageRef.current) return
      const w = stageRef.current.clientWidth
      const h = stageRef.current.clientHeight
      if (w > 0 && h > 0) chart.applyOptions({ width: w, height: Math.max(h, MIN_PANEL_HEIGHT) })
      if (seriesRef.current.strip) chart.panes()[1]?.setHeight(EDGE_STRIP_HEIGHT - PANE_SEPARATOR_PX)
      applySlots()
      requestAnimationFrame(refreshDecorations)
    }
    const onRange = () => requestAnimationFrame(refreshDecorations)
    const onMove = (param) => {
      const t = param?.time
      setHover(t == null ? null : { time: t, x: param.point?.x ?? 0, pane: param.paneIndex ?? 0 })
    }
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(handleResize) : null
    if (ro) ro.observe(stage)
    chart.timeScale().subscribeVisibleLogicalRangeChange(onRange)
    chart.subscribeCrosshairMove(onMove)
    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(onRange)
      chart.unsubscribeCrosshairMove(onMove)
      if (ro) ro.disconnect()
      chart.remove()
      chartRef.current = null
      seriesRef.current = {}
      lastAppliedLogicalRef.current = null
      refreshDecorationsRef.current = () => {}
    }
  }, [])

  // CR-DM item 2: slot layout of pane 0 (paneSlotMargins); rolling off = the CR-DC margins with volume in the middle.
  const applySlots = () => {
    const chart = chartRef.current
    const s = seriesRef.current
    if (!chart || !s.vol) return
    const { rollingOn: on, extra } = slotsRef.current
    // pane 0 from the height just applied (panes()[0].getHeight() lags until the chart's next layout pass)
    const h0 = chart.options().height - chart.timeScale().height() - (s.strip ? EDGE_STRIP_HEIGHT : 0)
    const m = paneSlotMargins(h0, on ? extra : 0)
    if ((s.rolling.options().priceScaleId === 'right') !== on) {
      s.vol.applyOptions({ priceScaleId: on ? 'vol' : 'right' })
      s.rolling.applyOptions({ priceScaleId: on ? 'right' : 'rolling' })
    }
    chart.priceScale('cvd').applyOptions({ scaleMargins: m.cvd })
    chart.priceScale('right').applyOptions({ scaleMargins: m.mid })
    chart.priceScale('delta').applyOptions({ scaleMargins: m.delta })
    if (on) chart.priceScale('vol').applyOptions({ visible: false, scaleMargins: m.low || m.mid })
    else chart.priceScale('rolling').applyOptions({ visible: false })
  }
  useEffect(() => {
    slotsRef.current = { rollingOn, extra: rollingExtra }
    fitToStage()
  }, [rollingOn, rollingExtra]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const s = seriesRef.current
    if (!s.delta) return
    const ws = (r) => ({ time: r.time })
    s.delta.setData(aligned.map((r) => (Number.isFinite(r.delta)
      ? { time: r.time, value: r.delta, color: r.delta >= 0 ? BUY_RGBA(0.75) : SELL_RGBA(0.6) }
      : ws(r))))
    s.rolling.setData(rolling ? aligned.map((r, i) => (rolling[i] != null
      ? { time: r.time, value: rolling[i], color: rolling[i] >= 0 ? BUY_RGBA(0.45) : SELL_RGBA(0.38) }
      : ws(r))) : [])
    s.vol.setData(aligned.map((r) => (Number.isFinite(r.vol) ? { time: r.time, value: r.vol } : ws(r))))
    s.cvd.setData(aligned.map((r, i) => (cvd[i] != null ? { time: r.time, value: cvd[i] } : ws(r))))
    const shared = normalizeLogicalRange(visibleLogicalRange)
    if (chartRef.current && candleTimes.length >= 2 && !shared) {
      chartRef.current.timeScale().fitContent()
      lastAppliedLogicalRef.current = normalizeLogicalRange(chartRef.current.timeScale().getVisibleLogicalRange?.())
    } else if (chartRef.current && shared) {
      try {
        chartRef.current.timeScale().setVisibleLogicalRange(shared)
        lastAppliedLogicalRef.current = shared
      } catch (err) {
        // ignore sync errors
      }
    }
    requestAnimationFrame(() => refreshDecorationsRef.current())
  }, [aligned, rolling, cvd, candleTimes, visibleLogicalRange])

  // CR-DI amendment 2: the pane (App) grows by EDGE_STRIP_HEIGHT with the strip, so size the chart to the stage and
  // the strip pane to its height now and on the next frame, before a ResizeObserver pass could split the new height
  // proportionally (which would shrink CVD / rolling / delta).
  const fitToStage = () => {
    const fit = () => {
      const chart = chartRef.current
      const st = stageRef.current
      if (!chart || !st) return
      const h = Math.max(st.clientHeight || MIN_PANEL_HEIGHT, MIN_PANEL_HEIGHT)
      if (chart.options().height !== h) chart.applyOptions({ height: h })
      if (seriesRef.current.strip) chart.panes()[1]?.setHeight(EDGE_STRIP_HEIGHT - PANE_SEPARATOR_PX)
      applySlots()
    }
    fit()
    requestAnimationFrame(fit)
  }

  // CR-DI: the strip's pane exists only while the strip is drawn, so with it off the chart is origin/main's.
  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return
    const s = seriesRef.current
    if (strip && !s.strip) {
      const carrier = chart.addSeries(LineSeries, {
        priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false, color: 'rgba(0,0,0,0)',
        lineWidth: 1, priceScaleId: 'edgeEffort',
      }, 1)
      chart.priceScale('edgeEffort', 1).applyOptions({ visible: false })
      chart.applyOptions({ layout: { panes: { separatorColor: 'rgba(148, 163, 184, 0.18)', separatorHoverColor: 'rgba(148, 163, 184, 0.18)', enableResize: false } } })
      const prim = new EdgeEffortPrimitive()
      carrier.attachPrimitive(prim)
      seriesRef.current = { ...s, strip: carrier, stripPrim: prim }
      fitToStage()
    } else if (!strip && s.strip) {
      chart.removeSeries(s.strip)
      if (chart.panes().length > 1) chart.removePane(1)
      const { strip: _c, stripPrim: _p, ...rest } = s
      seriesRef.current = rest
      fitToStage()
    }
    const cur = seriesRef.current
    if (strip && cur.strip) {
      cur.strip.setData(aligned.map((r) => ({ time: r.time })))
      cur.stripPrim.setData({ times: candleTimes, cols: strip.cols, max: strip.max, labels: visits?.labels })
    }
    requestAnimationFrame(() => refreshDecorationsRef.current())
  }, [strip, visits, aligned, candleTimes])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return
    const next = normalizeLogicalRange(visibleLogicalRange)
    const current = normalizeLogicalRange(chart.timeScale().getVisibleLogicalRange?.())
    if (!next || rangesClose(current, next) || rangesClose(lastAppliedLogicalRef.current, next)) {
      requestAnimationFrame(() => refreshDecorationsRef.current())
      return
    }
    requestAnimationFrame(() => {
      try {
        chart.timeScale().setVisibleLogicalRange(next)
        lastAppliedLogicalRef.current = next
      } catch (err) {
        // ignore sync errors
      }
      refreshDecorationsRef.current()
    })
  }, [visibleLogicalRange])

  useEffect(() => {
    requestAnimationFrame(() => refreshDecorationsRef.current())
  }, [linkedCrosshair, candleTimes])

  const hoverIdx = hover && hover.pane !== 1 ? aligned.findIndex((r) => r.time === hover.time) : -1
  // CR-DI strip hover: the candle's four sums, and the visit (running so far / total) of its last minute at each edge.
  const stripHover = useMemo(() => {
    if (!hover || hover.pane !== 1 || !strip || !visits) return null
    const idx = floorIndex(candleTimes, hover.time)
    if (idx < 0 || candleTimes[idx] !== hover.time) return null
    const c = strip.cols.get(idx) || [0, 0, 0, 0]
    let last = -1
    for (let i = 0; i < strip.idxOf.length; i += 1) if (strip.idxOf[i] === idx) last = i
    const visit = (e) => {
      if (last < 0) return null
      const k = visits.of[e][last]
      return k < 0 ? null : { ...visits[e][k], run: visits.run[e][last] }
    }
    const box = series.boxes.find((b) => shift(b.from) <= hover.time && hover.time <= shift(b.breakAt ?? b.to)) || null
    return { time: hover.time, x: hover.x, c, top: visit('top'), bottom: visit('bottom'), box }
  }, [hover, strip, series, visits, candleTimes, shift])
  const hoverRow = hoverIdx >= 0 ? aligned[hoverIdx] : null
  const setAnchorTo = (t) => {
    onAnchorArmedChange?.(false)
    if (t != null) onAnchorChange?.(t)
  }
  const swingAnchor = useMemo(() => lastSwingAnchor(sessionPayload?.legs, candleTimes, shift), [sessionPayload, candleTimes, shift])
  const openAnchor = useMemo(() => sessionOpenAnchor(sessionPayload?.segments, candleTimes, shift), [sessionPayload, candleTimes, shift])

  return (
    <div className="flow-shell" style={{ height: `${height}px`, position: 'relative', width: '100%' }} data-testid="flow-v3-panel">
      <div className="flow-panel-header" style={{ display: 'flex', alignItems: 'center', gap: '6px', justifyContent: 'flex-start', overflowX: 'auto' }}>
        <button type="button" style={iconStyle} title="Flow v3 settings" aria-label="Open Flow v3 settings" onClick={() => setSettingsOpen(true)}>⚙</button>
        <div className="flow-panel-title" style={{ whiteSpace: 'nowrap', flex: '0 0 auto' }}>Flow v3</div>
        {/* the header scrolls when narrow (review panel open): the mode toggle must never shrink away */}
        <span style={{ flex: '0 0 auto', display: 'inline-flex' }}><FlowModeToggle mode={flowMode} onChange={onFlowModeChange} /></span>
        {sources.map((s) => (
          <span key={s} data-testid="flow-v3-source" style={{ fontSize: '11px', padding: '2px 7px', borderRadius: '999px', border: '1px solid rgba(148,163,184,0.35)', color: s === 'databento' ? '#bfdbfe' : '#e5e7eb', whiteSpace: 'nowrap' }}>
            {SOURCE_LABEL[s] || s}
          </span>
        ))}
        <button type="button" style={chipStyle(layers.levels)} aria-pressed={Boolean(layers.levels)} data-testid="flow-v3-levels-toggle"
          onClick={() => onLayersChange?.((l) => ({ ...l, levels: !l.levels }))} title="Absorption levels (exact data only) — visual aid, untested">
          Absorption levels {layers.levels ? 'on' : 'off'}
        </button>
        <button type="button" style={chipStyle(layers.profile)} aria-pressed={layers.profile} data-testid="flow-v3-profile-toggle"
          onClick={() => onLayersChange?.((l) => ({ ...l, profile: !l.profile }))} title="Volume profile at the right edge, from the anchor (or session open) — visual aid, untested">
          Profile {layers.profile ? 'on' : 'off'}
        </button>
        <span style={{ width: 1, height: 16, background: 'rgba(148,163,184,0.25)', flex: '0 0 auto' }} />
        <button type="button" style={chipStyle(anchorArmed)} aria-pressed={anchorArmed} data-testid="flow-v3-anchor-pick"
          onClick={() => onAnchorArmedChange?.(!anchorArmed)} title="Click, then click a bar on the price chart to anchor CVD there (Esc cancels)">
          📍 {anchorArmed ? 'click a bar…' : 'set anchor'}
        </button>
        <button type="button" style={chipStyle(false)} data-testid="flow-v3-anchor-open"
          disabled={openAnchor == null} onClick={() => setAnchorTo(openAnchor)}>
          session open
        </button>
        <button type="button" style={chipStyle(false)} data-testid="flow-v3-anchor-swing" disabled={swingAnchor == null}
          onClick={() => setAnchorTo(swingAnchor)} title="The end of the last confirmed swing (Flow v2 swing size)">
          last swing
        </button>
        <button type="button" style={chipStyle(false)} data-testid="flow-v3-anchor-clear" disabled={anchor == null && !anchorArmed}
          onClick={() => { onAnchorArmedChange?.(false); onAnchorChange?.(null) }}>
          clear
        </button>
        <button type="button" style={chipStyle(layers.edges)} aria-pressed={Boolean(layers.edges)} data-testid="flow-v3-edges-toggle"
          onClick={() => onLayersChange?.((l) => ({ ...l, edges: !l.edges }))} title="Edge readout from the anchor: close box, contracts at each edge, held / broke — visual aid, untested">
          Edges {layers.edges ? 'on' : 'off'}
        </button>
        {layers.edges && (
          <span data-testid="flow-v3-edge-readout" style={{ fontSize: '11px', color: edges ? '#f5d0fe' : '#94a3b8', whiteSpace: 'nowrap', flex: '0 0 auto' }}>
            {anchor == null ? 'edges: set an anchor' : edges ? edgeLine : cellsLoading ? 'edges: loading…' : 'edges: no bars since the anchor'}
          </span>
        )}
        <button type="button" style={{ ...chipStyle(edgeEffortOn && !edgeEffortWhy), opacity: edgeEffortWhy ? 0.5 : 1, cursor: edgeEffortWhy ? 'default' : 'pointer' }}
          aria-pressed={Boolean(edgeEffortOn)} data-testid="flow-v3-edge-effort-toggle" disabled={Boolean(edgeEffortWhy)}
          onClick={() => onEdgeEffortChange?.(!edgeEffortOn)}
          title={edgeEffortWhy || 'Edge effort strip: aggressive contracts within 1 pt of each edge per minute, both sides — visual aid, untested'}>
          Edge effort {edgeEffortOn ? 'on' : 'off'}
        </button>
        {anyLayer && cellBadges.map((b) => (
          <span key={b.label} data-testid="flow-v3-cells-badge" title={b.exact ? 'each trade at its own price and side' : 'each second: sells at its low, buys at its high'}
            style={{ fontSize: '11px', padding: '2px 7px', borderRadius: '999px', border: `1px solid ${b.exact ? 'rgba(96,165,250,0.45)' : 'rgba(148,163,184,0.35)'}`, color: '#cbd5e1', whiteSpace: 'nowrap' }}>
            {b.label}
          </span>
        ))}
        {anyLayer && cellsLoading && <span style={{ fontSize: '11px', color: '#94a3b8', whiteSpace: 'nowrap' }}>loading cells…</span>}
        {anyLayer && cellsError && <span style={{ fontSize: '11px', color: '#cbd5e1', whiteSpace: 'nowrap' }} title={cellsError}>cells failed</span>}
        {(layers.marks || layers.levels) && marksNote && <span data-testid="flow-v3-marks-note" style={{ fontSize: '11px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{marksNote}</span>}
        <button type="button" style={chipStyle(effortOn)} aria-pressed={effortOn} data-testid="flow-v3-effort-toggle"
          onClick={() => onEffortChange?.(!effortOn)}
          title="Label each confirmed swing on the price chart: points · volume · contracts per point · × the previous swing the same way (the Flow v2 Effort switch)">
          Effort {effortOn ? 'on' : 'off'}
        </button>
        <button type="button" style={iconStyle} title="How to read Flow v3" aria-label="How to read Flow v3" onClick={() => setGuideOpen(true)}>?</button>
      </div>
      {guideOpen && <Modal title="How to read Flow v3" onClose={() => setGuideOpen(false)}><GuideBody /></Modal>}
      {settingsOpen && (
        <Settings value={settings} layers={layers} onClose={() => setSettingsOpen(false)}
          onApply={(lay, v) => { onLayersChange?.((l) => ({ ...l, ...lay })); onSettingsChange?.(v); setSettingsOpen(false) }} />
      )}
      <div ref={stageRef} className="flow-stage" style={{ width: '100%' }}>
        <div ref={hostRef} className="flow-host" style={{ width: '100%', height: '100%' }} />
        {sessionBands.map((band) => (
          <div key={band.key} className="session-band" style={{ left: `${band.left}px`, width: `${band.width}px` }} />
        ))}
        {Number.isFinite(anchorX) && (
          <div data-testid="flow-v3-anchor-line" style={{ position: 'absolute', top: 0, bottom: 0, left: `${Math.round(anchorX)}px`, width: 0, borderLeft: `1px dashed ${SLATE_RGBA(0.8)}`, pointerEvents: 'none', zIndex: 3 }} />
        )}
        {Number.isFinite(linkedCrosshairX) && (
          <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${Math.round(linkedCrosshairX)}px`, width: 0, borderLeft: '1px dashed rgba(191, 219, 254, 0.45)', pointerEvents: 'none', zIndex: 4 }} />
        )}
        <div data-testid="flow-v3-readout" style={{ position: 'absolute', top: 4, left: 8, zIndex: 4, pointerEvents: 'none', fontSize: '11px', color: '#cbd5e1', background: 'rgba(15,23,42,0.72)', borderRadius: 6, padding: '2px 6px' }}>
          {readout
            ? `since ${formatShiftedHHMM(readout.anchorTime)}: Δ ${sfmt(readout.delta)} · ${readout.dPrice == null ? '–' : `${readout.dPrice >= 0 ? '+' : ''}${readout.dPrice.toFixed(2)} pts`} · ${readout.minutes} min · ${readout.cpp == null ? '–' : kfmt(readout.cpp)}/pt`
              + mixedNote
            : `CVD from session open · ${rollingOn ? `rolling ${settings.rollingMin} min · ` : ''}volume per minute`}
        </div>
        {hoverRow && Number.isFinite(hoverRow.delta) && (
          <div data-testid="flow-v3-tooltip" style={{
            position: 'absolute', top: 24, left: Math.min(Math.max(0, hover.x + 12), (stageRef.current?.clientWidth || 600) - 260),
            maxWidth: 250, pointerEvents: 'none', zIndex: 5, background: 'rgba(2,6,23,0.92)', border: '1px solid #334155',
            borderRadius: 8, padding: '6px 8px', fontSize: '11px', color: '#e2e8f0', lineHeight: 1.4,
          }}>
            {(() => {
              const f = minuteFlow(hoverRow)
              return `${formatShiftedHHMM(hoverRow.time)} · vol ${f.vol == null ? '–' : kfmt(f.vol)} · Δ ${sfmt(hoverRow.delta)} · share ${shareFmt(f.share)}`
                + (rolling ? ` · ${settings.rollingMin}-min ${rolling[hoverIdx] == null ? '–' : sfmt(rolling[hoverIdx])}` : '')
                + ` · CVD ${cvd[hoverIdx] == null ? '–' : sfmt(cvd[hoverIdx])}`
            })()}
          </div>
        )}
        {stripHover && (
          <div data-testid="flow-v3-edge-effort-tooltip" style={{
            position: 'absolute', bottom: 70, left: Math.min(Math.max(0, stripHover.x + 12), (stageRef.current?.clientWidth || 600) - 330),
            maxWidth: 320, pointerEvents: 'none', zIndex: 5, background: 'rgba(2,6,23,0.92)', border: '1px solid #334155',
            borderRadius: 8, padding: '6px 8px', fontSize: '11px', color: '#e2e8f0', lineHeight: 1.4,
          }}>
            {(() => {
              const [tb, tsl, bs, bb] = stripHover.c
              const hm = (t) => formatShiftedHHMM(shift(t))
              const vis = (v, a, b) => (v ? ` · visit ${hm(v.start)}–${hm(v.end)}: so far ${kfmt(v.run[0])} ${a} / ${kfmt(v.run[1])} ${b} of ${kfmt(v.main)} / ${kfmt(v.opp)}` : '')
              return (
                <>
                  <div style={{ fontWeight: 700 }}>{formatShiftedHHMM(stripHover.time)} · edge effort{stripHover.box?.breakAt ? ` · box ${hm(stripHover.box.from)}–${hm(stripHover.box.breakAt)} (broke ${dirArrow(stripHover.box.dir)})` : ''}</div>
                  <div>Top {(stripHover.box || edges).top.price.toFixed(2)}: buyers {kfmt(tb)} / sellers {kfmt(tsl)} / net {sfmt(tb - tsl)}{vis(stripHover.top, 'buyers', 'sellers')}</div>
                  <div>Bottom {(stripHover.box || edges).bottom.price.toFixed(2)}: sellers {kfmt(bs)} / buyers {kfmt(bb)} / net {sfmt(bb - bs)}{vis(stripHover.bottom, 'sellers', 'buyers')}</div>
                </>
              )
            })()}
          </div>
        )}
        {(loading || error || candleTimes.length < 2 || !hasValues) && (
          <div className="flow-overlay">
            <div className="flow-overlay-title">{error ? 'Flow v3 could not load' : loading ? 'Loading Flow v3…' : 'No flow data returned'}</div>
            <div className="flow-overlay-text">{error || notes[0] || 'The loaded sessions did not return any aggressor flow data.'}</div>
          </div>
        )}
      </div>
    </div>
  )
}
