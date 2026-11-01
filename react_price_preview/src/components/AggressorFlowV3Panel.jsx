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
  alignV2ToCandles,
  DEFAULT_MARKER_SETTINGS,
  coordinateForLinkedCrosshair,
  computeSessionBands,
  formatShiftedHHMM,
  normalizeLogicalRange,
  normalizeShiftedCandleTimes,
  rangesClose,
  utcEpochShowingZoneTime,
} from '../flowShared'
import {
  BOX_RULES_RANGES,
  boxRules,
  BREAK_HOLD_MIN_RANGE,
  DEFAULT_V3_SETTINGS,
  LIVE_MIN_MOVE_RANGE,
  ROLLING_CHOICES,
  anchoredCvd,
  anchoredShare,
  clipShare,
  anchorReadout,
  EDGE_STRIP_HEIGHT,
  edgeEffortBlock,
  edgeEffortSeries,
  edgeStripDrawn,
  edgeStripColumns,
  edgeVisits,
  endWindowText,
  dirArrow,
  floorIndex,
  kfmt,
  lastSwingAnchor,
  makeShifter,
  minuteFlow,
  boundsFromFracs,
  boundsToMargins,
  dragSlotFracs,
  marginsToBounds,
  paneSlotMargins,
  slotGapPx,
  slotOrder,
  zoomRange,
  rollingDelta,
  rowCloses,
  segmentCvd,
  segmentShare,
  SHARE_GUIDES,
  SHARE_SCALE,
  readoutShare,
  mixedSourcesText,
  sharePctText,
  volumeCap,
  sessionOpenAnchor,
} from '../flowV3'
import { BoxStartTicksPrimitive, EdgeEffortPrimitive, ORANGE_RGB, ShareCapPrimitive, VolumeCapPrimitive } from './flowV3Primitives'

const BG_COLOR = '#1f2937'
const PANE_SEPARATOR_PX = 1 // CR-DI: the strip's 64 px include the 1-px pane separator, so pane 0 keeps its height
const MIN_PANEL_HEIGHT = 140
const BUY_RGBA = (alpha) => `rgba(96,165,250,${alpha})`
const SELL_RGBA = (alpha) => `rgba(229,231,235,${alpha})`
const SLATE_RGBA = (alpha) => `rgba(148,163,184,${alpha})`
const MAGENTA_RGBA = (alpha) => `rgba(232,121,249,${alpha})`
const ORANGE_RGBA = (alpha) => `rgba(${ORANGE_RGB},${alpha})`
const SOURCE_LABEL = { databento: 'Databento', ironbeam: 'Ironbeam' }
// CR-DP item 7b: the slot layout (heights as fractions, per slot count: '3' rolling off, '4' rolling on), per browser.
const SLOTS_STORAGE_KEY = 'ib-react-flow-v3-slots'
// CR-DP item 8b: per-slot y zoom factors for one trade date ({ date, f: { cvd | vol | rolling | delta: factor } }), per browser.
const ZOOM_STORAGE_KEY = 'ib-react-flow-v3-zoom'
const ZOOM_MIN = 0.2
const ZOOM_MAX = 20
const readZoom = (key) => {
  try {
    const v = JSON.parse(window.localStorage.getItem(ZOOM_STORAGE_KEY) || 'null')
    return v && v.date === key && v.f && typeof v.f === 'object' ? v.f : {}
  } catch (e) {
    return {}
  }
}
const readSlots = () => {
  try {
    const v = JSON.parse(window.localStorage.getItem(SLOTS_STORAGE_KEY) || 'null')
    return v && typeof v === 'object' ? v : {}
  } catch (e) {
    return {}
  }
}

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

// CR-DP item 3: one header group (its chips 6 px apart; the header puts 14 px between groups)
const groupStyle = { display: 'inline-flex', alignItems: 'center', gap: '6px', flex: '0 0 auto' }

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
      <div><b>Anchored line</b> (top): with an anchor, the <b>delta share</b> since the anchor: buying minus selling added up from the bar after the anchor, divided by all contracts traded over the same bars (the volume bars' volume), on a fixed ±10 % scale with faint guides at 0, ±2 % and ±5 %. A value beyond ±10 % is drawn at the edge with a small cap mark; hover shows the true value, Σ delta and Σ volume. ⚙ "Anchored line: raw CVD" plots the anchored CVD instead (buying minus selling from the anchor bar, starting at 0). Set the anchor with 📍 and a click on a price bar, or with "session open" / "last swing" (the end of the last confirmed swing; swing size: ⚙). Esc or "clear" removes it; without an anchor the line is the session CVD. With Edges off the readout at the top left shows delta, delta share, price change, minutes and contracts per point since the anchor. With Edges on, each drawn box is its own segment: the line restarts at each box's first bar (a gap at each boundary) and older boxes than the drawn ones aren't drawn; the current box's stats are in a popup on the price chart (below). Each box on the price chart is labelled "Δ · pts · minutes · contracts/pt" from the same numbers; hover a box line for its edges, contracts on both sides, held minutes, probes and how it ended. Delta share is a display aid with no research meaning.{tag}</div>
      <div><span style={sw(SLATE_RGBA(0.25))} /><b>Set end</b> (with an anchor): click "set end", then a price bar after the anchor (not after the last bar: now, or the replay / review until time). Anchored CVD and its readout, the Edges box(es), the Edge effort strip and the Profile then use anchor → end only; after the end CVD isn't drawn and the boxes and strip stop. Delta, volume and rolling bars and the absorption levels / marks are unchanged. The window is shaded lightly on the price chart and the header shows "hh:mm → hh:mm · N min". Esc while picking cancels; "clear end" removes only the end, "clear" removes anchor and end. On the 5-minute chart the end is the last minute of the clicked candle.{tag}</div>
      <div><span style={sw(SLATE_RGBA(0.55))} /><b>Volume per minute</b> (middle bars, slate grey): all contracts traded in each 1-minute bar (the flow's volume, every print, including the rare unknown-side ones), from the same data as the delta bars. Hover a minute for its volume, delta and delta share (delta ÷ volume). The scale tops out at the session's 98th-percentile minute (of the bars loaded), so one huge minute (e.g. the close) doesn't flatten the rest: taller minutes are drawn at full height with a brighter cap line, and their hover shows the true volume. A display scale only.{tag}</div>
      <div><span style={sw(BUY_RGBA(0.7))} /><span style={sw(SELL_RGBA(0.6))} /><b>Rolling net delta</b> (⚙, off by default): buying minus selling over the last N minutes (5 / 10 / 15). When on it draws in the middle slot, volume moves to the slot below it and the pane grows by one slot.{tag}</div>
      <div><b>Delta per bar</b> (bottom bars): the Flow v2 histogram, so this pane stands alone.{tag}</div>
      <div style={h}>On the price chart</div>
      <div><b>Live effort</b> (with Effort on): the leg in progress — from the end of the last confirmed swing to the running extreme since — gets a dimmed label at that extreme, e.g. "↑ +6.5 pts so far · 1.2× vs last ↑": contracts per point vs the previous leg the same way, as the confirmed labels. Shown once the leg has moved at least M pts (⚙ "Live effort min move", default 4) and a previous leg the same way exists. It moves with the extreme on each refresh and becomes the normal label once the swing is confirmed; in a review replay it uses bars up to the replay time only. Hover it for the numbers. The label and M are display aids, not tested thresholds.{tag}</div>
      <div><span style={sw(ORANGE_RGBA(0.9))} /><span style={sw(MAGENTA_RGBA(0.9))} /><b>Absorption levels</b> (orange and magenta lines): a price where aggressive selling ("sellers absorbed", a floor, orange) or buying ("buyers absorbed", a ceiling, magenta) hit the Absorbed rule below. The line starts at the minute the volume built up, appears once the hold window has passed, and ends when price trades through it (else it runs to now). Thicker = more contracts absorbed; it fades from full to 25 % over 60 minutes. At most 8 are drawn, the largest after fading. The label says side, contracts and minutes held; hover for start, opposite-side contracts and where it broke. Exact data only.{tag}</div>
      <div><span style={{ ...sw('transparent'), borderTop: `2px dashed ${MAGENTA_RGBA(0.8)}`, height: 0 }} /><b>Edges</b> (with an anchor; dashed magenta lines and a <b>box stats popup</b> near the current box: box from hh:mm (→ end) · Δ · Δ share · pts · minutes · contracts/pt, then each edge's price, contracts on both sides, held minutes and state — moved / probe / testing break; a last line "mixed sources (A + B)" when the box spans segments whose data differ (Databento / Ironbeam, or exact / approx). Drag it anywhere on the chart; it stays there until a new anchor; × hides it until the next anchor): the box starts at the anchor minute's close. A close up to 2 pts beyond an edge is a test and the edge stays; a close more than 2 pts beyond moves the edge or breaks the box (below). For each edge: aggressive contracts traded within 1 pt of it since the box started (buyers at the top, sellers at the bottom; exact data only) and minutes since price first came within 1 pt of it.{tag}
        <div style={{ marginTop: '4px' }}><b>Restart on a break</b> (CR-DI, CR-DN): while an edge is younger than N minutes (⚙, default 15) a close more than 2 pts beyond it moves the edge out ("moved hh:mm") and the box grows. Once the edge has held N minutes, such a close starts a <b>pending break</b> (popup "testing break hh:mm"; nothing grows or freezes meanwhile). If no 1-minute close comes back strictly inside the box (a close exactly on the edge does not count) within A minutes (⚙ "Break must hold A min", default 5), the break holds: the box freezes (drawn faded, ending at the break minute) and a new box starts at the break bar from that bar's close. If a close does come back inside, the break is cancelled: the box is unchanged and a faint magenta <b>probe</b> tick marks it at that edge (hover: the extreme reached, minutes outside, contracts traded beyond the edge). The popup and the Edge effort strip use the current box; the last 4 frozen boxes are drawn; clearing or moving the anchor removes all boxes. N and A are display defaults, not tested thresholds. An idea, untested: a failed probe can strengthen the fade.</div>
      </div>
      <div><span style={sw(BUY_RGBA(0.85))} /><span style={sw(SELL_RGBA(0.38))} /><span style={sw(SELL_RGBA(0.72))} /><span style={sw(BUY_RGBA(0.38))} /><b>Edge effort</b> (header chip, needs Edges and an anchor; a strip under the delta bars): per minute since the anchor, the same aggressive contracts the edge readout counts, within 1 pt of each edge, both sides. Top edge, drawn up: buyers (blue) with sellers beside them (thinner, lighter grey). Bottom edge, drawn down: sellers (grey) with buyers beside them (thinner, lighter blue). Each minute is counted against the box it belongs to (the drawn boxes: the last 4 frozen and the current one). Over the current box, the blue bars at the top and the grey bars at the bottom add up to the popup's numbers; when an edge of the current box moves, its minutes are re-counted against the new edge, as the popup is. A visit is a run of minutes with bars at that edge, allowing gaps of up to 2 minutes, and never crosses a break; its label says main side / other side (top buyers / sellers, bottom sellers / buyers). Hover a minute for both sides, the net and the visit's running totals. Exact data only. Visit sizes are effort, not a tested signal; Ironbeam live data understates volume.{tag}
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
      <div style={h}>Layout and zoom</div>
      <div>Drag the faint lines between this pane's slots to give a slot more or less height (each keeps at least 24 px; the pane's height stays the same), and the bar between the price chart and this pane to change the split. Both are remembered in this browser; ⚙ "Reset layout" restores the defaults. Drag up / down on this pane's right axis over a slot to stretch / compress that slot's scale; double-click there to fit it again. The price chart's right axis works the same way (drag, double-click to fit); a manual price range is kept for the date through refreshes and reloads. Zoom changes only the view, never the data.</div>
      <div style={h}>Data</div>
      <div><b>exact: trades</b> / <b>exact: es_flow_price_1m</b> — each trade at its own price and side. Ironbeam keeps raw trades for about 2 days, so this is usually only the live and previous session.</div>
      <div><b>approximate: 1-s data</b> — older sessions: each second's selling is placed at that second's low and its buying at its high. About 61 % of volume trades in seconds wider than 1 tick, so cells are off by a tick or two and selling leans toward lows, buying toward highs. <b>Absorption levels, Absorbed marks, edge contracts and the Edge effort strip are not drawn on approximate data</b>, because that lean manufactures them.</div>
      <div>Sessions through 2026-06-05 use Databento for RTH; later sessions use Ironbeam, which misses about 6 % of volume and understates buy − sell by about 30 %.</div>
    </div>
  )
}

// CR-DQ items 5 / 7 — the box panel's thresholds (one config object, flowV3.BOX_RULES_DEFAULTS). Display aids, untested.
function BoxRulesFields({ draft, setDraft, input, label }) {
  const r = { ...boxRules(null), ...(draft.boxRules || {}) }
  const set = (k) => (e) => setDraft((d) => ({ ...d, boxRules: { ...boxRules(null), ...(d.boxRules || {}), [k]: e.target.value } }))
  const num = (k, text, step, testid) => (
    <label style={{ display: 'block', flex: 1 }}>
      {label(text)}
      <input type="number" step={step} min={BOX_RULES_RANGES[k][0]} max={BOX_RULES_RANGES[k][1]} aria-label={text} data-testid={testid}
        value={r[k]} style={input} onChange={set(k)} />
    </label>
  )
  const note = { fontSize: '11px', color: '#94a3b8', marginTop: '2px', marginBottom: '10px' }
  return (
    <>
      <div style={{ fontSize: '12px', color: '#e2e8f0', fontWeight: 600, margin: '4px 0' }}>Box panel — percentile tiers</div>
      <div style={{ display: 'flex', gap: 8 }}>
        {num('notable', 'Notable from (percentile)', 1, 'flow-v3-tier-notable')}
        {num('strong', 'Strong from (percentile)', 1, 'flow-v3-tier-strong')}
      </div>
      <div style={note}>Display aid — untested. Badges compare a box's Δ share and contracts/pt with train RTH windows of similar length (defaults 80 / 95).</div>
      <div style={{ fontSize: '12px', color: '#e2e8f0', fontWeight: 600, margin: '4px 0' }}>Verdict rules</div>
      <div style={{ display: 'flex', gap: 8 }}>
        {num('nearPts', 'Near edge (pts)', 0.25, 'flow-v3-rule-near')}
        {num('progressPts', 'No advance (pts)', 0.25, 'flow-v3-rule-progress')}
        {num('absorbPts', 'Absorption within (pts)', 0.25, 'flow-v3-rule-absorb')}
      </div>
      <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
        {num('minAgeMin', 'Min box age (min)', 1, 'flow-v3-rule-min-age')}
        {num('minHeightPts', 'Min box height (pts)', 0.25, 'flow-v3-rule-min-height')}
      </div>
      <div style={note}>Draft rules (A19, A22, A23, A24), applied mechanically — untested, a display aid. "Notable" in the rules is the tier above. Absorption distance is to the box's high–low range (wicks). No verdict beyond NO TRADES until the box is this old and tall ("box still forming"). Defaults 2 / 2 / 2, 8 min / 2 pts. The box panel's verdict is the v1 rules engine's, which uses these values too.</div>
    </>
  )
}

function Settings({ value, layers, markers = DEFAULT_MARKER_SETTINGS, onApply, onClose, onResetLayout }) {
  const [draft, setDraft] = useState({
    ...value, heat: Boolean(layers?.heat), marks: Boolean(layers?.marks), swing: markers.swing, effortRatio: markers.effortRatio,
  })
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
      breakHoldMin: Math.round(n('breakHoldMin', BREAK_HOLD_MIN_RANGE[0], BREAK_HOLD_MIN_RANGE[1], DEFAULT_V3_SETTINGS.breakHoldMin)),
      rollingOn: Boolean(draft.rollingOn),
      liveMinMove: n('liveMinMove', LIVE_MIN_MOVE_RANGE[0], LIVE_MIN_MOVE_RANGE[1], DEFAULT_V3_SETTINGS.liveMinMove),
      anchoredLine: draft.anchoredLine === 'raw' ? 'raw' : 'share',
      boxRules: boxRules(draft.boxRules), // CR-DQ items 5 / 7: clamped to BOX_RULES_RANGES
      v1Alert: Boolean(draft.v1Alert), // CR-DU item 2
      contraMarks: Boolean(draft.contraMarks), // CR-DU item 4
    }, {
      // CR-DP amendment item 1: the Flow v2 marker settings, same ranges as v2's ⚙ (swing 1–50, ratio 1–10)
      ...markers,
      swing: n('swing', 1, 50, DEFAULT_MARKER_SETTINGS.swing),
      effortRatio: n('effortRatio', 1, 10, DEFAULT_MARKER_SETTINGS.effortRatio),
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
          {label('Anchored line (with an anchor)')}
          <select aria-label="Anchored line" data-testid="flow-v3-anchored-line" value={draft.anchoredLine === 'raw' ? 'raw' : 'share'} style={input}
            onChange={(e) => setDraft((d) => ({ ...d, anchoredLine: e.target.value }))}>
            <option value="share">delta share (Σ delta ÷ Σ volume, ±10 %)</option>
            <option value="raw">raw CVD</option>
          </select>
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
          {label('Break must hold A min')}
          <input type="number" step={1} min={BREAK_HOLD_MIN_RANGE[0]} max={BREAK_HOLD_MIN_RANGE[1]} aria-label="Break hold minutes" data-testid="flow-v3-break-hold"
            value={draft.breakHoldMin ?? DEFAULT_V3_SETTINGS.breakHoldMin} style={input} onChange={(e) => setDraft((d) => ({ ...d, breakHoldMin: e.target.value }))} />
          <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>Edges box, 1–15 (default 5): after a close beyond a held edge, the box freezes only if no 1-minute close comes back strictly inside within A minutes; otherwise the break is a probe. A display default, not a tested threshold.</div>
        </label>
        <label style={{ display: 'block', marginBottom: '10px' }}>
          {label('Live effort min move (pts)')}
          <input type="number" step={0.5} min={LIVE_MIN_MOVE_RANGE[0]} max={LIVE_MIN_MOVE_RANGE[1]} aria-label="Live effort min move" data-testid="flow-v3-live-min-move"
            value={draft.liveMinMove ?? DEFAULT_V3_SETTINGS.liveMinMove} style={input} onChange={(e) => setDraft((d) => ({ ...d, liveMinMove: e.target.value }))} />
          <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>With Effort on, the leg in progress is labelled once it has moved this far (2–12, default 4). A display default, not a tested threshold.</div>
        </label>
        <BoxRulesFields draft={draft} setDraft={setDraft} input={input} label={label} />
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', margin: '0 0 10px' }}>
          <input type="checkbox" aria-label="Alert on SHORT / LONG" data-testid="flow-v3-v1-alert" checked={Boolean(draft.v1Alert)}
            onChange={(e) => {
              const on = e.target.checked
              setDraft((d) => ({ ...d, v1Alert: on }))
              try { if (on && window.Notification && Notification.permission === 'default') Notification.requestPermission() } catch (err) { /* none */ }
            }} />
          <span style={{ fontSize: '12px', color: '#cbd5e1' }}>Alert when the v1 verdict turns SHORT / LONG (live session: tone + browser notification)</span>
        </label>
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', margin: '0 0 10px' }}>
          <input type="checkbox" aria-label="Contradiction bars" data-testid="flow-v3-contra-marks" checked={Boolean(draft.contraMarks)}
            onChange={(e) => setDraft((d) => ({ ...d, contraMarks: e.target.checked }))} />
          <span style={{ fontSize: '12px', color: '#cbd5e1' }}>Contradiction bars (markers) in the current box: heavy one-sided minute rejected at an edge (CR-DU item 4; untested)</span>
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
        <label style={{ display: 'block', marginBottom: 8 }}>
          {label('Swing size (pts)')}
          <input type="number" step={0.25} min={0} aria-label="Swing size (pts)" data-testid="flow-v3-swing-size"
            value={draft.swing} style={input} onChange={(e) => setDraft((d) => ({ ...d, swing: e.target.value }))} />
          <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>A swing high / low is confirmed after price retraces this far; sets "last swing" and the Effort labels (1–50, default 8). A display setting: A7's volatility-scaled swing is the research definition, not this.</div>
        </label>
        <label style={{ display: 'block', marginBottom: 8 }}>
          {label('Effort highlight ratio')}
          <input type="number" step={0.1} min={0} aria-label="Effort highlight ratio" data-testid="flow-v3-effort-ratio"
            value={draft.effortRatio} style={input} onChange={(e) => setDraft((d) => ({ ...d, effortRatio: e.target.value }))} />
          <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>An Effort label is highlighted when its contracts per point is at least this multiple of the previous swing the same way (1–10, default 1.5). A display setting.</div>
        </label>
        <div style={{ display: 'flex', gap: '8px', justifyContent: 'space-between', marginTop: '14px' }}>
          <span style={{ display: 'flex', gap: '6px' }}>
            <button type="button" style={chipStyle(false)} onClick={() => setDraft({
              ...DEFAULT_V3_SETTINGS, heat: false, marks: false, swing: DEFAULT_MARKER_SETTINGS.swing, effortRatio: DEFAULT_MARKER_SETTINGS.effortRatio,
            })}>Defaults</button>
            <button type="button" style={chipStyle(false)} data-testid="flow-v3-reset-layout" onClick={onResetLayout}
              title="Slot heights in this pane and the price chart / Flow split back to their defaults">Reset layout</button>
          </span>
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
  priceAxisWidth = null, // CR-DN item 7: the price chart's right axis width; this pane's right axis matches it
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
  settings = DEFAULT_V3_SETTINGS,
  onSettingsChange,
  layers = { heat: false, profile: false, marks: false, levels: false, edges: false },
  onLayersChange,
  edges = null, // CR-DF: flowV3.edgeReadout (App), null without an anchor or with the readout off
  segments = null, // CR-DN item 3: App's box segments { spans, readouts, dropped } (Edges on + anchor), else null
  anchor = null,
  onAnchorChange,
  anchorArmed = false,
  onAnchorArmedChange,
  end = null, // CR-DN item 1: the window's end (shifted epoch of its last 1-minute bar), null = to now
  onEndChange,
  endArmed = false,
  onEndArmedChange,
  effortOn = false, // CR-DG item 4: the Flow v2 Effort switch (same state), labels on the price chart
  onEffortChange,
  myTradesOn = false,
  onMyTradesChange,
  v1TradesOn = false,
  v1TradesBlocked = null,
  onV1TradesChange,
  edgeEffortOn = false, // CR-DI: the Edge effort strip (stored per viewer); draws only with Edges on, an anchor and exact data
  onEdgeEffortChange,
  zoomResetKey = null, // CR-DP item 8b: the trade date; a change resets the slot zoom
  markerSettings = DEFAULT_MARKER_SETTINGS, // CR-DP amendment item 1: swing size / effort ratio (the stored Flow v2 settings)
  onMarkerSettingsChange,
  onResetLayout, // CR-DP item 7: App restores the price chart / Flow split (7c); the pane clears its slot layout
}) {
  const stageRef = useRef(null)
  const hostRef = useRef(null)
  const chartRef = useRef(null)
  const priceAxisWidthRef = useRef(priceAxisWidth)
  priceAxisWidthRef.current = priceAxisWidth
  const seriesRef = useRef({})
  const lastAppliedLogicalRef = useRef(null)
  const candleTimesRef = useRef([])
  const linkedCrosshairRef = useRef(null)
  const anchorRef = useRef(null)
  const refreshDecorationsRef = useRef(() => {})
  const slotsRef = useRef({ rollingOn: false, extra: 0, fracs: {} })
  const [slotFracs, setSlotFracs] = useState(readSlots) // CR-DP item 7
  const [slotBounds, setSlotBounds] = useState(null) // pane-0 px bounds of the slots, for the dividers
  const slotDragRef = useRef(null)
  const [zoom, setZoom] = useState(() => readZoom(zoomResetKey)) // CR-DP item 8b: { seriesName: factor }
  const zoomKeyRef = useRef(zoomResetKey)
  const zoomDragRef = useRef(null)
  const slotBoundsRef = useRef(null)

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
  const closes = useMemo(() => rowCloses(aligned, candles, shift), [candles, aligned, shift])
  // CR-DN item 3: with box segments, the CVD restarts at 0 on each drawn box (same anchoredCvd per span); `cvdGaps` =
  // rows whose line to the next box is drawn transparent. Without segments, one anchored line as before.
  const cvdSeg = useMemo(() => (segments?.spans?.length ? segmentCvd(aligned, segments.lineSpans || segments.spans) : null), [aligned, segments])
  const plainCvd = useMemo(() => (cvdSeg ? null : anchoredCvd(aligned, anchor, end)), [cvdSeg, aligned, anchor, end])
  const cvd = cvdSeg ? cvdSeg.cvd : plainCvd
  const cvdGaps = cvdSeg ? cvdSeg.gapAt : null
  // CR-DP item 6: with an anchor the line plots cumulative delta share (Σ delta ÷ Σ volume from each segment's start: per
  // box with Edges on, else anchor → now / end) on a fixed ±10 % scale; ⚙ "raw CVD" keeps the CVD. No anchor: session CVD.
  const shareMode = anchor != null && settings.anchoredLine !== 'raw'
  const shareSeries = useMemo(() => {
    if (!shareMode) return null
    return segments?.spans?.length ? segmentShare(aligned, segments.lineSpans || segments.spans) : { ...anchoredShare(aligned, anchor, end), gapAt: null }
  }, [shareMode, aligned, segments, anchor, end])
  const rollingOn = Boolean(settings.rollingOn)
  const rolling = useMemo(() => (rollingOn ? rollingDelta(aligned, settings.rollingMin) : null), [aligned, rollingOn, settings.rollingMin])
  // CR-DN item 3c: with segments, the header reads the current box (the last one when an end is set).
  const readout = useMemo(() => {
    const ro = segments?.readouts?.length ? segments.readouts[segments.readouts.length - 1] : null
    return ro ? { ...ro, box: true } : anchorReadout(aligned, closes, anchor, end)
  }, [segments, aligned, closes, anchor, end])
  const windowText = useMemo(() => endWindowText(anchor, end, formatShiftedHHMM), [anchor, end])
  // An anchor in ETH sums Ironbeam ETH delta with the RTH segment's delta; say so when the sources differ.
  const mixedNote = useMemo(() => mixedSourcesText(readout, sessionPayload?.segments), [readout, sessionPayload])
  const sources = useMemo(() => {
    const s = new Set()
    for (const p of payloads || []) for (const seg of p?.segments || []) if (seg.name === 'rth') s.add(seg.source)
    return [...s]
  }, [payloads])
  // CR-DP item 2: one note when either segment's cells are approximate (1-s data); nothing on exact sessions.
  const approxCells = useMemo(() => (cellsPayload?.segments || []).some((seg) => seg.exact === false), [cellsPayload])
  const marksNote = useMemo(() => {
    const rth = (cellsPayload?.segments || []).find((s) => s.name === 'rth')
    return rth && !Array.isArray(rth.marks) ? rth.marks_note : null
  }, [cellsPayload])
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

  // DC5: Esc disarms a pending pick (anchor or, CR-DN, end), else clears the anchor (and with it the end).
  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== 'Escape' || e.defaultPrevented) return
      const tag = (e.target?.tagName || '').toLowerCase()
      if (tag === 'input' || tag === 'select' || tag === 'textarea') return
      if (endArmed) onEndArmedChange?.(false)
      else if (anchorArmed) onAnchorArmedChange?.(false)
      else if (anchor != null) onAnchorChange?.(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [anchor, anchorArmed, endArmed, onAnchorChange, onAnchorArmedChange, onEndArmedChange])

  useEffect(() => {
    if (!hostRef.current || !stageRef.current) return undefined
    const stage = stageRef.current
    const chart = createChart(hostRef.current, {
      width: stage.clientWidth || 900,
      height: Math.max(stage.clientHeight || MIN_PANEL_HEIGHT, MIN_PANEL_HEIGHT),
      layout: { background: { type: ColorType.Solid, color: BG_COLOR }, textColor: '#cbd5e1', attributionLogo: false },
      localization: { locale: 'en-US', priceFormatter: (v) => kfmt(Number(v)) },
      grid: { vertLines: { color: 'rgba(148, 163, 184, 0.08)' }, horzLines: { color: 'rgba(148, 163, 184, 0.08)' } },
      rightPriceScale: { visible: true, borderColor: 'rgba(148, 163, 184, 0.18)', scaleMargins: { top: 0.42, bottom: 0.30 }, minimumWidth: priceAxisWidthRef.current || 0 },
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
    const volCap = new VolumeCapPrimitive() // CR-DN item 6
    volS.attachPrimitive(volCap)
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
    const shareCap = new ShareCapPrimitive() // CR-DP item 6b
    cvdS.attachPrimitive(shareCap)
    const boxTicks = new BoxStartTicksPrimitive()
    cvdS.attachPrimitive(boxTicks)
    seriesRef.current = { delta, rolling: rollingS, cvd: cvdS, vol: volS, volCap, shareCap, boxTicks }

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

  // CR-DN item 7: the right axis is pinned to the price chart's, so both plots have the same width and the shared
  // logical range puts a bar at the same x in both panes.
  useEffect(() => {
    chartRef.current?.applyOptions({ rightPriceScale: { minimumWidth: priceAxisWidth || 0 } })
  }, [priceAxisWidth])

  // CR-DM item 2: slot layout of pane 0 (paneSlotMargins); rolling off = the CR-DC margins with volume in the middle.
  const applySlots = () => {
    const chart = chartRef.current
    const s = seriesRef.current
    if (!chart || !s.vol) return
    const { rollingOn: on, extra, fracs } = slotsRef.current
    // pane 0 from the height just applied (panes()[0].getHeight() lags until the chart's next layout pass)
    const h0 = chart.options().height - chart.timeScale().height() - (s.strip ? EDGE_STRIP_HEIGHT : 0)
    // CR-DP item 7: a stored layout for this slot count, else today's margins (paneSlotMargins)
    const order = slotOrder(on)
    const custom = boundsFromFracs(h0, fracs?.[order.length], slotGapPx(h0, on ? extra : 0))
    const m = custom ? boundsToMargins(h0, custom, order) : paneSlotMargins(h0, on ? extra : 0)
    const bounds = custom || marginsToBounds(h0, m, order)
    setSlotBounds((prev) => (prev && JSON.stringify(prev) === JSON.stringify({ order, bounds, h0 }) ? prev : { order, bounds, h0 }))
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
    slotsRef.current = { rollingOn, extra: rollingExtra, fracs: slotFracs }
    fitToStage()
  }, [rollingOn, rollingExtra, slotFracs]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    try {
      if (Object.keys(slotFracs).length) window.localStorage.setItem(SLOTS_STORAGE_KEY, JSON.stringify(slotFracs))
      else window.localStorage.removeItem(SLOTS_STORAGE_KEY)
    } catch (e) {
      // ignore
    }
  }, [slotFracs])
  // CR-DP item 7a: drag a divider between two slots (pointer capture on the handle); the pane's height is unchanged.
  const onDividerDown = (k) => (e) => {
    if (e.button !== 0 || !slotBounds) return
    e.preventDefault()
    e.stopPropagation()
    e.currentTarget.setPointerCapture?.(e.pointerId)
    slotDragRef.current = { k, y: e.clientY, bounds: slotBounds.bounds, n: slotBounds.order.length }
  }
  const onDividerMove = (e) => {
    const d = slotDragRef.current
    if (!d) return
    const fr = dragSlotFracs(d.bounds, d.k, e.clientY - d.y)
    setSlotFracs((prev) => ({ ...prev, [d.n]: fr }))
  }
  const onDividerUp = (e) => {
    if (!slotDragRef.current) return
    slotDragRef.current = null
    e.currentTarget.releasePointerCapture?.(e.pointerId)
  }
  const resetLayout = () => {
    setSlotFracs({})
    onResetLayout?.()
  }

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
    // CR-DN item 6: bars above the session's p98 minute volume are drawn at the cap (with a cap mark); hover reads the rows
    const cap = volumeCap(aligned)
    s.vol.setData(aligned.map((r) => (Number.isFinite(r.vol) ? { time: r.time, value: cap == null ? r.vol : Math.min(r.vol, cap) } : ws(r))))
    s.volCap.setData({ cap, color: SLATE_RGBA(0.95), times: cap == null ? [] : aligned.filter((r) => Number.isFinite(r.vol) && r.vol > cap).map((r) => r.time) })
    const gap = 'rgba(0,0,0,0)' // CR-DN item 3a: the line from a box's last bar to the next box's 0 is not drawn
    if (shareSeries) {
      // CR-DP item 6b: fixed ±10 % (values beyond are drawn at the clip with a cap mark; hover reads the true value)
      const sh = shareSeries.share
      s.cvd.setData(aligned.map((r, i) => (sh[i] != null
        ? (shareSeries.gapAt?.has(i) ? { time: r.time, value: clipShare(sh[i]), topLineColor: gap, bottomLineColor: gap } : { time: r.time, value: clipShare(sh[i]) })
        : ws(r))))
      s.shareCap.setData({ cap: SHARE_SCALE, color: SLATE_RGBA(0.95), marks: aligned.flatMap((r, i) => (sh[i] != null && Math.abs(sh[i]) > SHARE_SCALE ? [{ time: r.time, up: sh[i] > 0 }] : [])) })
    } else {
      s.cvd.setData(aligned.map((r, i) => (cvd[i] != null
        ? (cvdGaps?.has(i) ? { time: r.time, value: cvd[i], topLineColor: gap, bottomLineColor: gap } : { time: r.time, value: cvd[i] })
        : ws(r))))
      s.shareCap.setData({ marks: [] })
    }
    s.boxTicks.setData({ color: SLATE_RGBA(0.95), times: segments?.ticks || [] }) // CR-DU amendment 7 item 4
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
  }, [aligned, rolling, cvd, cvdGaps, shareSeries, candleTimes, visibleLogicalRange, segments])

  // CR-DP item 6b: in share mode the line's scale is fixed at ±10 % with faint slate guides at 0, ±2 % and ±5 % (no axis
  // labels); raw CVD autoscales as before. Item 8b: each slot's auto range (fixed ±10 % for the share line) is then
  // zoomed by that slot's factor; autoscale stays on, so live refreshes keep the zoom and never change the data.
  useEffect(() => {
    const s = seriesRef.current
    if (!s.cvd) return
    const provider = (name, fixed) => (original) => {
      const base = fixed ? { priceRange: { minValue: -SHARE_SCALE, maxValue: SHARE_SCALE } } : original()
      const f = zoom[name]
      if (!base?.priceRange || !(f > 0) || f === 1) return base
      return { ...base, priceRange: zoomRange(base.priceRange, f) }
    }
    s.cvd.applyOptions({ autoscaleInfoProvider: provider('cvd', shareMode) })
    s.vol.applyOptions({ autoscaleInfoProvider: provider('vol', false) })
    s.rolling.applyOptions({ autoscaleInfoProvider: provider('rolling', false) })
    s.delta.applyOptions({ autoscaleInfoProvider: provider('delta', false) })
  }, [shareMode, zoom])
  useEffect(() => {
    const s = seriesRef.current
    if (!s.cvd || !shareMode) return undefined
    const lines = SHARE_GUIDES.map((v) => s.cvd.createPriceLine({
      price: v, color: SLATE_RGBA(v === 0 ? 0.4 : 0.22), lineWidth: 1, lineStyle: v === 0 ? 0 : 2, axisLabelVisible: false, title: '',
    }))
    return () => { for (const l of lines) { try { s.cvd.removePriceLine(l) } catch (e) { /* chart gone */ } } }
  }, [shareMode])
  // Item 8b: a date change resets the slot zoom (a reload restores the date's stored factors); stored per browser.
  useEffect(() => {
    if (zoomKeyRef.current === zoomResetKey) return
    zoomKeyRef.current = zoomResetKey
    setZoom({})
  }, [zoomResetKey])
  useEffect(() => {
    try {
      if (Object.keys(zoom).length) window.localStorage.setItem(ZOOM_STORAGE_KEY, JSON.stringify({ date: zoomKeyRef.current, f: zoom }))
      else window.localStorage.removeItem(ZOOM_STORAGE_KEY)
    } catch (e) {
      // ignore
    }
  }, [zoom])
  useEffect(() => { slotBoundsRef.current = slotBounds }, [slotBounds])
  // Item 8b: drag on the pane's right axis over a slot stretches (up) / compresses (down) that slot only; double-click
  // there returns it to auto-fit. The library's own axis scaling is off in this pane (handleScale.axisPressedMouseMove).
  useEffect(() => {
    const stage = stageRef.current
    if (!stage) return undefined
    const slotAt = (e) => {
      const chart = chartRef.current
      const sb = slotBoundsRef.current
      if (!chart || !sb) return null
      const r = stage.getBoundingClientRect()
      const x = e.clientX - r.left
      const y = e.clientY - r.top
      if (x < chart.timeScale().width() || y > sb.h0) return null
      let best = 0
      let bestD = Infinity
      sb.bounds.forEach((b, i) => {
        const d = y < b.top ? b.top - y : y > b.bottom ? y - b.bottom : 0
        if (d < bestD) { bestD = d; best = i }
      })
      const key = sb.order[best]
      const on = sb.order.length === 4
      return key === 'cvd' ? 'cvd' : key === 'delta' ? 'delta' : key === 'low' ? 'vol' : on ? 'rolling' : 'vol'
    }
    const down = (e) => {
      if (e.button !== 0) return
      const name = slotAt(e)
      if (!name) return
      e.preventDefault()
      zoomDragRef.current = { name, y: e.clientY, f0: null }
    }
    const move = (e) => {
      const d = zoomDragRef.current
      if (!d) return
      setZoom((z) => {
        if (d.f0 == null) d.f0 = z[d.name] || 1
        const f = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, d.f0 * Math.exp(-(e.clientY - d.y) * 0.01)))
        return Math.abs(f - (z[d.name] || 1)) < 1e-6 ? z : { ...z, [d.name]: f }
      })
    }
    const up = () => { zoomDragRef.current = null }
    const dbl = (e) => {
      const name = slotAt(e)
      if (!name) return
      setZoom((z) => {
        if (!(name in z)) return z
        const { [name]: _drop, ...rest } = z
        return rest
      })
    }
    stage.addEventListener('mousedown', down, true)
    stage.addEventListener('dblclick', dbl, true)
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
    return () => {
      stage.removeEventListener('mousedown', down, true)
      stage.removeEventListener('dblclick', dbl, true)
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', up)
    }
  }, [])


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
    <div className="flow-shell" style={{ height: `${height}px`, position: 'relative', width: '100%' }} data-testid="flow-v3-panel"
      data-anchor={anchor ?? ''} data-end={end ?? ''} data-box-ticks={(segments?.ticks || []).join(',')}>
      {/* CR-DP item 3: Anchor · Edges · Display groups (a small gap between groups), status notes, then ⚙ and ? at the far
          right. Same buttons and behaviour; the anchored readout moved to the price chart's box stats popup (item 4). */}
      <div className="flow-panel-header" style={{ display: 'flex', alignItems: 'center', gap: '14px', justifyContent: 'flex-start', overflowX: 'auto' }}>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', flex: '0 0 auto' }}>
          <div className="flow-panel-title" style={{ whiteSpace: 'nowrap', flex: '0 0 auto' }}>Flow v3</div>
          {sources.map((s) => (
            <span key={s} data-testid="flow-v3-source" style={{ fontSize: '11px', padding: '2px 7px', borderRadius: '999px', border: '1px solid rgba(148,163,184,0.35)', color: s === 'databento' ? '#bfdbfe' : '#e5e7eb', whiteSpace: 'nowrap' }}>
              {SOURCE_LABEL[s] || s}
            </span>
          ))}
        </span>
        <span data-testid="flow-v3-group-anchor" style={groupStyle}>
          <button type="button" style={chipStyle(anchorArmed)} aria-pressed={anchorArmed} data-testid="flow-v3-anchor-pick"
            onClick={() => onAnchorArmedChange?.(!anchorArmed)} title="Click, then click a bar on the price chart to anchor CVD there (Esc cancels)">
            📍 {anchorArmed ? 'click a bar…' : 'set anchor'}
          </button>
          <button type="button" style={{ ...chipStyle(endArmed), opacity: anchor == null ? 0.5 : 1 }} aria-pressed={endArmed} data-testid="flow-v3-end-pick"
            disabled={anchor == null} onClick={() => onEndArmedChange?.(!endArmed)}
            title={anchor == null ? 'Set an anchor first' : 'Click, then click a bar after the anchor on the price chart to end the window there (Esc cancels)'}>
            {endArmed ? 'click a bar…' : 'set end'}
          </button>
          {windowText && (
            <span data-testid="flow-v3-window" style={{ fontSize: '11px', color: '#cbd5e1', whiteSpace: 'nowrap', flex: '0 0 auto' }}>{windowText}</span>
          )}
          {end != null && (
            <button type="button" data-testid="flow-v3-end-clear" onClick={() => { onEndArmedChange?.(false); onEndChange?.(null) }}
              style={{ ...chipStyle(false), fontSize: '10px', padding: '1px 6px' }} title="Remove the end (the anchor stays)">
              clear end
            </button>
          )}
          <button type="button" style={chipStyle(false)} data-testid="flow-v3-anchor-open"
            disabled={openAnchor == null} onClick={() => setAnchorTo(openAnchor)}>
            session open
          </button>
          <button type="button" style={chipStyle(false)} data-testid="flow-v3-anchor-swing" disabled={swingAnchor == null}
            onClick={() => setAnchorTo(swingAnchor)} title="The end of the last confirmed swing (swing size: ⚙)">
            last swing
          </button>
          <button type="button" style={chipStyle(false)} data-testid="flow-v3-anchor-clear" disabled={anchor == null && !anchorArmed && !endArmed}
            onClick={() => { onAnchorArmedChange?.(false); onEndArmedChange?.(false); onEndChange?.(null); onAnchorChange?.(null) }}>
            clear
          </button>
        </span>
        <span data-testid="flow-v3-group-edges" style={groupStyle}>
          <button type="button" style={chipStyle(layers.edges)} aria-pressed={Boolean(layers.edges)} data-testid="flow-v3-edges-toggle"
            onClick={() => onLayersChange?.((l) => ({ ...l, edges: !l.edges }))} title="Edges from the anchor: close box, contracts at each edge, held / broke; stats in a popup on the price chart — visual aid, untested">
            Edges {layers.edges ? 'on' : 'off'}
          </button>
          <button type="button" style={{ ...chipStyle(edgeEffortOn && !edgeEffortWhy), opacity: edgeEffortWhy ? 0.5 : 1, cursor: edgeEffortWhy ? 'default' : 'pointer' }}
            aria-pressed={Boolean(edgeEffortOn)} data-testid="flow-v3-edge-effort-toggle" disabled={Boolean(edgeEffortWhy)}
            onClick={() => onEdgeEffortChange?.(!edgeEffortOn)}
            title={edgeEffortWhy || 'Edge effort strip: aggressive contracts within 1 pt of each edge per minute, both sides — visual aid, untested'}>
            Edge effort {edgeEffortOn ? 'on' : 'off'}
          </button>
        </span>
        <span data-testid="flow-v3-group-display" style={groupStyle}>
          <button type="button" style={chipStyle(layers.levels)} aria-pressed={Boolean(layers.levels)} data-testid="flow-v3-levels-toggle"
            onClick={() => onLayersChange?.((l) => ({ ...l, levels: !l.levels }))} title="Absorption levels (exact data only) — visual aid, untested">
            Absorption levels {layers.levels ? 'on' : 'off'}
          </button>
          <button type="button" style={chipStyle(layers.profile)} aria-pressed={layers.profile} data-testid="flow-v3-profile-toggle"
            onClick={() => onLayersChange?.((l) => ({ ...l, profile: !l.profile }))} title="Volume profile at the right edge, from the anchor (or session open) — visual aid, untested">
            Profile {layers.profile ? 'on' : 'off'}
          </button>
          <button type="button" style={chipStyle(effortOn)} aria-pressed={effortOn} data-testid="flow-v3-effort-toggle"
            onClick={() => onEffortChange?.(!effortOn)}
            title="Label each confirmed swing on the price chart: points · volume · contracts per point · × the previous swing the same way">
            Effort {effortOn ? 'on' : 'off'}
          </button>
          <button type="button" style={chipStyle(myTradesOn)} aria-pressed={myTradesOn} data-testid="flow-v3-my-trades-toggle"
            onClick={() => onMyTradesChange?.(!myTradesOn)}
            title="Your logged trades on the price chart: ▲ / ▼ entry, × exit, resting stop (dashed) and target (dotted) while open; click a marker to review it">
            My trades {myTradesOn ? 'on' : 'off'}
          </button>
          <button type="button" style={{ ...chipStyle(v1TradesOn), ...(v1TradesBlocked ? { opacity: 0.5, cursor: 'not-allowed' } : {}) }}
            aria-pressed={v1TradesOn} disabled={Boolean(v1TradesBlocked)} data-testid="flow-v3-v1-trades-toggle" data-blocked={v1TradesBlocked || ''}
            onClick={() => onV1TradesChange?.(!v1TradesOn)}
            title={v1TradesBlocked ? `v1 trades: ${v1TradesBlocked}`
              : "The v1 rules engine's simulated trades for this session (teal): entry, exit, stop / +10 target while open, points; hover a minute for its verdict and failed rules. Run minute by minute without later bars. Engine's trades, not yours; train backtest in-sample only — untested."}>
            v1 trades {v1TradesBlocked ? `(${v1TradesBlocked})` : v1TradesOn ? 'on' : 'off'}
          </button>
        </span>
        {(anyLayer && (approxCells || cellsLoading || cellsError)) || ((layers.marks || layers.levels) && marksNote && !approxCells) ? (
          <span style={groupStyle}>
            {anyLayer && approxCells && (
              <span data-testid="flow-v3-approx-note" title="each second: sells at its low, buys at its high"
                style={{ fontSize: '11px', color: '#94a3b8', whiteSpace: 'nowrap', flex: '0 0 auto' }}>
                approx data — absorption &amp; edges off
              </span>
            )}
            {anyLayer && cellsLoading && <span style={{ fontSize: '11px', color: '#94a3b8', whiteSpace: 'nowrap' }}>loading cells…</span>}
            {anyLayer && cellsError && <span style={{ fontSize: '11px', color: '#cbd5e1', whiteSpace: 'nowrap' }} title={cellsError}>cells failed</span>}
            {(layers.marks || layers.levels) && marksNote && !approxCells && <span data-testid="flow-v3-marks-note" style={{ fontSize: '11px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{marksNote}</span>}
          </span>
        ) : null}
        <span style={{ ...groupStyle, marginLeft: 'auto' }}>
          <button type="button" style={iconStyle} title="Flow v3 settings" aria-label="Open Flow v3 settings" onClick={() => setSettingsOpen(true)}>⚙</button>
          <button type="button" style={iconStyle} title="How to read Flow v3" aria-label="How to read Flow v3" onClick={() => setGuideOpen(true)}>?</button>
        </span>
      </div>
      {guideOpen && <Modal title="How to read Flow v3" onClose={() => setGuideOpen(false)}><GuideBody /></Modal>}
      {settingsOpen && (
        <Settings value={settings} layers={layers} markers={markerSettings} onClose={() => setSettingsOpen(false)} onResetLayout={() => { resetLayout(); setSettingsOpen(false) }}
          onApply={(lay, v, mk) => { onLayersChange?.((l) => ({ ...l, ...lay })); onSettingsChange?.(v); onMarkerSettingsChange?.(mk); setSettingsOpen(false) }} />
      )}
      <div ref={stageRef} className="flow-stage" style={{ width: '100%' }}>
        <div ref={hostRef} className="flow-host" style={{ width: '100%', height: '100%' }} />
        {sessionBands.map((band) => (
          <div key={band.key} className="session-band" style={{ left: `${band.left}px`, width: `${band.width}px` }} />
        ))}
        {Number.isFinite(anchorX) && (
          <div data-testid="flow-v3-anchor-line" style={{ position: 'absolute', top: 0, bottom: 0, left: `${Math.round(anchorX)}px`, width: 0, borderLeft: `1px dashed ${SLATE_RGBA(0.8)}`, pointerEvents: 'none', zIndex: 3 }} />
        )}
        {slotBounds && slotBounds.bounds.slice(0, -1).map((b, k) => {
          const y = (b.bottom + slotBounds.bounds[k + 1].top) / 2
          // the handle stays inside the gap between the two slots, so hovering a slot keeps its tooltip
          const hit = Math.max(4, Math.min(8, Math.floor(slotBounds.bounds[k + 1].top - b.bottom)))
          return (
            <div key={`div-${slotBounds.order.length}-${k}`} data-testid={`flow-v3-slot-divider-${k}`} className="flow-v3-slot-divider"
              onPointerDown={onDividerDown(k)} onPointerMove={onDividerMove} onPointerUp={onDividerUp} onPointerCancel={onDividerUp}
              title="Drag to resize these slots (⚙ Reset layout restores)"
              style={{ position: 'absolute', left: 0, right: priceAxisWidth || 60, top: Math.round(y - hit / 2), height: hit, cursor: 'row-resize', zIndex: 4, touchAction: 'none' }}>
              <div style={{ position: 'absolute', left: 0, right: 0, top: Math.floor(hit / 2), height: 0, borderTop: `1px solid ${SLATE_RGBA(0.18)}` }} />
            </div>
          )
        })}
        {Number.isFinite(linkedCrosshairX) && (
          <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${Math.round(linkedCrosshairX)}px`, width: 0, borderLeft: '1px dashed rgba(191, 219, 254, 0.45)', pointerEvents: 'none', zIndex: 4 }} />
        )}
        {/* CR-DP item 4: with Edges on the box stats are the price chart's popup, so the in-pane "box from…" line is gone */}
        {!readout?.box && (
          <div data-testid="flow-v3-readout" style={{ position: 'absolute', top: 4, left: 8, zIndex: 4, pointerEvents: 'none', fontSize: '11px', color: '#cbd5e1', background: 'rgba(15,23,42,0.72)', borderRadius: 6, padding: '2px 6px' }}>
            {readout
              ? `since ${formatShiftedHHMM(readout.anchorTime)}: Δ ${sfmt(readout.delta)} · ${readout.dPrice == null ? '–' : `${readout.dPrice >= 0 ? '+' : ''}${readout.dPrice.toFixed(2)} pts`} · ${readout.minutes} min · ${readout.cpp == null ? '–' : kfmt(readout.cpp)}/pt`
                + (shareMode ? ` · share ${sharePctText(readoutShare(readout))} (line: Δ share, ±10 %)` : '')
                + mixedNote
              : `CVD from session open · ${rollingOn ? `rolling ${settings.rollingMin} min · ` : ''}volume per minute`}
          </div>
        )}
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
                + (shareSeries
                  ? ` · share so far ${sharePctText(shareSeries.share[hoverIdx])} · ΣΔ ${shareSeries.sumDelta[hoverIdx] == null ? '–' : sfmt(shareSeries.sumDelta[hoverIdx])} · Σvol ${shareSeries.sumVol[hoverIdx] == null ? '–' : kfmt(shareSeries.sumVol[hoverIdx])}`
                  : ` · CVD ${cvd[hoverIdx] == null ? '–' : sfmt(cvd[hoverIdx])}`)
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
