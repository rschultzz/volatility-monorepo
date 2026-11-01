// CR-DY — selectable boxes (specs/CR-DY-selectable-boxes.md): the drawn box and the engine's box per trade as data for
// the chart's rectangles and the box panel. Pure. The engine's own trade fields are read, never recomputed (Step 0 Q1).
import { floorIndex } from './flowV3.js'

export const DRAWN_BOX_ID = 'drawn'

// The one place a drawn box's rectangle and its saved fields come from: edgeReadout manual over the window (Step 0: the
// engine's rule — edges by closes, frozen once the box qualifies; wicks stay on the readout as high / low).
export function drawnBoxOf(r) {
  if (!r) return null
  return { fromT: r.from, toT: r.to, top: r.top.price, bottom: r.bottom.price }
}

// One entry per engine trade (open one last), ids as overlayMarks' so box and trade mark link: the box as it stood at the
// decision minute. fromT / toT = UTC bar starts.
export function engineTradeBoxes(sim) {
  if (!sim) return []
  const all = [...sim.trades, ...(sim.open ? [sim.open] : [])]
  return all.map((tr, k) => ({
    id: `v1-${k}`, k, fromT: tr.box.from, toT: tr.decisionT, top: tr.box.top, bottom: tr.box.bottom, trade: tr,
    open: k >= sim.trades.length,
  }))
}

// A box on the chart's candle times (shifted), or null when its start is before the first candle.
export function snapBox(id, box, times, shift, rgb) {
  const i0 = floorIndex(times, shift(box.fromT))
  const i1 = floorIndex(times, shift(box.toT))
  if (i0 < 0 || i1 < i0) return null
  return { id, from: times[i0], to: times[i1], top: box.top, bottom: box.bottom, rgb }
}

// The rectangle under a point (css px): the smallest one when they overlap. rects: { id, x0, x1, y0, y1 }.
export function hitBox(rects, pt) {
  if (!pt || !Array.isArray(rects)) return null
  let best = null
  let bestArea = Infinity
  for (const r of rects) {
    if (pt.x < r.x0 || pt.x > r.x1 || pt.y < r.y0 || pt.y > r.y1) continue
    const area = (r.x1 - r.x0) * (r.y1 - r.y0)
    if (area < bestArea) {
      bestArea = area
      best = r.id
    }
  }
  return best
}

// "Your box 07:49–08:12" / "Your box 07:49–now" (live, no end yet) / "Engine box 07:34–07:49 · trade 2 (short)". `fmt`
// takes a UTC bar time.
export function boxPanelTitle(sel, readout, fmt, { running = false, engineBox = null } = {}) {
  if (!readout) return ''
  if (sel?.kind === 'engine' && engineBox) {
    return `Engine box ${fmt(engineBox.fromT)}–${fmt(engineBox.toT)} · trade ${engineBox.k + 1} (${engineBox.trade.side})`
  }
  return `Your box ${fmt(readout.from)}–${running ? 'now' : fmt(readout.to)}`
}

// The engine box's trade line: "short · entry 7655.75 · stop 7665.75 · TP 7645.75 · target +10.00 at 07:58".
export function tradeText(box, fmt) {
  const tr = box?.trade
  if (!tr) return ''
  const px = (v) => Number(v).toFixed(2)
  const head = `${tr.side} · entry ${px(tr.entry)} at ${fmt(tr.entryT)} · stop ${px(tr.stop)} · TP ${px(tr.target)}`
  if (box.open) return `${head} · open`
  return `${head} · ${tr.reason} ${tr.pts >= 0 ? '+' : ''}${tr.pts.toFixed(2)} at ${fmt(tr.exitT)}`
}

// Goal 5 — the engine's read at the panel's end minute: "Engine at 12:59: NO ACTION — rule 6: box still forming". Its own
// box goes to the hover title.
export function engineAtLine(pe, fmt) {
  if (!pe?.ev) return 'Engine: no RTH bars yet'
  const f = pe.ev.failed?.[0]
  return `Engine at ${fmt(pe.ev.t)}: ${pe.ev.verdict}${f ? ` — rule ${f.rule}: ${f.text}` : ''}`
}
export function engineAtTitle(pe, fmt) {
  const b = pe?.ev?.box
  return b ? `The engine's own chained box at that minute: ${fmt(b.from)}–${fmt(pe.ev.t)}. It never moves your box.` : ''
}

// What Esc does next: cancel the armed pick, else clear the selection (the box stays), else remove the drawn box.
export function escAction({ endArmed = false, anchorArmed = false, selected = false, anchor = null }) {
  if (endArmed) return 'end'
  if (anchorArmed) return 'start'
  if (selected) return 'selection'
  return anchor != null ? 'box' : null
}

// A review save's box fields (setup_reviews): start, end, top, bottom of the drawn box; end / top / bottom null without one.
export function reviewBoxRecord(anchor, drawn, shift, toPt) {
  return {
    anchor_ts: toPt(anchor),
    box_end_ts: drawn ? toPt(shift(drawn.toT)) : null,
    box_top: drawn ? drawn.top : null,
    box_bottom: drawn ? drawn.bottom : null,
  }
}
