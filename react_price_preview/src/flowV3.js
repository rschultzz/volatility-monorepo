// CR-DC — Flow v3 pure helpers (no React, no chart): anchored CVD and its readout, rolling net delta,
// heatmap columns with percentile opacity, the volume profile, absorption marks for the chart, and the
// "last swing" anchor. Times: "shifted" = PT wall-clock epochs (the charts' candle times); "utc" = real
// epochs (the server's). Every input array is memoised by the caller, so these run once per data change.

export const TICK = 0.25
export const ALPHA_MIN = 0.08
export const ALPHA_SPAN = 0.62
export const CROWDED_TICK_PX = 3 // DC7: one tick shorter than this -> 0.5-pt cells
export const ROLLING_CHOICES = [5, 10, 15]
// CR-DM: rolling net delta is a ⚙ option, off by default (volume per minute takes its slot); liveMinMove = the live
// effort label's minimum move in points (⚙, a display default, not a tested threshold).
// CR-DP item 6d: anchoredLine = what the anchored line plots ('share' = cumulative delta share, the default; 'raw' = CVD).
export const DEFAULT_V3_SETTINGS = { rollingMin: 10, k: 3, m: 3, sellLayer: true, buyLayer: true, boxHoldMin: 15, rollingOn: false, liveMinMove: 4, breakHoldMin: 5, anchoredLine: 'share' }
// CR-DN item 2: a break must hold A minutes (no 1-minute close strictly back inside the box) before the box freezes (⚙;
// a display default, not a tested threshold).
export const BREAK_HOLD_MIN_RANGE = [1, 15]
export const LIVE_MIN_MOVE_RANGE = [2, 12]
// CR-DI amendment 9: an Edges box edge must hold this many minutes before a close beyond it freezes the box (⚙; a display
// default, not a tested threshold).
export const BOX_HOLD_MIN_RANGE = [5, 60]

// utc epoch -> shifted PT epoch. The PT offset is looked up once per UTC hour (offsets only change on the
// hour), not per timestamp: building an Intl.DateTimeFormat per value is what made CR-CH's re-renders costly.
export function makeShifter(shiftFn) {
  const cache = new Map()
  return (utc) => {
    const t = Number(utc)
    if (!Number.isFinite(t)) return NaN
    const hour = Math.floor(t / 3600) * 3600
    let off = cache.get(hour)
    if (off === undefined) {
      off = shiftFn(hour) - hour
      cache.set(hour, off)
    }
    return t + off
  }
}

// Index of the last element of sorted `times` that is <= t, or -1.
export function floorIndex(times, t) {
  let lo = 0
  let hi = times.length - 1
  if (!times.length || t < times[0]) return -1
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2)
    if (times[mid] <= t) lo = mid
    else hi = mid - 1
  }
  return lo
}

const fin = (v) => Number.isFinite(Number(v)) && v !== null

// DC5: CVD from 0 at the anchor bar, accumulating from the next bar; null before the anchor. With no
// anchor the v2 CVD (from the session open) is returned unchanged. CR-DN item 1: with `endTime` (shifted), null
// after the last row at or before it.
export function anchoredCvd(rows, anchorTime, endTime = null) {
  const list = Array.isArray(rows) ? rows : []
  if (anchorTime == null) return list.map((r) => (fin(r.cvd) ? Number(r.cvd) : null))
  const a = floorIndex(list.map((r) => r.time), anchorTime)
  if (a < 0) return list.map(() => null)
  const out = new Array(list.length).fill(null)
  let acc = 0
  out[a] = 0
  for (let i = a + 1; i < list.length; i += 1) {
    if (endTime != null && list[i].time > endTime) break
    if (!fin(list[i].delta)) continue
    acc += Number(list[i].delta)
    out[i] = acc
  }
  return out
}

// DC5 header readout: delta, price change, minutes and contracts per point from the anchor bar to the
// last bar with data. `closes[i]` is the candle close aligned with rows[i]. Null without an anchor. CR-DN item 1:
// with `endTime` (shifted), the last bar with data at or before it.
export function anchorReadout(rows, closes, anchorTime, endTime = null) {
  const list = Array.isArray(rows) ? rows : []
  if (anchorTime == null || !list.length) return null
  const a = floorIndex(list.map((r) => r.time), anchorTime)
  if (a < 0) return null
  let last = -1
  for (let i = list.length - 1; i > a; i -= 1) {
    if (endTime != null && list[i].time > endTime) continue
    if (fin(list[i].delta)) {
      last = i
      break
    }
  }
  const seg = (i) => list[i].seg ?? null
  if (last < 0) return { anchorTime: list[a].time, delta: 0, dPrice: 0, minutes: 0, vol: 0, cpp: null, fromSeg: seg(a), toSeg: seg(a) }
  let delta = 0
  let vol = 0
  for (let i = a + 1; i <= last; i += 1) {
    if (fin(list[i].delta)) delta += Number(list[i].delta)
    if (fin(list[i].vol)) vol += Number(list[i].vol)
  }
  const c0 = Number(closes?.[a])
  const c1 = Number(closes?.[last])
  const dPrice = Number.isFinite(c0) && Number.isFinite(c1) ? c1 - c0 : null
  return {
    anchorTime: list[a].time,
    delta,
    dPrice,
    minutes: Math.round((list[last].time - list[a].time) / 60),
    vol,
    cpp: dPrice == null || Math.abs(dPrice) < 1e-9 ? null : vol / Math.abs(dPrice),
    fromSeg: seg(a),   // 'eth' / 'rth': the caller flags a readout that spans two sources
    toSeg: seg(last),
  }
}

// Rolling Σ(buy − sell) over the last `minutes` minutes ending at each bar (DC6: minutes, also on 5-min
// charts); null for bars with no data.
export function rollingDelta(rows, minutes) {
  const list = Array.isArray(rows) ? rows : []
  const span = Number(minutes) * 60
  const out = new Array(list.length).fill(null)
  let lo = 0
  let acc = 0
  for (let i = 0; i < list.length; i += 1) {
    if (fin(list[i].delta)) acc += Number(list[i].delta)
    while (lo <= i && list[lo].time <= list[i].time - span) {
      if (fin(list[lo].delta)) acc -= Number(list[lo].delta)
      lo += 1
    }
    if (fin(list[i].delta)) out[i] = acc
  }
  return out
}

// CR-DM item 1: one minute's hover numbers from the same row the delta bars draw: volume (all prints, unknown side
// included), delta and the raw delta share (delta ÷ volume; the payload's `share` is its 3-bar EMA).
export function minuteFlow(row) {
  const vol = fin(row?.vol) ? Number(row.vol) : null
  const delta = fin(row?.delta) ? Number(row.delta) : null
  return { vol, delta, share: vol != null && delta != null && vol > 0 ? delta / vol : null }
}

// CR-DM item 2: the pane's slots as price-scale margins of pane 0. Rolling off: CVD, volume (the old rolling slot) and
// delta, exactly the CR-DC margins. Rolling on: the pane is `extra` px taller (App adds rollingExtraHeight) and the
// slots keep their pixel heights: CVD, rolling (its old slot), volume (the slot below), delta.
export const ROLLING_SLOT_FRAC = 0.32 // one slot (0.28) + its gap (0.04) of the pane's base height
// Pane 0 = the panel minus its header row (8 px padding + 24 px chips) and the time axis (28 px), measured at 220 px.
export const FLOW_V3_CHROME_PX = 60
export function rollingExtraHeight(panelHeight) {
  return Math.round(ROLLING_SLOT_FRAC * Math.max(0, Number(panelHeight) - FLOW_V3_CHROME_PX))
}
export function paneSlotMargins(pane0Px, extraPx) {
  const H = Number(pane0Px)
  const X = Number(extraPx) || 0
  if (!(X > 0) || !(H > X)) {
    return { cvd: { top: 0.04, bottom: 0.62 }, mid: { top: 0.42, bottom: 0.30 }, low: null, delta: { top: 0.74, bottom: 0.02 } }
  }
  const B = H - X // the base height: each slot keeps its px
  const m = (a, b) => ({ top: a / H, bottom: (H - b) / H })
  return {
    cvd: m(0.04 * B, 0.38 * B),
    mid: m(0.42 * B, 0.70 * B),
    low: m(0.74 * B, 0.70 * B + X),
    delta: m(0.74 * B + X, 0.98 * B + X),
  }
}

// CR-DP item 7 — adjustable slots. A slot = one price scale's band in pane 0, in px from the pane top. Without a stored
// layout the slots are exactly paneSlotMargins' (today's). A stored layout is the slots' heights as fractions of the
// content height C = H − (n + ½)·G, where G = 0.04 × the base height (H minus the rolling extra): the same top pad (G),
// gaps (G) and bottom pad (½G) as the default, so the pane's total height never changes. Each slot ≥ SLOT_MIN_PX.
export const SLOT_MIN_PX = 24
export function slotOrder(rollingOn) {
  return rollingOn ? ['cvd', 'mid', 'low', 'delta'] : ['cvd', 'mid', 'delta']
}
export function slotGapPx(pane0Px, extraPx) {
  return 0.04 * (Number(pane0Px) - (Number(extraPx) || 0))
}
// [{ top, bottom }] px of each slot from paneSlotMargins-shaped margins (in slotOrder).
export function marginsToBounds(pane0Px, margins, order) {
  const H = Number(pane0Px)
  return order.map((k) => ({ top: margins[k].top * H, bottom: (1 - margins[k].bottom) * H }))
}
export function boundsToMargins(pane0Px, bounds, order) {
  const H = Number(pane0Px)
  const out = { cvd: null, mid: null, low: null, delta: null }
  order.forEach((k, i) => { out[k] = { top: bounds[i].top / H, bottom: (H - bounds[i].bottom) / H } })
  return out
}
// Bounds from stored fractions (null / wrong length → null: use the default). Slots below the minimum are raised to
// it (taken from the largest) when the pane is big enough.
export function boundsFromFracs(pane0Px, fracs, gapPx, minPx = SLOT_MIN_PX) {
  const H = Number(pane0Px)
  const n = Array.isArray(fracs) ? fracs.length : 0
  if (!(n >= 2) || !(H > 0) || fracs.some((f) => !(Number(f) > 0))) return null
  const G = Number(gapPx)
  const C = H - (n + 0.5) * G
  if (!(C > 0)) return null
  const sum = fracs.reduce((a, f) => a + Number(f), 0)
  const h = fracs.map((f) => (Number(f) / sum) * C)
  if (C >= n * minPx) {
    for (let i = 0; i < n; i += 1) {
      if (h[i] >= minPx) continue
      const need = minPx - h[i]
      h[i] = minPx
      const j = h.indexOf(Math.max(...h))
      h[j] -= need
    }
  }
  const out = []
  let y = G
  for (let i = 0; i < n; i += 1) {
    out.push({ top: y, bottom: y + h[i] })
    y += h[i] + G
  }
  return out
}
// Move the divider between slot k and slot k+1 by dy px (down = slot k grows): both stay ≥ minPx, the rest unchanged.
// Returns the new heights as fractions of their sum (what is stored).
export function dragSlotFracs(bounds, k, dy, minPx = SLOT_MIN_PX) {
  const h = bounds.map((b) => b.bottom - b.top)
  const d = Math.max(minPx - h[k], Math.min(Number(dy) || 0, h[k + 1] - minPx))
  h[k] += d
  h[k + 1] -= d
  const sum = h.reduce((a, x) => a + x, 0)
  return h.map((x) => x / sum)
}

// CR-DP item 8b — a slot's y zoom: its auto range ({ minValue, maxValue }) scaled by `f` (> 1 stretches) about 0 when 0
// is in range (bars, delta, delta share), else about the middle (raw CVD). The view only.
export function zoomRange(range, f) {
  if (!range || !(f > 0) || f === 1) return range
  const { minValue: lo, maxValue: hi } = range
  const a = lo <= 0 && hi >= 0 ? 0 : (lo + hi) / 2
  return { minValue: a + (lo - a) / f, maxValue: a + (hi - a) / f }
}

// CR-DN item 6: the volume bars' scale cap = the 98th percentile (nearest rank) of the loaded rows' positive volumes
// (null with no volume). Bars are drawn at min(vol, cap) and the scale keeps autoscaling, so it tops out at
// min(visible max, cap): a stretch with no minute above the cap draws as before. A display scale, no research meaning.
export const VOLUME_CAP_Q = 0.98
export function volumeCap(rows, q = VOLUME_CAP_Q) {
  const v = (Array.isArray(rows) ? rows : []).map((r) => Number(r?.vol)).filter((x) => Number.isFinite(x) && x > 0).sort((a, b) => a - b)
  if (!v.length) return null
  return v[Math.min(v.length - 1, Math.max(0, Math.ceil(q * v.length) - 1))]
}

// CR-DM item 3: the live effort label for the leg in progress (the payload's `live_leg`: swing_legs' end state with
// leg_effort's numbers). Drawn only once the leg has moved at least `minMove` pts and a previous leg the same way exists
// (prev_cpp); else null. A display aid, not a tested threshold.
export const LIVE_EFFORT_ID = 'cr-dm-live-effort'
export function liveEffortLabel(live, minMove) {
  if (!live || !fin(live.pts) || !fin(live.end_time)) return null
  const pts = Number(live.pts)
  if (Math.abs(pts) < Number(minMove) || live.prev_cpp == null) return null
  const arrow = live.dir === 'up' ? '↑' : '↓'
  const ratio = fin(live.ratio) ? `${Number(live.ratio).toFixed(1)}×` : '–'
  return { leg: live, up: live.dir === 'up', text: `${arrow} ${pts >= 0 ? '+' : '-'}${Math.abs(pts).toFixed(1)} pts so far · ${ratio} vs last ${arrow}` }
}

// Flatten the cells payload's segments into shifted-time arrays (once per payload).
export function flattenCells(payload, shift) {
  const segs = Array.isArray(payload?.segments) ? payload.segments : []
  const out = { t: [], p: [], buy: [], sell: [], seg: [], segments: segs }
  segs.forEach((s, si) => {
    const c = s?.cells || {}
    const n = Array.isArray(c.t) ? c.t.length : 0
    for (let i = 0; i < n; i += 1) {
      out.t.push(shift(c.t[i]))
      out.p.push(Number(c.p[i]))
      out.buy.push(Number(c.buy[i]) || 0)
      out.sell.push(Number(c.sell[i]) || 0)
      out.seg.push(si)
    }
  })
  return out
}

function upperBound(sorted, v) {
  let lo = 0
  let hi = sorted.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (sorted[mid] <= v) lo = mid + 1
    else hi = mid
  }
  return lo
}

// Opacity from a value's percentile in its distribution (DC8): ALPHA_MIN .. ALPHA_MIN + ALPHA_SPAN.
export function percentileAlpha(sorted, v) {
  if (!(v > 0) || !sorted.length) return 0
  return ALPHA_MIN + ALPHA_SPAN * (upperBound(sorted, v) / sorted.length)
}

// One column per candle (DC15: cells summed per candle), at 0.25 pt (res 1) and 0.5 pt (res 2). Each
// column holds price ticks (cell floor) and per-side opacities from that side's per-segment percentile.
export function buildHeatColumns(flat, candleTimes, stepSec) {
  const times = Array.isArray(candleTimes) ? candleTimes : []
  const result = {}
  for (const res of [1, 2]) {
    const acc = new Map() // key `${idx}|${seg}|${p}` -> [buy, sell]
    for (let i = 0; i < flat.t.length; i += 1) {
      const idx = floorIndex(times, flat.t[i])
      if (idx < 0) continue
      const end = idx + 1 < times.length ? Math.min(times[idx + 1], times[idx] + stepSec) : times[idx] + stepSec
      if (flat.t[i] >= end) continue
      const p = Math.floor(flat.p[i] / res) * res
      const key = `${idx}|${flat.seg[i]}|${p}`
      const v = acc.get(key)
      if (v) {
        v[0] += flat.buy[i]
        v[1] += flat.sell[i]
      } else acc.set(key, [flat.buy[i], flat.sell[i]])
    }
    const dist = new Map() // seg -> {buy: [], sell: []}
    for (const [key, [b, s]] of acc) {
      const seg = key.split('|')[1]
      if (!dist.has(seg)) dist.set(seg, { buy: [], sell: [] })
      if (b > 0) dist.get(seg).buy.push(b)
      if (s > 0) dist.get(seg).sell.push(s)
    }
    for (const d of dist.values()) {
      d.buy.sort((x, y) => x - y)
      d.sell.sort((x, y) => x - y)
    }
    const cols = new Array(times.length).fill(null)
    const tmp = new Map()
    for (const [key, [b, s]] of acc) {
      const [idxS, seg, pS] = key.split('|')
      const idx = Number(idxS)
      const d = dist.get(seg)
      if (!tmp.has(idx)) tmp.set(idx, [])
      tmp.get(idx).push([Number(pS), percentileAlpha(d.buy, b), percentileAlpha(d.sell, s), b, s])
    }
    for (const [idx, list] of tmp) {
      list.sort((x, y) => x[0] - y[0])
      cols[idx] = {
        p: Int32Array.from(list, (x) => x[0]),
        ba: Float32Array.from(list, (x) => x[1]),
        sa: Float32Array.from(list, (x) => x[2]),
        buy: Float64Array.from(list, (x) => x[3]),
        sell: Float64Array.from(list, (x) => x[4]),
      }
    }
    result[res] = cols
  }
  return result
}

// Volume profile (section 4): per-tick buy / sell totals over cells with shifted time in [from, to).
export function profileTotals(flat, fromShifted, toShifted = Infinity) {
  const m = new Map()
  for (let i = 0; i < flat.t.length; i += 1) {
    if (flat.t[i] < fromShifted || flat.t[i] >= toShifted) continue
    const v = m.get(flat.p[i])
    if (v) {
      v.buy += flat.buy[i]
      v.sell += flat.sell[i]
    } else m.set(flat.p[i], { buy: flat.buy[i], sell: flat.sell[i] })
  }
  return [...m.entries()].map(([p, v]) => ({ p, ...v })).sort((a, b) => a.p - b.p)
}

// Merge profile rows to res ticks (DC7 crowding).
export function mergeRows(rows, res) {
  if (res === 1) return rows
  const m = new Map()
  for (const r of rows) {
    const p = Math.floor(r.p / res) * res
    const v = m.get(p)
    if (v) {
      v.buy += r.buy
      v.sell += r.sell
    } else m.set(p, { p, buy: r.buy, sell: r.sell })
  }
  return [...m.values()].sort((a, b) => a.p - b.p)
}

// Absorption marks for the chart (A1.2: present only on exact segments), snapped to their candle; the
// slate line runs to the break bar or the segment end (now / `until`). CR-DG: `heldMin` (the hover) runs from the
// mark's minute to its break, else to the payload's last bar — the server's `held_min` stops at the segment end,
// which is 13:00 PT on a live session.
export function chartMarks(payload, candleTimes, shift) {
  const now = payloadNow(payload)
  const out = []
  for (const s of Array.isArray(payload?.segments) ? payload.segments : []) {
    if (!Array.isArray(s.marks)) continue
    for (const mk of s.marks) {
      const st = shift(mk.time)
      const idx = floorIndex(candleTimes, st)
      if (idx < 0) continue
      const endUtc = mk.break_time != null ? mk.break_time : s.end
      const endShifted = shift(endUtc)
      const stop = mk.break_time != null ? Number(mk.break_time) : now ?? Number(s.end)
      const heldMin = Math.max(0, Math.round((stop - Number(mk.time)) / 60))
      out.push({ ...mk, heldMin, idx, candleTime: candleTimes[idx], endShifted, endIdx: Math.max(idx, floorIndex(candleTimes, endShifted)) })
    }
  }
  return out
}

// DC4 (A1.1): "anchor = last swing" — the end of the last confirmed swing leg (start of the current,
// unconfirmed leg), snapped to its candle; null when no leg is confirmed yet.
export function lastSwingAnchor(legs, candleTimes, shift) {
  let best = null
  for (const leg of Array.isArray(legs) ? legs : []) {
    const t = Number(leg?.end_time)
    if (Number.isFinite(t) && (best == null || t > best)) best = t
  }
  if (best == null) return null
  const idx = floorIndex(candleTimes, shift(best))
  return idx < 0 ? null : candleTimes[idx]
}

// "anchor = session open": the earliest segment start of the selected session (06:30 PT in RTH view,
// 15:00 PT the day before in FULL view), snapped to its candle.
export function sessionOpenAnchor(segments, candleTimes, shift) {
  let best = null
  for (const s of Array.isArray(segments) ? segments : []) {
    const t = Number(s?.start)
    if (Number.isFinite(t) && (best == null || t < best)) best = t
  }
  if (best == null) return null
  const st = shift(best)
  const i = candleTimes.findIndex((ct) => ct >= st)
  return i < 0 ? null : candleTimes[i]
}

export const kfmt = (v) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(1)}k` : `${Math.round(v)}`)

// ---------------------------------------------------------------------------------------------------------
// CR-DF — absorption levels and the edge readout (exact data only). Inputs are the cells payload (UTC epochs);
// outputs carry shifted times / candle indexes for the primitives. "Now" is the end of the payload's last
// 1-minute bar (live: the last completed data; REVIEW replay: `until`, since the payload is cut there).

export const LEVELS_MAX = 8 // DF3
export const LEVEL_FADE_MIN = 60 // DF2: opacity 1 at 0 min -> LEVEL_ALPHA_MIN at 60 min, then flat
export const LEVEL_ALPHA_MIN = 0.25
// CR-DN item 5c: an unbroken level is *drawn* fading to this floor (broken levels keep LEVEL_ALPHA_MIN). The selection
// score keeps levelAlpha with LEVEL_ALPHA_MIN, so the drawn set of LEVELS_MAX levels is unchanged.
export const LEVEL_ALPHA_MIN_UNBROKEN = 0.5
export const EDGE_BREAK_PTS = 2 // DF4 (CR-CP A1.1 / A1.2): only a close > 2 pts beyond moves the edge
export const EDGE_NEAR_PTS = 1 // contracts and first touch within 1 pt of the edge

// Line width (px) from contracts absorbed: log scale, 1 px at <= 50, 5 px at >= 1,500 (Step 0: p50 350, max 1,426).
export function levelWidth(contracts) {
  const c = Number(contracts)
  if (!(c > 50)) return 1
  return Math.min(5, 1 + (4 * Math.log(c / 50)) / Math.log(30))
}

export function levelAlpha(ageMin, floor = LEVEL_ALPHA_MIN) {
  const a = Math.max(0, Number(ageMin) || 0)
  return 1 - (1 - floor) * (Math.min(a, LEVEL_FADE_MIN) / LEVEL_FADE_MIN)
}

// End of the last 1-minute bar in the payload (UTC epoch), or null.
export function payloadNow(payload) {
  let now = null
  for (const s of Array.isArray(payload?.segments) ? payload.segments : []) {
    const t = s?.bars?.t
    if (Array.isArray(t) && t.length) {
      const e = Math.min(Number(t[t.length - 1]) + 60, Number(s.end))
      if (now == null || e > now) now = e
    }
  }
  return now
}

// CR-DG: an open hover `{ [field]: obj, pt }` re-reads its mark / level from a new list (same kind, price and
// `idKey`), so held minutes update on a refresh while the pointer is still. Unchanged object = same hover; gone
// from the new list = null (hover closed).
export function relinkHover(hover, field, list, idKey) {
  const old = hover?.[field]
  if (!old || !Array.isArray(list)) return null
  const hit = list.find((x) => x.kind === old.kind && x[idKey] === old[idKey] && x.price === old.price)
  if (!hit) return null
  return hit === old ? hover : { ...hover, [field]: hit }
}

// DF1–DF3: one level per absorption mark that is confirmed by now; it starts at the mark's minute and ends at
// its break bar or now; opacity fades with age; the top LEVELS_MAX by contracts x opacity are kept.
export function absorptionLevels(payload, candleTimes, shift, max = LEVELS_MAX) {
  const now = payloadNow(payload)
  if (now == null) return []
  const all = []
  for (const s of Array.isArray(payload?.segments) ? payload.segments : []) {
    if (!s?.exact || !Array.isArray(s.marks)) continue
    for (const mk of s.marks) {
      if (!(Number(mk.confirm_time) <= now)) continue
      const start = Number(mk.time)
      const broken = mk.break_time != null && Number(mk.break_time) <= now
      const end = broken ? Number(mk.break_time) : now
      const contracts = Number(mk.aggressor) + (Number(mk.merged_vol) || 0)
      const ageMin = (now - start) / 60
      const alpha = levelAlpha(ageMin)
      const held = Math.max(0, Math.round((end - start) / 60))
      const side = mk.kind === 'sellers_absorbed' ? 'sellers' : 'buyers'
      all.push({
        kind: mk.kind, price: Number(mk.price), start, end, broken, contracts, opposite: Number(mk.opposite),
        heldMin: held, ageMin, alpha, width: levelWidth(contracts), score: contracts * alpha,
        drawAlpha: broken ? alpha : levelAlpha(ageMin, LEVEL_ALPHA_MIN_UNBROKEN), // CR-DN item 5c
        label: `${side} absorbed ${kfmt(contracts)} · ${held} min`,
      })
    }
  }
  all.sort((a, b) => b.score - a.score || b.start - a.start)
  const kept = all.slice(0, max).sort((a, b) => a.start - b.start || a.price - b.price)
  const times = Array.isArray(candleTimes) ? candleTimes : []
  const out = []
  for (const lv of kept) {
    const st = shift(lv.start)
    const idx = floorIndex(times, st)
    if (idx < 0) continue
    const endShifted = shift(lv.end)
    out.push({ ...lv, startShifted: st, endShifted, idx, endIdx: Math.max(idx, floorIndex(times, endShifted)) })
  }
  return out
}

// The payload's 1-minute bars across segments, time-ordered, with each bar's segment exactness.
export function flattenBars(payload) {
  const rows = []
  for (const s of Array.isArray(payload?.segments) ? payload.segments : []) {
    const b = s?.bars
    const n = Array.isArray(b?.t) ? b.t.length : 0
    for (let i = 0; i < n; i += 1) rows.push([Number(b.t[i]), Number(b.h[i]), Number(b.l[i]), Number(b.c[i]), Boolean(s.exact)])
  }
  rows.sort((x, y) => x[0] - y[0])
  return {
    t: rows.map((r) => r[0]), h: rows.map((r) => r[1]), l: rows.map((r) => r[2]), c: rows.map((r) => r[3]),
    exact: rows.map((r) => r[4]),
  }
}

// Cells of [t0, t1] (UTC minute starts, inclusive) within EDGE_NEAR_PTS of `top` / `bottom`, per minute, both sides
// at each edge: topBuy / bottomSell are the readout's sides, topSell / bottomBuy the opposite ones (CR-DI).
export function edgeCounts(payload, t0, t1, top, bottom) {
  const near = EDGE_NEAR_PTS / TICK
  const topT = Math.round(top / TICK)
  const botT = Math.round(bottom / TICK)
  const per = new Map() // minute -> [top buy, top sell, bottom sell, bottom buy]
  for (const s of Array.isArray(payload?.segments) ? payload.segments : []) {
    const c = s?.cells || {}
    const n = Array.isArray(c.t) ? c.t.length : 0
    for (let i = 0; i < n; i += 1) {
      const t = Number(c.t[i])
      if (t < t0 || t > t1) continue
      const p = Number(c.p[i])
      const atTop = Math.abs(p - topT) <= near
      const atBottom = Math.abs(p - botT) <= near
      if (!atTop && !atBottom) continue
      let m = per.get(t)
      if (!m) {
        m = [0, 0, 0, 0]
        per.set(t, m)
      }
      const b = Number(c.buy[i]) || 0
      const sl = Number(c.sell[i]) || 0
      if (atTop) {
        m[0] += b
        m[1] += sl
      }
      if (atBottom) {
        m[2] += sl
        m[3] += b
      }
    }
  }
  const ts = [...per.keys()].sort((x, y) => x - y)
  const col = (k) => ts.map((t) => per.get(t)[k])
  return { t: ts, topBuy: col(0), topSell: col(1), bottomSell: col(2), bottomBuy: col(3) }
}

const total = (a) => a.reduce((x, v) => x + v, 0)

// CR-DN item 2: one cancelled break (a probe) from break bar `p.i` to the bar before the close back inside (`back`):
// the extreme beyond the edge, minutes outside, and the aggressor contracts of cells strictly beyond the edge over those
// minutes (null when not exact). `box` = the start (UTC) of the box it belongs to.
function probeStats(payload, bars, p, back, edge, box, exact) {
  const up = p.dir === 'up'
  let ext = up ? -Infinity : Infinity
  for (let k = p.i; k < back; k += 1) ext = up ? Math.max(ext, bars.h[k]) : Math.min(ext, bars.l[k])
  let buy = null
  let sell = null
  if (exact) {
    buy = 0
    sell = 0
    const e = Math.round(edge / TICK)
    const t0 = bars.t[p.i]
    const t1 = bars.t[back - 1]
    for (const s of Array.isArray(payload?.segments) ? payload.segments : []) {
      const c = s?.cells || {}
      const n = Array.isArray(c.t) ? c.t.length : 0
      for (let j = 0; j < n; j += 1) {
        const t = Number(c.t[j])
        if (t < t0 || t > t1) continue
        const pr = Number(c.p[j])
        if (up ? pr > e : pr < e) {
          buy += Number(c.buy[j]) || 0
          sell += Number(c.sell[j]) || 0
        }
      }
    }
  }
  return { at: bars.t[p.i], dir: p.dir, edge, extreme: ext, minutesOutside: back - p.i, back: bars.t[back], buy, sell, box }
}

// One box over bars [j0, j1] with its final edges `e` as edgeReadout's loop left them (CR-DN item 2: a probe's closes
// never move an edge, so the edges can't be re-derived from the closes): { top, bottom, topMoved, bottomMoved,
// topMoves, bottomMoves } (growth moves); first touch within EDGE_NEAR_PTS; contracts from edgeCounts against the final
// edges (null when not exact).
function boxStats(payload, bars, j0, j1, exact, e) {
  const { top, bottom, topMoved, bottomMoved, topMoves, bottomMoves } = e
  let topTouch = null
  let bottomTouch = null
  for (let i = j0; i <= j1 && (topTouch == null || bottomTouch == null); i += 1) {
    if (topTouch == null && bars.h[i] >= top - EDGE_NEAR_PTS - 1e-9) topTouch = bars.t[i]
    if (bottomTouch == null && bars.l[i] <= bottom + EDGE_NEAR_PTS + 1e-9) bottomTouch = bars.t[i]
  }
  const minutes = exact ? edgeCounts(payload, bars.t[j0], bars.t[j1], top, bottom) : null
  const sum = (k) => (minutes ? total(minutes[k]) : null)
  // CR-DQ amendment 1: the box's full high–low range over its bars (wicks), for the verdict's absorption distance
  let high = -Infinity
  let low = Infinity
  for (let i = j0; i <= j1; i += 1) {
    high = Math.max(high, bars.h[i])
    low = Math.min(low, bars.l[i])
  }
  return {
    from: bars.t[j0], to: bars.t[j1], minutes, high, low,
    top: { price: top, contracts: sum('topBuy'), opposite: sum('topSell'), firstTouch: topTouch, brokeAt: topMoved, breaks: topMoves },
    bottom: { price: bottom, contracts: sum('bottomSell'), opposite: sum('bottomBuy'), firstTouch: bottomTouch, brokeAt: bottomMoved, breaks: bottomMoves },
  }
}

// DF4 edge readout from the anchor (shifted candle time) to the last bar, as a box. The edges start at the box's
// first minute's close; a close up to EDGE_BREAK_PTS beyond an edge is a test (edge unchanged). CR-DI amendment 9:
// a close more than EDGE_BREAK_PTS beyond an edge moves that edge to it (the box grows) while the edge has stood
// less than `holdMin` minutes since it was set; once it has stood `holdMin` minutes, such a close is a break: the box
// freezes (`frozen`, oldest first, each with `breakAt` and `dir` 'up' / 'down') and a new box starts at the break bar
// with top = bottom = its close (both edges set there). A box spans its first bar to the bar before its break, so no
// minute is counted twice. The returned object is the current box: contracts = aggressor volume of cells within
// EDGE_NEAR_PTS of its edges since it started (buys at the top, sells at the bottom; `opposite` = the other side),
// only when every bar since the anchor is on an exact segment (else null and `exact` false); `minutes` are the same
// cells per minute (CR-DI strip), so the contracts are their sums. First touch: the first minute of the box whose high
// (top) / low (bottom) came within EDGE_NEAR_PTS of the edge; `brokeAt` / `breaks` on an edge = its growth moves.
// `holdMin` is a display default (⚙), not a tested threshold. Null without an anchor or bars. CR-DN item 1: with
// `endShifted`, bars after it are ignored (the last box ends on the last bar at or before it; `end` = that bar).
// CR-DN item 2 (replaces the single-close break): such a close starts a *pending* break; the box freezes only if no
// close comes back strictly inside it (bottom < close < top) within `breakHoldMin` (A) bars, and then the freeze and
// the new box's start are the original break bar. A close back inside cancels it: the box is unchanged and the break
// is a probe (`probes` on its box: break bar, extreme, minutes outside, contracts beyond the edge). While pending,
// nothing grows or breaks; `pending` is the unresolved break at the last bar (or the end), else null.
export function edgeReadout(payload, bars, anchorShifted, shift, holdMin = DEFAULT_V3_SETTINGS.boxHoldMin, endShifted = null,
  breakHoldMin = DEFAULT_V3_SETTINGS.breakHoldMin) {
  if (anchorShifted == null || !bars?.t?.length) return null
  let i0 = -1
  for (let i = 0; i < bars.t.length; i += 1) {
    if (shift(bars.t[i]) >= anchorShifted) {
      i0 = i
      break
    }
  }
  if (i0 < 0) return null
  let last = bars.t.length - 1
  if (endShifted != null) while (last > i0 && shift(bars.t[last]) > endShifted) last -= 1
  let exact = true
  for (let i = i0; i <= last; i += 1) if (!bars.exact[i]) exact = false
  const hold = Math.max(0, Number(holdMin) || 0) * 60
  const accept = Math.max(1, Math.round(Number(breakHoldMin) || 1))
  const spans = [] // [first bar, last bar, dir of the break that ended it]
  const probes = [] // CR-DN item 2: cancelled breaks, oldest first
  let pending = null // CR-DN item 2: { i: break bar, dir }
  let j0 = i0
  let top = bars.c[i0]
  let bottom = bars.c[i0]
  let topSet = bars.t[i0]
  let bottomSet = bars.t[i0]
  let moves = { topMoved: null, bottomMoved: null, topMoves: 0, bottomMoves: 0 }
  const edgesNow = () => ({ top, bottom, ...moves })
  for (let i = i0 + 1; i <= last; i += 1) {
    const c = bars.c[i]
    if (pending) {
      // Strictly inside (Amendment 1): a close on the edge is not back inside. Inside within A bars = a probe.
      if (c < top - 1e-9 && c > bottom + 1e-9) {
        probes.push(probeStats(payload, bars, pending, i, pending.dir === 'up' ? top : bottom, bars.t[j0], exact))
        pending = null
        continue
      }
      if (i - pending.i < accept) continue
      // held: the box freezes on the bar before the break bar, the new box starts at the break bar and replays from it
      const ib = pending.i
      spans.push([j0, ib - 1, pending.dir, edgesNow()])
      pending = null
      j0 = ib
      top = bars.c[ib]
      bottom = bars.c[ib]
      topSet = bars.t[ib]
      bottomSet = bars.t[ib]
      moves = { topMoved: null, bottomMoved: null, topMoves: 0, bottomMoves: 0 }
      i = ib
      continue
    }
    const up = c > top + EDGE_BREAK_PTS + 1e-9
    const down = c < bottom - EDGE_BREAK_PTS - 1e-9
    if ((up && bars.t[i] - topSet >= hold) || (down && bars.t[i] - bottomSet >= hold)) {
      pending = { i, dir: up ? 'up' : 'down' }
      continue
    }
    if (up) {
      top = c
      topSet = bars.t[i]
      moves = { ...moves, topMoved: bars.t[i], topMoves: moves.topMoves + 1 }
    }
    if (down) {
      bottom = c
      bottomSet = bars.t[i]
      moves = { ...moves, bottomMoved: bars.t[i], bottomMoves: moves.bottomMoves + 1 }
    }
  }
  const tLast = bars.t[last]
  const probesOf = (from) => probes.filter((p) => p.box === from)
  const frozen = spans.map(([a, z, dir, e]) => {
    const box = { ...boxStats(payload, bars, a, z, exact, e), probes: probesOf(bars.t[a]) }
    const end = bars.t[z + 1]
    const held = (e) => (e.firstTouch == null ? null : Math.round((end - e.firstTouch) / 60))
    return {
      ...box, breakAt: end, dir, heldMin: Math.round((end - box.from) / 60),
      top: { ...box.top, heldMin: held(box.top) }, bottom: { ...box.bottom, heldMin: held(box.bottom) },
    }
  })
  const cur = { ...boxStats(payload, bars, j0, last, exact, edgesNow()), probes: probesOf(bars.t[j0]) }
  const held = (e) => (e.firstTouch == null ? null : Math.round((tLast - e.firstTouch) / 60))
  return {
    ...cur, anchor: bars.t[i0], exact, holdMin: Number(holdMin), breakHoldMin: accept, frozen, end: endShifted == null ? null : bars.t[last],
    lastClose: bars.c[last], // CR-DQ item 7: where price sits in the current box (its last bar's close)
    pending: pending ? { at: bars.t[pending.i], dir: pending.dir, edge: pending.dir === 'up' ? top : bottom } : null,
    startDir: frozen.length ? frozen[frozen.length - 1].dir : null,
    top: { ...cur.top, heldMin: held(cur.top) }, bottom: { ...cur.bottom, heldMin: held(cur.bottom) },
  }
}

// CR-DI: the boxes the chart draws (the last FROZEN_BOXES_MAX frozen + the current), oldest first, and how many
// older frozen boxes were dropped from the drawing.
export const FROZEN_BOXES_MAX = 4
export function drawnBoxes(r, max = FROZEN_BOXES_MAX) {
  if (!r) return { boxes: [], dropped: 0 }
  const frozen = Array.isArray(r.frozen) ? r.frozen : []
  const kept = frozen.slice(Math.max(0, frozen.length - max))
  return { boxes: [...kept, r], dropped: frozen.length - kept.length }
}

// CR-DI strip series: the per-minute cells of the drawn boxes, each against its own box, concatenated in time with
// `box` = the box's index in `boxes`. Null when not exact.
export function edgeEffortSeries(r, max = FROZEN_BOXES_MAX) {
  if (!r?.exact) return null
  const { boxes, dropped } = drawnBoxes(r, max)
  const out = { t: [], topBuy: [], topSell: [], bottomSell: [], bottomBuy: [], box: [], boxes, dropped }
  boxes.forEach((bx, k) => {
    const m = bx.minutes
    if (!m) return
    for (let i = 0; i < m.t.length; i += 1) {
      out.t.push(m.t[i])
      out.topBuy.push(m.topBuy[i])
      out.topSell.push(m.topSell[i])
      out.bottomSell.push(m.bottomSell[i])
      out.bottomBuy.push(m.bottomBuy[i])
      out.box.push(k)
    }
  })
  return out
}

// CR-DI visits: at each edge, a run of minutes with a non-zero main bar (top buy / bottom sell), joined across up
// to `gap` empty minutes. Each visit sums both sides over its minutes [start, end] (the opposite side includes the
// visit's quiet minutes). `of[e][i]` = visit index of minute i at edge e (or -1); `run[e][i]` = [main, opp] summed
// from the visit's start through minute i. Input: edgeReadout(...).minutes or edgeEffortSeries(...) (its `box` keeps
// visits inside one box).
export const VISIT_GAP_MIN = 2
export function edgeVisits(minutes, gap = VISIT_GAP_MIN) {
  const t = Array.isArray(minutes?.t) ? minutes.t : []
  const box = Array.isArray(minutes?.box) ? minutes.box : null // CR-DI amendment 9: visits never join across a break
  const one = (main, opp) => {
    const visits = []
    let cur = null
    for (let i = 0; i < t.length; i += 1) {
      if (!(main[i] > 0)) continue
      const sameBox = !box || box[i] === box[cur?.j1]
      if (cur && sameBox && t[i] - t[cur.j1] <= (gap + 1) * 60) cur.j1 = i
      else {
        cur = { j0: i, j1: i }
        visits.push(cur)
      }
    }
    const of = new Array(t.length).fill(-1)
    const run = new Array(t.length).fill(null)
    const out = visits.map((v, k) => {
      let a = 0
      let b = 0
      for (let i = v.j0; i <= v.j1; i += 1) {
        a += main[i]
        b += opp[i]
        of[i] = k
        run[i] = [a, b]
      }
      return { start: t[v.j0], end: t[v.j1], main: a, opp: b }
    })
    return { visits: out, of, run }
  }
  const top = one(minutes?.topBuy || [], minutes?.topSell || [])
  const bottom = one(minutes?.bottomSell || [], minutes?.bottomBuy || [])
  return { top: top.visits, bottom: bottom.visits, of: { top: top.of, bottom: bottom.of }, run: { top: top.run, bottom: bottom.run } }
}

// CR-DI strip columns: edgeReadout(...).minutes bucketed onto the pane's candles (shifted times; one minute per
// candle on the 1-minute chart). cols: Map candle idx -> [top buy, top sell, bottom sell, bottom buy]; idxOf[i] =
// candle idx of minute i (or -1); max = the largest single bar, the strip's scale.
export function edgeStripColumns(minutes, candleTimes, shift) {
  const t = Array.isArray(minutes?.t) ? minutes.t : []
  const times = Array.isArray(candleTimes) ? candleTimes : []
  const cols = new Map()
  const idxOf = new Array(t.length).fill(-1)
  for (let i = 0; i < t.length; i += 1) {
    const idx = floorIndex(times, shift(t[i]))
    if (idx < 0) continue
    idxOf[i] = idx
    let c = cols.get(idx)
    if (!c) {
      c = [0, 0, 0, 0]
      cols.set(idx, c)
    }
    c[0] += minutes.topBuy[i]
    c[1] += minutes.topSell[i]
    c[2] += minutes.bottomSell[i]
    c[3] += minutes.bottomBuy[i]
  }
  let max = 0
  for (const c of cols.values()) for (const v of c) if (v > max) max = v
  return { cols, idxOf, max }
}

// CR-DI: the strip is drawn (and, amendment 2, the Flow v3 pane grows by EDGE_STRIP_HEIGHT) only with the chip on,
// Edges on, an anchor and an exact readout. One condition for App (pane height) and the panel (the strip's chart pane).
export const EDGE_STRIP_HEIGHT = 64
export function edgeStripDrawn(edgeEffortOn, edgesOn, anchor, edges) {
  return Boolean(edgeEffortOn && !edgeEffortBlock(edgesOn, anchor, edges) && edges?.exact && edges?.minutes)
}

// CR-DI: why the Edge effort chip can't act (its tooltip), or '' when it can.
export function edgeEffortBlock(edgesOn, anchor, edges) {
  if (!edgesOn) return 'Edge effort needs Edges on'
  if (anchor == null) return 'Edge effort needs an anchor (📍 set anchor, session open or last swing)'
  if (edges && !edges.exact) return 'Edge effort is drawn on exact data only: the bars since the anchor include approximate (1-s) cells, which place each second\'s selling at its low and buying at its high'
  return ''
}

export const dirArrow = (dir) => (dir === 'up' ? '↑' : '↓')

// "Top 7724.75 · 1.8k buyers · held 9 m | Bottom 7715.25 · 4.1k sellers · held 6 m" (+ " · moved hh:mm" for an edge's
// last growth move). CR-DI: after a break, prefixed "Box from hh:mm ↑ · " (the current box only).
export function edgeText(r, hhmm) {
  if (!r) return ''
  const one = (name, e, who) => {
    const vol = e.contracts == null ? 'exact data only' : `${kfmt(e.contracts)} ${who}`
    const held = e.heldMin == null ? '' : ` · held ${e.heldMin} m`
    const moved = e.brokeAt == null ? '' : ` · moved ${hhmm(e.brokeAt)}`
    return `${name} ${e.price.toFixed(2)} · ${vol}${held}${moved}`
  }
  const box = r.frozen?.length ? `Box from ${hhmm(r.from)} ${dirArrow(r.startDir)} · ` : ''
  const testing = r.pending ? `testing break ${hhmm(r.pending.at)} ${dirArrow(r.pending.dir)} · ` : '' // CR-DN item 2
  return `${testing}${box}${one('Top', r.top, 'buyers')} | ${one('Bottom', r.bottom, 'sellers')}`
}

// CR-DN item 3 — boxes as segments. The drawn boxes (drawnBoxes: the last FROZEN_BOXES_MAX frozen + the current) as
// CVD segments in shifted time, oldest first: from = the box's first bar, to = its last bar (a frozen box ends on the
// bar before its break; the current one on the last bar or the end). `dropped` = older boxes whose segments aren't drawn.
export function boxSegments(r, shift, max = FROZEN_BOXES_MAX) {
  if (!r) return null
  const { boxes, dropped } = drawnBoxes(r, max)
  return { spans: boxes.map((b) => ({ from: shift(b.from), to: shift(b.to) })), dropped }
}

// The candle close aligned with each pane row (rows[i].time = shifted candle time), as the anchored readout uses.
export function rowCloses(rows, candles, shift) {
  const m = new Map()
  for (const c of Array.isArray(candles) ? candles : []) m.set(shift(c.time), Number(c.close))
  return (Array.isArray(rows) ? rows : []).map((r) => m.get(r.time))
}

// Per-segment anchored CVD through the same anchoredCvd: each span restarts at 0 on its first bar and runs to its last
// bar; rows outside every span are null. `gapAt` = the last row of each span but the newest: the pane draws the line
// from there to the next span's 0 transparent, so segments don't join.
export function segmentCvd(rows, spans) {
  const list = Array.isArray(rows) ? rows : []
  const out = new Array(list.length).fill(null)
  const owner = new Array(list.length).fill(-1)
  ;(spans || []).forEach((sp, k) => {
    const c = anchoredCvd(list, sp.from, sp.to)
    for (let i = 0; i < c.length; i += 1) {
      if (c[i] == null) continue
      out[i] = c[i] // a later span's 0 overwrites the previous span's last value on a shared (5-min) row
      owner[i] = k
    }
  })
  const gapAt = new Set()
  for (let i = 0; i + 1 < list.length; i += 1) if (owner[i] >= 0 && owner[i + 1] >= 0 && owner[i + 1] !== owner[i]) gapAt.add(i)
  return { cvd: out, gapAt }
}

// CR-DP item 6a — cumulative delta share from a segment start: at each row after the start bar, Σ delta ÷ Σ volume over
// the bars after the start bar up to that row (the bars anchorReadout sums, so a segment's last value = readoutShare of
// its readout). The start bar itself is null (nothing summed yet), as are rows without delta, rows after `endTime` and
// rows where Σ volume is still 0. Also returns the running sums for the hover. A display aid, no research meaning.
export function anchoredShare(rows, anchorTime, endTime = null) {
  const list = Array.isArray(rows) ? rows : []
  const share = new Array(list.length).fill(null)
  const sumDelta = new Array(list.length).fill(null)
  const sumVol = new Array(list.length).fill(null)
  if (anchorTime == null) return { share, sumDelta, sumVol }
  const a = floorIndex(list.map((r) => r.time), anchorTime)
  if (a < 0) return { share, sumDelta, sumVol }
  let d = 0
  let v = 0
  for (let i = a + 1; i < list.length; i += 1) {
    if (endTime != null && list[i].time > endTime) break
    if (fin(list[i].vol)) v += Number(list[i].vol)
    if (!fin(list[i].delta)) continue
    d += Number(list[i].delta)
    sumDelta[i] = d
    sumVol[i] = v
    share[i] = v > 0 ? d / v : null
  }
  return { share, sumDelta, sumVol }
}

// Per-segment delta share (Edges on: one segment per drawn box, as segmentCvd): each span restarts at its own start
// bar; `gapAt` = each span's last row with a value when the next row with a value belongs to another span: the chart
// would join them across the empty start bar, so the pane draws that segment transparent (as segmentCvd).
export function segmentShare(rows, spans) {
  const list = Array.isArray(rows) ? rows : []
  const share = new Array(list.length).fill(null)
  const sumDelta = new Array(list.length).fill(null)
  const sumVol = new Array(list.length).fill(null)
  const owner = new Array(list.length).fill(-1)
  ;(spans || []).forEach((sp, k) => {
    const s = anchoredShare(list, sp.from, sp.to)
    for (let i = 0; i < list.length; i += 1) {
      if (s.share[i] == null && s.sumDelta[i] == null) continue
      share[i] = s.share[i]
      sumDelta[i] = s.sumDelta[i]
      sumVol[i] = s.sumVol[i]
      owner[i] = k
    }
    const a = floorIndex(list.map((r) => r.time), sp.from)
    if (a >= 0) { share[a] = null; sumDelta[a] = null; sumVol[a] = null; owner[a] = -1 } // a new span's start bar
  })
  const gapAt = new Set()
  let prev = -1
  for (let i = 0; i < list.length; i += 1) {
    if (share[i] == null) continue
    if (prev >= 0 && owner[i] !== owner[prev]) gapAt.add(prev)
    prev = i
  }
  return { share, sumDelta, sumVol, gapAt }
}

// CR-DP item 6b — the share line's fixed scale (±10 %) and guides (0, ±2 %, ±5 %); a value beyond the scale is drawn at
// ±SHARE_SCALE with a cap mark, its hover keeps the true value.
export const SHARE_SCALE = 0.10
export const SHARE_GUIDES = [0, 0.02, -0.02, 0.05, -0.05]
export const clipShare = (v) => Math.max(-SHARE_SCALE, Math.min(SHARE_SCALE, v))

// The anchored readout of each span (anchorReadout over [from, to]), in span order.
export function segmentReadouts(rows, closes, spans) {
  return (spans || []).map((sp) => anchorReadout(rows, closes, sp.from, sp.to))
}

// Box label: "Δ +1.2k · +7.5 pts · 45 m · 1.8k/pt". CR-DP item 4: built from boxLabelParts, which the popup shares.
export function boxLabelParts(ro) {
  const d = Number(ro.delta) || 0
  return {
    delta: `Δ ${d >= 0 ? '+' : '-'}${kfmt(Math.abs(d))}`,
    pts: ro.dPrice == null ? '–' : `${ro.dPrice >= 0 ? '+' : ''}${ro.dPrice.toFixed(1)} pts`,
    minutes: `${ro.minutes} m`,
    cpp: `${ro.cpp == null ? '–' : kfmt(ro.cpp)}/pt`,
  }
}
export function boxLabelText(ro) {
  if (!ro) return ''
  const p = boxLabelParts(ro)
  return `${p.delta} · ${p.pts} · ${p.minutes} · ${p.cpp}`
}

// CR-DP items 4 / 6 — delta share = Σ delta ÷ Σ volume over the readout's bars (anchorReadout's `vol`: the same volume
// field as the CR-DM volume bars); null without volume. Shown as a percentage with two decimals. A display aid.
export function readoutShare(ro) {
  return ro && Number(ro.vol) > 0 ? Number(ro.delta) / Number(ro.vol) : null
}
export const sharePctText = (v) => (v == null || !Number.isFinite(v) ? '–' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(2)} %`)

// CR-DP item 4 — one edge's state for the box stats popup: "testing break hh:mm ↑" while a break at that edge is pending,
// else "moved hh:mm" (its last growth move) and / or "probe hh:mm" (its last cancelled break, "(n)" when several); ''
// when none. Same fields as the box hover and edgeText.
export function edgeStateText(box, side, hhmm) {
  const dir = side === 'top' ? 'up' : 'down'
  if (box.pending && box.pending.dir === dir) return `testing break ${hhmm(box.pending.at)} ${dirArrow(dir)}`
  const e = box[side]
  const probes = (box.probes || []).filter((p) => p.dir === dir)
  const out = []
  if (e?.brokeAt != null) out.push(`moved ${hhmm(e.brokeAt)}`)
  if (probes.length) out.push(`probe ${hhmm(probes[probes.length - 1].at)}${probes.length > 1 ? ` (${probes.length})` : ''}`)
  return out.join(' · ')
}

// CR-DP item 4 — the popup's three lines for the current box (PriceChart's v3Edges box: drawnBoxes' box + its segment
// readout + ended / pending). `hhmm` formats a UTC epoch as PT hh:mm.
//   "box from 06:49 → 07:10 · Δ +1.2k · share +1.02 % · +7.5 pts · 45 m · 1.8k/pt"
//   "Top 7724.75 · buyers 1.8k / sellers 900 · held 9 m · probe 07:34"
//   "Bottom 7715.25 · sellers 4.1k / buyers 2.0k · held 6 m"
// CR-DP amendment item 2: " · mixed sources (A + B)" when a readout spans two segments whose data differ, by source
// (Databento / Ironbeam) or by exactness (exact / approx 1-s cells); '' otherwise. `segments`: a payload's segments
// ({ name, source, exact? }) — the cells payload (source and exactness) or the v2 payload (source only).
const SOURCE_LABEL = { databento: 'Databento', ironbeam: 'Ironbeam' }
export function mixedSourcesText(readout, segments) {
  if (!readout || readout.fromSeg == null || readout.fromSeg === readout.toSeg) return ''
  const by = Object.fromEntries((Array.isArray(segments) ? segments : []).map((g) => [g.name, g]))
  const a = by[readout.fromSeg]
  const b = by[readout.toSeg]
  if (!a?.source || !b?.source) return ''
  const name = (g) => `${SOURCE_LABEL[g.source] || g.source}${g.exact === false ? ' approx' : ''}`
  return name(a) !== name(b) ? ` · mixed sources (${name(a)} + ${name(b)})` : ''
}

export function boxPopupLines(box, hhmm, segments = null) {
  if (!box) return null
  const ro = box.readout
  const n = (v) => (v == null ? 'exact data only' : kfmt(v))
  const held = (e) => (e.heldMin == null ? 'not touched' : `held ${e.heldMin} m`)
  const end = box.ended?.kind === 'end' ? ` → ${hhmm(box.ended.at)}` : ''
  const p = ro ? boxLabelParts(ro) : null
  const stats = p ? ` · ${p.delta} · share ${sharePctText(readoutShare(ro))} · ${p.pts} · ${p.minutes} · ${p.cpp}` : ''
  const edge = (name, side, main, opp) => {
    const e = box[side]
    const st = edgeStateText(box, side, hhmm)
    return `${name} ${e.price.toFixed(2)} · ${main} ${n(e.contracts)} / ${opp} ${n(e.opposite)} · ${held(e)}${st ? ` · ${st}` : ''}`
  }
  return {
    head: `box from ${hhmm(box.from)}${end}${stats}`,
    // CR-DQ item 5: the head's pieces, so the panel can put a percentile badge after the share and after contracts/pt
    // (joined with ' · ' they are `head`)
    parts: p ? [`box from ${hhmm(box.from)}${end}`, p.delta, `share ${sharePctText(readoutShare(ro))}`, p.pts, p.minutes, p.cpp]
      : [`box from ${hhmm(box.from)}${end}`],
    top: edge('Top', 'top', 'buyers', 'sellers'),
    bottom: edge('Bottom', 'bottom', 'sellers', 'buyers'),
    mixed: ro ? mixedSourcesText(ro, segments).replace(/^ · /, '') : '',
  }
}

// CR-DQ item 2 — the box panel: one section per drawn box (PriceChart's v3Edges.boxes, oldest first, the current last),
// each from the same box fields as boxPopupLines (one source with CR-DP's popup and the removed box hover).
//   summary (one line): "Box 10:12–10:41 · broke ↓ 10:41 · share -2.39 %" (frozen) / "Box 10:41–11:00 · current"
//   lines: boxPopupLines (head / top / bottom / mixed); probes: "2 probes · testing break 10:58 ↑"
// `dropped` = older frozen boxes not drawn ("n older boxes not shown").
export function boxPanelSections(entries, hhmm, segments = null, pctOf = null) {
  const list = Array.isArray(entries) ? entries : []
  const sections = list.map((en) => {
    const b = en.box
    const frozen = b.ended?.kind === 'broke'
    const until = frozen ? b.ended.at : b.to
    const state = frozen ? `broke ${dirArrow(b.ended.dir)} ${hhmm(b.ended.at)}`
      : b.ended?.kind === 'end' ? `set end ${hhmm(b.ended.at)}` : 'current'
    const share = b.readout ? ` · share ${sharePctText(readoutShare(b.readout))}` : ''
    const n = b.probes?.length || 0
    const pend = b.pending ? ` · testing break ${hhmm(b.pending.at)} ${dirArrow(b.pending.dir)}${b.ended?.kind === 'end' ? ' (pending at the end)' : ''}` : ''
    return {
      key: String(b.from), current: !frozen,
      summary: `Box ${hhmm(b.from)}–${hhmm(until)} · ${state}${frozen ? share : ''}`,
      lines: boxPopupLines(b, hhmm, segments),
      probes: `${n} probe${n === 1 ? '' : 's'}${pend}`,
      pct: pctOf ? pctOf(b) : null, // CR-DQ item 5: boxPercentiles of this box (null: no badge)
    }
  })
  const dropped = list.length ? Number(list[list.length - 1].box.dropped) || 0 : 0
  return { sections, dropped, droppedText: dropped ? `${dropped} older box${dropped === 1 ? '' : 'es'} not shown` : '' }
}

// CR-DQ items 5 / 7 — one config object for the panel's display thresholds (⚙ "Percentile tiers" and "Verdict rules";
// display aids, untested): tier cut-offs (percentiles) and the verdict's distances (pts).
export const BOX_RULES_DEFAULTS = { notable: 80, strong: 95, nearPts: 2, progressPts: 2, absorbPts: 2, minAgeMin: 10, minHeightPts: 2 }
export const BOX_RULES_RANGES = { notable: [50, 99], strong: [51, 100], nearPts: [0.25, 10], progressPts: [0, 20], absorbPts: [0, 10], minAgeMin: [0, 120], minHeightPts: [0, 20] }
export function boxRules(stored) {
  const out = { ...BOX_RULES_DEFAULTS }
  for (const [k, [lo, hi]] of Object.entries(BOX_RULES_RANGES)) {
    const v = Number(stored?.[k])
    if (stored?.[k] != null && Number.isFinite(v)) out[k] = Math.min(hi, Math.max(lo, v))
  }
  if (out.strong <= out.notable) out.strong = Math.min(100, out.notable + 1)
  return out
}

// CR-DQ item 5 — percentile badges against the train baseline (scripts/cr_dq_baseline.py → src/data/boxBaseline.json):
// same source as the box (Ryan, Step 0 decision 2), length bucket by the box's minutes, |Δ share| by the box's start
// time of day (decision 1), contracts/pt all day. Percentile = linear interpolation in the 101-point grid (numpy.interp).
export const BOX_PCT_MIN_MINUTES = 3
export function quantilePct(grid, v) {
  const g = Array.isArray(grid) ? grid : []
  const x = Number(v)
  if (g.length < 2 || !Number.isFinite(x)) return null
  const n = g.length - 1
  if (x <= g[0]) return 0
  if (x >= g[n]) return 100
  let lo = 0
  let hi = n
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (g[mid] <= x) lo = mid
    else hi = mid
  }
  const span = g[hi] - g[lo]
  return (100 / n) * (lo + (span > 0 ? (x - g[lo]) / span : 0))
}
export function baselineBucket(baseline, minutes) {
  const m = Number(minutes)
  const bs = baseline?.meta?.buckets || []
  if (!Number.isFinite(m) || m < BOX_PCT_MIN_MINUTES || !bs.length) return null
  return bs.find((b) => m >= b.min && m <= b.max) || (m > bs[bs.length - 1].max ? bs[bs.length - 1] : null)
}
export function baselineTod(baseline, ptMinuteOfDay) {
  const rel = Number(ptMinuteOfDay) - 390 // minutes after 06:30 PT
  const ts = baseline?.meta?.tod || []
  return ts.find((t) => rel >= t.fromMin && rel < t.toMin) || null
}
// `ro` = the box's readout, `ptMinuteOfDay` = its first bar's PT minute of day, `segments` = the payload's segments
// ({ name, source }). Null (no badge) for ETH, a box spanning two segments, an unknown source or under 3 minutes.
export function boxPercentiles(ro, ptMinuteOfDay, segments, baseline) {
  if (!ro || !baseline || ro.fromSeg !== 'rth' || ro.toSeg !== 'rth') return null
  const seg = (Array.isArray(segments) ? segments : []).find((g) => g.name === 'rth')
  const src = seg?.source
  const base = src ? baseline[src] : null
  const bucket = baselineBucket(baseline, ro.minutes)
  const tod = baselineTod(baseline, ptMinuteOfDay)
  if (!base || !bucket || !tod || !(Number(ro.vol) > 0)) return null
  const share = Number(ro.delta) / Number(ro.vol)
  const cpp = Number(ro.vol) / Math.max(Math.abs(Number(ro.dPrice) || 0), baseline.meta.tick || TICK)
  return {
    source: src, bucket: bucket.name, bucketRange: [bucket.min, bucket.max], tod: tod.name,
    sessions: base.sessions, first: base.first, last: base.last, excluded: baseline.meta.excluded || [],
    share: { value: share, p: quantilePct(base.absShare?.[tod.name]?.[bucket.name], Math.abs(share)) },
    cpp: { value: cpp, p: quantilePct(base.cpp?.[bucket.name], cpp) },
  }
}
export function pctTier(p, rules = BOX_RULES_DEFAULTS) {
  if (p == null) return null
  return p >= rules.strong ? 'strong' : p >= rules.notable ? 'notable' : 'normal'
}
export const pctText = (p) => (p == null ? '' : `p${Math.min(99, Math.floor(p))}`)
const TOD_TEXT = { open: '06:30–07:30', mid: '07:30–11:00', late: '11:00–13:00' }
// The badge's hover: what the percentile means, the bucket, the baseline sessions and date range.
export function pctBadgeTitle(kind, pc, rules = BOX_RULES_DEFAULTS) {
  const src = SOURCE_LABEL[pc.source] || pc.source
  const [lo, hi] = pc.bucketRange
  const len = hi >= 240 ? `${lo}+ min` : `${lo}–${hi} min`
  const p = kind === 'share' ? pc.share.p : pc.cpp.p
  const what = kind === 'share'
    ? `|Δ share| is above ${Math.floor(p)} % of RTH windows of ${len} starting ${TOD_TEXT[pc.tod] || pc.tod} PT`
    : `contracts per point is above ${Math.floor(p)} % of RTH windows of ${len} (all day)`
  const noise = kind === 'share'
    ? ' The same window read from Ironbeam and from Databento differs by a median ~9 percentile points on Δ share, so a badge near a cut-off can flip.'
    : ''
  return `${pctText(p)}: this box's ${what}, in the ${src} baseline (${pc.sessions} train sessions ${pc.first} → ${pc.last}; `
    + `FOMC days and the March 2026 roll week left out; same source as the box, no adjustment).${noise} `
    + `Tiers: notable ≥ p${rules.notable}, strong ≥ p${rules.strong}. Display aid — untested.`
}

// CR-DQ item 7 — verdict line for the current box (Ryan's draft rules applied mechanically; a display aid, untested).
// FOMC decision days (A22): the last day of each scheduled 2026 meeting, from federalreserve.gov/monetarypolicy/
// fomccalendars.htm (checked 2026-10-31), until an events table exists.
export const FOMC_2026_DECISION_DAYS = ['2026-01-28', '2026-03-18', '2026-04-29', '2026-06-17', '2026-07-29', '2026-09-16', '2026-10-28', '2026-12-09']
const levelSide = (lv) => (lv.kind === 'sellers_absorbed' ? 'sellers' : 'buyers')
// Absorption levels (absorptionLevels: the chart's top 8) standing at the box's last bar: started at or before it and not
// broken by then. CR-DQ amendment 1: distances are to the box's full high–low range (wicks; `box.high` / `box.low`, the
// close-based edges when absent): `where` = 'top' / 'bottom' (within absorbPts of the high / low, the nearer if both) or
// 'in' (between them), else not counted.
export function boxLevels(levels, box, lastT, absorbPts) {
  const hi = Number.isFinite(box.high) ? box.high : box.top
  const lo = Number.isFinite(box.low) ? box.low : box.bottom
  const out = []
  for (const lv of Array.isArray(levels) ? levels : []) {
    if (!(lv.start <= lastT) || (lv.broken && lv.end <= lastT)) continue
    const nearTop = Math.abs(lv.price - hi) <= absorbPts + 1e-9
    const nearBottom = Math.abs(lv.price - lo) <= absorbPts + 1e-9
    const inside = lv.price <= hi + 1e-9 && lv.price >= lo - 1e-9
    const where = nearTop && (!nearBottom || Math.abs(lv.price - hi) <= Math.abs(lv.price - lo)) ? 'top'
      : nearBottom ? 'bottom' : inside ? 'in' : null
    if (where) out.push({ ...lv, side: levelSide(lv), where })
  }
  return out
}
// Verdict, checked in order (spec item 7b + Ryan's Step 0 decision 3):
//   NO TRADES (A22) → CONFLICT (A24) → NO ACTION two-sided absorption → SHORT / LONG READ (A19 + A23) → WATCH BREAK (idea)
//   → NO ACTION nothing notable.
// `box` = the current box (top / bottom, readout, lastClose, to), `pct` = boxPercentiles (or null), `levels` =
// absorptionLevels, `hhmm` formats a UTC epoch as PT hh:mm. Returns { kind, label, evidence, rules, title }.
export function boxVerdict({ box: b0, pct, levels, tradeDate, rules: r0 = null, hhmm = String, fomc = FOMC_2026_DECISION_DAYS }) {
  const r = boxRules(r0)
  // edges as prices: the panel's boxes carry edge objects ({ price, contracts, … }), tests may pass numbers
  const edgePx = (e) => Number(e != null && typeof e === 'object' ? e.price : e)
  const box = b0 ? { ...b0, top: edgePx(b0.top), bottom: edgePx(b0.bottom) } : null
  const ro = box?.readout || null
  const title = (ids) => `Rules: ${ids.join(' + ') || 'none'}. Thresholds: notable ≥ p${r.notable} (Δ share and contracts/pt), `
    + `near an edge ≤ ${r.nearPts} pts, no advance ≤ ${r.progressPts} pts, absorption within ${r.absorbPts} pts of the box's high–low range, `
    + `box formed at ≥ ${r.minAgeMin} min and ≥ ${r.minHeightPts} pts. `
    + 'Order: A22 → box still forming → A24 → two-sided absorption → A19 + A23 reads → watch break (idea) → nothing notable. Draft rules, untested — a display aid.'
  const out = (kind, label, evidence, ids) => ({ kind, label, evidence, rules: ids, title: title(ids) })
  if (tradeDate && fomc.includes(String(tradeDate).slice(0, 10))) {
    return out('no-trades', 'NO TRADES — event day', `FOMC decision day ${String(tradeDate).slice(0, 10)}: no new trades all session`, ['A22'])
  }
  if (!box || !ro) return out('none', 'NO ACTION — nothing notable', 'no box stats yet', [])
  // CR-DQ amendment 2: nothing beyond NO TRADES until the box is old and tall enough (close-based height)
  const age = Number(ro.minutes) || 0
  const height = box.top - box.bottom
  if (age < r.minAgeMin || !(height >= r.minHeightPts - 1e-9)) {
    return out('forming', 'NO ACTION — box still forming',
      `box ${age} min (min ${r.minAgeMin}) · ${Number.isFinite(height) ? height.toFixed(2) : '–'} pts tall (min ${r.minHeightPts})`, [])
  }
  const share = readoutShare(ro)
  const sp = pct?.share?.p ?? null
  const cp = pct?.cpp?.p ?? null
  const lean = share != null && sp != null && sp >= r.notable ? Math.sign(share) : 0
  const shareTxt = `share ${sharePctText(share)}${sp == null ? '' : ` ${pctText(sp)}`}`
  const cppTxt = `c/pt ${cp == null ? '–' : pctText(cp)}`
  const lastT = box.ended?.kind === 'broke' ? box.ended.at : box.to
  const lv = boxLevels(levels, box, lastT, r.absorbPts)
  const lvTxt = (l) => `${l.side} absorbed ${l.price.toFixed(2)} (${hhmm(l.start)})${l.where === 'in' ? ' in the box' : ` at the ${l.where}`}`
  const buyersAbs = lv.filter((l) => l.side === 'buyers')
  const sellersAbs = lv.filter((l) => l.side === 'sellers')
  // A24: flow lean and absorption point opposite ways
  const against = lean > 0 ? sellersAbs : lean < 0 ? buyersAbs : []
  if (against.length) {
    return out('conflict', 'CONFLICT → NO ACTION', `${shareTxt} (${lean > 0 ? 'buyers' : 'sellers'} leaning) vs ${against.map(lvTxt).join(', ')}`, ['A24'])
  }
  // Ryan, Step 0 decision 3 (not A24): absorption on both sides and no notable flow lean
  if (lean === 0 && buyersAbs.length && sellersAbs.length) {
    return out('two-sided', 'NO ACTION — two-sided absorption', `${lvTxt(buyersAbs[0])} · ${lvTxt(sellersAbs[0])}; no notable flow lean (${shareTxt})`, [])
  }
  const px = Number(box.lastClose)
  const dp = Number(ro.dPrice)
  const nearTop = Number.isFinite(px) && box.top - px <= r.nearPts + 1e-9
  const nearBottom = Number.isFinite(px) && px - box.bottom <= r.nearPts + 1e-9
  const effort = cp != null && cp >= r.notable
  // CR-DQ amendment 3: which side of the edge price is on
  const posTxt = (side) => {
    const e = side === 'top' ? box.top : box.bottom
    if (Math.abs(px - e) < 1e-9) return `price at the ${side}`
    return `price ${Math.abs(px - e).toFixed(2)} ${px > e ? 'above' : 'below'} the ${side}`
  }
  const absTop = buyersAbs.filter((l) => l.where === 'top')
  const absBottom = sellersAbs.filter((l) => l.where === 'bottom')
  // A19 + A23: short at the top — buyers leaning and / or buyers absorbed at the top, no advance
  const shortWhy = lean > 0 || absTop.length
  const shortStall = effort || (Number.isFinite(dp) && dp <= r.progressPts + 1e-9)
  if (nearTop && shortWhy && shortStall) {
    const ev = [posTxt('top'), lean > 0 ? `${shareTxt} buyers leaning` : null, ...absTop.map(lvTxt),
      effort ? `${cppTxt} (effort)` : `no advance ${Number.isFinite(dp) ? `${dp >= 0 ? '+' : ''}${dp.toFixed(1)} pts` : ''}`]
    return out('short', 'SHORT READ at top', ev.filter(Boolean).join(' · '), absTop.length ? ['A19', 'A23'] : ['A19'])
  }
  const longWhy = lean < 0 || absBottom.length
  const longStall = effort || (Number.isFinite(dp) && dp >= -r.progressPts - 1e-9)
  if (nearBottom && longWhy && longStall) {
    const ev = [posTxt('bottom'), lean < 0 ? `${shareTxt} sellers leaning` : null, ...absBottom.map(lvTxt),
      effort ? `${cppTxt} (effort)` : `no decline ${Number.isFinite(dp) ? `${dp >= 0 ? '+' : ''}${dp.toFixed(1)} pts` : ''}`]
    return out('long', 'LONG READ at bottom', ev.filter(Boolean).join(' · '), absBottom.length ? ['A19', 'A23'] : ['A19'])
  }
  // idea only: the pushing side winning at an edge (moving easily, nothing absorbing it)
  if (cp != null && cp < r.notable) {
    if (nearTop && lean > 0 && !absTop.length) {
      return out('watch-up', 'WATCH BREAK ↑ (idea)', `${posTxt('top')} · ${shareTxt} · ${cppTxt} (moving easily) · no buyers absorbed at the top — wait for an accepted break`, ['idea'])
    }
    if (nearBottom && lean < 0 && !absBottom.length) {
      return out('watch-down', 'WATCH BREAK ↓ (idea)', `${posTxt('bottom')} · ${shareTxt} · ${cppTxt} (moving easily) · no sellers absorbed at the bottom — wait for an accepted break`, ['idea'])
    }
  }
  const where = !Number.isFinite(px) ? '' : nearTop ? posTxt('top') : nearBottom ? posTxt('bottom') : 'price mid-box'
  const lvs = lv.length ? ` · ${lv.map(lvTxt).join(', ')}` : ''
  return out('none', 'NO ACTION — nothing notable', [where, shareTxt, cppTxt].filter(Boolean).join(' · ') + lvs, [])
}

// CR-DQ item 3 — the panel's place. Stored as px from the plot's left and bottom edges (so it stays bottom-anchored when
// the plot height changes); default bottom-left, 4 px inset. Returns stage px clamped inside the plot.
export const BOX_PANEL_INSET = 4
export function boxPanelPlace(pane, stored, w, h) {
  if (!pane) return null
  const off = stored && Number.isFinite(stored.left) && Number.isFinite(stored.bottom) ? stored : { left: BOX_PANEL_INSET, bottom: BOX_PANEL_INSET }
  const left = pane.left + off.left
  const top = pane.top + pane.height - off.bottom - h
  return {
    left: Math.max(pane.left, Math.min(left, pane.left + pane.width - w)),
    top: Math.max(pane.top, Math.min(top, pane.top + pane.height - h)),
  }
}
export function boxPanelOffset(pane, left, top, h) {
  return { left: left - pane.left, bottom: pane.top + pane.height - (top + h) }
}

// CR-DN item 1 — set end. The picked end (shifted candle time) is stored as the last 1-minute bar inside that candle
// (`stepSec` = the chart's candle step: 60 on the 1-minute chart, 300 on 5-minute), so a 5-minute pick keeps the whole
// candle. Valid only after the anchor and not after the payload's last bar (`lastShifted`: live = now, replay /
// review = the until time); else null (the pick is ignored).
export function endFromPick(t, stepSec, anchor, lastShifted) {
  const v = Number(t)
  if (!Number.isFinite(v) || anchor == null) return null
  const end = v + Math.max(0, (Number(stepSec) || 60) - 60)
  if (!(v > anchor)) return null
  if (lastShifted != null && v > lastShifted) return null
  return lastShifted != null ? Math.min(end, lastShifted) : end
}

// "06:49 → 06:57 · 8 min" for the header (hhmm formats a shifted epoch).
export function endWindowText(anchor, end, hhmm) {
  if (anchor == null || end == null) return ''
  return `${hhmm(anchor)} → ${hhmm(end)} · ${Math.round((end - anchor) / 60)} min`
}
