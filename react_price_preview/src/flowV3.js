// CR-DC — Flow v3 pure helpers (no React, no chart): anchored CVD and its readout, rolling net delta,
// heatmap columns with percentile opacity, the volume profile, absorption marks for the chart, and the
// "last swing" anchor. Times: "shifted" = PT wall-clock epochs (the charts' candle times); "utc" = real
// epochs (the server's). Every input array is memoised by the caller, so these run once per data change.

export const TICK = 0.25
export const ALPHA_MIN = 0.08
export const ALPHA_SPAN = 0.62
export const CROWDED_TICK_PX = 3 // DC7: one tick shorter than this -> 0.5-pt cells
export const ROLLING_CHOICES = [5, 10, 15]
export const DEFAULT_V3_SETTINGS = { rollingMin: 10, k: 3, m: 3, sellLayer: true, buyLayer: true, boxHoldMin: 15 }
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
// anchor the v2 CVD (from the session open) is returned unchanged.
export function anchoredCvd(rows, anchorTime) {
  const list = Array.isArray(rows) ? rows : []
  if (anchorTime == null) return list.map((r) => (fin(r.cvd) ? Number(r.cvd) : null))
  const a = floorIndex(list.map((r) => r.time), anchorTime)
  if (a < 0) return list.map(() => null)
  const out = new Array(list.length).fill(null)
  let acc = 0
  out[a] = 0
  for (let i = a + 1; i < list.length; i += 1) {
    if (!fin(list[i].delta)) continue
    acc += Number(list[i].delta)
    out[i] = acc
  }
  return out
}

// DC5 header readout: delta, price change, minutes and contracts per point from the anchor bar to the
// last bar with data. `closes[i]` is the candle close aligned with rows[i]. Null without an anchor.
export function anchorReadout(rows, closes, anchorTime) {
  const list = Array.isArray(rows) ? rows : []
  if (anchorTime == null || !list.length) return null
  const a = floorIndex(list.map((r) => r.time), anchorTime)
  if (a < 0) return null
  let last = -1
  for (let i = list.length - 1; i > a; i -= 1) {
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
export const EDGE_BREAK_PTS = 2 // DF4 (CR-CP A1.1 / A1.2): only a close > 2 pts beyond moves the edge
export const EDGE_NEAR_PTS = 1 // contracts and first touch within 1 pt of the edge

// Line width (px) from contracts absorbed: log scale, 1 px at <= 50, 5 px at >= 1,500 (Step 0: p50 350, max 1,426).
export function levelWidth(contracts) {
  const c = Number(contracts)
  if (!(c > 50)) return 1
  return Math.min(5, 1 + (4 * Math.log(c / 50)) / Math.log(30))
}

export function levelAlpha(ageMin) {
  const a = Math.max(0, Number(ageMin) || 0)
  return 1 - (1 - LEVEL_ALPHA_MIN) * (Math.min(a, LEVEL_FADE_MIN) / LEVEL_FADE_MIN)
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

// One box over bars [j0, j1]: edges from j0's close, moved by closes > EDGE_BREAK_PTS beyond (growth); first touch
// within EDGE_NEAR_PTS; contracts from edgeCounts against the final edges (null when not exact).
function boxStats(payload, bars, j0, j1, exact) {
  let top = bars.c[j0]
  let bottom = bars.c[j0]
  let topMoved = null
  let bottomMoved = null
  let topMoves = 0
  let bottomMoves = 0
  for (let i = j0 + 1; i <= j1; i += 1) {
    const c = bars.c[i]
    if (c > top + EDGE_BREAK_PTS + 1e-9) {
      top = c
      topMoved = bars.t[i]
      topMoves += 1
    }
    if (c < bottom - EDGE_BREAK_PTS - 1e-9) {
      bottom = c
      bottomMoved = bars.t[i]
      bottomMoves += 1
    }
  }
  let topTouch = null
  let bottomTouch = null
  for (let i = j0; i <= j1 && (topTouch == null || bottomTouch == null); i += 1) {
    if (topTouch == null && bars.h[i] >= top - EDGE_NEAR_PTS - 1e-9) topTouch = bars.t[i]
    if (bottomTouch == null && bars.l[i] <= bottom + EDGE_NEAR_PTS + 1e-9) bottomTouch = bars.t[i]
  }
  const minutes = exact ? edgeCounts(payload, bars.t[j0], bars.t[j1], top, bottom) : null
  const sum = (k) => (minutes ? total(minutes[k]) : null)
  return {
    from: bars.t[j0], to: bars.t[j1], minutes,
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
// `holdMin` is a display default (⚙), not a tested threshold. Null without an anchor or bars.
export function edgeReadout(payload, bars, anchorShifted, shift, holdMin = DEFAULT_V3_SETTINGS.boxHoldMin) {
  if (anchorShifted == null || !bars?.t?.length) return null
  let i0 = -1
  for (let i = 0; i < bars.t.length; i += 1) {
    if (shift(bars.t[i]) >= anchorShifted) {
      i0 = i
      break
    }
  }
  if (i0 < 0) return null
  const last = bars.t.length - 1
  let exact = true
  for (let i = i0; i <= last; i += 1) if (!bars.exact[i]) exact = false
  const hold = Math.max(0, Number(holdMin) || 0) * 60
  const spans = [] // [first bar, last bar, dir of the break that ended it]
  let j0 = i0
  let top = bars.c[i0]
  let bottom = bars.c[i0]
  let topSet = bars.t[i0]
  let bottomSet = bars.t[i0]
  for (let i = i0 + 1; i <= last; i += 1) {
    const c = bars.c[i]
    const up = c > top + EDGE_BREAK_PTS + 1e-9
    const down = c < bottom - EDGE_BREAK_PTS - 1e-9
    if ((up && bars.t[i] - topSet >= hold) || (down && bars.t[i] - bottomSet >= hold)) {
      spans.push([j0, i - 1, up ? 'up' : 'down'])
      j0 = i
      top = c
      bottom = c
      topSet = bars.t[i]
      bottomSet = bars.t[i]
      continue
    }
    if (up) {
      top = c
      topSet = bars.t[i]
    }
    if (down) {
      bottom = c
      bottomSet = bars.t[i]
    }
  }
  const tLast = bars.t[last]
  const frozen = spans.map(([a, z, dir]) => {
    const box = boxStats(payload, bars, a, z, exact)
    const end = bars.t[z + 1]
    const held = (e) => (e.firstTouch == null ? null : Math.round((end - e.firstTouch) / 60))
    return {
      ...box, breakAt: end, dir, heldMin: Math.round((end - box.from) / 60),
      top: { ...box.top, heldMin: held(box.top) }, bottom: { ...box.bottom, heldMin: held(box.bottom) },
    }
  })
  const cur = boxStats(payload, bars, j0, last, exact)
  const held = (e) => (e.firstTouch == null ? null : Math.round((tLast - e.firstTouch) / 60))
  return {
    ...cur, anchor: bars.t[i0], exact, holdMin: Number(holdMin), frozen,
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
  return `${box}${one('Top', r.top, 'buyers')} | ${one('Bottom', r.bottom, 'sellers')}`
}
