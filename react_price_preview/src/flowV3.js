// CR-DC — Flow v3 pure helpers (no React, no chart): anchored CVD and its readout, rolling net delta,
// heatmap columns with percentile opacity, the volume profile, absorption marks for the chart, and the
// "last swing" anchor. Times: "shifted" = PT wall-clock epochs (the charts' candle times); "utc" = real
// epochs (the server's). Every input array is memoised by the caller, so these run once per data change.

export const TICK = 0.25
export const ALPHA_MIN = 0.08
export const ALPHA_SPAN = 0.62
export const CROWDED_TICK_PX = 3 // DC7: one tick shorter than this -> 0.5-pt cells
export const ROLLING_CHOICES = [5, 10, 15]
export const DEFAULT_V3_SETTINGS = { rollingMin: 10, k: 3, m: 3, sellLayer: true, buyLayer: true }

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
// slate line runs to the break bar or the segment end (now / `until`).
export function chartMarks(payload, candleTimes, shift) {
  const out = []
  for (const s of Array.isArray(payload?.segments) ? payload.segments : []) {
    if (!Array.isArray(s.marks)) continue
    for (const mk of s.marks) {
      const st = shift(mk.time)
      const idx = floorIndex(candleTimes, st)
      if (idx < 0) continue
      const endUtc = mk.break_time != null ? mk.break_time : s.end
      const endShifted = shift(endUtc)
      out.push({ ...mk, idx, candleTime: candleTimes[idx], endShifted, endIdx: Math.max(idx, floorIndex(candleTimes, endShifted)) })
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
