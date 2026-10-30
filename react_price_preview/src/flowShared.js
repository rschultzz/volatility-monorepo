// CR-DP item 1 — helpers the Flow v3 pane and App share, moved unchanged from the retired Flow Current
// (AggressorFlowPanel) and Flow v2 (AggressorFlowV2Panel) panes: PT-shifted time, logical-range sync, session bands,
// the linked crosshair, the v2-row alignment and the swing effort labels / markers.

function partsForZone(epochSec, timeZone = 'America/Los_Angeles') {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone,
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(new Date(epochSec * 1000))

  const out = {}
  for (const p of parts) {
    if (p.type !== 'literal') out[p.type] = p.value
  }
  return out
}

export function utcEpochShowingZoneTime(originalEpochSec, timeZone = 'America/Los_Angeles') {
  const p = partsForZone(originalEpochSec, timeZone)
  return (
    Date.UTC(
      Number(p.year),
      Number(p.month) - 1,
      Number(p.day),
      Number(p.hour),
      Number(p.minute),
      Number(p.second)
    ) / 1000
  )
}

function shiftedEpochToDate(epochSec) {
  return new Date(epochSec * 1000)
}

export function formatShiftedHHMM(epochSec) {
  const dt = shiftedEpochToDate(epochSec)
  const hh = String(dt.getUTCHours()).padStart(2, '0')
  const mm = String(dt.getUTCMinutes()).padStart(2, '0')
  return `${hh}:${mm}`
}

export function normalizeLogicalRange(range) {
  if (!range) return null
  const from = Number(range.from)
  const to = Number(range.to)
  if (!Number.isFinite(from) || !Number.isFinite(to)) return null
  return { from, to }
}

export function rangesClose(a, b, eps = 0.001) {
  if (!a || !b) return false
  return (
    Math.abs(Number(a.from) - Number(b.from)) <= eps &&
    Math.abs(Number(a.to) - Number(b.to)) <= eps
  )
}

export function normalizeShiftedCandleTimes(candles) {
  const seen = new Set()
  const out = []

  for (const bar of Array.isArray(candles) ? candles : []) {
    const rawTime = Number(bar?.time)
    if (!Number.isFinite(rawTime)) continue
    const shifted = utcEpochShowingZoneTime(rawTime, 'America/Los_Angeles')
    if (seen.has(shifted)) continue
    seen.add(shifted)
    out.push(shifted)
  }

  out.sort((a, b) => a - b)
  return out
}

export function inferStepSeconds(times, fallback = 60) {
  if (!Array.isArray(times) || times.length < 2) return fallback

  const diffs = []
  for (let i = 1; i < times.length && diffs.length < 24; i += 1) {
    const diff = Number(times[i]) - Number(times[i - 1])
    if (Number.isFinite(diff) && diff > 0) diffs.push(diff)
  }

  if (!diffs.length) return fallback
  diffs.sort((a, b) => a - b)
  return diffs[Math.floor(diffs.length / 2)]
}

function isRthShiftedEpoch(epochSec) {
  const hhmm = formatShiftedHHMM(epochSec)
  return hhmm >= '06:30' && hhmm <= '13:00'
}

function shiftedDateKey(epochSec) {
  const dt = shiftedEpochToDate(epochSec)
  const yyyy = dt.getUTCFullYear()
  const mm = String(dt.getUTCMonth() + 1).padStart(2, '0')
  const dd = String(dt.getUTCDate()).padStart(2, '0')
  return `${yyyy}-${mm}-${dd}`
}

export function computeSessionBands(chart, shiftedTimes, viewportWidth) {
  if (!chart || !Array.isArray(shiftedTimes) || !shiftedTimes.length) {
    return []
  }

  const width = Math.max(0, Number(viewportWidth) || 0)
  if (!width) return []

  const groups = []
  let current = null

  for (const time of shiftedTimes) {
    if (typeof time !== 'number') continue
    if (!isRthShiftedEpoch(time)) continue

    const key = shiftedDateKey(time)

    if (!current || current.key !== key) {
      current = { key, times: [time] }
      groups.push(current)
    } else {
      current.times.push(time)
    }
  }

  if (!groups.length) return []

  const barSpacing = chart.timeScale().options().barSpacing || 6
  const out = []

  for (const group of groups) {
    const firstTime = group.times[0]
    const lastTime = group.times[group.times.length - 1]
    const startX = chart.timeScale().timeToCoordinate(firstTime)
    const endX = chart.timeScale().timeToCoordinate(lastTime)

    if (startX == null || endX == null) continue

    const rawLeft = Math.min(startX, endX) - barSpacing * 0.5
    const rawRight = Math.max(startX, endX) + barSpacing * 0.5

    if (!Number.isFinite(rawLeft) || !Number.isFinite(rawRight)) continue
    if (rawRight <= 0 || rawLeft >= width) continue

    const clippedLeft = Math.max(0, rawLeft)
    const clippedRight = Math.min(width, rawRight)
    const clippedWidth = clippedRight - clippedLeft

    if (!(clippedWidth > 0)) continue

    out.push({
      key: group.key,
      left: clippedLeft,
      width: clippedWidth,
    })
  }

  return out
}

export function coordinateForLinkedCrosshair(chart, linkedCrosshair) {
  if (!chart || !linkedCrosshair) return null

  const logical = Number(linkedCrosshair.logical)
  if (Number.isFinite(logical) && typeof chart.timeScale().logicalToCoordinate === 'function') {
    const x = chart.timeScale().logicalToCoordinate(logical)
    if (Number.isFinite(x)) return x
  }

  const shiftedTime = Number(linkedCrosshair.shiftedTime)
  if (Number.isFinite(shiftedTime)) {
    const x = chart.timeScale().timeToCoordinate(shiftedTime)
    if (Number.isFinite(x)) return x
  }

  return null
}

const BUY_RGBA = (alpha) => `rgba(96,165,250,${alpha})`
const SELL_RGBA = (alpha) => `rgba(229,231,235,${alpha})`

export const DEFAULT_MARKER_SETTINGS = { show: false, swing: 8, push: 500, div: 300, fade: 0.1, gap: 30, effortRatio: 1.5 }

// One entry per candle time (shifted PT epochs), so the pane stays index-aligned with the price chart.
// 1-min candles take the server's minute values; wider candles sum delta / volume, take the last CVD,
// recompute delta share from the sums (A1 D6) and mark absorption if any minute absorbed.
export function alignV2ToCandles(bars, candles) {
  const times = normalizeShiftedCandleTimes(candles)
  const pts = (Array.isArray(bars) ? bars : [])
    .map((b) => ({ ...b, st: utcEpochShowingZoneTime(Number(b.time)) }))
    .filter((b) => Number.isFinite(b.st))
    .sort((a, b) => a.st - b.st)
  if (!times.length) return []
  const step = inferStepSeconds(times, 60)
  const out = []
  let j = 0
  for (let i = 0; i < times.length; i += 1) {
    const start = times[i]
    const end = i + 1 < times.length ? Math.min(times[i + 1], start + step) : start + step
    while (j < pts.length && pts[j].st < start) j += 1
    const bucket = []
    let k = j
    while (k < pts.length && pts[k].st < end) {
      bucket.push(pts[k])
      k += 1
    }
    j = k
    if (!bucket.length) {
      out.push({ time: start })
      continue
    }
    const sum = (key) => bucket.reduce((acc, b) => acc + (Number(b[key]) || 0), 0)
    const delta = sum('delta')
    const vol = sum('vol')
    const last = bucket[bucket.length - 1]
    out.push({
      time: start,
      delta,
      vol,
      cvd: Number(last.cvd),
      share: bucket.length === 1 ? last.share : vol > 0 ? delta / vol : null,
      absorb: bucket.some((b) => b.absorb),
      seg: last.seg,
    })
  }
  return out
}

// CR-CD A5 — swing effort labels: "+12.0 · 18.4k · 1.5k/pt · ×2.1" (pts · volume · contracts per point ·
// contracts per point vs the previous leg in the same direction; the ratio is left out when it is null).
const kfmt = (v) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(1)}k` : `${Math.round(v)}`)

export function effortLabel(leg) {
  const pts = Number(leg.pts)
  const parts = [`${pts >= 0 ? '+' : '-'}${Math.abs(pts).toFixed(1)}`, kfmt(Number(leg.vol)),
    leg.cpp == null ? '–/pt' : `${kfmt(Number(leg.cpp))}/pt`]
  if (leg.ratio != null) parts.push(`×${Number(leg.ratio).toFixed(1)}`)
  return parts.join(' · ')
}

export function effortIsHot(leg, threshold) {
  return leg.ratio != null && Number(leg.ratio) >= Number(threshold)
}

// Price-chart markers (time = shifted PT epoch of the leg's end bar; the chart snaps it to its candles).
// Pane palette (A1.5): blue up legs, light grey down legs; dimmed unless the ratio reaches the threshold.
export function effortChartMarkers(legs, threshold = DEFAULT_MARKER_SETTINGS.effortRatio) {
  return (Array.isArray(legs) ? legs : []).map((leg) => {
    const up = leg.dir === 'up'
    const hot = effortIsHot(leg, threshold)
    return {
      time: utcEpochShowingZoneTime(Number(leg.end_time)),
      position: up ? 'aboveBar' : 'belowBar',
      shape: 'circle',
      color: up ? BUY_RGBA(hot ? 1 : 0.55) : SELL_RGBA(hot ? 1 : 0.5),
      text: effortLabel(leg),
      size: hot ? 1 : 0.5,
    }
  }).filter((m) => Number.isFinite(m.time))
}
