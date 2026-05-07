import { useEffect, useMemo, useRef, useState } from 'react'
import {
  createChart,
  CrosshairMode,
  ColorType,
  CandlestickSeries,
  LineSeries,
} from 'lightweight-charts'
import SmileChart from './SmileChart'
import SignalPanel from './SignalPanel'
import TradeAnnotationPanel from './TradeAnnotationPanel'

const ETH_BG_COLOR = '#1f2937'
const PRICE_AXIS_HIT_WIDTH = 72
const TIME_AXIS_HEIGHT = 24
const MIN_PRICE_RANGE = 0.25
const MIN_CHART_HEIGHT = 180

const TOOLTIP_OFFSET_X = 14
const TOOLTIP_OFFSET_Y = 14
const TOOLTIP_EDGE_PAD = 8

const LIVE_OVERLAY_UP_BORDER = '#60a5fa'
const LIVE_OVERLAY_DOWN_BORDER = '#e5e7eb'
const LIVE_OVERLAY_UP_FILL = 'rgba(96,165,250,0.18)'
const LIVE_OVERLAY_DOWN_FILL = 'rgba(229,231,235,0.18)'

const EXPECTED_MOVE_COLOR = '#ef4444'

function coerceGexMinAbsB(value, fallback = 10) {
  const num = Number(value)
  if (!Number.isFinite(num)) return fallback
  return Math.max(0, Math.min(200, Math.round(num)))
}

// Shared formatters for GEX popup & panel
function fmtGammaB(v) {
  if (!Number.isFinite(v)) return '—'
  const b = v / 1e9
  const sign = b > 0 ? '+' : (b < 0 ? '' : '')
  return `${sign}${b.toFixed(2)}B`
}

function fmtGexExpDate(iso) {
  const m = String(iso || '').match(/^(\d{4})-(\d{2})-(\d{2})$/)
  return m ? `${m[2]}/${m[3]}` : String(iso || '—')
}

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

function utcEpochShowingZoneTime(originalEpochSec, timeZone = 'America/Los_Angeles') {
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

function pad2(value) {
  return String(value).padStart(2, '0')
}

function formatShiftedHHMM(epochSec) {
  const dt = shiftedEpochToDate(epochSec)
  const hh = String(dt.getUTCHours()).padStart(2, '0')
  const mm = String(dt.getUTCMinutes()).padStart(2, '0')
  return `${hh}:${mm}`
}

function formatShiftedTimestamp(epochSec) {
  if (!Number.isFinite(epochSec)) return '--'
  const dt = shiftedEpochToDate(Math.round(epochSec))
  const mm = pad2(dt.getUTCMonth() + 1)
  const dd = pad2(dt.getUTCDate())
  const hh = pad2(dt.getUTCHours())
  const mi = pad2(dt.getUTCMinutes())
  const ss = pad2(dt.getUTCSeconds())
  return `${mm}/${dd} ${hh}:${mi}:${ss} PT`
}

function toPtHHMM(chartTime, interval) {
  const base = typeof chartTime === 'number' ? chartTime : NaN
  if (!Number.isFinite(base)) return ''
  let epoch = base
  if (interval === '1min') {
    epoch += 60
  }
  return formatShiftedHHMM(epoch)
}

function normalizeTimes(value) {
  if (!Array.isArray(value)) return []
  const seen = new Set()
  const out = []
  for (const item of value) {
    const s = String(item || '').trim()
    if (!s) continue
    if (!/^\d{2}:\d{2}$/.test(s)) continue
    if (seen.has(s)) continue
    seen.add(s)
    out.push(s)
  }
  return out.sort()
}

function applySelectionColors(candles, selectedSet, interval) {
  return candles.map((bar) => {
    const hhmm = toPtHHMM(bar.time, interval)
    if (!selectedSet.has(hhmm)) {
      return {
        ...bar,
        color: undefined,
        borderColor: undefined,
        wickColor: undefined,
      }
    }
    return {
      ...bar,
      color: '#ef4444',
      borderColor: '#ef4444',
      wickColor: '#ef4444',
    }
  })
}

function safeLineWidth(value) {
  const n = Number(value)
  if (!Number.isFinite(n)) return 2
  return Math.max(1, Math.min(5, Math.round(n)))
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

function computeSessionBands(chart, shiftedCandles, viewportWidth) {
  if (!chart || !Array.isArray(shiftedCandles) || !shiftedCandles.length) {
    return []
  }

  const width = Math.max(0, Number(viewportWidth) || 0)
  if (!width) return []

  // timeToCoordinate returns x relative to the chart's main pane, which
  // starts AFTER the left price scale when one is visible. Our band divs
  // are positioned absolutely from the stage's left edge, which includes
  // the axis area — so we need to add the left scale's width back.
  let leftAxisWidth = 0
  try {
    leftAxisWidth = Number(chart.priceScale('left').width()) || 0
  } catch (err) {
    leftAxisWidth = 0
  }

  const groups = []
  let current = null

  for (const bar of shiftedCandles) {
    if (typeof bar?.time !== 'number') continue
    if (!isRthShiftedEpoch(bar.time)) continue

    const key = shiftedDateKey(bar.time)

    if (!current || current.key !== key) {
      current = { key, bars: [bar] }
      groups.push(current)
    } else {
      current.bars.push(bar)
    }
  }

  if (!groups.length) return []

  const barSpacing = chart.timeScale().options().barSpacing || 6
  const out = []

  for (const group of groups) {
    const firstTime = group.bars[0].time
    const lastTime = group.bars[group.bars.length - 1].time
    const startX = chart.timeScale().timeToCoordinate(firstTime)
    const endX = chart.timeScale().timeToCoordinate(lastTime)

    if (startX == null || endX == null) continue

    const rawLeft = Math.min(startX, endX) - barSpacing * 0.5 + leftAxisWidth
    const rawRight = Math.max(startX, endX) + barSpacing * 0.5 + leftAxisWidth

    if (!Number.isFinite(rawLeft) || !Number.isFinite(rawRight)) continue
    if (rawRight <= 0 || rawLeft >= width) continue

    const clippedLeft = Math.max(0, rawLeft)
    const clippedRight = Math.min(width, rawRight)
    const clippedWidth = clippedRight - clippedLeft

    if (!(clippedWidth > 0)) continue

    out.push({
      key: group.key,
      visible: true,
      left: clippedLeft,
      width: clippedWidth,
      startX,
      endX,
    })
  }

  return out
}

function getPlotHeight(container) {
  if (!container) return 400
  return Math.max(80, container.clientHeight - TIME_AXIS_HEIGHT)
}

function clampVisibleRange(range) {
  if (!range) return null
  let from = Number(range.from)
  let to = Number(range.to)
  if (!Number.isFinite(from) || !Number.isFinite(to)) return null
  if (to < from) {
    const tmp = from
    from = to
    to = tmp
  }
  if (to - from < MIN_PRICE_RANGE) {
    const mid = (from + to) / 2
    from = mid - MIN_PRICE_RANGE / 2
    to = mid + MIN_PRICE_RANGE / 2
  }
  return { from, to }
}

function normalizeLogicalRange(range) {
  if (!range) return null
  const from = Number(range.from)
  const to = Number(range.to)
  if (!Number.isFinite(from) || !Number.isFinite(to)) return null
  return { from, to }
}

function rangesClose(a, b, eps = 1) {
  if (!a || !b) return false
  return Math.abs(Number(a.from) - Number(b.from)) <= eps && Math.abs(Number(a.to) - Number(b.to)) <= eps
}

function pointerInfo(evt, container, leftAxisWidth = 0) {
  if (!evt || !container) return null
  const rect = container.getBoundingClientRect()
  const x = evt.clientX - rect.left
  const y = evt.clientY - rect.top
  const overTimeAxis = y > rect.height - TIME_AXIS_HEIGHT
  const overPriceAxis = x >= rect.width - PRICE_AXIS_HIT_WIDTH
  const overLeftPriceAxis = leftAxisWidth > 0 && x >= 0 && x < leftAxisWidth
  return { rect, x, y, overTimeAxis, overPriceAxis, overLeftPriceAxis }
}

function normalizeIntervalValue(value, fallback = '1min') {
  const s = String(value || '').trim().toLowerCase()
  if (s === '5min' || s === '5m') return '5min'
  if (s === '1min' || s === '1m') return '1min'
  return fallback
}

function findParentControlRoot(controlId) {
  try {
    const parentWin = window.parent
    if (!parentWin || parentWin === window) return null
    return parentWin.document.getElementById(controlId)
  } catch (err) {
    return null
  }
}

function findParentControlGroup(controlId, labelText) {
  const root = findParentControlRoot(controlId)
  if (!root) return null

  let node = root
  const normalizedLabel = String(labelText || '').trim().toLowerCase()

  for (let depth = 0; depth < 5 && node; depth += 1) {
    const text = String(node.textContent || '').replace(/\s+/g, ' ').trim().toLowerCase()
    if (normalizedLabel && text.includes(normalizedLabel)) {
      return node
    }
    node = node.parentElement
  }

  return root.parentElement || root
}

function setParentRadioValue(controlId, nextValue) {
  try {
    const root = findParentControlRoot(controlId)
    if (!root) return false

    const selector = `input[type="radio"][value="${String(nextValue)}"]`
    const input = root.querySelector(selector)
    if (!input) return false

    if (!input.checked) {
      input.click()
      input.dispatchEvent(new Event('input', { bubbles: true }))
      input.dispatchEvent(new Event('change', { bubbles: true }))
    }
    return true
  } catch (err) {
    return false
  }
}

function setParentControlGroupVisibility(controlId, labelText, visible) {
  try {
    const group = findParentControlGroup(controlId, labelText)
    if (!group) return false
    group.style.display = visible ? '' : 'none'
    group.style.visibility = visible ? '' : 'hidden'
    group.style.pointerEvents = visible ? '' : 'none'
    return true
  } catch (err) {
    return false
  }
}

function hideParentTopControls() {
  setParentControlGroupVisibility('ironbeam-bar-interval', 'Bar Interval', false)
  setParentControlGroupVisibility('ib-chart-mode-toggle', 'Chart Mode', false)
}

function showParentTopControls() {
  setParentControlGroupVisibility('ironbeam-bar-interval', 'Bar Interval', true)
}

function intervalToSeconds(interval) {
  const s = String(interval || '').trim().toLowerCase()
  if (!s) return null

  if (s === '1min' || s === '1m') return 60
  if (s === '5min' || s === '5m') return 300
  if (s === '15min' || s === '15m') return 900
  if (s === '30min' || s === '30m') return 1800
  if (s === '60min' || s === '60m' || s === '1h') return 3600

  const m = s.match(/^(\d+)\s*(min|m)$/)
  if (m) return Number(m[1]) * 60

  const h = s.match(/^(\d+)\s*(hour|hours|hr|hrs|h)$/)
  if (h) return Number(h[1]) * 3600

  return null
}

function inferStepSeconds(candles, interval) {
  if (Array.isArray(candles) && candles.length >= 2) {
    const diffs = []
    for (let i = 1; i < candles.length && diffs.length < 12; i += 1) {
      const diff = Number(candles[i]?.time) - Number(candles[i - 1]?.time)
      if (Number.isFinite(diff) && diff > 0) diffs.push(diff)
    }
    if (diffs.length) {
      diffs.sort((a, b) => a - b)
      return diffs[Math.floor(diffs.length / 2)]
    }
  }
  return intervalToSeconds(interval) || 60
}

function interpolateShiftedEpoch(logical, candles, interval) {
  if (!Number.isFinite(logical) || !Array.isArray(candles) || !candles.length) {
    return null
  }

  if (candles.length === 1) {
    return Number(candles[0]?.time) || null
  }

  const step = inferStepSeconds(candles, interval)
  const lastIndex = candles.length - 1

  if (logical <= 0) {
    return Number(candles[0].time) + logical * step
  }
  if (logical >= lastIndex) {
    return Number(candles[lastIndex].time) + (logical - lastIndex) * step
  }

  const leftIndex = Math.floor(logical)
  const rightIndex = Math.ceil(logical)
  const frac = logical - leftIndex

  const leftTime = Number(candles[leftIndex]?.time)
  const rightTime = Number(candles[rightIndex]?.time)

  if (Number.isFinite(leftTime) && Number.isFinite(rightTime)) {
    return leftTime + (rightTime - leftTime) * frac
  }
  if (Number.isFinite(leftTime)) {
    return leftTime + frac * step
  }
  if (Number.isFinite(rightTime)) {
    return rightTime - (1 - frac) * step
  }

  return null
}

function logicalRangeToShiftedTimeRange(range, candles, interval) {
  const logicalRange = normalizeLogicalRange(range)
  if (!logicalRange || !Array.isArray(candles) || !candles.length) return null

  const from = interpolateShiftedEpoch(logicalRange.from, candles, interval)
  const to = interpolateShiftedEpoch(logicalRange.to, candles, interval)
  if (!Number.isFinite(from) || !Number.isFinite(to)) return null

  return from <= to ? { from, to } : { from: to, to: from }
}

function countPrependedBars(previousCandles, nextCandles) {
  if (!Array.isArray(previousCandles) || !previousCandles.length) return 0
  if (!Array.isArray(nextCandles) || !nextCandles.length) return 0

  const prevFirstTime = Number(previousCandles[0]?.time)
  if (!Number.isFinite(prevFirstTime)) return 0

  let count = 0
  for (const candle of nextCandles) {
    const t = Number(candle?.time)
    if (!Number.isFinite(t)) continue
    if (t < prevFirstTime) {
      count += 1
      continue
    }
    break
  }
  return count
}

function shiftLogicalRange(range, delta) {
  const logicalRange = normalizeLogicalRange(range)
  const shift = Number(delta)
  if (!logicalRange || !Number.isFinite(shift) || shift === 0) return logicalRange
  return {
    from: logicalRange.from + shift,
    to: logicalRange.to + shift,
  }
}

function clampLogicalToData(logical, candles) {
  const value = Number(logical)
  if (!Number.isFinite(value) || !Array.isArray(candles) || !candles.length) return null
  const lastIndex = candles.length - 1
  if (value < 0) return 0
  if (value > lastIndex) return lastIndex
  return value
}

function normalizeGexSegments(value) {
  if (!Array.isArray(value)) return []
  const out = []
  for (const seg of value) {
    const level = Number(seg?.level)
    const startTime = Number(seg?.start_time)
    const endTime = Number(seg?.end_time)
    if (!Number.isFinite(level) || !Number.isFinite(startTime) || !Number.isFinite(endTime)) {
      continue
    }

    const shiftedStart = utcEpochShowingZoneTime(startTime, 'America/Los_Angeles')
    const shiftedEnd = utcEpochShowingZoneTime(endTime, 'America/Los_Angeles')

    out.push({
      ...seg,
      level,
      start_time: startTime,
      end_time: endTime,
      shiftedStart,
      shiftedEnd,
    })
  }
  return out
}

function normalizeExpectedMoveLevels(value) {
  if (!Array.isArray(value)) return []
  const out = []
  for (const item of value) {
    const upper = Number(item?.upper)
    const lower = Number(item?.lower)
    const startTime = Number(item?.start_time)
    const endTime = Number(item?.end_time)
    if (!Number.isFinite(upper) || !Number.isFinite(lower) || !Number.isFinite(startTime) || !Number.isFinite(endTime)) {
      continue
    }

    const shiftedStart = utcEpochShowingZoneTime(startTime, 'America/Los_Angeles')
    const shiftedEnd = utcEpochShowingZoneTime(endTime, 'America/Los_Angeles')

    out.push({
      ...item,
      upper,
      lower,
      shiftedStart,
      shiftedEnd,
    })
  }
  return out
}

function buildShiftedEpoch(year, month, day, hour = 0, minute = 0, second = 0) {
  return Date.UTC(year, month - 1, day, hour, minute, second) / 1000
}

function computeTradeSessionTimeRange(tradeDate) {
  const raw = String(tradeDate || '').trim()
  const m = raw.match(/^(\d{4})-(\d{2})-(\d{2})$/)
  if (!m) return null

  const year = Number(m[1])
  const month = Number(m[2])
  const day = Number(m[3])
  if (!Number.isFinite(year) || !Number.isFinite(month) || !Number.isFinite(day)) return null

  const baseUtcMs = Date.UTC(year, month - 1, day, 0, 0, 0)
  const prev = new Date(baseUtcMs - 24 * 60 * 60 * 1000)

  return {
    from: buildShiftedEpoch(
      prev.getUTCFullYear(),
      prev.getUTCMonth() + 1,
      prev.getUTCDate(),
      15,
      0,
      0
    ),
    to: buildShiftedEpoch(year, month, day, 13, 0, 0),
    key: `${raw}:1500-1300`,
  }
}

function computeCenterLogicalRange(candles, sessionTimeRange) {
  if (!Array.isArray(candles) || !candles.length) return null

  let firstIdx = -1
  let lastIdx = -1

  for (let i = 0; i < candles.length; i += 1) {
    if (candles[i]?.is_center === true) {
      if (firstIdx === -1) firstIdx = i
      lastIdx = i
    }
  }

  if (firstIdx === -1 || lastIdx === -1) {
    const fromTime = Number(sessionTimeRange?.from)
    const toTime = Number(sessionTimeRange?.to)
    if (Number.isFinite(fromTime) && Number.isFinite(toTime)) {
      for (let i = 0; i < candles.length; i += 1) {
        const candleTime = Number(candles[i]?.time)
        if (!Number.isFinite(candleTime)) continue
        if (candleTime < fromTime || candleTime > toTime) continue
        if (firstIdx === -1) firstIdx = i
        lastIdx = i
      }
    }
  }

  if (firstIdx === -1 || lastIdx === -1) {
    firstIdx = 0
    lastIdx = candles.length - 1
  }

  return {
    from: firstIdx - 0.5,
    to: lastIdx + 0.5,
    key: `${firstIdx}:${lastIdx}:${candles.length}`,
  }
}

export default function PriceChart({
  candles,
  liveTradeCandles,
  tradeDate,
  interval,
  selectedTimes: parentSelectedTimes = [],
  gexSegments,
  gexEnabled,
  gexMinAbsB = 50,
  expectedMoveLevels,
  onApplyGexMinAbsB,
  onApplyIntervalChange,
  onVisibleLogicalRangeChange,
  onLinkedCrosshairChange,
  onInteractionActiveChange,
  skewData = [],
  smileData = null,
  // Snapshots and anchor controls
  smileSnapshots = null,
  activeBandsAnchorTime = null,
  onBandsAnchorChange = null,
  // Computed band levels to draw on chart (ES coords, basis-corrected)
  // Shape: { shiftedStart, shiftedEnd, sigmaUpper, sigmaLower, shortPut, longPut, shortCall, longCall, anchorEs }
  bandsLevels = null,
  // Per-minute 0DTE ATM IV series for the line overlay.
  // Shape: [{ time: "HH:MM", atm_iv_pct: number }, ...]
  atmIvSeries = [],
}) {
  const stageRef = useRef(null)
  const hostRef = useRef(null)
  const chartRef = useRef(null)
  const seriesRef = useRef(null)
  const liveTradeSeriesRef = useRef(null)
  const gexSeriesRefs = useRef([])
  // Map<LineSeries, segment> — lets the click handler look up which segment was hit
  const gexSegmentBySeriesRef = useRef(new Map())
  const expectedMoveSeriesRefs = useRef([])
  const bandsSeriesRefs = useRef([])
  const atmIvSeriesRefs = useRef([])
  // Tracks whether the IV scale is currently mounted/visible. Used by the
  // render effect to apply `autoScale: true` only on the initial enable —
  // subsequent re-runs (polling, new bars, interval changes) must not reset
  // the user's manual zoom/pan on the IV scale.
  const atmIvScaleVisibleRef = useRef(false)
  const intervalRef = useRef(interval)
  const shiftedCandlesRef = useRef([])
  const dragRef = useRef({ active: false, lastY: 0 })
  const hasUserInteractedRef = useRef(false)
  const currentCenterKeyRef = useRef('')
  const appliedInitialRangeKeyRef = useRef('')
  const lastReportedRangeRef = useRef(null)
  const lastManualTimeRangeRef = useRef(null)
  const suppressRangeReportRef = useRef(0)
  const previousDisplayCandlesRef = useRef([])

  const tooltipRef = useRef(null)
  const tooltipTimeRef = useRef(null)
  const tooltipPriceRef = useRef(null)
  const interactionReleaseTimerRef = useRef(null)
  const interactionActiveRef = useRef(false)
  const previousLiveTradeCandlesRef = useRef([])

  const [localSelectedTimes, setLocalSelectedTimes] = useState(normalizeTimes(parentSelectedTimes))
  const selectedTimes = useMemo(() => normalizeTimes(parentSelectedTimes), [parentSelectedTimes])

  const [sessionBands, setSessionBands] = useState([])
  const [settingsOpen, setSettingsOpen] = useState(false)
  // GEX click-popup state: which segment is being shown, and where (in chart-relative pixels)
  const [clickedGexSegment, setClickedGexSegment] = useState(null)
  const [gexPopupPos, setGexPopupPos] = useState({ left: 0, top: 0 })

  // GEX legend panel state: open/closed (always starts closed on reload — only the
  // drag position is persisted), hovered row (highlights line on chart),
  // expanded row (shows the per-expiration breakdown inline)
  const [gexPanelOpen, setGexPanelOpen] = useState(false)
  const [hoveredPanelLevel, setHoveredPanelLevel] = useState(null)
  const [expandedPanelLevel, setExpandedPanelLevel] = useState(null)
  // Max DTE filter for the panel. null = show all (no upper bound).
  // Persisted as a string: '' for "show all", or a non-negative integer.
  const [gexPanelMaxDte, setGexPanelMaxDte] = useState(() => {
    try {
      const raw = window.localStorage.getItem('ib-react-gex-panel-max-dte')
      if (raw == null || raw === '') return null
      const n = parseInt(raw, 10)
      return Number.isFinite(n) && n >= 0 ? n : null
    } catch (e) {
      return null
    }
  })

  // GEX panel drag-position state (chart-relative pixels). null = use default top-right anchor.
  const [gexPanelPos, setGexPanelPos] = useState(() => {
    try {
      const raw = window.localStorage.getItem('ib-react-gex-panel-pos')
      if (!raw) return null
      const parsed = JSON.parse(raw)
      const left = Number(parsed?.left)
      const top = Number(parsed?.top)
      if (Number.isFinite(left) && Number.isFinite(top)) return { left, top }
    } catch (e) {
      // ignore parse errors
    }
    return null
  })
  // Drag-handle bookkeeping — captures starting cursor and panel position on mousedown,
  // computes new offsets in mousemove, persists final position on mouseup.
  const gexPanelDragRef = useRef(null)

  // Dismiss the GEX popup or panel on Escape (popup first if both are open)
  useEffect(() => {
    if (!clickedGexSegment && !gexPanelOpen) return undefined
    const onKey = (e) => {
      if (e.key !== 'Escape') return
      if (clickedGexSegment) {
        setClickedGexSegment(null)
      } else if (gexPanelOpen) {
        setGexPanelOpen(false)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [clickedGexSegment, gexPanelOpen])

  // Clean up any stale "GEX panel open" persistence from older builds — we no
  // longer persist this state. Position is still persisted under a different key.
  useEffect(() => {
    try {
      window.localStorage.removeItem('ib-react-gex-panel-open')
    } catch (e) {
      // ignore
    }
  }, [])

  // Persist panel DTE filter across sessions ('' = show all)
  useEffect(() => {
    try {
      window.localStorage.setItem(
        'ib-react-gex-panel-max-dte',
        gexPanelMaxDte == null ? '' : String(gexPanelMaxDte),
      )
    } catch (e) {
      // ignore quota / disabled storage
    }
  }, [gexPanelMaxDte])

  // Persist panel drag position (null = no persisted position, fall back to default anchor)
  useEffect(() => {
    try {
      if (gexPanelPos) {
        window.localStorage.setItem('ib-react-gex-panel-pos', JSON.stringify(gexPanelPos))
      } else {
        window.localStorage.removeItem('ib-react-gex-panel-pos')
      }
    } catch (e) {
      // ignore quota / disabled storage
    }
  }, [gexPanelPos])

  // Highlight the chart line when a panel row is hovered. The cleanup function
  // restores the original line width so rapid hover changes don't leave a stuck
  // highlight, and so unmount/segment-rebuild can't leave dangling state.
  useEffect(() => {
    if (hoveredPanelLevel == null) return undefined
    const targetLevel = Number(hoveredPanelLevel)
    if (!Number.isFinite(targetLevel)) return undefined

    const map = gexSegmentBySeriesRef.current
    let targetSeries = null
    let originalWidth = null
    for (const [series, seg] of map.entries()) {
      if (Number(seg?.level) === targetLevel) {
        targetSeries = series
        originalWidth = safeLineWidth(seg?.line_width)
        break
      }
    }
    if (!targetSeries) return undefined

    try {
      targetSeries.applyOptions({ lineWidth: Math.min(5, originalWidth + 2) })
    } catch (e) {
      // ignore stale series
    }

    return () => {
      try {
        targetSeries.applyOptions({ lineWidth: originalWidth })
      } catch (e) {
        // ignore stale series
      }
    }
  }, [hoveredPanelLevel])
  const [draftGexMinAbsB, setDraftGexMinAbsB] = useState(() => coerceGexMinAbsB(gexMinAbsB, 10))
  const [draftInterval, setDraftInterval] = useState(() => normalizeIntervalValue(interval, '1min'))
  const [settingsError, setSettingsError] = useState('')

  // ATM IV (0DTE) line overlay toggle — persisted in localStorage so the user's
  // preference survives reloads. Draft state is only used while the settings
  // modal is open; applySettings commits it to atmIvLineEnabled.
  const [atmIvLineEnabled, setAtmIvLineEnabled] = useState(() => {
    try {
      return window.localStorage.getItem('ib-react-atm-iv-line') === '1'
    } catch (e) {
      return false
    }
  })
  const [draftAtmIvLineEnabled, setDraftAtmIvLineEnabled] = useState(atmIvLineEnabled)

  const [floatingPos, setFloatingPos] = useState(() => {
    try {
      const saved = window.localStorage.getItem('ib-react-skew-window-pos')
      if (saved) return JSON.parse(saved)
    } catch (e) {}
    return { bottom: 12, left: 12 }
  })
  const floatingDragRef = useRef(null)

  const [smileWindowSize, setSmileWindowSize] = useState(() => {
    try {
      const saved = window.localStorage.getItem('ib-react-smile-window-size')
      if (saved) return JSON.parse(saved)
    } catch (e) {}
    return { width: 0, height: 0 } // 0 means calculate from stage
  })
  // SMILE panel always starts collapsed on reload — position/size still persist
  // under different keys, just not the open/closed state.
  const [smileCollapsed, setSmileCollapsed] = useState(true)
  const smileResizeRef = useRef(null)

  // ── Trade annotation mode ─────────────────────────────────────
  const [annotationState, setAnnotationState] = useState(null)
  const annotationStateRef = useRef(null)

  // Poll annotation state every 1.5s
  useEffect(() => {
    let cancelled = false
    async function poll() {
      try {
        const res = await fetch('/api/trade-log/annotation-state')
        const data = await res.json()
        if (cancelled) return
        const next = (data.ok && data.state) ? data.state : null
        annotationStateRef.current = next
        setAnnotationState(next)
      } catch { /* non-critical — silently skip */ }
    }
    poll()
    const id = setInterval(poll, 1500)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  // Clear annotation mode when user navigates to a different trade date
  useEffect(() => {
    const ann = annotationStateRef.current
    if (!ann || !tradeDate) return
    if (ann.trade_date !== tradeDate) {
      fetch('/api/trade-log/annotation-state', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      }).catch(() => {})
      annotationStateRef.current = null
      setAnnotationState(null)
    }
  }, [tradeDate])

  function clearAnnotationState() {
    fetch('/api/trade-log/annotation-state', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    }).catch(() => {})
    annotationStateRef.current = null
    setAnnotationState(null)
  }

  useEffect(() => {
    if (!stageRef.current || (smileWindowSize.width > 0 && smileWindowSize.height > 0)) return
    const stage = stageRef.current
    const w = Math.round(stage.clientWidth / 2.4)
    const h = Math.round(stage.clientHeight / 2.4)
    setSmileWindowSize({ width: w, height: h })
  }, [smileWindowSize])

  // Sync internal state with external prop
  useEffect(() => {
    setLocalSelectedTimes(selectedTimes)
  }, [selectedTimes])

  useEffect(() => {
    try {
      window.parent.postMessage({ type: 'ib-react-request-timeslices' }, '*')
    } catch (err) {
      // ignore
    }
  }, [])

  useEffect(() => {
    if (!settingsOpen) {
      setDraftGexMinAbsB(coerceGexMinAbsB(gexMinAbsB, 10))
      setDraftInterval(normalizeIntervalValue(interval, '1min'))
      setDraftAtmIvLineEnabled(atmIvLineEnabled)
      setSettingsError('')
    }
  }, [gexMinAbsB, interval, atmIvLineEnabled, settingsOpen])

  useEffect(() => {
    hideParentTopControls()
    return () => {
      showParentTopControls()
    }
  }, [])

  useEffect(() => {
    intervalRef.current = interval
  }, [interval])

  const shiftedCandles = useMemo(() => {
    return (Array.isArray(candles) ? candles : []).map((bar) => ({
      ...bar,
      time: utcEpochShowingZoneTime(Number(bar.time), 'America/Los_Angeles'),
    }))
  }, [candles])

  const shiftedLiveTradeCandles = useMemo(() => {
    return (Array.isArray(liveTradeCandles) ? liveTradeCandles : []).map((bar) => ({
      ...bar,
      time: utcEpochShowingZoneTime(Number(bar.time), 'America/Los_Angeles'),
    }))
  }, [liveTradeCandles])

  const normalizedGexSegments = useMemo(
    () => normalizeGexSegments(gexSegments),
    [gexSegments]
  )

  // Highest DTE seen in the data, used to set the slider's upper bound.
  // Negatives (already-expired rows per the API's dte field) are excluded —
  // slider runs 0..maxDteAvailable.
  const maxDteAvailable = useMemo(() => {
    if (!Array.isArray(normalizedGexSegments) || normalizedGexSegments.length === 0) return 0
    let max = 0
    for (const seg of normalizedGexSegments) {
      const exps = Array.isArray(seg?.expirations) ? seg.expirations : []
      for (const e of exps) {
        const d = Number(e?.dte)
        if (Number.isFinite(d) && d > max) max = d
      }
    }
    return max
  }, [normalizedGexSegments])

  // Filtered & sorted segments for the panel.
  // - Restricted to segments whose session_date matches the current trade_date,
  //   so the panel acts as a legend for "the date you've selected" (clicking a
  //   line from another session still works — the popup uses that segment's own
  //   data directly and ignores this filter).
  // - Always drops expired (dte < 0) rows.
  // - When maxDte is set: drops expirations with dte > maxDte, then recomputes
  //   net/call/put gamma per level, then drops levels whose |net| < gexMinAbsB
  //   (matching the existing visibility threshold from the URL params).
  // - When maxDte is null (show all): uses original totals untouched.
  // - Sorted by |net_gamma| descending in both cases.
  const sortedGexSegmentsForPanel = useMemo(() => {
    if (!Array.isArray(normalizedGexSegments)) return []
    const showAll = gexPanelMaxDte == null
    const minAbs = Math.max(0, Number(gexMinAbsB) || 0) * 1e9 // threshold in raw $
    const tradeDateStr = String(tradeDate || '').trim()

    const out = []
    for (const seg of normalizedGexSegments) {
      // Restrict the legend to the currently selected trade_date's segments.
      // If a segment doesn't carry a session_date, we keep it (defensive — better
      // to show too much than nothing if the field is ever missing).
      const segSession = String(seg?.session_date || '').trim()
      if (tradeDateStr && segSession && segSession !== tradeDateStr) continue

      const allExp = Array.isArray(seg?.expirations) ? seg.expirations : []
      // Always strip expired rows; honor slider when active. Trust the API's
      // dte field — it's already correct relative to the trade_date being viewed.
      const visibleExp = allExp.filter((e) => {
        const d = Number(e?.dte)
        if (!Number.isFinite(d) || d < 0) return false
        if (!showAll && d > gexPanelMaxDte) return false
        return true
      })

      if (showAll) {
        // No recomputation — original headline numbers and visibleExp ride along
        out.push({ ...seg, expirations: visibleExp })
        continue
      }

      // Slider active: recompute headline numbers from the surviving expirations
      let net = 0
      let call = 0
      let put = 0
      for (const e of visibleExp) {
        net += Number(e?.net_gamma) || 0
        call += Number(e?.call_gamma) || 0
        put += Number(e?.put_gamma) || 0
      }
      // Drop levels that fall below the visibility threshold under this filter
      if (Math.abs(net) < minAbs) continue

      out.push({
        ...seg,
        expirations: visibleExp,
        net_gamma: net,
        call_gamma: call,
        put_gamma: put,
      })
    }

    out.sort(
      (a, b) => Math.abs(Number(b?.net_gamma) || 0) - Math.abs(Number(a?.net_gamma) || 0),
    )
    return out
  }, [normalizedGexSegments, gexPanelMaxDte, gexMinAbsB, tradeDate])

  const normalizedExpectedMoveLevels = useMemo(
    () => normalizeExpectedMoveLevels(expectedMoveLevels),
    [expectedMoveLevels]
  )

  const sessionTimeRange = useMemo(
    () => computeTradeSessionTimeRange(tradeDate),
    [tradeDate]
  )

  const centerLogicalRange = useMemo(
    () => computeCenterLogicalRange(shiftedCandles, sessionTimeRange),
    [shiftedCandles, sessionTimeRange]
  )

  useEffect(() => {
    shiftedCandlesRef.current = shiftedCandles
  }, [shiftedCandles])

  useEffect(() => {
    const nextKey = `${tradeDate || ''}:${interval || ''}`
    if (nextKey !== currentCenterKeyRef.current) {
      currentCenterKeyRef.current = nextKey
      appliedInitialRangeKeyRef.current = ''
      hasUserInteractedRef.current = false
      lastReportedRangeRef.current = null
      lastManualTimeRangeRef.current = null
      previousDisplayCandlesRef.current = []
    }
  }, [tradeDate, interval])

  const selectedSet = useMemo(() => new Set(localSelectedTimes), [localSelectedTimes])

  const displayCandles = useMemo(
    () => applySelectionColors(shiftedCandles, selectedSet, interval),
    [shiftedCandles, selectedSet, interval]
  )

  useEffect(() => {
    if (!hostRef.current || !stageRef.current) return undefined

    const container = hostRef.current
    const stage = stageRef.current

    const hideTooltip = () => {
      const el = tooltipRef.current
      if (!el) return
      el.style.opacity = '0'
      el.style.transform = 'translate(-9999px, -9999px)'
    }

    const showTooltipAtPoint = (x, y, priceText, timeText) => {
      const el = tooltipRef.current
      if (!el || !stageRef.current) return

      if (tooltipTimeRef.current) tooltipTimeRef.current.textContent = timeText
      if (tooltipPriceRef.current) tooltipPriceRef.current.textContent = priceText

      const plotHeight = getPlotHeight(stageRef.current)
      const boxWidth = el.offsetWidth || 140
      const boxHeight = el.offsetHeight || 54

      let left = x + TOOLTIP_OFFSET_X
      let top = y + TOOLTIP_OFFSET_Y

      if (left + boxWidth > stageRef.current.clientWidth - TOOLTIP_EDGE_PAD) {
        left = x - boxWidth - TOOLTIP_OFFSET_X
      }
      if (left < TOOLTIP_EDGE_PAD) {
        left = TOOLTIP_EDGE_PAD
      }

      if (top + boxHeight > plotHeight - TOOLTIP_EDGE_PAD) {
        top = y - boxHeight - TOOLTIP_OFFSET_Y
      }
      if (top < TOOLTIP_EDGE_PAD) {
        top = TOOLTIP_EDGE_PAD
      }

      el.style.opacity = '1'
      el.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`
    }

    const chart = createChart(container, {
      width: container.clientWidth || stage.clientWidth || 900,
      height: Math.max(container.clientHeight || stage.clientHeight || 0, MIN_CHART_HEIGHT),
      layout: {
        background: { type: ColorType.Solid, color: ETH_BG_COLOR },
        textColor: '#cbd5e1',
        attributionLogo: false,
      },
      localization: {
        locale: 'en-US',
      },
      grid: {
        vertLines: { color: 'rgba(148, 163, 184, 0.08)' },
        horzLines: { color: 'rgba(148, 163, 184, 0.08)' },
      },
      rightPriceScale: {
        borderColor: 'rgba(148, 163, 184, 0.18)',
        autoScale: true,
      },
      timeScale: {
        borderColor: 'rgba(148, 163, 184, 0.18)',
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (time) => formatShiftedHHMM(time),
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: 'rgba(191, 219, 254, 0.45)',
          width: 1,
          style: 2,
          labelBackgroundColor: '#1e293b',
        },
        horzLine: {
          color: 'rgba(191, 219, 254, 0.45)',
          width: 1,
          style: 2,
          labelBackgroundColor: '#1e293b',
        },
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
    })

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#60a5fa',
      downColor: '#e5e7eb',
      wickUpColor: '#60a5fa',
      wickDownColor: '#e5e7eb',
      borderUpColor: '#60a5fa',
      borderDownColor: '#e5e7eb',
      borderVisible: true,
      priceLineVisible: false,
      lastValueVisible: true,
    })

    const liveTradeSeries = chart.addSeries(CandlestickSeries, {
      upColor: LIVE_OVERLAY_UP_FILL,
      downColor: LIVE_OVERLAY_DOWN_FILL,
      wickUpColor: LIVE_OVERLAY_UP_BORDER,
      wickDownColor: LIVE_OVERLAY_DOWN_BORDER,
      borderUpColor: LIVE_OVERLAY_UP_BORDER,
      borderDownColor: LIVE_OVERLAY_DOWN_BORDER,
      borderVisible: true,
      priceLineVisible: false,
      lastValueVisible: false,
    })

    chartRef.current = chart
    seriesRef.current = candleSeries
    liveTradeSeriesRef.current = liveTradeSeries

    const setManualPriceScale = () => {
      try {
        chart.priceScale('right').applyOptions({ autoScale: false })
      } catch (err) {
        chart.applyOptions({ rightPriceScale: { autoScale: false } })
      }
    }

    const panPriceByPixels = (deltaY) => {
      if (!Number.isFinite(deltaY) || deltaY === 0) return
      const priceScale = chart.priceScale('right')
      const vr = clampVisibleRange(priceScale.getVisibleRange())
      if (!vr) return
      const plotHeight = getPlotHeight(stage)
      const deltaPrice = (deltaY / plotHeight) * (vr.to - vr.from)
      setManualPriceScale()
      priceScale.setVisibleRange({ from: vr.from + deltaPrice, to: vr.to + deltaPrice })
    }

    const zoomPriceAtY = (deltaY, yCoord) => {
      const priceScale = chart.priceScale('right')
      const vr = clampVisibleRange(priceScale.getVisibleRange())
      if (!vr) return
      const plotHeight = getPlotHeight(stage)
      const clampedY = Math.max(0, Math.min(plotHeight, yCoord))
      let anchorPrice = candleSeries.coordinateToPrice(clampedY)
      if (!Number.isFinite(anchorPrice)) {
        anchorPrice = (vr.from + vr.to) / 2
      }
      const factor = Math.exp(deltaY * 0.0015)
      let next = {
        from: anchorPrice - (anchorPrice - vr.from) * factor,
        to: anchorPrice + (vr.to - anchorPrice) * factor,
      }
      next = clampVisibleRange(next)
      if (!next) return
      setManualPriceScale()
      priceScale.setVisibleRange(next)
    }

    // Vertical-only zoom for the IV (left) scale. Independent of the candle
    // scale and the time scale — wheeling on the IV axis must not pan the
    // chart left/right. Anchors at the cursor's IV value when an IV series
    // is mounted (so the line under the cursor stays fixed during zoom),
    // otherwise falls back to the visible range midpoint.
    const zoomIvAtY = (deltaY, yCoord) => {
      const ivScale = chart.priceScale('left')
      let vr = null
      try {
        vr = ivScale.getVisibleRange()
      } catch (err) {
        vr = null
      }
      if (!vr || !Number.isFinite(vr.from) || !Number.isFinite(vr.to) || vr.to <= vr.from) {
        return
      }
      const plotHeight = getPlotHeight(stage)
      const clampedY = Math.max(0, Math.min(plotHeight, yCoord))

      let anchorPrice = (vr.from + vr.to) / 2
      const ivSeries = atmIvSeriesRefs.current[0]
      if (ivSeries) {
        const p = ivSeries.coordinateToPrice(clampedY)
        if (Number.isFinite(p)) anchorPrice = p
      }

      const factor = Math.exp(deltaY * 0.0015)
      const next = {
        from: anchorPrice - (anchorPrice - vr.from) * factor,
        to: anchorPrice + (vr.to - anchorPrice) * factor,
      }
      if (!Number.isFinite(next.from) || !Number.isFinite(next.to) || next.to <= next.from) return
      try {
        ivScale.applyOptions({ autoScale: false })
        ivScale.setVisibleRange(next)
      } catch (err) {
        // ignore — scale may have been hidden mid-event
      }
    }

    const panIvByPixels = (deltaY) => {
      if (!Number.isFinite(deltaY) || deltaY === 0) return
      const ivScale = chart.priceScale('left')
      let vr = null
      try {
        vr = ivScale.getVisibleRange()
      } catch (err) {
        vr = null
      }
      if (!vr || !Number.isFinite(vr.from) || !Number.isFinite(vr.to) || vr.to <= vr.from) return
      const plotHeight = getPlotHeight(stage)
      const deltaPrice = (deltaY / plotHeight) * (vr.to - vr.from)
      try {
        ivScale.applyOptions({ autoScale: false })
        ivScale.setVisibleRange({ from: vr.from + deltaPrice, to: vr.to + deltaPrice })
      } catch (err) {
        // ignore
      }
    }

    const updateBand = () => {
      const next = computeSessionBands(
        chart,
        shiftedCandlesRef.current,
        stageRef.current?.clientWidth || 0
      )
      setSessionBands(next)
    }

    const reportLogicalRange = (range) => {
      if (typeof onVisibleLogicalRangeChange !== 'function') return
      const next = normalizeLogicalRange(range)
      if (!next) return
      if (rangesClose(lastReportedRangeRef.current, next)) return
      lastReportedRangeRef.current = next
      onVisibleLogicalRangeChange(next)
    }

    const applyLogicalViewport = (nextRange) => {
      const normalized = normalizeLogicalRange(nextRange)
      if (!normalized || !chartRef.current) return null
      suppressRangeReportRef.current += 1
      chartRef.current.timeScale().setVisibleLogicalRange({
        from: normalized.from,
        to: normalized.to,
      })
      requestAnimationFrame(() => {
        suppressRangeReportRef.current = Math.max(0, suppressRangeReportRef.current - 1)
      })
      return normalized
    }

    const applyTimeViewport = (nextRange) => {
      const from = Number(nextRange?.from)
      const to = Number(nextRange?.to)
      if (!chartRef.current || !Number.isFinite(from) || !Number.isFinite(to)) return false
      suppressRangeReportRef.current += 1
      chartRef.current.timeScale().setVisibleRange({
        from: Math.min(from, to),
        to: Math.max(from, to),
      })
      requestAnimationFrame(() => {
        suppressRangeReportRef.current = Math.max(0, suppressRangeReportRef.current - 1)
      })
      return true
    }

    const reportLinkedCrosshair = (nextValue) => {
      if (typeof onLinkedCrosshairChange !== 'function') return
      onLinkedCrosshairChange(nextValue || null)
    }

    const updateFloatingTooltip = (param) => {
      if (!param?.point || !stageRef.current || !seriesRef.current || !chartRef.current) {
        hideTooltip()
        reportLinkedCrosshair(null)
        return
      }

      const stageEl = stageRef.current
      const x = Number(param.point.x)
      const y = Number(param.point.y)
      const plotHeight = getPlotHeight(stageEl)

      if (
        !Number.isFinite(x) ||
        !Number.isFinite(y) ||
        x < 0 ||
        y < 0 ||
        x > stageEl.clientWidth ||
        y > plotHeight
      ) {
        hideTooltip()
        reportLinkedCrosshair(null)
        return
      }

      const price = seriesRef.current.coordinateToPrice(y)

      let logical = null
      if (Number.isFinite(param.logical)) {
        logical = param.logical
      } else if (typeof chartRef.current.timeScale().coordinateToLogical === 'function') {
        logical = chartRef.current.timeScale().coordinateToLogical(x)
      }

      const clampedLogical = clampLogicalToData(logical, shiftedCandlesRef.current)
      const shiftedEpoch = interpolateShiftedEpoch(
        clampedLogical,
        shiftedCandlesRef.current,
        intervalRef.current
      )

      if (!Number.isFinite(price) || !Number.isFinite(shiftedEpoch) || !Number.isFinite(clampedLogical)) {
        hideTooltip()
        reportLinkedCrosshair(null)
        return
      }

      const formatter = seriesRef.current.priceFormatter?.()
      const priceText =
        formatter && typeof formatter.format === 'function'
          ? formatter.format(price)
          : price.toFixed(2)

      const timeText = formatShiftedTimestamp(shiftedEpoch)
      showTooltipAtPoint(x, y, priceText, timeText)
      reportLinkedCrosshair({ logical: clampedLogical, shiftedTime: shiftedEpoch })
    }

    const handleResize = () => {
      if (!chart || !container) return
      const w = container.clientWidth || stage.clientWidth
      const h = container.clientHeight || stage.clientHeight
      if (w > 0 && h > 0) {
        chart.applyOptions({
          width: w,
          height: Math.max(h, MIN_CHART_HEIGHT),
        })
      }
      hideTooltip()
      requestAnimationFrame(updateBand)
      requestAnimationFrame(() => {
        const vr = normalizeLogicalRange(chart.timeScale().getVisibleLogicalRange?.())
        reportLogicalRange(vr)
      })
    }

    const resizeObserver =
      typeof ResizeObserver !== 'undefined'
        ? new ResizeObserver(() => {
            handleResize()
          })
        : null

    if (resizeObserver) {
      resizeObserver.observe(stage)
    }

    const handleClick = (param) => {
      // GEX line hit-test first: if the click landed near a GEX horizontal line,
      // show the per-expiration popup and short-circuit before toggling a timeslice.
      if (param?.point && gexSeriesRefs.current.length > 0) {
        const HIT_TOLERANCE_PX = 8
        let bestSeries = null
        let bestDist = Infinity
        const clickY = Number(param.point.y)
        if (Number.isFinite(clickY)) {
          for (const series of gexSeriesRefs.current) {
            const seg = gexSegmentBySeriesRef.current.get(series)
            if (!seg) continue
            const lineY = series.priceToCoordinate(seg.level)
            if (lineY == null) continue
            const dist = Math.abs(lineY - clickY)
            if (dist < bestDist) {
              bestDist = dist
              bestSeries = series
            }
          }
        }
        if (bestSeries && bestDist <= HIT_TOLERANCE_PX) {
          const seg = gexSegmentBySeriesRef.current.get(bestSeries)
          setClickedGexSegment(seg)
          setGexPopupPos({
            left: Number(param.point.x) || 0,
            top: Number(param.point.y) || 0,
          })
          return
        }
      }

      // Click landed away from any GEX line — close any open popup before
      // running the existing timeslice-toggle behavior.
      setClickedGexSegment(null)

      if (!param?.time) return
      const hhmm = toPtHHMM(param.time, intervalRef.current)

      setLocalSelectedTimes((prev) => {
        const exists = prev.includes(hhmm)
        const next = normalizeTimes(exists ? prev.filter((x) => x !== hhmm) : [...prev, hhmm])
        try {
          window.parent.postMessage({ type: 'ib-react-timeslices', times: next }, '*')
        } catch (err) {
          console.error('postMessage failed', err)
        }
        return next
      })
    }

    const handleVisibleRange = (range) => {
      requestAnimationFrame(updateBand)

      if (hasUserInteractedRef.current) {
        const nextManualTimeRange = logicalRangeToShiftedTimeRange(
          range,
          shiftedCandlesRef.current,
          intervalRef.current
        )
        if (nextManualTimeRange) {
          lastManualTimeRangeRef.current = nextManualTimeRange
        }
      }

      if (suppressRangeReportRef.current > 0) return
      reportLogicalRange(range)
    }

    const handleCrosshairMove = (param) => {
      updateFloatingTooltip(param)
    }

    const setInteractionActive = (active, releaseDelay = 0) => {
      if (interactionReleaseTimerRef.current) {
        window.clearTimeout(interactionReleaseTimerRef.current)
        interactionReleaseTimerRef.current = null
      }

      if (active) {
        if (!interactionActiveRef.current) {
          interactionActiveRef.current = true
          if (typeof onInteractionActiveChange === 'function') {
            onInteractionActiveChange(true)
          }
        }
        return
      }

      const release = () => {
        if (!interactionActiveRef.current) return
        interactionActiveRef.current = false
        if (typeof onInteractionActiveChange === 'function') {
          onInteractionActiveChange(false)
        }
      }

      if (releaseDelay > 0) {
        interactionReleaseTimerRef.current = window.setTimeout(release, releaseDelay)
      } else {
        release()
      }
    }

    const getLeftAxisWidth = () => {
      try {
        const w = Number(chart.priceScale('left').width()) || 0
        return w
      } catch (err) {
        return 0
      }
    }

    const handleWheel = (evt) => {
      hasUserInteractedRef.current = true
      setInteractionActive(true)
      lastManualTimeRangeRef.current = logicalRangeToShiftedTimeRange(
        chart.timeScale().getVisibleLogicalRange?.(),
        shiftedCandlesRef.current,
        intervalRef.current
      )
      const info = pointerInfo(evt, stage, getLeftAxisWidth())
      if (!info || info.overTimeAxis) {
        setInteractionActive(false, 180)
        return
      }
      // Wheel over the IV axis → zoom only the IV scale; never the time
      // scale, never the price scale.
      if (info.overLeftPriceAxis) {
        evt.preventDefault()
        evt.stopPropagation()
        zoomIvAtY(evt.deltaY, info.y)
        setInteractionActive(false, 180)
        return
      }
      if (!info.overPriceAxis) {
        setInteractionActive(false, 180)
        return
      }
      evt.preventDefault()
      evt.stopPropagation()
      zoomPriceAtY(evt.deltaY, info.y)
      setInteractionActive(false, 180)
    }

    const handleMouseDown = (evt) => {
      if (evt.button !== 0) return
      hasUserInteractedRef.current = true
      setInteractionActive(true)
      lastManualTimeRangeRef.current = logicalRangeToShiftedTimeRange(
        chart.timeScale().getVisibleLogicalRange?.(),
        shiftedCandlesRef.current,
        intervalRef.current
      )
      const info = pointerInfo(evt, stage, getLeftAxisWidth())
      if (!info || info.overTimeAxis || info.overPriceAxis) return
      // Drag on the IV axis → pan IV scale only (mirrors the right-axis
      // native zoom behavior). Don't fall through to the candle-pan path.
      if (info.overLeftPriceAxis) {
        dragRef.current = { active: true, lastY: evt.clientY, scale: 'left' }
        return
      }
      // Shift+drag in the plot area pans the IV line up/down independently
      // of the candles. The candles and IV overlap the same plot area, so
      // we need an explicit modifier to disambiguate intent.
      if (evt.shiftKey && atmIvSeriesRefs.current.length > 0) {
        evt.preventDefault()
        dragRef.current = { active: true, lastY: evt.clientY, scale: 'left' }
        return
      }
      dragRef.current = { active: true, lastY: evt.clientY, scale: 'right' }
    }

    const handleMouseMove = (evt) => {
      if (!dragRef.current.active) return
      setInteractionActive(true)
      const deltaY = evt.clientY - dragRef.current.lastY
      dragRef.current.lastY = evt.clientY
      if (deltaY !== 0) {
        if (dragRef.current.scale === 'left') {
          panIvByPixels(deltaY)
        } else {
          panPriceByPixels(deltaY)
        }
      }
    }

    const handleMouseUp = () => {
      dragRef.current.active = false
      setInteractionActive(false, 180)
    }

    const handleMouseLeave = () => {
      hideTooltip()
      reportLinkedCrosshair(null)
      dragRef.current.active = false
      setInteractionActive(false, 180)
    }

    chart.subscribeClick(handleClick)
    chart.subscribeCrosshairMove(handleCrosshairMove)
    chart.timeScale().subscribeVisibleLogicalRangeChange(handleVisibleRange)

    window.addEventListener('resize', handleResize)
    stage.addEventListener('wheel', handleWheel, { passive: false, capture: true })
    stage.addEventListener('mousedown', handleMouseDown)
    stage.addEventListener('mouseleave', handleMouseLeave)
    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)

    requestAnimationFrame(updateBand)

    return () => {
      window.removeEventListener('resize', handleResize)
      if (resizeObserver) {
        resizeObserver.disconnect()
      }
      stage.removeEventListener('wheel', handleWheel, { capture: true })
      stage.removeEventListener('mousedown', handleMouseDown)
      stage.removeEventListener('mouseleave', handleMouseLeave)
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)

      chart.unsubscribeClick(handleClick)
      chart.unsubscribeCrosshairMove(handleCrosshairMove)
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(handleVisibleRange)

      chart.remove()
      chartRef.current = null
      seriesRef.current = null
      liveTradeSeriesRef.current = null
      gexSeriesRefs.current = []
      gexSegmentBySeriesRef.current = new Map()
      expectedMoveSeriesRefs.current = []
      dragRef.current.active = false
      reportLinkedCrosshair(null)
      if (interactionReleaseTimerRef.current) {
        window.clearTimeout(interactionReleaseTimerRef.current)
        interactionReleaseTimerRef.current = null
      }
      interactionActiveRef.current = false
      if (typeof onInteractionActiveChange === 'function') {
        onInteractionActiveChange(false)
      }
    }
  }, [onVisibleLogicalRangeChange, onInteractionActiveChange])

  useEffect(() => {
    function handleParentMessage(event) {
      const data = event && event.data
      if (!data || data.type !== 'ib-parent-timeslices') return
      setLocalSelectedTimes(normalizeTimes(data.times || []))
    }

    window.addEventListener('message', handleParentMessage)
    return () => window.removeEventListener('message', handleParentMessage)
  }, [])

  useEffect(() => {
    if (!seriesRef.current || !chartRef.current) return

    const chart = chartRef.current
    const previousCandles = Array.isArray(previousDisplayCandlesRef.current)
      ? previousDisplayCandlesRef.current
      : []
    const currentLogicalBeforeUpdate = normalizeLogicalRange(
      chart.timeScale().getVisibleLogicalRange?.()
    )

    const prependCount = countPrependedBars(previousCandles, displayCandles)
    const preservedManualLogicalRange =
      hasUserInteractedRef.current && currentLogicalBeforeUpdate
        ? shiftLogicalRange(currentLogicalBeforeUpdate, prependCount)
        : null

    if (hasUserInteractedRef.current && currentLogicalBeforeUpdate && previousCandles.length) {
      const preservedManualTimeRange = logicalRangeToShiftedTimeRange(
        currentLogicalBeforeUpdate,
        previousCandles,
        intervalRef.current
      )
      if (preservedManualTimeRange) {
        lastManualTimeRangeRef.current = preservedManualTimeRange
      }
    }

    seriesRef.current.setData(displayCandles)
    previousDisplayCandlesRef.current = displayCandles

    const syncBands = () => {
      setSessionBands(
        computeSessionBands(
          chart,
          shiftedCandles,
          stageRef.current?.clientWidth || 0
        )
      )
    }

    const syncReportedRange = () => {
      const vr = normalizeLogicalRange(chart.timeScale().getVisibleLogicalRange?.())
      if (vr && typeof onVisibleLogicalRangeChange === 'function') {
        if (!rangesClose(lastReportedRangeRef.current, vr)) {
          lastReportedRangeRef.current = vr
          onVisibleLogicalRangeChange(vr)
        }
      }
    }

    syncBands()

    if (hasUserInteractedRef.current) {
      if (preservedManualLogicalRange) {
        suppressRangeReportRef.current += 1
        chart.timeScale().setVisibleLogicalRange({
          from: preservedManualLogicalRange.from,
          to: preservedManualLogicalRange.to,
        })
        requestAnimationFrame(() => {
          suppressRangeReportRef.current = Math.max(0, suppressRangeReportRef.current - 1)
        })
        requestAnimationFrame(() => {
          if (!chartRef.current) return
          syncBands()
          syncReportedRange()
        })
      } else if (lastManualTimeRangeRef.current) {
        suppressRangeReportRef.current += 1
        chart.timeScale().setVisibleRange({
          from: Math.min(lastManualTimeRangeRef.current.from, lastManualTimeRangeRef.current.to),
          to: Math.max(lastManualTimeRangeRef.current.from, lastManualTimeRangeRef.current.to),
        })
        requestAnimationFrame(() => {
          suppressRangeReportRef.current = Math.max(0, suppressRangeReportRef.current - 1)
        })
        requestAnimationFrame(() => {
          if (!chartRef.current) return
          syncBands()
          syncReportedRange()
        })
      } else {
        requestAnimationFrame(() => {
          if (!chartRef.current) return
          syncBands()
          syncReportedRange()
        })
      }
      return
    }

    if (centerLogicalRange) {
      const current = normalizeLogicalRange(chart.timeScale().getVisibleLogicalRange?.())

      if (!rangesClose(current, centerLogicalRange)) {
        suppressRangeReportRef.current += 1
        chart.timeScale().setVisibleLogicalRange({
          from: centerLogicalRange.from,
          to: centerLogicalRange.to,
        })
        requestAnimationFrame(() => {
          suppressRangeReportRef.current = Math.max(0, suppressRangeReportRef.current - 1)
        })
      }

      appliedInitialRangeKeyRef.current = centerLogicalRange.key

      requestAnimationFrame(() => {
        if (!chartRef.current) return

        const secondPassCurrent = normalizeLogicalRange(
          chartRef.current.timeScale().getVisibleLogicalRange?.()
        )

        if (!rangesClose(secondPassCurrent, centerLogicalRange)) {
          suppressRangeReportRef.current += 1
        chart.timeScale().setVisibleLogicalRange({
          from: centerLogicalRange.from,
          to: centerLogicalRange.to,
        })
        requestAnimationFrame(() => {
          suppressRangeReportRef.current = Math.max(0, suppressRangeReportRef.current - 1)
        })
        }

        syncBands()
        syncReportedRange()
      })
    } else {
      chart.timeScale().fitContent()
      requestAnimationFrame(() => {
        if (!chartRef.current) return
        syncBands()
        syncReportedRange()
      })
    }
  }, [displayCandles, shiftedCandles, sessionTimeRange, centerLogicalRange, onVisibleLogicalRangeChange])

  useEffect(() => {
    if (!settingsOpen) return undefined

    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        setSettingsOpen(false)
        setSettingsError('')
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [settingsOpen])

  function openSettings() {
    setDraftGexMinAbsB(coerceGexMinAbsB(gexMinAbsB, 10))
    setDraftInterval(normalizeIntervalValue(interval, '1min'))
    setDraftAtmIvLineEnabled(atmIvLineEnabled)
    setSettingsError('')
    setSettingsOpen(true)
  }

  function closeSettings() {
    setSettingsOpen(false)
    setSettingsError('')
    setDraftGexMinAbsB(coerceGexMinAbsB(gexMinAbsB, 10))
    setDraftInterval(normalizeIntervalValue(interval, '1min'))
    setDraftAtmIvLineEnabled(atmIvLineEnabled)
  }

  function applySettings(event) {
    if (event) event.preventDefault()
    const next = coerceGexMinAbsB(draftGexMinAbsB, NaN)
    if (!Number.isFinite(next) || next < 0) {
      setSettingsError('Choose a value from 0 to 200.')
      return
    }

    const nextInterval = normalizeIntervalValue(draftInterval, '1min')

    if (typeof onApplyGexMinAbsB === 'function') {
      onApplyGexMinAbsB(next)
    }
    if (typeof onApplyIntervalChange === 'function') {
      onApplyIntervalChange(nextInterval)
    }

    // Commit pure-React settings (IV line) before the parent-Dash sync below,
    // so they still take effect when the React app runs standalone (no parent
    // Dash controls present, e.g. dev/preview mode).
    setAtmIvLineEnabled(draftAtmIvLineEnabled)
    try {
      window.localStorage.setItem('ib-react-atm-iv-line', draftAtmIvLineEnabled ? '1' : '0')
    } catch (e) {
      // ignore quota / privacy-mode errors
    }

    const intervalOk = setParentRadioValue('ironbeam-bar-interval', nextInterval)

    if (!intervalOk) {
      setSettingsError('Could not sync bar interval with the parent Dash controls.')
      hideParentTopControls()
      return
    }

    hideParentTopControls()
    setSettingsOpen(false)
    setSettingsError('')
  }

  useEffect(() => {
    if (!liveTradeSeriesRef.current) return

    const next = Array.isArray(shiftedLiveTradeCandles) ? shiftedLiveTradeCandles : []
    const hasLive = next.length > 0

    if (seriesRef.current) {
      seriesRef.current.applyOptions({
        lastValueVisible: !hasLive,
      })
    }

    liveTradeSeriesRef.current.applyOptions({
      lastValueVisible: hasLive,
      priceLineVisible: false,
    })

    const prev = Array.isArray(previousLiveTradeCandlesRef.current)
      ? previousLiveTradeCandlesRef.current
      : []

    const canIncremental =
      prev.length > 0 &&
      next.length > 0 &&
      next.length >= prev.length &&
      prev
        .slice(0, Math.max(0, prev.length - 1))
        .every((bar, idx) => Number(bar?.time) === Number(next[idx]?.time))

    if (!canIncremental) {
      liveTradeSeriesRef.current.setData(next)
      previousLiveTradeCandlesRef.current = next
      return
    }

    const startIdx = Math.max(0, prev.length - 1)
    for (let i = startIdx; i < next.length; i += 1) {
      liveTradeSeriesRef.current.update(next[i])
    }
    previousLiveTradeCandlesRef.current = next
  }, [shiftedLiveTradeCandles])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return

    for (const series of gexSeriesRefs.current) {
      try {
        chart.removeSeries(series)
      } catch (err) {
        // ignore stale references
      }
    }
    gexSeriesRefs.current = []
    gexSegmentBySeriesRef.current = new Map()

    if (!gexEnabled || !Array.isArray(normalizedGexSegments) || !normalizedGexSegments.length) {
      return
    }

    const nextRefs = []
    const nextMap = new Map()

    for (const seg of normalizedGexSegments) {
      const level = Number(seg?.level)
      const shiftedStart = Number(seg?.shiftedStart)
      const shiftedEnd = Number(seg?.shiftedEnd)

      if (
        !Number.isFinite(level) ||
        !Number.isFinite(shiftedStart) ||
        !Number.isFinite(shiftedEnd)
      ) {
        continue
      }

      try {
        const lineSeries = chart.addSeries(LineSeries, {
          color: seg?.color || 'rgba(148,163,184,0.55)',
          lineWidth: safeLineWidth(seg?.line_width),
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
          pointMarkersVisible: false,
        })

        lineSeries.setData([
          { time: shiftedStart, value: level },
          { time: shiftedEnd, value: level },
        ])

        nextRefs.push(lineSeries)
        nextMap.set(lineSeries, seg)
      } catch (err) {
        // ignore bad segment
      }
    }

    gexSeriesRefs.current = nextRefs
    gexSegmentBySeriesRef.current = nextMap
  }, [normalizedGexSegments, gexEnabled])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return

    for (const series of expectedMoveSeriesRefs.current) {
      try {
        chart.removeSeries(series)
      } catch (err) {
        // ignore stale references
      }
    }
    expectedMoveSeriesRefs.current = []

    if (!Array.isArray(normalizedExpectedMoveLevels) || !normalizedExpectedMoveLevels.length) {
      return
    }

    const nextRefs = []

    for (const item of normalizedExpectedMoveLevels) {
      const shiftedStart = Number(item?.shiftedStart)
      const shiftedEnd = Number(item?.shiftedEnd)
      const upper = Number(item?.upper)
      const lower = Number(item?.lower)

      if (!Number.isFinite(shiftedStart) || !Number.isFinite(shiftedEnd)) {
        continue
      }

      const commonOptions = {
        color: EXPECTED_MOVE_COLOR,
        lineWidth: 1,
        lineStyle: 2, // Dotted
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
        pointMarkersVisible: false,
      }

      try {
        if (Number.isFinite(upper)) {
          const upperSeries = chart.addSeries(LineSeries, commonOptions)
          upperSeries.setData([
            { time: shiftedStart, value: upper },
            { time: shiftedEnd, value: upper },
          ])
          nextRefs.push(upperSeries)
        }
        if (Number.isFinite(lower)) {
          const lowerSeries = chart.addSeries(LineSeries, commonOptions)
          lowerSeries.setData([
            { time: shiftedStart, value: lower },
            { time: shiftedEnd, value: lower },
          ])
          nextRefs.push(lowerSeries)
        }
      } catch (err) {
        // ignore errors
      }
    }

    expectedMoveSeriesRefs.current = nextRefs
  }, [normalizedExpectedMoveLevels])

  // ── Sigma bands + condor strike lines ────────────────────────────────
  // Renders ±1σ horizontal lines and 4 condor strike lines (ES coords) when
  // an anchor is active. Pattern mirrors the expectedMoveLevels effect above.
  // Colors: σ lines in cyan-ish, short strikes in amber, long strikes in muted gray.
  useEffect(() => {
    const chart = chartRef.current
    if (!chart) {
      console.log('[bands-render] no chart ref yet', { bandsLevels })
      return
    }

    // Tear down any prior series
    for (const series of bandsSeriesRefs.current) {
      try {
        chart.removeSeries(series)
      } catch (err) {
        // ignore
      }
    }
    bandsSeriesRefs.current = []

    if (!bandsLevels) {
      console.log('[bands-render] no bandsLevels — clearing only')
      return
    }

    console.log('[bands-render] drawing lines', bandsLevels)

    const shiftedStart = Number(bandsLevels.shiftedStart)
    const shiftedEnd = Number(bandsLevels.shiftedEnd)
    if (!Number.isFinite(shiftedStart) || !Number.isFinite(shiftedEnd) || shiftedEnd <= shiftedStart) {
      return
    }

    const baseOptions = {
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
      pointMarkersVisible: false,
    }

    // Neon-orange σ band lines — bright enough to stand out from candles, GEX zones,
    // and the red expected-move lines. Strike lines use amber/gray to stay readable
    // but distinguishable from σ bands.
    const SIGMA_COLOR = '#ff6a00'         // neon orange
    const SHORT_STRIKE_COLOR = '#fcd34d'  // amber — strike "defense" levels
    const LONG_STRIKE_COLOR = '#94a3b8'   // muted gray — defensive wings

    const linesToDraw = [
      { value: bandsLevels.sigmaUpper, color: SIGMA_COLOR, width: 3, style: 0, label: '+1σ' },
      { value: bandsLevels.sigmaLower, color: SIGMA_COLOR, width: 3, style: 0, label: '-1σ' },
      { value: bandsLevels.shortCall,  color: SHORT_STRIKE_COLOR, width: 1, style: 2, label: 'Short Call' },
      { value: bandsLevels.shortPut,   color: SHORT_STRIKE_COLOR, width: 1, style: 2, label: 'Short Put' },
      { value: bandsLevels.longCall,   color: LONG_STRIKE_COLOR, width: 1, style: 2, label: 'Long Call' },
      { value: bandsLevels.longPut,    color: LONG_STRIKE_COLOR, width: 1, style: 2, label: 'Long Put' },
    ]

    const nextRefs = []

    for (const line of linesToDraw) {
      const v = Number(line.value)
      if (!Number.isFinite(v)) {
        console.log('[bands-render] skipping line with bad value', line)
        continue
      }
      try {
        const opts = {
          ...baseOptions,
          color: line.color,
          lineWidth: line.width,
          lastValueVisible: true,
          priceLineVisible: false,
        }
        // Only set lineStyle when explicitly non-solid; solid (0) is default.
        // Skipping when 0 avoids any chance the API rejects 0 in some versions.
        if (line.style && line.style !== 0) {
          opts.lineStyle = line.style
        }
        const series = chart.addSeries(LineSeries, opts)
        series.setData([
          { time: shiftedStart, value: v },
          { time: shiftedEnd, value: v },
        ])
        nextRefs.push(series)
      } catch (err) {
        console.error('[bands-render] failed to add line', line, err)
      }
    }

    console.log(`[bands-render] added ${nextRefs.length} of ${linesToDraw.length} lines`)

    bandsSeriesRefs.current = nextRefs
  }, [bandsLevels])

  // ── ATM IV (0DTE) line overlay ────────────────────────────────────────
  // Maps atmIvSeries entries (PT HH:MM → atm_iv_pct) onto the rendered
  // candles' shifted-epoch times so the IV line aligns to the chart's bars.
  // Switching 1min↔5min auto-thins because we iterate the chart's bars.
  //
  // Scale: drawn on the built-in 'left' price scale. lightweight-charts only
  // supports two visible price axes ('left' and 'right'); 'right' is taken
  // by candles, so the IV axis goes on the LEFT side, with the same
  // axisPressedMouseMove + mouseWheel zoom/pan behavior as the price axis.
  //
  // Session breaks: the candles backend filters out overnight bars, so we
  // can't rely on whitespace-at-non-RTH to break the line. Instead we group
  // the IV points by session date and create one LineSeries per session —
  // independent series never connect across the gap.
  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return

    const tearDown = () => {
      for (const s of atmIvSeriesRefs.current) {
        try {
          chart.removeSeries(s)
        } catch (err) {
          // ignore stale reference
        }
      }
      atmIvSeriesRefs.current = []
      try {
        chart.priceScale('left').applyOptions({ visible: false })
      } catch (err) {
        // ignore
      }
      // Forget that we ever fit the scale, so the next time the toggle is
      // turned on we re-fit (autoScale: true) once.
      atmIvScaleVisibleRef.current = false
    }

    if (!atmIvLineEnabled || !Array.isArray(atmIvSeries) || atmIvSeries.length === 0) {
      tearDown()
      return
    }

    // (session date, PT HH:MM) → IV %, RTH-only (06:30–13:00 PT inclusive).
    // Keying on the date too is critical: yesterday's 08:30 IV must not be
    // looked up for today's 08:30 candle.
    const ivByDateHHMM = new Map()
    for (const item of atmIvSeries) {
      const date = String(item?.date || '').trim()
      const t = String(item?.time || '').trim()
      const v = Number(item?.atm_iv_pct)
      if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) continue
      if (!/^\d{2}:\d{2}$/.test(t) || !Number.isFinite(v)) continue
      if (t < '06:30' || t > '13:00') continue
      ivByDateHHMM.set(`${date}|${t}`, v)
    }

    // Group points by shifted-date (one bucket per session).
    const sessions = new Map()
    for (const bar of shiftedCandles) {
      const t = Number(bar?.time)
      if (!Number.isFinite(t)) continue
      const hhmm = toPtHHMM(t, interval)
      if (!hhmm || hhmm < '06:30' || hhmm > '13:00') continue
      const date = shiftedDateKey(t)
      const iv = ivByDateHHMM.get(`${date}|${hhmm}`)
      if (!Number.isFinite(iv)) continue
      if (!sessions.has(date)) sessions.set(date, [])
      sessions.get(date).push({ time: t, value: iv })
    }

    // Always tear down and rebuild — sessions can drop in/out of view as
    // candles arrive, and lightweight-charts requires sorted unique times
    // per series. Per-session rebuild is simplest and cheap.
    tearDown.skipScale = true
    for (const s of atmIvSeriesRefs.current) {
      try { chart.removeSeries(s) } catch (err) { /* ignore */ }
    }
    atmIvSeriesRefs.current = []

    if (sessions.size === 0) {
      try {
        chart.priceScale('left').applyOptions({ visible: false })
      } catch (err) { /* ignore */ }
      return
    }

    const seriesOpts = {
      color: '#06b6d4',
      lineWidth: 2,
      priceScaleId: 'left',
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: true,
      pointMarkersVisible: false,
      priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
    }

    for (const [, points] of sessions) {
      if (points.length === 0) continue
      try {
        const series = chart.addSeries(LineSeries, seriesOpts)
        series.setData(points)
        atmIvSeriesRefs.current.push(series)
      } catch (err) {
        console.error('[atm-iv] failed to add session series', err)
      }
    }

    if (atmIvSeriesRefs.current.length === 0) {
      try {
        chart.priceScale('left').applyOptions({ visible: false })
      } catch (err) { /* ignore */ }
      atmIvScaleVisibleRef.current = false
      return
    }

    // Only apply `autoScale: true` on the initial enable. Subsequent runs
    // (polling, new candles, interval flips) must preserve the user's
    // manual IV zoom/pan — re-applying autoScale would refit the scale and
    // erase that interaction.
    const isInitialEnable = !atmIvScaleVisibleRef.current
    try {
      const opts = {
        visible: true,
        borderVisible: true,
        borderColor: 'rgba(6, 182, 212, 0.45)',
        scaleMargins: { top: 0.08, bottom: 0.08 },
      }
      if (isInitialEnable) opts.autoScale = true
      chart.priceScale('left').applyOptions(opts)
      atmIvScaleVisibleRef.current = true
    } catch (err) {
      console.error('[atm-iv] failed to configure price scale', err)
    }

    // The left axis becoming visible/hidden shifts the chart's main pane
    // horizontally, which moves where session-band divs should sit. Recompute
    // them on the next frame so the band offsets pick up the new axis width.
    requestAnimationFrame(() => {
      try {
        const next = computeSessionBands(
          chart,
          shiftedCandlesRef.current,
          stageRef.current?.clientWidth || 0
        )
        setSessionBands(next)
      } catch (err) {
        // ignore
      }
    })
  }, [atmIvLineEnabled, atmIvSeries, shiftedCandles, interval])

  const handleFloatingMouseDown = (e) => {
    // Only drag if not clicking on collapse button
    if (e.target.closest('.smile-collapse-btn')) return
    e.stopPropagation()

    floatingDragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startTop: floatingPos.top ?? NaN,
      startRight: floatingPos.right ?? NaN,
      startBottom: floatingPos.bottom ?? NaN,
      startLeft: floatingPos.left ?? NaN,
    }
    window.addEventListener('mousemove', handleFloatingMouseMove)
    window.addEventListener('mouseup', handleFloatingMouseUp)
  }

  const handleFloatingMouseMove = (e) => {
    if (!floatingDragRef.current) return
    const { startX, startY, startTop, startRight, startBottom, startLeft } = floatingDragRef.current
    const dx = e.clientX - startX
    const dy = e.clientY - startY

    const next = {}
    if (Number.isFinite(startTop)) next.top = startTop + dy
    if (Number.isFinite(startBottom)) next.bottom = startBottom - dy
    if (Number.isFinite(startLeft)) next.left = startLeft + dx
    if (Number.isFinite(startRight)) next.right = startRight - dx

    setFloatingPos(next)
  }

  const handleFloatingMouseUp = () => {
    window.removeEventListener('mousemove', handleFloatingMouseMove)
    window.removeEventListener('mouseup', handleFloatingMouseUp)
    if (floatingPos) {
      window.localStorage.setItem('ib-react-skew-window-pos', JSON.stringify(floatingPos))
    }
    floatingDragRef.current = null
  }

  const handleSmileResizeMouseDown = (e, corner) => {
    e.stopPropagation()
    smileResizeRef.current = {
      corner,
      startX: e.clientX,
      startY: e.clientY,
      startWidth: smileWindowSize.width,
      startHeight: smileWindowSize.height,
      startTop: floatingPos.top ?? NaN,
      startLeft: floatingPos.left ?? NaN,
      startBottom: floatingPos.bottom ?? NaN,
      startRight: floatingPos.right ?? NaN,
    }
    window.addEventListener('mousemove', handleSmileResizeMouseMove)
    window.addEventListener('mouseup', handleSmileResizeMouseUp)
  }

  const handleSmileResizeMouseMove = (e) => {
    if (!smileResizeRef.current) return
    const { corner, startX, startY, startWidth, startHeight, startTop, startLeft, startBottom, startRight } = smileResizeRef.current
    const dx = e.clientX - startX
    const dy = e.clientY - startY

    let nw = startWidth
    let nh = startHeight
    const nextPos = { ...floatingPos }

    // Width and Horizontal Position
    if (corner.includes('right')) {
      const adjDx = Math.max(200 - startWidth, dx)
      nw = startWidth + adjDx
      if (Number.isFinite(startRight)) nextPos.right = startRight - adjDx
    } else if (corner.includes('left')) {
      const adjDx = Math.min(startWidth - 200, dx)
      nw = startWidth - adjDx
      if (Number.isFinite(startLeft)) nextPos.left = startLeft + adjDx
    }

    // Height and Vertical Position
    if (corner.includes('bottom')) {
      const adjDy = Math.max(150 - startHeight, dy)
      nh = startHeight + adjDy
      if (Number.isFinite(startBottom)) nextPos.bottom = startBottom - adjDy
    } else if (corner.includes('top')) {
      const adjDy = Math.min(startHeight - 150, dy)
      nh = startHeight - adjDy
      if (Number.isFinite(startTop)) nextPos.top = startTop + adjDy
    }

    setSmileWindowSize({ width: nw, height: nh })
    setFloatingPos(nextPos)
  }

  const handleSmileResizeMouseUp = () => {
    window.removeEventListener('mousemove', handleSmileResizeMouseMove)
    window.removeEventListener('mouseup', handleSmileResizeMouseUp)
    window.localStorage.setItem('ib-react-smile-window-size', JSON.stringify(smileWindowSize))
    window.localStorage.setItem('ib-react-skew-window-pos', JSON.stringify(floatingPos))
    smileResizeRef.current = null
  }

  const toggleSmileCollapsed = (e) => {
    e.stopPropagation()
    setSmileCollapsed(prev => {
      const next = !prev
      // Open/closed state is intentionally not persisted — always starts closed on reload
      try {
        window.localStorage.removeItem('ib-react-smile-collapsed')
      } catch (e) {}
      return next
    })
  }

  const handleRemoveTimeslice = (timeToRemove) => {
    setLocalSelectedTimes((prev) => {
      const next = normalizeTimes(prev.filter((x) => x !== timeToRemove))
      try {
        window.parent.postMessage({ type: 'ib-react-timeslices', times: next }, '*')
      } catch (err) {
        console.error('postMessage failed', err)
      }
      return next
    })
  }

  return (
    <div className="chart-shell chart-shell-compact">
      <div className="chart-frame chart-frame-compact">
        <div
          ref={stageRef}
          className="chart-stage chart-stage-compact"
          style={{ position: 'relative' }}
        >
          <div
            style={{
              position: 'absolute',
              top: '8px',
              left: '12px',
              zIndex: 6,
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
          >
            <button
              type="button"
              onClick={openSettings}
              title="Price chart settings"
              aria-label="Open price chart settings"
              style={{
                width: '44px',
                height: '44px',
                borderRadius: '10px',
                border: '1px solid rgba(148, 163, 184, 0.22)',
                background: 'rgba(15, 23, 42, 0.96)',
                color: '#cbd5e1',
                cursor: 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '24px',
                lineHeight: 1,
                padding: 0,
              }}
            >
              ⚙
            </button>
          </div>

          <div
            onMouseDown={handleFloatingMouseDown}
            onWheel={(e) => e.stopPropagation()}
            onClick={smileCollapsed ? toggleSmileCollapsed : (e) => e.stopPropagation()}
            style={{
              position: 'absolute',
              zIndex: 10,
              cursor: smileCollapsed ? 'pointer' : 'grab',
              background: 'rgba(15, 23, 42, 0.92)',
              border: '1px solid #1f2937',
              borderRadius: '12px',
              padding: smileCollapsed ? '6px 14px' : '10px 14px',
              boxShadow: '0 10px 25px rgba(0,0,0,0.4)',
              minWidth: smileCollapsed ? 'auto' : '280px',
              width: smileCollapsed ? 'auto' : smileWindowSize.width || 'auto',
              height: smileCollapsed ? 'auto' : smileWindowSize.height || 'auto',
              color: '#e2e8f0',
              fontSize: '13px',
              pointerEvents: 'auto',
              userSelect: 'none',
              ...(smileCollapsed ? { top: 8, left: 64, borderRadius: '10px' } : floatingPos),
              transition: 'all 0.2s ease',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden'
            }}
          >
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: smileCollapsed ? '0' : '8px',
              height: smileCollapsed ? '32px' : 'auto'
            }}>
              <div style={{
                fontWeight: 800,
                color: smileCollapsed ? '#60a5fa' : '#94a3b8',
                fontSize: smileCollapsed ? '13px' : '11px',
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}>
                {smileCollapsed ? 'SMILE' : 'Overview'}
              </div>
              {!smileCollapsed && (
                <button
                  className="smile-collapse-btn"
                  onClick={toggleSmileCollapsed}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: '#94a3b8',
                    cursor: 'pointer',
                    padding: '2px',
                    fontSize: '14px',
                    lineHeight: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                >
                  −
                </button>
              )}
            </div>

            {!smileCollapsed && (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
                {smileData && smileData.traces && smileData.traces.length > 0 && (
                  <div style={{ flex: '1', minHeight: 0, marginBottom: '4px' }}>
                    <SmileChart
                      data={smileData}
                      width={smileWindowSize.width - 28}
                      height={Math.max(150, smileWindowSize.height - (skewData?.length ? (skewData.length * 25 + 80) : 60))}
                      snapshots={smileSnapshots}
                      activeAnchorTime={activeBandsAnchorTime}
                      onAnchorChange={onBandsAnchorChange}
                    />
                  </div>
                )}

                <div style={{ flex: '0 0 auto', overflowY: 'auto' }}>
                  {skewData && skewData.length > 0 ? (
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ color: '#94a3b8', fontSize: '10px', borderBottom: '1px solid #334155' }}>
                          <th style={{ textAlign: 'left', padding: '4px 0' }}>Time</th>
                          <th style={{ textAlign: 'right', padding: '4px 0' }}>Δ ATM%</th>
                          <th style={{ textAlign: 'right', padding: '4px 0' }}>Δ Call%</th>
                          <th style={{ textAlign: 'right', padding: '4px 0' }}>Δ Put%</th>
                          <th style={{ textAlign: 'right', padding: '4px 0' }}>Exp Move</th>
                          <th style={{ width: '20px' }}></th>
                        </tr>
                      </thead>
                      <tbody style={{ fontFamily: 'ui-monospace, monospace', fontSize: '12px' }}>
                        {skewData.map((row, idx) => (
                          <tr key={idx} style={{ borderBottom: idx === skewData.length - 1 ? '0' : '1px solid #1e293b' }}>
                            <td style={{ padding: '6px 0', fontWeight: 600 }}>{row.time}</td>
                            <td style={{ textAlign: 'right', color: row.d_atm > 0 ? '#4ade80' : row.d_atm < 0 ? '#f87171' : 'inherit' }}>
                              {row.d_atm != null ? row.d_atm.toFixed(2) : '--'}
                            </td>
                            <td style={{ textAlign: 'right', color: row.d_call > 0 ? '#4ade80' : row.d_call < 0 ? '#f87171' : 'inherit' }}>
                              {row.d_call != null ? row.d_call.toFixed(2) : '--'}
                            </td>
                            <td style={{ textAlign: 'right', color: row.d_put > 0 ? '#4ade80' : row.d_put < 0 ? '#f87171' : 'inherit' }}>
                              {row.d_put != null ? row.d_put.toFixed(2) : '--'}
                            </td>
                            <td style={{ textAlign: 'right', color: '#94a3b8' }}>
                              {row.exp_move != null ? row.exp_move.toFixed(2) : '--'}
                            </td>
                            <td style={{ textAlign: 'center', padding: '6px 0' }}>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation()
                                  handleRemoveTimeslice(row.time)
                                }}
                                style={{
                                  background: 'transparent',
                                  border: 'none',
                                  color: '#64748b',
                                  cursor: 'pointer',
                                  padding: '0 4px',
                                  fontSize: '11px',
                                  lineHeight: 1,
                                  display: 'inline-flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                  transition: 'color 0.15s'
                                }}
                                onMouseEnter={(e) => e.currentTarget.style.color = '#f87171'}
                                onMouseLeave={(e) => e.currentTarget.style.color = '#64748b'}
                                title={`Remove ${row.time}`}
                              >
                                ✕
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <div style={{ color: '#64748b', fontSize: '12px', fontStyle: 'italic', textAlign: 'center', padding: '10px 0' }}>
                      Select time slices to view data
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Resize Handles */}
            {!smileCollapsed && (
              <>
                <div onMouseDown={(e) => handleSmileResizeMouseDown(e, 'top-left')} style={{ position: 'absolute', top: 0, left: 0, width: '100px', height: '10px', cursor: 'nwse-resize', zIndex: 11 }} />
                <div onMouseDown={(e) => handleSmileResizeMouseDown(e, 'top-right')} style={{ position: 'absolute', top: 0, right: 0, width: '100px', height: '10px', cursor: 'nesw-resize', zIndex: 11 }} />
                <div onMouseDown={(e) => handleSmileResizeMouseDown(e, 'bottom-left')} style={{ position: 'absolute', bottom: 0, left: 0, width: '100px', height: '10px', cursor: 'nesw-resize', zIndex: 11 }} />
                <div onMouseDown={(e) => handleSmileResizeMouseDown(e, 'bottom-right')} style={{ position: 'absolute', bottom: 0, right: 0, width: '100px', height: '10px', cursor: 'nwse-resize', zIndex: 11 }} />
              </>
            )}
          </div>

          {/* Trade Annotation Panel — appears when a Trade Log trade is being annotated */}
          {annotationState && (
            <TradeAnnotationPanel
              annotationState={annotationState}
              selectedTimes={localSelectedTimes}
              tradeDate={tradeDate}
              onSaved={clearAnnotationState}
              onCancel={clearAnnotationState}
            />
          )}

          {/* Signal Panel — separate floating window, self-managed fetch */}
          <SignalPanel
            tradeDate={tradeDate}
            isToday={tradeDate === new Date().toISOString().slice(0, 10)}
          />

          {settingsOpen && (
            <div
              onClick={closeSettings}
              style={{
                position: 'fixed',
                inset: 0,
                background: 'rgba(2, 6, 23, 0.58)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 1200,
                padding: '24px',
              }}
            >
              <form
                onSubmit={applySettings}
                onClick={(event) => event.stopPropagation()}
                style={{
                  width: 'min(420px, calc(100vw - 32px))',
                  borderRadius: '16px',
                  border: '1px solid #1f2937',
                  background: '#0b1220',
                  boxShadow: '0 20px 60px rgba(0, 0, 0, 0.42)',
                  padding: '18px',
                  color: '#e2e8f0',
                }}
              >
                <div style={{ fontSize: '16px', fontWeight: 800, marginBottom: '14px' }}>
                  Price chart settings
                </div>

                <div style={{ fontSize: '12px', color: '#cbd5e1', marginBottom: '8px' }}>
                  Bar Interval
                </div>
                <div style={{ display: 'flex', gap: '8px', marginBottom: '14px' }}>
                  {['1min', '5min'].map((value) => {
                    const active = draftInterval === value
                    return (
                      <button
                        key={value}
                        type="button"
                        onClick={() => setDraftInterval(value)}
                        style={{
                          borderRadius: '10px',
                          border: `1px solid ${active ? '#3b82f6' : '#334155'}`,
                          background: active ? 'rgba(37, 99, 235, 0.20)' : '#020617',
                          color: active ? '#dbeafe' : '#cbd5e1',
                          padding: '8px 12px',
                          cursor: 'pointer',
                          fontWeight: 700,
                        }}
                      >
                        {value === '1min' ? '1 min' : '5 min'}
                      </button>
                    )
                  })}
                </div>

                <label
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    marginBottom: '14px',
                    fontSize: '12px',
                    color: '#cbd5e1',
                    cursor: 'pointer',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={draftAtmIvLineEnabled}
                    onChange={(event) => setDraftAtmIvLineEnabled(event.target.checked)}
                    style={{ accentColor: '#06b6d4' }}
                  />
                  <span>
                    Show 0DTE ATM IV line{' '}
                    <span style={{ color: '#06b6d4', fontWeight: 700 }}>(cyan)</span>
                  </span>
                </label>

                <label style={{ display: 'block', fontSize: '12px', color: '#cbd5e1', marginBottom: '8px' }}>
                  Min |GEX| (B)
                </label>

                <input
                  type="range"
                  min={0}
                  max={200}
                  step={1}
                  value={draftGexMinAbsB}
                  onChange={(event) => {
                    setDraftGexMinAbsB(event.target.value)
                    if (settingsError) setSettingsError('')
                  }}
                  style={{ width: '100%', marginBottom: '8px' }}
                />

                <div
                  style={{
                    position: 'relative',
                    height: '16px',
                    marginBottom: '8px',
                    color: '#94a3b8',
                    fontSize: '11px',
                  }}
                >
                  {[0, 10, 25, 50, 100, 150, 200].map((mark) => {
                    const left = `${(mark / 200) * 100}%`
                    let transform = 'translateX(-50%)'
                    if (mark === 0) transform = 'translateX(0)'
                    if (mark === 200) transform = 'translateX(-100%)'
                    return (
                      <span
                        key={mark}
                        style={{
                          position: 'absolute',
                          left,
                          top: 0,
                          transform,
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {mark}
                      </span>
                    )
                  })}
                </div>

                <div
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minWidth: '72px',
                    padding: '8px 12px',
                    borderRadius: '10px',
                    background: '#020617',
                    border: '1px solid #334155',
                    fontSize: '18px',
                    fontWeight: 800,
                    fontVariantNumeric: 'tabular-nums',
                    marginBottom: '10px',
                  }}
                >
                  {coerceGexMinAbsB(draftGexMinAbsB, 10)}
                </div>

                <div style={{ fontSize: '11px', color: '#94a3b8', lineHeight: 1.45 }}>
                  Only plot GEX levels whose absolute net gamma is at or above this threshold, in billions.
                </div>

                {settingsError && (
                  <div style={{ marginTop: '10px', fontSize: '12px', color: '#fca5a5' }}>
                    {settingsError}
                  </div>
                )}

                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '18px' }}>
                  <button
                    type="button"
                    onClick={closeSettings}
                    style={{
                      borderRadius: '10px',
                      border: '1px solid #334155',
                      background: 'transparent',
                      color: '#cbd5e1',
                      padding: '8px 12px',
                      cursor: 'pointer',
                    }}
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    style={{
                      borderRadius: '10px',
                      border: '1px solid #2563eb',
                      background: '#1d4ed8',
                      color: 'white',
                      padding: '8px 12px',
                      cursor: 'pointer',
                      fontWeight: 700,
                        }}
                  >
                    Apply
                  </button>
                </div>
              </form>
            </div>
          )}

          {sessionBands.map((band) => (
            <div
              key={band.key}
              className="session-band"
              style={{ left: `${band.left}px`, width: `${band.width}px` }}
            />
          ))}

          <div
            ref={tooltipRef}
            style={{
              position: 'absolute',
              left: 0,
              top: 0,
              transform: 'translate(-9999px, -9999px)',
              opacity: 0,
              pointerEvents: 'none',
              zIndex: 4,
              minWidth: '132px',
              padding: '8px 10px',
              borderRadius: '8px',
              background: 'rgba(15, 23, 42, 0.94)',
              border: '1px solid rgba(148, 163, 184, 0.28)',
              boxShadow: '0 8px 24px rgba(0, 0, 0, 0.28)',
              color: '#e5e7eb',
              fontSize: '12px',
              lineHeight: 1.25,
              whiteSpace: 'nowrap',
            }}
          >
            <div style={{ color: '#94a3b8', marginBottom: '2px' }}>Cursor</div>
            <div ref={tooltipTimeRef} style={{ fontVariantNumeric: 'tabular-nums' }}>
              --
            </div>
            <div
              ref={tooltipPriceRef}
              style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}
            >
              --
            </div>
          </div>

          {clickedGexSegment && (() => {
            // Layout constants for the GEX popup
            const POPUP_WIDTH = 280
            const POPUP_MAX_HEIGHT = 360
            const OFFSET = 14
            const PAD = 8
            const stageRect = stageRef.current?.getBoundingClientRect()
            const stageW = stageRect?.width || 0
            const stageH = stageRect?.height || 0

            // Anchor near the click, flipping/clamping so the popup stays inside the chart
            let left = gexPopupPos.left + OFFSET
            let top = gexPopupPos.top + OFFSET
            if (left + POPUP_WIDTH > stageW - PAD) {
              left = gexPopupPos.left - POPUP_WIDTH - OFFSET
            }
            if (top + POPUP_MAX_HEIGHT > stageH - PAD) {
              top = Math.max(PAD, stageH - POPUP_MAX_HEIGHT - PAD)
            }
            if (left < PAD) left = PAD
            if (top < PAD) top = PAD

            // Format helpers are hoisted to module scope: fmtGammaB, fmtGexExpDate

            // Show all expirations from the API. Rows whose expir_date is before
            // the segment's session_date (or before today as a fallback) are still
            // shown but visually marked as expired so you can see what made up the
            // headline numbers without anything being silently filtered out.
            const allExpirations = Array.isArray(clickedGexSegment.expirations)
              ? clickedGexSegment.expirations
              : []
            // Sort chronologically (earliest first) for predictable reading
            const visibleExpirations = [...allExpirations].sort((a, b) => {
              const ad = String(a?.expir_date || '')
              const bd = String(b?.expir_date || '')
              return ad < bd ? -1 : (ad > bd ? 1 : 0)
            })
            // Use the segment's own session_date as the "as-of" for expired marking,
            // falling back to today if it's missing. This way clicking a Friday line
            // shows its 04/25 row as not-expired (which is correct for that session).
            const segSession = String(clickedGexSegment.session_date || '').trim()
            const todayIso = new Date().toISOString().slice(0, 10)
            const asOfDate = segSession || todayIso
            const hiddenCount = 0  // No longer hiding anything

            const netVal = Number(clickedGexSegment.net_gamma)
            const callVal = Number(clickedGexSegment.call_gamma)
            const putVal = Number(clickedGexSegment.put_gamma)
            const netColor = netVal > 0 ? '#86efac' : (netVal < 0 ? '#fca5a5' : '#e5e7eb')

            return (
              <div
                style={{
                  position: 'absolute',
                  left: `${left}px`,
                  top: `${top}px`,
                  width: `${POPUP_WIDTH}px`,
                  maxHeight: `${POPUP_MAX_HEIGHT}px`,
                  display: 'flex',
                  flexDirection: 'column',
                  zIndex: 6,
                  borderRadius: '10px',
                  background: 'rgba(15, 23, 42, 0.96)',
                  border: '1px solid rgba(148, 163, 184, 0.32)',
                  boxShadow: '0 12px 32px rgba(0, 0, 0, 0.42)',
                  color: '#e5e7eb',
                  fontSize: '12px',
                  lineHeight: 1.35,
                  pointerEvents: 'auto',
                }}
                onClick={(e) => e.stopPropagation()}
              >
                {/* Header */}
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '10px 12px 8px',
                    borderBottom: '1px solid rgba(148,163,184,0.18)',
                  }}
                >
                  <div>
                    <div style={{ color: '#94a3b8', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                      GEX Level
                    </div>
                    <div style={{ fontWeight: 700, fontSize: '15px', fontVariantNumeric: 'tabular-nums' }}>
                      {Number(clickedGexSegment.level).toFixed(0)}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setClickedGexSegment(null)}
                    aria-label="Close"
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: '#94a3b8',
                      fontSize: '18px',
                      lineHeight: 1,
                      cursor: 'pointer',
                      padding: '2px 6px',
                    }}
                  >
                    ×
                  </button>
                </div>

                {/* Totals */}
                <div style={{ padding: '8px 12px', borderBottom: '1px solid rgba(148,163,184,0.18)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <span style={{ color: '#94a3b8' }}>Net</span>
                    <span style={{ fontWeight: 700, color: netColor, fontVariantNumeric: 'tabular-nums' }}>
                      {fmtGammaB(netVal)}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
                    <span style={{ color: '#94a3b8' }}>Calls</span>
                    <span style={{ color: '#86efac', fontVariantNumeric: 'tabular-nums' }}>{fmtGammaB(callVal)}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#94a3b8' }}>Puts</span>
                    <span style={{ color: '#fca5a5', fontVariantNumeric: 'tabular-nums' }}>{fmtGammaB(putVal)}</span>
                  </div>
                </div>

                {/* Expirations table */}
                <div
                  style={{
                    flex: 1,
                    overflowY: 'auto',
                    padding: '6px 12px 10px',
                  }}
                >
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '1fr 36px 1fr',
                      columnGap: '8px',
                      color: '#94a3b8',
                      fontSize: '10.5px',
                      textTransform: 'uppercase',
                      letterSpacing: '0.04em',
                      paddingBottom: '4px',
                      borderBottom: '1px dashed rgba(148,163,184,0.18)',
                      marginBottom: '4px',
                    }}
                  >
                    <span>Exp</span>
                    <span style={{ textAlign: 'right' }}>DTE</span>
                    <span style={{ textAlign: 'right' }}>Net</span>
                  </div>

                  {visibleExpirations.length === 0 ? (
                    <div style={{ color: '#94a3b8', padding: '8px 0' }}>
                      No expirations to show.
                    </div>
                  ) : (
                    visibleExpirations.map((row, idx) => {
                      const rowNet = Number(row?.net_gamma)
                      const rowColor = rowNet > 0 ? '#86efac' : (rowNet < 0 ? '#fca5a5' : '#e5e7eb')
                      // A row is "expired" when its expir_date is strictly before the
                      // session/today as-of date. We use string compare on ISO dates,
                      // which is chronologically correct for YYYY-MM-DD.
                      const expIso = String(row?.expir_date || '')
                      const isExpired = expIso && asOfDate && expIso < asOfDate
                      return (
                        <div
                          key={`${row?.expir_date || idx}-${idx}`}
                          style={{
                            display: 'grid',
                            gridTemplateColumns: '1fr 36px 1fr',
                            columnGap: '8px',
                            padding: '3px 0',
                            fontVariantNumeric: 'tabular-nums',
                            opacity: isExpired ? 0.45 : 1,
                            textDecoration: isExpired ? 'line-through' : 'none',
                          }}
                          title={isExpired ? 'Expired before this session\'s as-of date' : undefined}
                        >
                          <span>
                            {fmtGexExpDate(row?.expir_date)}
                            {isExpired && (
                              <span style={{ marginLeft: '6px', fontSize: '10px', color: '#94a3b8', fontStyle: 'italic', textDecoration: 'none' }}>
                                exp
                              </span>
                            )}
                          </span>
                          <span style={{ textAlign: 'right', color: '#cbd5e1' }}>
                            {Number.isFinite(Number(row?.dte)) ? Number(row.dte) : "—"}
                          </span>
                          <span style={{ textAlign: 'right', color: rowColor, fontWeight: 600 }}>
                            {fmtGammaB(rowNet)}
                          </span>
                        </div>
                      )
                    })
                  )}

                  {hiddenCount > 0 && (
                    <div
                      style={{
                        marginTop: '6px',
                        paddingTop: '4px',
                        borderTop: '1px dashed rgba(148,163,184,0.18)',
                        color: '#64748b',
                        fontSize: '10.5px',
                        fontStyle: 'italic',
                      }}
                    >
                      {hiddenCount} expired row{hiddenCount === 1 ? '' : 's'} hidden
                    </div>
                  )}
                </div>
              </div>
            )
          })()}

          {/* GEX legend panel toggle (only shown when panel is closed and there are levels to list).
              Matches the SMILE / SIGNALS pill structure exactly: outer container with auto height,
              inner wrapper with fixed 32px height that defines the pill's vertical dimension. */}
          {!gexPanelOpen && Array.isArray(normalizedGexSegments) && normalizedGexSegments.length > 0 && gexEnabled && (
            <div
              onClick={() => setGexPanelOpen(true)}
              style={{
                position: 'absolute',
                top: 8,
                left: 240,
                zIndex: 10,
                cursor: 'pointer',
                background: 'rgba(15, 23, 42, 0.92)',
                border: '1px solid #1f2937',
                borderRadius: '10px',
                padding: '6px 14px',
                boxShadow: '0 10px 25px rgba(0,0,0,0.4)',
                color: '#e2e8f0',
                fontSize: '13px',
                pointerEvents: 'auto',
                userSelect: 'none',
                transition: 'all 0.2s ease',
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden',
              }}
              title="Show GEX legend"
            >
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: 0,
                  height: '32px',
                }}
              >
                <div
                  style={{
                    fontWeight: 800,
                    color: '#60a5fa',
                    fontSize: '13px',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                  }}
                >
                  GEX
                </div>
              </div>
            </div>
          )}

          {/* GEX legend panel — defaults to right-edge dock, but draggable anywhere via the header.
              Position persists across reloads. */}
          {gexPanelOpen && (
            <div
              onClick={(e) => e.stopPropagation()}
              style={{
                position: 'absolute',
                // When user has dragged, use their saved position; otherwise default top-right dock
                ...(gexPanelPos
                  ? { left: `${gexPanelPos.left}px`, top: `${gexPanelPos.top}px` }
                  : { top: '8px', right: `${PRICE_AXIS_HIT_WIDTH + 8}px`, bottom: `${TIME_AXIS_HEIGHT + 8}px` }),
                width: '320px',
                maxHeight: gexPanelPos ? '70vh' : 'calc(100% - 48px)',
                zIndex: 10,
                display: 'flex',
                flexDirection: 'column',
                borderRadius: '10px',
                background: 'rgba(15, 23, 42, 0.96)',
                border: '1px solid rgba(148, 163, 184, 0.32)',
                boxShadow: '0 12px 32px rgba(0, 0, 0, 0.42)',
                color: '#e5e7eb',
                fontSize: '12px',
                pointerEvents: 'auto',
              }}
            >
              {/* Header (also serves as drag handle — grab anywhere except the close button) */}
              <div
                onMouseDown={(e) => {
                  // Skip drag when the mousedown is on an interactive control —
                  // close button, slider, stepper buttons. Without this, dragging
                  // the slider thumb or clicking the +/− buttons would move the
                  // whole panel instead of using the control.
                  if (e.target.closest('.gex-panel-close')) return
                  if (e.target.closest('input, button')) return
                  e.preventDefault()
                  const stageRect = stageRef.current?.getBoundingClientRect()
                  if (!stageRect) return
                  // Compute current panel-left in chart-relative coordinates so default-anchored
                  // panels (no gexPanelPos yet) get a sensible starting offset to drag from.
                  const panelEl = e.currentTarget.parentElement
                  const panelRect = panelEl?.getBoundingClientRect()
                  const startLeft = panelRect ? panelRect.left - stageRect.left : 0
                  const startTop = panelRect ? panelRect.top - stageRect.top : 0
                  gexPanelDragRef.current = {
                    startCursorX: e.clientX,
                    startCursorY: e.clientY,
                    startLeft,
                    startTop,
                    panelW: panelRect?.width || 320,
                    panelH: panelRect?.height || 0,
                    stageW: stageRect.width,
                    stageH: stageRect.height,
                  }
                  const onMove = (ev) => {
                    const d = gexPanelDragRef.current
                    if (!d) return
                    const dx = ev.clientX - d.startCursorX
                    const dy = ev.clientY - d.startCursorY
                    // Clamp to keep at least 40px of header on screen so the user can always grab it back
                    const minLeft = 40 - d.panelW
                    const maxLeft = d.stageW - 40
                    const minTop = 0
                    const maxTop = d.stageH - 40
                    const left = Math.max(minLeft, Math.min(maxLeft, d.startLeft + dx))
                    const top = Math.max(minTop, Math.min(maxTop, d.startTop + dy))
                    setGexPanelPos({ left, top })
                  }
                  const onUp = () => {
                    gexPanelDragRef.current = null
                    window.removeEventListener('mousemove', onMove)
                    window.removeEventListener('mouseup', onUp)
                  }
                  window.addEventListener('mousemove', onMove)
                  window.addEventListener('mouseup', onUp)
                }}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  padding: '10px 12px 10px',
                  borderBottom: '1px solid rgba(148,163,184,0.18)',
                  flexShrink: 0,
                  gap: '10px',
                  cursor: 'grab',
                  userSelect: 'none',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                >
                  <div>
                    <div
                      style={{
                        color: '#94a3b8',
                        fontSize: '10px',
                        textTransform: 'uppercase',
                        letterSpacing: '0.06em',
                        fontWeight: 700,
                      }}
                    >
                      GEX Legend
                    </div>
                    <div style={{ fontSize: '12px', color: '#cbd5e1' }}>
                      {sortedGexSegmentsForPanel.length} level{sortedGexSegmentsForPanel.length === 1 ? '' : 's'}
                      <span style={{ color: '#64748b' }}> · sorted by |γ|</span>
                    </div>
                  </div>
                  <button
                    type="button"
                    className="gex-panel-close"
                    onClick={() => setGexPanelOpen(false)}
                    aria-label="Close"
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: '#94a3b8',
                      fontSize: '18px',
                      lineHeight: 1,
                      cursor: 'pointer',
                      padding: '2px 6px',
                    }}
                  >
                    ×
                  </button>
                </div>

                {/* Max DTE slider */}
                {maxDteAvailable > 0 && (
                  <div>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'baseline',
                        marginBottom: '4px',
                      }}
                    >
                      <span
                        style={{
                          color: '#94a3b8',
                          fontSize: '10px',
                          textTransform: 'uppercase',
                          letterSpacing: '0.06em',
                          fontWeight: 700,
                        }}
                      >
                        Max DTE
                      </span>
                      <span
                        style={{
                          fontSize: '11px',
                          color: '#cbd5e1',
                          fontVariantNumeric: 'tabular-nums',
                        }}
                      >
                        {gexPanelMaxDte == null
                          ? 'all'
                          : (gexPanelMaxDte === 0 ? '0DTE only' : `≤ ${gexPanelMaxDte}d`)}
                      </span>
                    </div>
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                      }}
                    >
                      <button
                        type="button"
                        onClick={() => {
                          // Decrement by 1, clamped to 0. If we were at "all", drop to maxDte - 1.
                          const current = gexPanelMaxDte == null ? maxDteAvailable : gexPanelMaxDte
                          const next = Math.max(0, current - 1)
                          setGexPanelMaxDte(next)
                        }}
                        disabled={(gexPanelMaxDte != null && gexPanelMaxDte <= 0)}
                        aria-label="Decrease max DTE by 1"
                        style={{
                          width: '24px',
                          height: '24px',
                          flexShrink: 0,
                          background: 'rgba(148, 163, 184, 0.12)',
                          border: '1px solid rgba(148, 163, 184, 0.24)',
                          borderRadius: '6px',
                          color: '#cbd5e1',
                          fontSize: '14px',
                          fontWeight: 700,
                          lineHeight: 1,
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          padding: 0,
                          opacity: (gexPanelMaxDte != null && gexPanelMaxDte <= 0) ? 0.4 : 1,
                        }}
                        title="Decrease by 1 day"
                      >
                        −
                      </button>
                      <input
                        type="range"
                        min={0}
                        max={maxDteAvailable}
                        step={1}
                        // Slider sits at maxDteAvailable when "all" is selected
                        value={gexPanelMaxDte == null ? maxDteAvailable : gexPanelMaxDte}
                        onChange={(e) => {
                          const n = parseInt(e.target.value, 10)
                          // Treat the rightmost position as "show all"
                          if (n >= maxDteAvailable) {
                            setGexPanelMaxDte(null)
                          } else {
                            setGexPanelMaxDte(Math.max(0, n))
                          }
                        }}
                        style={{
                          flex: 1,
                          accentColor: '#60a5fa',
                          cursor: 'pointer',
                          minWidth: 0,
                        }}
                      />
                      <button
                        type="button"
                        onClick={() => {
                          // Increment by 1. If we were at "all", stay there. If incrementing
                          // hits the max, snap to "all" so the slider visually reaches the end.
                          if (gexPanelMaxDte == null) return
                          const next = gexPanelMaxDte + 1
                          if (next >= maxDteAvailable) {
                            setGexPanelMaxDte(null)
                          } else {
                            setGexPanelMaxDte(next)
                          }
                        }}
                        disabled={gexPanelMaxDte == null}
                        aria-label="Increase max DTE by 1"
                        style={{
                          width: '24px',
                          height: '24px',
                          flexShrink: 0,
                          background: 'rgba(148, 163, 184, 0.12)',
                          border: '1px solid rgba(148, 163, 184, 0.24)',
                          borderRadius: '6px',
                          color: '#cbd5e1',
                          fontSize: '14px',
                          fontWeight: 700,
                          lineHeight: 1,
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          padding: 0,
                          opacity: gexPanelMaxDte == null ? 0.4 : 1,
                        }}
                        title="Increase by 1 day"
                      >
                        +
                      </button>
                    </div>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        color: '#64748b',
                        fontSize: '10px',
                        marginTop: '2px',
                        fontVariantNumeric: 'tabular-nums',
                      }}
                    >
                      <span>0</span>
                      <span>{maxDteAvailable}d (all)</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Rows */}
              <div style={{ flex: 1, overflowY: 'auto', overflowX: 'hidden' }}>
                {sortedGexSegmentsForPanel.length === 0 ? (
                  <div style={{ padding: '14px 12px', color: '#94a3b8' }}>
                    No GEX levels in view.
                  </div>
                ) : (
                  sortedGexSegmentsForPanel.map((seg, idx) => {
                    const lvl = Number(seg?.level)
                    if (!Number.isFinite(lvl)) return null

                    const netVal = Number(seg?.net_gamma)
                    const isExpanded = expandedPanelLevel === lvl
                    const netColor = netVal > 0 ? '#86efac' : (netVal < 0 ? '#fca5a5' : '#e5e7eb')

                    const allExp = Array.isArray(seg?.expirations) ? seg.expirations : []
                    const visibleExp = allExp.filter(
                      (e) => !Number.isFinite(Number(e?.dte)) || Number(e.dte) >= 0
                    )
                    const hidden = allExp.length - visibleExp.length

                    return (
                      <div
                        key={`gex-row-${lvl}-${idx}`}
                        onMouseEnter={() => setHoveredPanelLevel(lvl)}
                        onMouseLeave={() => setHoveredPanelLevel((cur) => (cur === lvl ? null : cur))}
                        style={{
                          borderBottom: '1px solid rgba(148, 163, 184, 0.10)',
                        }}
                      >
                        {/* Row summary (clickable to expand) */}
                        <div
                          onClick={() => setExpandedPanelLevel(isExpanded ? null : lvl)}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            padding: '8px 12px',
                            cursor: 'pointer',
                            background: isExpanded ? 'rgba(148,163,184,0.08)' : 'transparent',
                            transition: 'background 0.12s ease',
                          }}
                        >
                          <div
                            style={{
                              width: '8px',
                              height: '14px',
                              background: seg?.color || '#94a3b8',
                              borderRadius: '2px',
                              flexShrink: 0,
                            }}
                          />
                          <div
                            style={{
                              fontWeight: 700,
                              fontSize: '13px',
                              fontVariantNumeric: 'tabular-nums',
                              minWidth: '52px',
                            }}
                          >
                            {lvl.toFixed(0)}
                          </div>
                          <div
                            style={{
                              flex: 1,
                              textAlign: 'right',
                              color: netColor,
                              fontVariantNumeric: 'tabular-nums',
                              fontWeight: 600,
                            }}
                          >
                            {fmtGammaB(netVal)}
                          </div>
                          <div
                            style={{
                              color: '#64748b',
                              fontSize: '11px',
                              marginLeft: '4px',
                              width: '12px',
                              textAlign: 'center',
                            }}
                          >
                            {isExpanded ? '▾' : '▸'}
                          </div>
                        </div>

                        {/* Expanded expirations breakdown */}
                        {isExpanded && (
                          <div
                            style={{
                              padding: '6px 12px 10px',
                              background: 'rgba(2, 6, 23, 0.40)',
                            }}
                          >
                            <div
                              style={{
                                display: 'grid',
                                gridTemplateColumns: '1fr 36px 1fr',
                                columnGap: '8px',
                                color: '#94a3b8',
                                fontSize: '10.5px',
                                textTransform: 'uppercase',
                                letterSpacing: '0.04em',
                                paddingBottom: '4px',
                                borderBottom: '1px dashed rgba(148,163,184,0.18)',
                                marginBottom: '4px',
                              }}
                            >
                              <span>Exp</span>
                              <span style={{ textAlign: 'right' }}>DTE</span>
                              <span style={{ textAlign: 'right' }}>Net</span>
                            </div>

                            {visibleExp.length === 0 ? (
                              <div style={{ color: '#94a3b8', padding: '6px 0' }}>
                                No expirations to show.
                              </div>
                            ) : (
                              visibleExp.map((row, i) => {
                                const rowNet = Number(row?.net_gamma)
                                const rowColor =
                                  rowNet > 0 ? '#86efac' : (rowNet < 0 ? '#fca5a5' : '#e5e7eb')
                                return (
                                  <div
                                    key={`gex-exp-${lvl}-${row?.expir_date || i}-${i}`}
                                    style={{
                                      display: 'grid',
                                      gridTemplateColumns: '1fr 36px 1fr',
                                      columnGap: '8px',
                                      padding: '3px 0',
                                      fontVariantNumeric: 'tabular-nums',
                                    }}
                                  >
                                    <span>{fmtGexExpDate(row?.expir_date)}</span>
                                    <span style={{ textAlign: 'right', color: '#cbd5e1' }}>
                                      {Number.isFinite(Number(row?.dte)) ? Number(row.dte) : "—"}
                                    </span>
                                    <span
                                      style={{
                                        textAlign: 'right',
                                        color: rowColor,
                                        fontWeight: 600,
                                      }}
                                    >
                                      {fmtGammaB(rowNet)}
                                    </span>
                                  </div>
                                )
                              })
                            )}

                            {hidden > 0 && (
                              <div
                                style={{
                                  marginTop: '6px',
                                  paddingTop: '4px',
                                  borderTop: '1px dashed rgba(148,163,184,0.18)',
                                  color: '#64748b',
                                  fontSize: '10.5px',
                                  fontStyle: 'italic',
                                }}
                              >
                                {hidden} expired row{hidden === 1 ? '' : 's'} hidden
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )
                  })
                )}
              </div>
            </div>
          )}

          <div ref={hostRef} className="chart-host" style={{ width: '100%', height: '100%' }} />
        </div>
      </div>
    </div>
  )
}
