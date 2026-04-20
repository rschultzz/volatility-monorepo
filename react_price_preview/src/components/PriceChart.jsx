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

function pointerInfo(evt, container) {
  if (!evt || !container) return null
  const rect = container.getBoundingClientRect()
  const x = evt.clientX - rect.left
  const y = evt.clientY - rect.top
  const overTimeAxis = y > rect.height - TIME_AXIS_HEIGHT
  const overPriceAxis = x >= rect.width - PRICE_AXIS_HIT_WIDTH
  return { rect, x, y, overTimeAxis, overPriceAxis }
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
  gexMinAbsB = 10,
  expectedMoveLevels,
  onApplyGexMinAbsB,
  onApplyIntervalChange,
  onVisibleLogicalRangeChange,
  onLinkedCrosshairChange,
  onInteractionActiveChange,
  skewData = [],
  smileData = null,
}) {
  const stageRef = useRef(null)
  const hostRef = useRef(null)
  const chartRef = useRef(null)
  const seriesRef = useRef(null)
  const liveTradeSeriesRef = useRef(null)
  const gexSeriesRefs = useRef([])
  const expectedMoveSeriesRefs = useRef([])
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
  const [draftGexMinAbsB, setDraftGexMinAbsB] = useState(() => coerceGexMinAbsB(gexMinAbsB, 10))
  const [draftInterval, setDraftInterval] = useState(() => normalizeIntervalValue(interval, '1min'))
  const [settingsError, setSettingsError] = useState('')

  const [floatingPos, setFloatingPos] = useState(() => {
    try {
      const saved = window.localStorage.getItem('ib-react-skew-window-pos')
      if (saved) return JSON.parse(saved)
    } catch (e) {}
    return { top: 12, right: 12 }
  })
  const floatingDragRef = useRef(null)

  const [smileWindowSize, setSmileWindowSize] = useState(() => {
    try {
      const saved = window.localStorage.getItem('ib-react-smile-window-size')
      if (saved) return JSON.parse(saved)
    } catch (e) {}
    return { width: 0, height: 0 } // 0 means calculate from stage
  })
  const [smileCollapsed, setSmileCollapsed] = useState(() => {
    try {
      const saved = window.localStorage.getItem('ib-react-smile-collapsed')
      if (saved) return saved === 'true'
    } catch (e) {}
    return false
  })
  const smileResizeRef = useRef(null)

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
      setSettingsError('')
    }
  }, [gexMinAbsB, interval, settingsOpen])

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

    const handleWheel = (evt) => {
      hasUserInteractedRef.current = true
      setInteractionActive(true)
      lastManualTimeRangeRef.current = logicalRangeToShiftedTimeRange(
        chart.timeScale().getVisibleLogicalRange?.(),
        shiftedCandlesRef.current,
        intervalRef.current
      )
      const info = pointerInfo(evt, stage)
      if (!info || info.overTimeAxis || !info.overPriceAxis) {
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
      const info = pointerInfo(evt, stage)
      if (!info || info.overTimeAxis || info.overPriceAxis) return
      dragRef.current = { active: true, lastY: evt.clientY }
    }

    const handleMouseMove = (evt) => {
      if (!dragRef.current.active) return
      setInteractionActive(true)
      const deltaY = evt.clientY - dragRef.current.lastY
      dragRef.current.lastY = evt.clientY
      if (deltaY !== 0) {
        panPriceByPixels(deltaY)
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
    setSettingsError('')
    setSettingsOpen(true)
  }

  function closeSettings() {
    setSettingsOpen(false)
    setSettingsError('')
    setDraftGexMinAbsB(coerceGexMinAbsB(gexMinAbsB, 10))
    setDraftInterval(normalizeIntervalValue(interval, '1min'))
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

    if (!gexEnabled || !Array.isArray(normalizedGexSegments) || !normalizedGexSegments.length) {
      return
    }

    const nextRefs = []

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
      } catch (err) {
        // ignore bad segment
      }
    }

    gexSeriesRefs.current = nextRefs
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
      window.localStorage.setItem('ib-react-smile-collapsed', String(next))
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
            onClick={(e) => e.stopPropagation()}
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
            onClick={smileCollapsed ? toggleSmileCollapsed : (e) => e.stopPropagation()}
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

          <div ref={hostRef} className="chart-host" style={{ width: '100%', height: '100%' }} />
        </div>
      </div>
    </div>
  )
}
