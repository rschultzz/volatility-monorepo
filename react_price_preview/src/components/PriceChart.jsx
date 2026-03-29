import { useEffect, useMemo, useRef, useState } from 'react'
import {
  createChart,
  CrosshairMode,
  ColorType,
  CandlestickSeries,
  LineSeries,
} from 'lightweight-charts'

const ETH_BG_COLOR = '#1f2937'
const PRICE_AXIS_HIT_WIDTH = 72
const TIME_AXIS_HEIGHT = 24
const MIN_PRICE_RANGE = 0.25
const MIN_CHART_HEIGHT = 180

const TOOLTIP_OFFSET_X = 14
const TOOLTIP_OFFSET_Y = 14
const TOOLTIP_EDGE_PAD = 8

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

function computeCenterLogicalRange(candles) {
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
  interval,
  initialSelectedTimes,
  gexSegments,
  gexEnabled,
  onVisibleLogicalRangeChange,
  onLinkedCrosshairChange,
}) {
  const stageRef = useRef(null)
  const hostRef = useRef(null)
  const chartRef = useRef(null)
  const seriesRef = useRef(null)
  const gexSeriesRefs = useRef([])
  const intervalRef = useRef(interval)
  const shiftedCandlesRef = useRef([])
  const dragRef = useRef({ active: false, lastY: 0 })
  const hasUserInteractedRef = useRef(false)
  const currentCenterKeyRef = useRef('')
  const appliedInitialRangeKeyRef = useRef('')
  const lastReportedRangeRef = useRef(null)

  const tooltipRef = useRef(null)
  const tooltipTimeRef = useRef(null)
  const tooltipPriceRef = useRef(null)

  const [selectedTimes, setSelectedTimes] = useState(normalizeTimes(initialSelectedTimes || []))
  const [sessionBands, setSessionBands] = useState([])

  useEffect(() => {
    setSelectedTimes(normalizeTimes(initialSelectedTimes || []))
  }, [initialSelectedTimes])

  useEffect(() => {
    intervalRef.current = interval
  }, [interval])

  const shiftedCandles = useMemo(() => {
    return (Array.isArray(candles) ? candles : []).map((bar) => ({
      ...bar,
      time: utcEpochShowingZoneTime(Number(bar.time), 'America/Los_Angeles'),
    }))
  }, [candles])

  const normalizedGexSegments = useMemo(
    () => normalizeGexSegments(gexSegments),
    [gexSegments]
  )

  const centerLogicalRange = useMemo(
    () => computeCenterLogicalRange(shiftedCandles),
    [shiftedCandles]
  )

  useEffect(() => {
    shiftedCandlesRef.current = shiftedCandles
  }, [shiftedCandles])

  useEffect(() => {
    const nextKey = centerLogicalRange?.key || ''
    if (nextKey && nextKey !== currentCenterKeyRef.current) {
      currentCenterKeyRef.current = nextKey
      appliedInitialRangeKeyRef.current = ''
      hasUserInteractedRef.current = false
      lastReportedRangeRef.current = null
    }
  }, [centerLogicalRange])

  const selectedSet = useMemo(() => new Set(selectedTimes), [selectedTimes])

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
      width: container.clientWidth || 900,
      height: Math.max(container.clientHeight || 0, MIN_CHART_HEIGHT),
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

    chartRef.current = chart
    seriesRef.current = candleSeries

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

      const shiftedEpoch = interpolateShiftedEpoch(
        logical,
        shiftedCandlesRef.current,
        intervalRef.current
      )

      if (!Number.isFinite(price) || !Number.isFinite(shiftedEpoch)) {
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
      reportLinkedCrosshair({ logical, shiftedTime: shiftedEpoch })
    }

    const handleResize = () => {
      chart.applyOptions({
        width: container.clientWidth || 900,
        height: Math.max(container.clientHeight || 0, MIN_CHART_HEIGHT),
      })
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
      resizeObserver.observe(container)
    }

    const handleClick = (param) => {
      if (!param?.time) return
      const hhmm = toPtHHMM(param.time, intervalRef.current)

      setSelectedTimes((prev) => {
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
      reportLogicalRange(range)
    }

    const handleCrosshairMove = (param) => {
      updateFloatingTooltip(param)
    }

    const handleWheel = (evt) => {
      hasUserInteractedRef.current = true
      const info = pointerInfo(evt, stage)
      if (!info || info.overTimeAxis || !info.overPriceAxis) return
      evt.preventDefault()
      evt.stopPropagation()
      zoomPriceAtY(evt.deltaY, info.y)
    }

    const handleMouseDown = (evt) => {
      if (evt.button !== 0) return
      hasUserInteractedRef.current = true
      const info = pointerInfo(evt, stage)
      if (!info || info.overTimeAxis || info.overPriceAxis) return
      dragRef.current = { active: true, lastY: evt.clientY }
    }

    const handleMouseMove = (evt) => {
      if (!dragRef.current.active) return
      const deltaY = evt.clientY - dragRef.current.lastY
      dragRef.current.lastY = evt.clientY
      if (deltaY !== 0) {
        panPriceByPixels(deltaY)
      }
    }

    const handleMouseUp = () => {
      dragRef.current.active = false
    }

    const handleMouseLeave = () => {
      hideTooltip()
      reportLinkedCrosshair(null)
      dragRef.current.active = false
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
      gexSeriesRefs.current = []
      dragRef.current.active = false
      reportLinkedCrosshair(null)
    }
  }, [onVisibleLogicalRangeChange])

  useEffect(() => {
    function handleParentMessage(event) {
      const data = event && event.data
      if (!data || data.type !== 'ib-parent-timeslices') return
      setSelectedTimes(normalizeTimes(data.times || []))
    }

    window.addEventListener('message', handleParentMessage)
    return () => window.removeEventListener('message', handleParentMessage)
  }, [])

  useEffect(() => {
    if (!seriesRef.current || !chartRef.current) return
    seriesRef.current.setData(displayCandles)

    requestAnimationFrame(() => {
      if (!chartRef.current) return

      setSessionBands(
        computeSessionBands(
          chartRef.current,
          shiftedCandles,
          stageRef.current?.clientWidth || 0
        )
      )

      if (centerLogicalRange && !hasUserInteractedRef.current) {
        if (appliedInitialRangeKeyRef.current !== centerLogicalRange.key) {
          chartRef.current.timeScale().setVisibleLogicalRange({
            from: centerLogicalRange.from,
            to: centerLogicalRange.to,
          })
          appliedInitialRangeKeyRef.current = centerLogicalRange.key

          const vr = normalizeLogicalRange(chartRef.current.timeScale().getVisibleLogicalRange?.())
          if (vr && typeof onVisibleLogicalRangeChange === 'function') {
            if (!rangesClose(lastReportedRangeRef.current, vr)) {
              lastReportedRangeRef.current = vr
              onVisibleLogicalRangeChange(vr)
            }
          }
        }
      } else if (!centerLogicalRange && !hasUserInteractedRef.current) {
        chartRef.current.timeScale().fitContent()

        const vr = normalizeLogicalRange(chartRef.current.timeScale().getVisibleLogicalRange?.())
        if (vr && typeof onVisibleLogicalRangeChange === 'function') {
          if (!rangesClose(lastReportedRangeRef.current, vr)) {
            lastReportedRangeRef.current = vr
            onVisibleLogicalRangeChange(vr)
          }
        }
      }
    })
  }, [displayCandles, shiftedCandles, centerLogicalRange, onVisibleLogicalRangeChange])

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

  return (
    <div className="chart-shell chart-shell-compact">
      <div className="chart-frame chart-frame-compact">
        <div
          ref={stageRef}
          className="chart-stage chart-stage-compact"
          style={{ position: 'relative' }}
        >
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

          <div ref={hostRef} className="chart-host" />
        </div>
      </div>
    </div>
  )
}