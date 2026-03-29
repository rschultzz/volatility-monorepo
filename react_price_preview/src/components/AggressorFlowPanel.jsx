import { useEffect, useMemo, useRef, useState } from 'react'
import {
  createChart,
  CrosshairMode,
  ColorType,
  BaselineSeries,
  LineSeries,
} from 'lightweight-charts'

const BG_COLOR = '#1f2937'
const ZERO_COLOR = 'rgba(255,255,255,0.20)'
const MIN_PANEL_HEIGHT = 140
const POS_RGBA = (alpha) => `rgba(96,165,250,${alpha})`
const NEG_RGBA = (alpha) => `rgba(229,231,235,${alpha})`

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

function formatShiftedHHMM(epochSec) {
  const dt = shiftedEpochToDate(epochSec)
  const hh = String(dt.getUTCHours()).padStart(2, '0')
  const mm = String(dt.getUTCMinutes()).padStart(2, '0')
  return `${hh}:${mm}`
}

function normalizeLogicalRange(range) {
  if (!range) return null
  const from = Number(range.from)
  const to = Number(range.to)
  if (!Number.isFinite(from) || !Number.isFinite(to)) return null
  return { from, to }
}

function rangesClose(a, b, eps = 0.001) {
  if (!a || !b) return false
  return (
    Math.abs(Number(a.from) - Number(b.from)) <= eps &&
    Math.abs(Number(a.to) - Number(b.to)) <= eps
  )
}

function normalizeShiftedFlowPoints(points) {
  return (Array.isArray(points) ? points : [])
    .map((pt) => {
      const rawTime = Number(pt?.time)
      const value = Number(pt?.value)
      if (!Number.isFinite(rawTime) || !Number.isFinite(value)) return null
      return {
        time: utcEpochShowingZoneTime(rawTime, 'America/Los_Angeles'),
        value,
      }
    })
    .filter(Boolean)
    .sort((a, b) => a.time - b.time)
}

function normalizeShiftedCandleTimes(candles) {
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

function inferStepSeconds(times, fallback = 60) {
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

function bucketFlowToCandles(flowPoints, candleTimes) {
  const points = normalizeShiftedFlowPoints(flowPoints)
  const times = normalizeShiftedCandleTimes(candleTimes)

  if (!times.length) {
    return points.map((pt) => ({
      time: pt.time,
      value: pt.value,
    }))
  }

  const step = inferStepSeconds(times, 60)
  const out = []
  let pointIdx = 0

  for (let i = 0; i < times.length; i += 1) {
    const bucketStart = times[i]
    const bucketEnd = i + 1 < times.length ? times[i + 1] : bucketStart + step

    while (pointIdx < points.length && points[pointIdx].time < bucketStart) {
      pointIdx += 1
    }

    let sum = 0
    let count = 0
    let scanIdx = pointIdx

    while (scanIdx < points.length && points[scanIdx].time < bucketEnd) {
      sum += points[scanIdx].value
      count += 1
      scanIdx += 1
    }

    pointIdx = scanIdx

    if (count > 0) {
      out.push({
        time: bucketStart,
        value: sum / count,
      })
    } else {
      out.push({ time: bucketStart })
    }
  }

  return out
}

function hasFlowValues(points) {
  return (Array.isArray(points) ? points : []).some((pt) => Number.isFinite(Number(pt?.value)))
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

function computeSessionBands(chart, shiftedTimes, viewportWidth) {
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

function coordinateForLinkedCrosshair(chart, linkedCrosshair) {
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

export default function AggressorFlowPanel({
  candles,
  dataPoints,
  visibleLogicalRange,
  linkedCrosshair,
  height = 220,
  loading = false,
  error = '',
  histAlpha = 0.30,
}) {
  const stageRef = useRef(null)
  const hostRef = useRef(null)
  const chartRef = useRef(null)
  const flowSeriesRef = useRef(null)
  const zeroSeriesRef = useRef(null)
  const lastAppliedLogicalRef = useRef(null)
  const candleTimesRef = useRef([])
  const linkedCrosshairRef = useRef(null)
  const refreshDecorationsRef = useRef(() => {})

  const [sessionBands, setSessionBands] = useState([])
  const [linkedCrosshairX, setLinkedCrosshairX] = useState(null)

  const candleTimes = useMemo(() => normalizeShiftedCandleTimes(candles), [candles])

  const alignedPoints = useMemo(
    () => bucketFlowToCandles(dataPoints, candles),
    [dataPoints, candles]
  )

  const hasRenderablePoints = candleTimes.length >= 2
  const hasAnyFlowValue = useMemo(() => hasFlowValues(alignedPoints), [alignedPoints])

  useEffect(() => {
    candleTimesRef.current = candleTimes
  }, [candleTimes])

  useEffect(() => {
    linkedCrosshairRef.current = linkedCrosshair || null
  }, [linkedCrosshair])

  useEffect(() => {
    if (!hostRef.current || !stageRef.current) return undefined

    const container = hostRef.current
    const chart = createChart(container, {
      width: container.clientWidth || 900,
      height: Math.max(container.clientHeight || MIN_PANEL_HEIGHT, MIN_PANEL_HEIGHT),
      layout: {
        background: { type: ColorType.Solid, color: BG_COLOR },
        textColor: '#cbd5e1',
        attributionLogo: false,
      },
      localization: { locale: 'en-US' },
      grid: {
        vertLines: { color: 'rgba(148, 163, 184, 0.08)' },
        horzLines: { color: 'rgba(148, 163, 184, 0.08)' },
      },
      rightPriceScale: {
        borderColor: 'rgba(148, 163, 184, 0.18)',
        autoScale: true,
        scaleMargins: { top: 0.12, bottom: 0.12 },
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
          visible: false,
          labelVisible: false,
        },
        horzLine: {
          visible: false,
          labelVisible: false,
        },
      },
      handleScroll: {
        mouseWheel: false,
        pressedMouseMove: false,
        horzTouchDrag: false,
        vertTouchDrag: false,
      },
      handleScale: {
        axisPressedMouseMove: false,
        mouseWheel: false,
        pinch: false,
      },
    })

    const fillAlpha = Number.isFinite(Number(histAlpha))
      ? Math.max(0.05, Math.min(1, Number(histAlpha)))
      : 0.30

    const flowSeries = chart.addSeries(BaselineSeries, {
      baseValue: { type: 'price', price: 0 },
      topLineColor: POS_RGBA(0.95),
      topFillColor1: POS_RGBA(fillAlpha),
      topFillColor2: POS_RGBA(fillAlpha * 0.65),
      bottomLineColor: NEG_RGBA(0.95),
      bottomFillColor1: NEG_RGBA(fillAlpha),
      bottomFillColor2: NEG_RGBA(fillAlpha * 0.65),
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
      pointMarkersVisible: false,
    })

    const zeroSeries = chart.addSeries(LineSeries, {
      color: ZERO_COLOR,
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
      pointMarkersVisible: false,
    })

    chartRef.current = chart
    flowSeriesRef.current = flowSeries
    zeroSeriesRef.current = zeroSeries

    const refreshDecorations = () => {
      const stageWidth = stageRef.current?.clientWidth || 0
      const nextBands = computeSessionBands(chart, candleTimesRef.current, stageWidth)
      setSessionBands(nextBands)

      const x = coordinateForLinkedCrosshair(chart, linkedCrosshairRef.current)
      if (!Number.isFinite(x) || x < 0 || x > stageWidth) {
        setLinkedCrosshairX(null)
      } else {
        setLinkedCrosshairX(x)
      }
    }

    refreshDecorationsRef.current = refreshDecorations

    const handleResize = () => {
      chart.applyOptions({
        width: container.clientWidth || 900,
        height: Math.max(container.clientHeight || MIN_PANEL_HEIGHT, MIN_PANEL_HEIGHT),
      })
      requestAnimationFrame(refreshDecorations)
    }

    const handleVisibleLogicalRangeChange = () => {
      requestAnimationFrame(refreshDecorations)
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

    chart.timeScale().subscribeVisibleLogicalRangeChange(handleVisibleLogicalRangeChange)
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(handleVisibleLogicalRangeChange)
      if (resizeObserver) {
        resizeObserver.disconnect()
      }
      chart.remove()
      chartRef.current = null
      flowSeriesRef.current = null
      zeroSeriesRef.current = null
      lastAppliedLogicalRef.current = null
      refreshDecorationsRef.current = () => {}
    }
  }, [histAlpha])

  useEffect(() => {
    if (!flowSeriesRef.current || !zeroSeriesRef.current) return

    flowSeriesRef.current.setData(alignedPoints)

    if (candleTimes.length >= 2) {
      zeroSeriesRef.current.setData([
        { time: candleTimes[0], value: 0 },
        { time: candleTimes[candleTimes.length - 1], value: 0 },
      ])
    } else {
      zeroSeriesRef.current.setData([])
    }

    if (chartRef.current && candleTimes.length >= 2 && !visibleLogicalRange) {
      chartRef.current.timeScale().fitContent()
      lastAppliedLogicalRef.current = normalizeLogicalRange(
        chartRef.current.timeScale().getVisibleLogicalRange?.()
      )
    }

    requestAnimationFrame(() => {
      refreshDecorationsRef.current()
    })
  }, [alignedPoints, candleTimes, visibleLogicalRange])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return

    const next = normalizeLogicalRange(visibleLogicalRange)
    if (!next) {
      requestAnimationFrame(() => {
        refreshDecorationsRef.current()
      })
      return
    }

    const current = normalizeLogicalRange(chart.timeScale().getVisibleLogicalRange?.())
    if (rangesClose(current, next) || rangesClose(lastAppliedLogicalRef.current, next)) {
      requestAnimationFrame(() => {
        refreshDecorationsRef.current()
      })
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
    requestAnimationFrame(() => {
      refreshDecorationsRef.current()
    })
  }, [linkedCrosshair, candleTimes])

  return (
    <div className="flow-shell" style={{ height: `${height}px` }}>
      <div className="flow-panel-header">
        <div className="flow-panel-title">Aggressor Flow</div>
      </div>
      <div ref={stageRef} className="flow-stage">
        <div ref={hostRef} className="flow-host" />
        {sessionBands.map((band) => (
          <div
            key={band.key}
            className="session-band"
            style={{ left: `${band.left}px`, width: `${band.width}px` }}
          />
        ))}
        {Number.isFinite(linkedCrosshairX) && (
          <div
            style={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: `${Math.round(linkedCrosshairX)}px`,
              width: 0,
              borderLeft: '1px dashed rgba(191, 219, 254, 0.45)',
              pointerEvents: 'none',
              zIndex: 4,
            }}
          />
        )}
        {(loading || error || !hasRenderablePoints || !hasAnyFlowValue) && (
          <div className="flow-overlay">
            <div className="flow-overlay-title">
              {error
                ? 'Aggressor Flow could not load'
                : loading
                  ? 'Loading Aggressor Flow…'
                  : 'No flow data returned'}
            </div>
            <div className="flow-overlay-text">
              {error || 'The loaded sessions did not return any aggressor flow data.'}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
