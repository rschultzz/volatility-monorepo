import { useEffect, useMemo, useRef } from 'react'
import {
  createChart,
  CrosshairMode,
  ColorType,
  HistogramSeries,
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

function bucketFlowToCandles(flowPoints, candleTimes, histAlpha) {
  const alpha = Number.isFinite(Number(histAlpha))
    ? Math.max(0.05, Math.min(1, Number(histAlpha)))
    : 0.30

  const points = normalizeShiftedFlowPoints(flowPoints)
  const times = normalizeShiftedCandleTimes(candleTimes)

  if (!times.length) {
    return points.map((pt) => ({
      time: pt.time,
      value: pt.value,
      color: pt.value >= 0 ? POS_RGBA(alpha) : NEG_RGBA(alpha),
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

    const value = count > 0 ? sum / count : 0
    out.push({
      time: bucketStart,
      value,
      color: value >= 0 ? POS_RGBA(alpha) : NEG_RGBA(alpha),
    })
  }

  return out
}

export default function AggressorFlowPanel({
  candles,
  dataPoints,
  visibleLogicalRange,
  height = 220,
  loading = false,
  error = '',
  histAlpha = 0.30,
}) {
  const hostRef = useRef(null)
  const chartRef = useRef(null)
  const histSeriesRef = useRef(null)
  const zeroSeriesRef = useRef(null)
  const lastAppliedLogicalRef = useRef(null)

  const alignedPoints = useMemo(
    () => bucketFlowToCandles(dataPoints, candles, histAlpha),
    [dataPoints, candles, histAlpha]
  )

  const hasRenderablePoints = alignedPoints.length >= 2

  useEffect(() => {
    if (!hostRef.current) return undefined

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

    const histSeries = chart.addSeries(HistogramSeries, {
      priceLineVisible: false,
      lastValueVisible: false,
      base: 0,
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
    histSeriesRef.current = histSeries
    zeroSeriesRef.current = zeroSeries

    const handleResize = () => {
      chart.applyOptions({
        width: container.clientWidth || 900,
        height: Math.max(container.clientHeight || MIN_PANEL_HEIGHT, MIN_PANEL_HEIGHT),
      })
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
      chartRef.current = null
      histSeriesRef.current = null
      zeroSeriesRef.current = null
      lastAppliedLogicalRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!histSeriesRef.current || !zeroSeriesRef.current) return

    histSeriesRef.current.setData(alignedPoints)

    if (hasRenderablePoints) {
      zeroSeriesRef.current.setData([
        { time: alignedPoints[0].time, value: 0 },
        { time: alignedPoints[alignedPoints.length - 1].time, value: 0 },
      ])
    } else {
      zeroSeriesRef.current.setData([])
    }

    if (chartRef.current && hasRenderablePoints && !visibleLogicalRange) {
      chartRef.current.timeScale().fitContent()
      lastAppliedLogicalRef.current = normalizeLogicalRange(
        chartRef.current.timeScale().getVisibleLogicalRange?.()
      )
    }
  }, [alignedPoints, hasRenderablePoints, visibleLogicalRange])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return

    const next = normalizeLogicalRange(visibleLogicalRange)
    if (!next) return

    const current = normalizeLogicalRange(chart.timeScale().getVisibleLogicalRange?.())
    if (rangesClose(current, next) || rangesClose(lastAppliedLogicalRef.current, next)) return

    requestAnimationFrame(() => {
      try {
        chart.timeScale().setVisibleLogicalRange(next)
        lastAppliedLogicalRef.current = next
      } catch (err) {
        // ignore sync errors
      }
    })
  }, [visibleLogicalRange])

  return (
    <div className="flow-shell" style={{ height: `${height}px` }}>
      <div className="flow-panel-header">
        <div className="flow-panel-title">Aggressor Flow</div>
      </div>
      <div className="flow-stage">
        <div ref={hostRef} className="flow-host" />
        {(loading || error || !alignedPoints.length) && (
          <div className="flow-overlay">
            <div className="flow-overlay-title">
              {error
                ? 'Aggressor Flow could not load'
                : loading
                  ? 'Loading Aggressor Flow…'
                  : 'No flow data returned'}
            </div>
            <div className="flow-overlay-text">
              {error || 'The backend responded, but this session did not return any flow data.'}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
