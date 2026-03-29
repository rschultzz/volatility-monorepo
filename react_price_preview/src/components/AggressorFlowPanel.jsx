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

function normalizeRange(range) {
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

function normalizePoints(points, histAlpha) {
  const alpha = Number.isFinite(Number(histAlpha))
    ? Math.max(0.05, Math.min(1, Number(histAlpha)))
    : 0.30

  return (Array.isArray(points) ? points : [])
    .map((pt) => {
      const rawTime = Number(pt?.time)
      const value = Number(pt?.value)
      if (!Number.isFinite(rawTime) || !Number.isFinite(value)) return null
      const time = utcEpochShowingZoneTime(rawTime, 'America/Los_Angeles')
      const color = value >= 0 ? `rgba(96,165,250,${alpha})` : `rgba(229,231,235,${alpha})`
      return { time, value, color }
    })
    .filter(Boolean)
}

export default function AggressorFlowPanel({
  dataPoints,
  visibleTimeRange,
  height = 220,
  loading = false,
  error = '',
  histAlpha = 0.30,
}) {
  const hostRef = useRef(null)
  const chartRef = useRef(null)
  const histSeriesRef = useRef(null)
  const zeroSeriesRef = useRef(null)
  const lastAppliedRangeRef = useRef(null)

  const normalizedPoints = useMemo(
    () => normalizePoints(dataPoints, histAlpha),
    [dataPoints, histAlpha]
  )

  const hasRenderablePoints = normalizedPoints.length >= 2

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
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false,
      },
      handleScale: {
        axisPressedMouseMove: false,
        mouseWheel: true,
        pinch: true,
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
    }
  }, [])

  useEffect(() => {
    if (!histSeriesRef.current || !zeroSeriesRef.current) return

    histSeriesRef.current.setData(normalizedPoints)

    if (hasRenderablePoints) {
      zeroSeriesRef.current.setData([
        { time: normalizedPoints[0].time, value: 0 },
        { time: normalizedPoints[normalizedPoints.length - 1].time, value: 0 },
      ])
    } else {
      zeroSeriesRef.current.setData([])
    }

    if (chartRef.current && hasRenderablePoints && !visibleTimeRange) {
      chartRef.current.timeScale().fitContent()
    }
  }, [normalizedPoints, visibleTimeRange, hasRenderablePoints])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart || !hasRenderablePoints) return

    const next = normalizeRange(visibleTimeRange)
    if (!next) return

    const current = normalizeRange(chart.timeScale().getVisibleRange?.())
    if (rangesClose(current, next) || rangesClose(lastAppliedRangeRef.current, next)) return

    try {
      chart.timeScale().setVisibleRange(next)
      lastAppliedRangeRef.current = next
    } catch (err) {
      // ignore sync errors
    }
  }, [visibleTimeRange, hasRenderablePoints])

  return (
    <div className="flow-shell" style={{ height: `${height}px` }}>
      <div className="flow-panel-header">
        <div className="flow-panel-title">Aggressor Flow</div>
      </div>
      <div className="flow-stage">
        <div ref={hostRef} className="flow-host" />
        {(loading || error || !normalizedPoints.length) && (
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