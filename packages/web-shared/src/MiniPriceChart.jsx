// MiniPriceChart — RTH-only 1-min candlestick chart with GEX cluster overlays.
// Used by react_today_setup to show intraday price action alongside the landscape.
// Props:
//   date                string  "YYYY-MM-DD"
//   ticker              string  e.g. "SPX"
//   apiBase             string  base URL for the backend (e.g. "http://localhost:8050")
//   clusters            array   [{ center_price, max_gex }] — drawn as horizontal price lines
//   height              number  default 200
//   onPriceRangeChange  fn      ({ priceBot, priceTop }) => void — fires after bars + clusters
//                               both contribute to the union range; lets DayView sync
//                               GexLandscape's visiblePriceRange to the same Y window.
//   externalRange       { priceBot, priceTop } | null — when non-null, the chart's visible
//                               price scale is set to this range (used for landscape→chart zoom).
import { useEffect, useRef, useState } from 'react'
import { createChart, CandlestickSeries, LineSeries, LineStyle, ColorType } from 'lightweight-charts'
import { utcEpochShowingZoneTime } from './timezone.js'

const CLUSTER_POS_COLOR = '#d946ef'  // magenta — positive GEX
const CLUSTER_NEG_COLOR = '#06b6d4'  // teal — negative GEX

export default function MiniPriceChart({
  date, ticker = 'SPX', apiBase = '', clusters = [], height = 200,
  onPriceRangeChange, externalRange = null,
}) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)
  const seriesRef = useRef(null)
  // Hidden LineSeries whose data points bracket the union price range so
  // lightweight-charts' auto-scale extends to cover clusters outside bar range.
  const rangeAnchorRef = useRef(null)
  // Latest loaded bars (stored after setData so the union-range effect can
  // read them when clusters change independently of a bars re-fetch).
  const barsRef = useRef([])
  // Track the last range we emitted so we don't react to our own emission
  // when externalRange echoes back through DayView state.
  const lastEmittedRef = useRef(null)
  // Current visible range — kept as a ref so the wheel handler can read it
  // without a stale closure. Updated whenever we call setVisibleRange.
  const currentRangeRef = useRef(null)
  // Keep onPriceRangeChange stable in the wheel handler without listing it
  // as a useEffect dep (it can change reference on every render).
  const onRangeChangeRef = useRef(onPriceRangeChange)
  useEffect(() => { onRangeChangeRef.current = onPriceRangeChange }, [onPriceRangeChange])
  const [status, setStatus] = useState('idle')  // idle | loading | empty | ok | error

  // Initialise lightweight-charts once on mount.
  useEffect(() => {
    if (!containerRef.current) return
    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#1f2937' },
        textColor: '#cbd5e1',
      },
      grid: {
        vertLines: { color: 'rgba(148, 163, 184, 0.08)' },
        horzLines: { color: 'rgba(148, 163, 184, 0.08)' },
      },
      timeScale: {
        borderColor: 'rgba(148, 163, 184, 0.18)',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: 'rgba(148, 163, 184, 0.18)',
        scaleMargins: { top: 0, bottom: 0 },
      },
      crosshair: { mode: 1 },
      handleScroll: false,
      handleScale: false,
    })
    const series = chart.addSeries(CandlestickSeries, {
      upColor: '#60a5fa',
      downColor: '#e5e7eb',
      borderVisible: false,
      wickUpColor: '#60a5fa',
      wickDownColor: '#e5e7eb',
    })
    // Invisible range anchor — two data points set to priceBot/priceTop so
    // auto-scale picks them up and expands the Y axis to include cluster lines.
    const rangeAnchor = chart.addSeries(LineSeries, {
      color: 'rgba(0,0,0,0)',
      lineVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    })
    chartRef.current = chart
    seriesRef.current = series
    rangeAnchorRef.current = rangeAnchor

    const ro = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect
      if (rect && chart) {
        chart.applyOptions({ width: rect.width })
      }
    })
    ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      chart.remove()
      chartRef.current = null
      seriesRef.current = null
      rangeAnchorRef.current = null
      barsRef.current = []
    }
  }, [])

  // Fetch bars when date/ticker change.
  useEffect(() => {
    if (!date || !seriesRef.current) return
    let cancelled = false
    setStatus('loading')

    const params = new URLSearchParams({ date, ticker, session: 'rth' })
    fetch(`${apiBase}/api/bars?${params}`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((bars) => {
        if (cancelled) return
        if (!Array.isArray(bars) || bars.length === 0) {
          seriesRef.current?.setData([])
          barsRef.current = []
          setStatus('empty')
          return
        }
        // Shift bar epochs so the X axis shows PT wall-clock times
        // (06:30–13:00) instead of UTC (13:30–20:00).
        const ptBars = bars.map(b => ({
          ...b,
          time: utcEpochShowingZoneTime(b.time, 'America/Los_Angeles'),
        }))
        seriesRef.current?.setData(ptBars)
        barsRef.current = ptBars
        // Add 10 minutes of left-pad before the first bar so the opening
        // candle isn't crammed against the left axis.
        const LEFT_PAD_SECONDS = 600
        chartRef.current?.timeScale().setVisibleRange({
          from: ptBars[0].time - LEFT_PAD_SECONDS,
          to: ptBars[ptBars.length - 1].time,
        })
        setStatus('ok')
      })
      .catch(() => {
        if (!cancelled) setStatus('error')
      })

    return () => { cancelled = true }
  }, [date, ticker, apiBase])

  // Compute union Y range (bars high/low + cluster centers) whenever either changes.
  // Runs after status→'ok' (new bars loaded) and whenever clusters update.
  // Pins the chart's price scale to exactly this range (no auto-scale padding)
  // and fires onPriceRangeChange so DayView can sync GexLandscape.
  useEffect(() => {
    const bars = barsRef.current
    if (!bars.length || status !== 'ok') return
    const lows = bars.map(b => b.low)
    const highs = bars.map(b => b.high)
    const clusterPrices = (clusters || [])
      .map(c => Number(c.center_price))
      .filter(Number.isFinite)
    const unionMin = Math.min(...lows, ...clusterPrices)
    const unionMax = Math.max(...highs, ...clusterPrices)
    const pad = (unionMax - unionMin) * 0.04
    const priceBot = unionMin - pad
    const priceTop = unionMax + pad
    if (rangeAnchorRef.current) {
      rangeAnchorRef.current.setData([
        { time: bars[0].time, value: priceBot },
        { time: bars[bars.length - 1].time, value: priceTop },
      ])
    }
    // Pin the chart to exactly priceBot..priceTop so the displayed range
    // matches what we emit (no additional lightweight-charts scale margin).
    chartRef.current?.priceScale('right').setVisibleRange({ from: priceBot, to: priceTop })
    const range = { priceBot, priceTop }
    lastEmittedRef.current = range
    currentRangeRef.current = range
    onPriceRangeChange?.(range)
  }, [clusters, status, onPriceRangeChange])

  // Apply externalRange to the chart when the landscape (or DayView) drives a zoom.
  // Skip if we just emitted this range ourselves to avoid feedback loops.
  useEffect(() => {
    if (!externalRange || !chartRef.current) return
    const last = lastEmittedRef.current
    if (
      last &&
      Math.abs(last.priceBot - externalRange.priceBot) < 0.01 &&
      Math.abs(last.priceTop - externalRange.priceTop) < 0.01
    ) return
    chartRef.current.priceScale('right').setVisibleRange({
      from: externalRange.priceBot,
      to: externalRange.priceTop,
    })
  }, [externalRange])

  // Wheel-zoom handler on the chart container — zooms the shared Y axis.
  // Registered as non-passive so we can call preventDefault() and prevent
  // the page from scrolling while the user zooms the chart.
  const outerRef = useRef(null)
  useEffect(() => {
    const el = outerRef.current
    if (!el) return
    function onWheel(e) {
      const range = currentRangeRef.current
      if (!range || !chartRef.current) return
      e.preventDefault()
      const { priceBot, priceTop } = range
      const span = priceTop - priceBot
      const MIN_SPAN = 5
      // deltaY > 0 → scroll down → zoom out; < 0 → zoom in
      const factor = e.deltaY > 0 ? 1.06 : 1 / 1.06
      const newSpan = Math.max(MIN_SPAN, span * factor)
      const mid = (priceBot + priceTop) / 2
      const newBot = mid - newSpan / 2
      const newTop = mid + newSpan / 2
      chartRef.current.priceScale('right').setVisibleRange({ from: newBot, to: newTop })
      const newRange = { priceBot: newBot, priceTop: newTop }
      lastEmittedRef.current = newRange
      currentRangeRef.current = newRange
      onRangeChangeRef.current?.(newRange)
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  // Price lines for cluster centers — recreated whenever clusters or status change.
  const priceLineRefs = useRef([])
  useEffect(() => {
    if (!seriesRef.current) return
    priceLineRefs.current.forEach((pl) => {
      try { seriesRef.current?.removePriceLine(pl) } catch (_) {}
    })
    priceLineRefs.current = []
    if (status !== 'ok') return
    const lines = (clusters || []).map((c) => {
      const isPos = Number(c.max_gex) >= 0
      return seriesRef.current?.createPriceLine({
        price: Number(c.center_price),
        color: isPos ? CLUSTER_POS_COLOR : CLUSTER_NEG_COLOR,
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: false,
        title: '',
      })
    }).filter(Boolean)
    priceLineRefs.current = lines
  }, [clusters, status])

  return (
    <div ref={outerRef} style={{ position: 'relative', width: '100%', height }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
      {status === 'loading' && <div style={overlay}>Loading chart…</div>}
      {status === 'empty' && <div style={overlay}>No bar data for {date}</div>}
      {status === 'error' && <div style={overlay}>Chart unavailable</div>}
    </div>
  )
}

const overlay = {
  position: 'absolute',
  inset: 0,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  fontSize: 12,
  color: '#64748b',
  pointerEvents: 'none',
}
