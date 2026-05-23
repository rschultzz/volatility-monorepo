// MiniPriceChart — RTH-only 1-min candlestick chart with GEX cluster overlays.
// Used by react_today_setup to show intraday price action alongside the landscape.
// Props:
//   date     string  "YYYY-MM-DD"
//   ticker   string  e.g. "SPX"
//   apiBase  string  base URL for the backend (e.g. "http://localhost:8050")
//   clusters array   [{ center_price, max_gex }] — drawn as horizontal price lines
//   height   number  default 200
import { useEffect, useRef, useState } from 'react'
import { createChart, CandlestickSeries, LineStyle, ColorType } from 'lightweight-charts'

const CLUSTER_POS_COLOR = '#d946ef'  // magenta — positive GEX
const CLUSTER_NEG_COLOR = '#06b6d4'  // teal — negative GEX

export default function MiniPriceChart({ date, ticker = 'SPX', apiBase = '', clusters = [], height = 200 }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)
  const seriesRef = useRef(null)
  const [status, setStatus] = useState('idle')  // idle | loading | empty | error

  // Initialise lightweight-charts once on mount.
  useEffect(() => {
    if (!containerRef.current) return
    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#0b1220' },
        textColor: '#64748b',
      },
      grid: {
        vertLines: { color: '#1e293b' },
        horzLines: { color: '#1e293b' },
      },
      timeScale: {
        borderColor: '#334155',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: { borderColor: '#334155' },
      crosshair: { mode: 1 },
      handleScroll: false,
      handleScale: false,
    })
    const series = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    })
    chartRef.current = chart
    seriesRef.current = series

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
          setStatus('empty')
          return
        }
        seriesRef.current?.setData(bars)
        chartRef.current?.timeScale().fitContent()
        setStatus('ok')
      })
      .catch(() => {
        if (!cancelled) setStatus('error')
      })

    return () => { cancelled = true }
  }, [date, ticker, apiBase])

  // Update cluster price lines when clusters or status change.
  useEffect(() => {
    if (!seriesRef.current || status !== 'ok') return
    // Remove existing price lines then add new ones.
    // lightweight-charts v5 series holds priceLine refs we keep in a closure.
    // Simple approach: recreate them by rebuilding the series isn't viable.
    // Instead track refs via a side-effect ref.
  }, [clusters, status])

  // Separate ref-based approach: add price lines after data loads.
  const priceLineRefs = useRef([])
  useEffect(() => {
    if (!seriesRef.current) return
    // Remove old lines.
    priceLineRefs.current.forEach((pl) => {
      try { seriesRef.current?.removePriceLine(pl) } catch (_) {}
    })
    priceLineRefs.current = []
    if (status !== 'ok') return
    // Add new lines.
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
    <div style={{ position: 'relative', width: '100%', height }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
      {status === 'loading' && (
        <div style={overlay}>Loading chart…</div>
      )}
      {status === 'empty' && (
        <div style={overlay}>No bar data for {date}</div>
      )}
      {status === 'error' && (
        <div style={overlay}>Chart unavailable</div>
      )}
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
