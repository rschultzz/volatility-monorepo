// CR-CD — Flow v2 pane: delta per bar, CVD, delta share, absorption band and (optional) markers,
// from /api/ironbeam/flow?mode=v2. Same pane slot, height, time-scale sync, crosshair link and
// session bands as AggressorFlowPanel. Colours are the pane's / candles' own palette (A1.5):
// blue = buying, light grey = selling, slate = absorption.
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  createChart,
  createSeriesMarkers,
  CrosshairMode,
  ColorType,
  BaselineSeries,
  HistogramSeries,
  LineSeries,
  LineStyle,
} from 'lightweight-charts'
import {
  FlowModeToggle,
  coordinateForLinkedCrosshair,
  computeSessionBands,
  formatShiftedHHMM,
  normalizeLogicalRange,
  normalizeShiftedCandleTimes,
  rangesClose,
  utcEpochShowingZoneTime,
  inferStepSeconds,
} from './AggressorFlowPanel'

const BG_COLOR = '#1f2937'
const MIN_PANEL_HEIGHT = 140
const BUY_RGBA = (alpha) => `rgba(96,165,250,${alpha})`
const SELL_RGBA = (alpha) => `rgba(229,231,235,${alpha})`
const ABSORB_RGBA = 'rgba(148,163,184,0.20)'
const SHARE_COLOR = 'rgba(203,213,225,0.85)'
const SHARE_REF = 0.15

export const DEFAULT_MARKER_SETTINGS = { show: false, swing: 8, push: 500, div: 300, fade: 0.1, gap: 30 }
export const A1_MARKER_SETTINGS = { swing: 4, push: 0, div: 0, fade: 0, gap: 15 }

const KIND_LETTER = { divergence: 'D', push_decay: 'P', fading: 'F' }
const SOURCE_LABEL = { databento: 'Databento', ironbeam: 'Ironbeam' }

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

function snapMarkers(markers, candles) {
  const times = normalizeShiftedCandleTimes(candles)
  if (!times.length) return []
  const out = []
  for (const m of Array.isArray(markers) ? markers : []) {
    const st = utcEpochShowingZoneTime(Number(m.time))
    let lo = 0
    let hi = times.length - 1
    if (st < times[0]) continue
    while (lo < hi) {
      const mid = Math.ceil((lo + hi) / 2)
      if (times[mid] <= st) lo = mid
      else hi = mid - 1
    }
    out.push({
      time: times[lo],
      position: m.side === 'bull' ? 'belowBar' : 'aboveBar',
      shape: m.side === 'bull' ? 'arrowUp' : 'arrowDown',
      color: m.side === 'bull' ? BUY_RGBA(1) : SELL_RGBA(1),
      text: KIND_LETTER[m.kind] || '?',
      size: 1,
    })
  }
  return out.sort((a, b) => a.time - b.time)
}

const chipStyle = (active) => ({
  height: '24px',
  borderRadius: '8px',
  border: `1px solid ${active ? 'rgba(96,165,250,0.6)' : 'rgba(148, 163, 184, 0.22)'}`,
  background: active ? 'rgba(96,165,250,0.18)' : 'rgba(15, 23, 42, 0.96)',
  color: '#cbd5e1',
  cursor: 'pointer',
  fontSize: '12px',
  padding: '0 8px',
})

const iconStyle = {
  width: '24px',
  height: '24px',
  borderRadius: '8px',
  border: '1px solid rgba(148, 163, 184, 0.22)',
  background: 'rgba(15, 23, 42, 0.96)',
  color: '#cbd5e1',
  cursor: 'pointer',
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  fontSize: '13px',
  lineHeight: 1,
  padding: 0,
}

function Modal({ title, onClose, children, width = 520 }) {
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(2, 6, 23, 0.58)', display: 'flex',
        alignItems: 'center', justifyContent: 'center', zIndex: 1200, padding: '24px',
      }}
    >
      <div
        role="dialog"
        aria-label={title}
        onClick={(e) => e.stopPropagation()}
        style={{
          width: `min(${width}px, calc(100vw - 32px))`, maxHeight: 'calc(100vh - 48px)', overflowY: 'auto',
          borderRadius: '16px', border: '1px solid #1f2937', background: '#0b1220',
          boxShadow: '0 20px 60px rgba(0, 0, 0, 0.42)', padding: '18px', color: '#e2e8f0',
          fontSize: '13px', lineHeight: 1.5,
        }}
      >
        <div style={{ fontSize: '16px', fontWeight: 800, marginBottom: '12px' }}>{title}</div>
        {children}
      </div>
    </div>
  )
}

function GuideBody() {
  const h = { fontWeight: 700, marginTop: '12px', color: '#f1f5f9' }
  const sw = (bg) => ({ display: 'inline-block', width: 10, height: 10, background: bg, marginRight: 6, verticalAlign: 'middle' })
  return (
    <div>
      <div style={h}>What is drawn</div>
      <div><span style={sw(BUY_RGBA(0.8))} /><span style={sw(SELL_RGBA(0.8))} /><b>Delta per bar</b> (bars, lower part): contracts bought at the offer minus sold at the bid in each bar. Blue = buyers were the aggressors; grey = sellers.</div>
      <div><b>CVD</b> (solid line, upper part; value in the hover box): delta added up from the session open (06:30 PT for RTH, 15:00 PT for the overnight session). Blue above zero, grey below.</div>
      <div><span style={sw(SHARE_COLOR)} /><b>Delta share</b> (dotted line, right axis): delta as a share of volume, smoothed over 3 bars. ±1 would mean every trade on one side; most bars sit within ±0.15 (the faint guide lines).</div>
      <div><span style={sw(ABSORB_RGBA)} /><b>Absorption</b> (shaded columns, RTH only): over the last 5 minutes price moved unusually little for the net buying or selling (bottom 20 % of the prior 20 sessions), while one side clearly dominated (|share| ≥ 0.15). Someone is taking the other side.</div>

      <div style={h}>How to read it against price</div>
      <div><b>Divergence:</b> price makes a new low below the last confirmed swing low, but CVD is higher than it was at that low. The new low came on less selling: sellers may be running out. The mirror at highs.</div>
      <div><b>Absorption at a level:</b> heavy aggression with price not moving. If price then turns, the passive side won. If price breaks through after the absorption ends, it did not.</div>
      <div><b>Push decay:</b> each retest of the same swing high (low) brings in less buying (selling) than the last.</div>
      <div><b>Fading:</b> the side that has dominated the last 30 minutes has a smaller share of volume in the last 10 minutes than in the first 10.</div>
      <div style={{ marginTop: 6, color: '#94a3b8' }}>Markers (D / P / F) are <b>visual aids</b> to help you learn the read. They are <b>not</b> the CR-CB test features and have not been tested as signals. They are off by default; ⚙ sets how strict they are.</div>

      <div style={h}>Data source</div>
      <div>Sessions through 2026-06-05 use <b>Databento</b> (the complete feed) for RTH; later sessions, and every live session, use <b>Ironbeam</b>. The overnight session is always Ironbeam.</div>
      <div><b>Ironbeam misses about 6 % of volume and understates buy − sell by about 30 %</b>, more in busy minutes. On live (Ironbeam) data a CVD divergence can even have the opposite sign to Databento. On 2026-05-01 at the 08:02 low, Databento CVD was +187 above the 07:37 low and Ironbeam −379 below it. <b>Ratio measures</b> (delta share, absorption against an Ironbeam baseline) are more robust than CVD levels.</div>
    </div>
  )
}

function MarkerSettings({ value, onApply, onClose }) {
  const [draft, setDraft] = useState({ ...value })
  const field = (key, label, step, hint) => (
    <label style={{ display: 'block', marginBottom: '10px' }}>
      <div style={{ fontSize: '12px', color: '#cbd5e1', marginBottom: '4px' }}>{label}</div>
      <input
        type="number"
        step={step}
        min={0}
        value={draft[key]}
        aria-label={label}
        onChange={(e) => setDraft((d) => ({ ...d, [key]: e.target.value }))}
        style={{
          width: '100%', background: '#020617', color: '#e2e8f0', border: '1px solid #334155',
          borderRadius: '10px', padding: '8px 10px', fontSize: '14px',
        }}
      />
      <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>{hint}</div>
    </label>
  )
  const apply = (e) => {
    e.preventDefault()
    const n = (k, lo, hi, d) => {
      const v = Number(draft[k])
      return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d
    }
    onApply({
      ...value,
      swing: n('swing', 1, 50, DEFAULT_MARKER_SETTINGS.swing),
      push: n('push', 0, 100000, DEFAULT_MARKER_SETTINGS.push),
      div: n('div', 0, 100000, DEFAULT_MARKER_SETTINGS.div),
      fade: n('fade', 0, 1, DEFAULT_MARKER_SETTINGS.fade),
      gap: Math.round(n('gap', 1, 390, DEFAULT_MARKER_SETTINGS.gap)),
    })
  }
  return (
    <Modal title="Flow v2 marker settings" onClose={onClose} width={380}>
      <form onSubmit={apply}>
        {field('swing', 'Swing size (pts)', 0.25, 'A swing high/low is confirmed after price retraces this far.')}
        {field('push', 'Push decay: minimum previous push (contracts)', 50, 'The earlier push must be at least this big to compare.')}
        {field('div', 'Divergence: minimum CVD difference (contracts)', 50, 'Also at least 10 % of the CVD range since the swing.')}
        {field('fade', 'Fading: minimum 30-min delta share', 0.01, 'The dominant side must hold at least this share first.')}
        {field('gap', 'At most one marker of a kind per (minutes)', 1, '')}
        <div style={{ display: 'flex', gap: '8px', justifyContent: 'space-between', marginTop: '14px' }}>
          <div style={{ display: 'flex', gap: '6px' }}>
            <button type="button" style={chipStyle(false)} onClick={() => setDraft((d) => ({ ...d, ...DEFAULT_MARKER_SETTINGS, show: d.show }))}>Defaults</button>
            <button type="button" style={chipStyle(false)} onClick={() => setDraft((d) => ({ ...d, ...A1_MARKER_SETTINGS }))} title="The A1 definitions as first written (many more markers)">Loose (A1)</button>
          </div>
          <div style={{ display: 'flex', gap: '6px' }}>
            <button type="button" style={chipStyle(false)} onClick={onClose}>Cancel</button>
            <button type="submit" style={{ ...chipStyle(true), fontWeight: 700 }}>Apply</button>
          </div>
        </div>
      </form>
    </Modal>
  )
}

export default function AggressorFlowV2Panel({
  candles,
  payloads,
  visibleLogicalRange,
  linkedCrosshair,
  height = 220,
  loading = false,
  error = '',
  flowMode = 'v2',
  onFlowModeChange,
  markerSettings = DEFAULT_MARKER_SETTINGS,
  onMarkerSettingsChange,
}) {
  const stageRef = useRef(null)
  const hostRef = useRef(null)
  const chartRef = useRef(null)
  const seriesRef = useRef({})
  const markersRef = useRef(null)
  const lastAppliedLogicalRef = useRef(null)
  const candleTimesRef = useRef([])
  const linkedCrosshairRef = useRef(null)
  const refreshDecorationsRef = useRef(() => {})

  const [sessionBands, setSessionBands] = useState([])
  const [linkedCrosshairX, setLinkedCrosshairX] = useState(null)
  const [guideOpen, setGuideOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [hover, setHover] = useState(null)

  const candleTimes = useMemo(() => normalizeShiftedCandleTimes(candles), [candles])
  const allBars = useMemo(() => (payloads || []).flatMap((p) => (Array.isArray(p?.bars) ? p.bars : [])), [payloads])
  const allMarkers = useMemo(() => (payloads || []).flatMap((p) => (Array.isArray(p?.markers) ? p.markers : [])), [payloads])
  const aligned = useMemo(() => alignV2ToCandles(allBars, candles), [allBars, candles])
  const snapped = useMemo(() => snapMarkers(allMarkers, candles), [allMarkers, candles])
  const markerByTime = useMemo(() => {
    const m = new Map()
    for (const mk of allMarkers) {
      const st = utcEpochShowingZoneTime(Number(mk.time))
      m.set(st, [...(m.get(st) || []), mk])
    }
    return m
  }, [allMarkers])
  const sources = useMemo(() => {
    const s = new Set()
    for (const p of payloads || []) {
      for (const seg of p?.segments || []) if (seg.name === 'rth') s.add(seg.source)
    }
    return [...s]
  }, [payloads])
  const notes = useMemo(() => [...new Set((payloads || []).flatMap((p) => p?.notes || []))], [payloads])
  const hasValues = aligned.some((r) => Number.isFinite(r.delta))

  useEffect(() => {
    candleTimesRef.current = candleTimes
  }, [candleTimes])

  useEffect(() => {
    linkedCrosshairRef.current = linkedCrosshair || null
  }, [linkedCrosshair])

  useEffect(() => {
    if (!hostRef.current || !stageRef.current) return undefined
    const stage = stageRef.current
    const chart = createChart(hostRef.current, {
      width: stage.clientWidth || 900,
      height: Math.max(stage.clientHeight || MIN_PANEL_HEIGHT, MIN_PANEL_HEIGHT),
      layout: { background: { type: ColorType.Solid, color: BG_COLOR }, textColor: '#cbd5e1', attributionLogo: false },
      localization: { locale: 'en-US' },
      grid: { vertLines: { color: 'rgba(148, 163, 184, 0.08)' }, horzLines: { color: 'rgba(148, 163, 184, 0.08)' } },
      rightPriceScale: { visible: true, borderColor: 'rgba(148, 163, 184, 0.18)', scaleMargins: { top: 0.08, bottom: 0.08 } },
      timeScale: { borderColor: 'rgba(148, 163, 184, 0.18)', timeVisible: true, secondsVisible: false, tickMarkFormatter: (t) => formatShiftedHHMM(t) },
      crosshair: { mode: CrosshairMode.Normal, vertLine: { visible: false, labelVisible: false }, horzLine: { visible: false, labelVisible: false } },
      handleScroll: { mouseWheel: false, pressedMouseMove: false, horzTouchDrag: false, vertTouchDrag: false },
      handleScale: { axisPressedMouseMove: false, mouseWheel: false, pinch: false },
    })
    const common = { priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false }
    const absorb = chart.addSeries(HistogramSeries, { ...common, priceScaleId: 'absorb', base: 0 })
    chart.priceScale('absorb').applyOptions({ visible: false, scaleMargins: { top: 0, bottom: 0 } })
    const delta = chart.addSeries(HistogramSeries, { ...common, priceScaleId: 'delta', base: 0 })
    chart.priceScale('delta').applyOptions({ visible: false, scaleMargins: { top: 0.66, bottom: 0.02 } })
    const cvd = chart.addSeries(BaselineSeries, {
      ...common,
      priceScaleId: 'cvd',
      baseValue: { type: 'price', price: 0 },
      topLineColor: BUY_RGBA(0.95),
      bottomLineColor: SELL_RGBA(0.95),
      topFillColor1: 'rgba(0,0,0,0)', topFillColor2: 'rgba(0,0,0,0)',
      bottomFillColor1: 'rgba(0,0,0,0)', bottomFillColor2: 'rgba(0,0,0,0)',
      lineWidth: 2,
    })
    // No left axis: it would narrow the plot area and break the bar-for-bar alignment with the
    // price chart. CVD has its own unlabelled scale (its value is in the hover tooltip).
    chart.priceScale('cvd').applyOptions({ visible: false, scaleMargins: { top: 0.08, bottom: 0.38 } })
    const share = chart.addSeries(LineSeries, {
      ...common, priceScaleId: 'right', color: SHARE_COLOR, lineWidth: 1, lineStyle: LineStyle.Dotted,
      autoscaleInfoProvider: (orig) => {
        const r = orig()
        const m = Math.max(SHARE_REF * 1.6, Math.abs(r?.priceRange?.minValue || 0), Math.abs(r?.priceRange?.maxValue || 0))
        return { priceRange: { minValue: -m, maxValue: m } }
      },
    })
    for (const v of [SHARE_REF, -SHARE_REF]) {
      share.createPriceLine({ price: v, color: 'rgba(203,213,225,0.22)', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: false })
    }
    chartRef.current = chart
    seriesRef.current = { absorb, delta, cvd, share }
    markersRef.current = createSeriesMarkers(cvd, [])

    const refreshDecorations = () => {
      const w = stageRef.current?.clientWidth || 0
      setSessionBands(computeSessionBands(chart, candleTimesRef.current, w))
      const x = coordinateForLinkedCrosshair(chart, linkedCrosshairRef.current)
      setLinkedCrosshairX(!Number.isFinite(x) || x < 0 || x > w ? null : x)
    }
    refreshDecorationsRef.current = refreshDecorations
    const handleResize = () => {
      if (!stageRef.current) return
      const w = stageRef.current.clientWidth
      const h = stageRef.current.clientHeight
      if (w > 0 && h > 0) chart.applyOptions({ width: w, height: Math.max(h, MIN_PANEL_HEIGHT) })
      requestAnimationFrame(refreshDecorations)
    }
    const onRange = () => requestAnimationFrame(refreshDecorations)
    const onMove = (param) => {
      const t = param?.time
      if (t == null) {
        setHover(null)
        return
      }
      setHover({ time: t, x: param.point?.x ?? 0 })
    }
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(handleResize) : null
    if (ro) ro.observe(stage)
    chart.timeScale().subscribeVisibleLogicalRangeChange(onRange)
    chart.subscribeCrosshairMove(onMove)
    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(onRange)
      chart.unsubscribeCrosshairMove(onMove)
      if (ro) ro.disconnect()
      chart.remove()
      chartRef.current = null
      seriesRef.current = {}
      markersRef.current = null
      lastAppliedLogicalRef.current = null
      refreshDecorationsRef.current = () => {}
    }
  }, [])

  useEffect(() => {
    const s = seriesRef.current
    if (!s.delta) return
    const ws = (r) => ({ time: r.time })
    s.absorb.setData(aligned.map((r) => (r.absorb ? { time: r.time, value: 1, color: ABSORB_RGBA } : ws(r))))
    s.delta.setData(aligned.map((r) => (Number.isFinite(r.delta)
      ? { time: r.time, value: r.delta, color: r.delta >= 0 ? BUY_RGBA(0.75) : SELL_RGBA(0.6) }
      : ws(r))))
    s.cvd.setData(aligned.map((r) => (Number.isFinite(r.cvd) ? { time: r.time, value: r.cvd } : ws(r))))
    s.share.setData(aligned.map((r) => (Number.isFinite(r.share) && r.share !== null ? { time: r.time, value: r.share } : ws(r))))
    markersRef.current?.setMarkers(markerSettings.show ? snapped : [])
    const shared = normalizeLogicalRange(visibleLogicalRange)
    if (chartRef.current && candleTimes.length >= 2 && !shared) {
      chartRef.current.timeScale().fitContent()
      lastAppliedLogicalRef.current = normalizeLogicalRange(chartRef.current.timeScale().getVisibleLogicalRange?.())
    } else if (chartRef.current && shared) {
      // v2 data usually lands after the shared range: re-apply it so the panes stay bar-aligned.
      try {
        chartRef.current.timeScale().setVisibleLogicalRange(shared)
        lastAppliedLogicalRef.current = shared
      } catch (err) {
        // ignore sync errors
      }
    }
    requestAnimationFrame(() => refreshDecorationsRef.current())
  }, [aligned, snapped, markerSettings.show, candleTimes, visibleLogicalRange])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return
    const next = normalizeLogicalRange(visibleLogicalRange)
    const current = normalizeLogicalRange(chart.timeScale().getVisibleLogicalRange?.())
    if (!next || rangesClose(current, next) || rangesClose(lastAppliedLogicalRef.current, next)) {
      requestAnimationFrame(() => refreshDecorationsRef.current())
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
    requestAnimationFrame(() => refreshDecorationsRef.current())
  }, [linkedCrosshair, candleTimes])

  const hoverRow = hover ? aligned.find((r) => r.time === hover.time) : null
  const hoverMarks = hover && markerSettings.show ? markerByTime.get(hover.time) || [] : []

  return (
    <div className="flow-shell" style={{ height: `${height}px`, position: 'relative', width: '100%' }} data-testid="flow-v2-panel">
      <div className="flow-panel-header" style={{ display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'flex-start' }}>
        <button type="button" style={iconStyle} title="Flow v2 marker settings" aria-label="Open Flow v2 marker settings" onClick={() => setSettingsOpen(true)}>⚙</button>
        <div className="flow-panel-title">Flow v2</div>
        <FlowModeToggle mode={flowMode} onChange={onFlowModeChange} />
        {sources.map((s) => (
          <span key={s} data-testid="flow-v2-source" title={s === 'ironbeam' ? 'Ironbeam undercounts buy − sell by ~30 %; see ?' : 'Databento: complete trade feed'}
            style={{ fontSize: '11px', padding: '2px 7px', borderRadius: '999px', border: '1px solid rgba(148,163,184,0.35)', color: s === 'databento' ? '#bfdbfe' : '#e5e7eb' }}>
            {SOURCE_LABEL[s] || s}
          </span>
        ))}
        <button type="button" style={chipStyle(markerSettings.show)} aria-pressed={markerSettings.show} data-testid="flow-v2-markers-toggle"
          onClick={() => onMarkerSettingsChange?.({ ...markerSettings, show: !markerSettings.show })}
          title="Show divergence (D), push decay (P) and fading (F) markers — visual aids, not tested signals">
          Markers {markerSettings.show ? 'on' : 'off'}
        </button>
        <button type="button" style={iconStyle} title="How to read Flow v2" aria-label="How to read Flow v2" onClick={() => setGuideOpen(true)}>?</button>
        {notes.length > 0 && (
          <span style={{ fontSize: '11px', color: '#94a3b8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={notes.join('\n')}>
            {notes[0]}
          </span>
        )}
      </div>
      {guideOpen && <Modal title="How to read Flow v2" onClose={() => setGuideOpen(false)}><GuideBody /></Modal>}
      {settingsOpen && (
        <MarkerSettings
          value={markerSettings}
          onClose={() => setSettingsOpen(false)}
          onApply={(v) => {
            onMarkerSettingsChange?.(v)
            setSettingsOpen(false)
          }}
        />
      )}
      <div ref={stageRef} className="flow-stage" style={{ width: '100%' }}>
        <div ref={hostRef} className="flow-host" style={{ width: '100%', height: '100%' }} />
        {sessionBands.map((band) => (
          <div key={band.key} className="session-band" style={{ left: `${band.left}px`, width: `${band.width}px` }} />
        ))}
        {Number.isFinite(linkedCrosshairX) && (
          <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${Math.round(linkedCrosshairX)}px`, width: 0, borderLeft: '1px dashed rgba(191, 219, 254, 0.45)', pointerEvents: 'none', zIndex: 4 }} />
        )}
        {hoverRow && Number.isFinite(hoverRow.delta) && (
          <div data-testid="flow-v2-tooltip" style={{
            position: 'absolute', top: 4, left: Math.min(Math.max(0, hover.x + 12), (stageRef.current?.clientWidth || 600) - 300),
            maxWidth: 290, pointerEvents: 'none', zIndex: 5, background: 'rgba(2,6,23,0.92)', border: '1px solid #334155',
            borderRadius: 8, padding: '6px 8px', fontSize: '11px', color: '#e2e8f0', lineHeight: 1.4,
          }}>
            <div>{formatShiftedHHMM(hoverRow.time)} · Δ {Math.round(hoverRow.delta).toLocaleString()} · CVD {Math.round(hoverRow.cvd).toLocaleString()} · share {Number.isFinite(hoverRow.share) ? hoverRow.share.toFixed(2) : '–'}{hoverRow.absorb ? ' · absorption' : ''}</div>
            {hoverMarks.map((m, i) => <div key={i} style={{ marginTop: 3, color: '#cbd5e1' }}>{m.text}</div>)}
          </div>
        )}
        {(loading || error || candleTimes.length < 2 || !hasValues) && (
          <div className="flow-overlay">
            <div className="flow-overlay-title">{error ? 'Flow v2 could not load' : loading ? 'Loading Flow v2…' : 'No flow data returned'}</div>
            <div className="flow-overlay-text">{error || notes[0] || 'The loaded sessions did not return any aggressor flow data.'}</div>
          </div>
        )}
      </div>
    </div>
  )
}
