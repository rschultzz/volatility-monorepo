// CR-DR item 2 — Ryan's real trades on the price chart, in ONE series primitive (CR-CH lesson): per trade an entry
// marker (▲ long / ▼ short) and an exit marker (×) at the fill time and price, a thin line between them, and the
// exit-side orders resting while the position was open (stop dashed, target dotted), each from its placing (or the
// entry) to its fill / cancel (or the exit). Palette (C1): blue for long, slate for short; no red / green.
// Fill times are seconds inside a candle: x = the candle's x + (offset in the bar − ½) × bar spacing, so a fill sits
// left-to-right inside its own bar (offset clamped to 10–90 % of the bar).
import { floorIndex } from '../flowV3.js'

export const TRADE_LONG_RGB = '96,165,250'
export const TRADE_SHORT_RGB = '148,163,184'
const HIT_R2 = 100 // 10 px

export function tradeRgb(direction) {
  return direction === 'long' ? TRADE_LONG_RGB : TRADE_SHORT_RGB
}

// x (css px) of a shifted epoch `t` on candles `times` (shifted, ascending, bar length `step` s); null off the data.
export function timeX(ts, times, step, t) {
  if (t == null || !times.length) return null
  const i = floorIndex(times, t)
  if (i < 0) return null
  const x = ts.timeToCoordinate(times[i])
  if (x == null) return null
  const spacing = Number(ts.options().barSpacing) || 6
  // kept 10 % inside the bar's edges, so a fill at :00 is drawn on its own bar, not the boundary with the previous one
  const frac = Math.min(0.9, Math.max(0.1, (t - times[i]) / (step || 60)))
  return x + (frac - 0.5) * spacing
}

// data: { trades: [{ id, direction, entryT, entryPrice, exitT, exitPrice, lines: [{ kind, price, fromT, toT }] }],
//         times, step, selectedId }
export class TradeMarksPrimitive {
  constructor() {
    this._chart = null
    this._series = null
    this._requestUpdate = null
    this._data = null
    this._hit = []
    const self = this
    this._view = {
      zOrder: () => 'top',
      renderer: () => ({ draw: (target) => self._draw(target) }),
    }
  }

  // The trade id of the marker within 10 px of a chart click point, or null. Called by the chart's own click handler,
  // which then skips its timeslice toggle.
  hitTest(pt) {
    if (!pt || !this._hit.length) return null
    let best = null
    let bestD = HIT_R2
    for (const h of this._hit) {
      const dd = (h.x - pt.x) ** 2 + (h.y - pt.y) ** 2
      if (dd <= bestD) {
        bestD = dd
        best = h
      }
    }
    return best ? best.id : null
  }

  attached({ chart, series, requestUpdate }) {
    this._chart = chart
    this._series = series
    this._requestUpdate = requestUpdate
  }

  detached() {
    this._chart = null
    this._series = null
    this._requestUpdate = null
  }

  paneViews() {
    return [this._view]
  }

  updateAllViews() {}

  setData(data) {
    this._data = data
    this._requestUpdate?.()
  }

  // Marker positions (css px) as last drawn — the click hit list; also read by the smoke spec.
  hits() {
    return this._hit.slice()
  }

  _draw(target) {
    const chart = this._chart
    const series = this._series
    const d = this._data
    if (!chart || !series || !d) return
    target.useBitmapCoordinateSpace((scope) => this.paint(scope, chart, series, d))
  }

  paint({ context: ctx, horizontalPixelRatio: hr, verticalPixelRatio: vr }, chart, series, d) {
    this._hit = []
    const ts = chart.timeScale()
    const times = Array.isArray(d.times) ? d.times : []
    for (const tr of d.trades || []) {
      const rgb = tradeRgb(tr.direction)
      const sel = tr.id === d.selectedId
      const xe = timeX(ts, times, d.step, tr.entryT)
      const xx = timeX(ts, times, d.step, tr.exitT)
      const ye = series.priceToCoordinate(tr.entryPrice)
      const yx = series.priceToCoordinate(tr.exitPrice)

      // resting stop / target lines, under the markers
      for (const ln of tr.lines || []) {
        const x0 = timeX(ts, times, d.step, ln.fromT)
        const x1 = timeX(ts, times, d.step, ln.toT)
        const y = series.priceToCoordinate(ln.price)
        if (x0 == null || x1 == null || y == null) continue
        ctx.save()
        ctx.strokeStyle = `rgba(${rgb},${sel ? 0.95 : 0.7})`
        ctx.lineWidth = Math.max(1, Math.round(hr))
        ctx.setLineDash(ln.kind === 'stop' ? [5 * hr, 3 * hr] : [1.5 * hr, 2.5 * hr])
        const yy = Math.round(y * vr) + 0.5
        ctx.beginPath()
        ctx.moveTo(Math.round(x0 * hr), yy)
        ctx.lineTo(Math.round(x1 * hr), yy)
        ctx.stroke()
        ctx.restore()
        ctx.fillStyle = `rgba(${rgb},0.95)`
        ctx.font = `${Math.round(10 * vr)}px sans-serif`
        ctx.textBaseline = 'bottom'
        ctx.fillText(ln.kind === 'stop' ? 'stop' : 'TP', Math.round((x1 + 3) * hr), Math.round((y - 1) * vr))
      }

      if (xe != null && xx != null && ye != null && yx != null) {
        ctx.save()
        ctx.strokeStyle = `rgba(${rgb},${sel ? 1 : 0.75})`
        ctx.lineWidth = Math.max(1, Math.round((sel ? 2 : 1) * hr))
        ctx.beginPath()
        ctx.moveTo(xe * hr, ye * vr)
        ctx.lineTo(xx * hr, yx * vr)
        ctx.stroke()
        ctx.restore()
      }

      const s = sel ? 7 : 6
      if (xe != null && ye != null) {
        // ▲ below the fill price for a long, ▼ above it for a short; its tip on the price
        const up = tr.direction === 'long'
        ctx.fillStyle = `rgba(${rgb},1)`
        ctx.strokeStyle = 'rgba(15,23,42,0.9)'
        ctx.lineWidth = Math.max(1, Math.round(hr))
        ctx.beginPath()
        if (up) {
          ctx.moveTo(xe * hr, ye * vr)
          ctx.lineTo((xe - s) * hr, (ye + 2 * s) * vr)
          ctx.lineTo((xe + s) * hr, (ye + 2 * s) * vr)
        } else {
          ctx.moveTo(xe * hr, ye * vr)
          ctx.lineTo((xe - s) * hr, (ye - 2 * s) * vr)
          ctx.lineTo((xe + s) * hr, (ye - 2 * s) * vr)
        }
        ctx.closePath()
        ctx.fill()
        ctx.stroke()
        this._hit.push({ id: tr.id, kind: 'entry', x: xe, y: up ? ye + s : ye - s, priceY: ye })
      }
      if (xx != null && yx != null) {
        const a = s - 1
        ctx.save()
        ctx.strokeStyle = 'rgba(15,23,42,0.9)'
        ctx.lineWidth = Math.max(1, Math.round(4.5 * hr))
        ctx.beginPath()
        ctx.moveTo((xx - a) * hr, (yx - a) * vr)
        ctx.lineTo((xx + a) * hr, (yx + a) * vr)
        ctx.moveTo((xx + a) * hr, (yx - a) * vr)
        ctx.lineTo((xx - a) * hr, (yx + a) * vr)
        ctx.stroke()
        ctx.strokeStyle = `rgba(${rgb},1)`
        ctx.lineWidth = Math.max(1, Math.round(2 * hr))
        ctx.stroke()
        ctx.restore()
        this._hit.push({ id: tr.id, kind: 'exit', x: xx, y: yx, priceY: yx })
      }
    }
  }
}
