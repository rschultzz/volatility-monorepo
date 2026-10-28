// CR-DC — Flow v3 canvas primitives on the price chart's candle series (spec section 6, CR-CH lesson):
// the heatmap, the absorption marks and the volume profile each draw in ONE series primitive (no series,
// price lines or markers per cell), only for the visible time and price range. Their inputs are prepared
// once per data change (flowV3.js, memoised by the caller); a draw only reads them.
// Palette (C1): sell = light grey, buy = blue, marks / lines = slate. No green, no red.
import { CROWDED_TICK_PX, TICK, floorIndex } from '../flowV3.js'

const SELL_RGB = '229,231,235'
const BUY_RGB = '96,165,250'
const SLATE = '#94a3b8'

function visibleIndexRange(chart, times) {
  const vr = chart.timeScale().getVisibleRange?.()
  if (!vr || !times.length) return null
  const from = Math.max(0, floorIndex(times, Number(vr.from)))
  const to = Math.min(times.length - 1, Math.max(0, floorIndex(times, Number(vr.to)) + 1))
  return from <= to ? [from, to] : null
}

function tickPx(series, price) {
  const a = series.priceToCoordinate(price)
  const b = series.priceToCoordinate(price + TICK)
  return a == null || b == null ? null : Math.abs(a - b)
}

class BasePrimitive {
  constructor() {
    this._chart = null
    this._series = null
    this._requestUpdate = null
    this._data = null
    const self = this
    this._view = {
      zOrder: () => self.zOrder(),
      renderer: () => ({ draw: (target) => self._draw(target) }),
    }
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

  zOrder() {
    return 'normal'
  }

  _draw(target) {
    const chart = this._chart
    const series = this._series
    const d = this._data
    if (!chart || !series || !d) return
    target.useBitmapCoordinateSpace((scope) => this.paint(scope, chart, series, d))
  }
}

// Heatmap behind the candles. data: { columns: {1: cols, 2: cols}, times, sell, buy }.
export class HeatmapPrimitive extends BasePrimitive {
  zOrder() {
    return 'bottom'
  }

  paint({ context: ctx, horizontalPixelRatio: hr, verticalPixelRatio: vr }, chart, series, d) {
    if (!d.sell && !d.buy) return
    const range = visibleIndexRange(chart, d.times)
    if (!range) return
    const ts = chart.timeScale()
    const spacing = Number(ts.options().barSpacing) || 6
    const w = Math.max(1, spacing * 0.92)
    let probe = null
    for (let i = range[0]; i <= range[1] && probe == null; i += 1) {
      const c = d.columns[1][i]
      if (c && c.p.length) probe = c.p[0] * TICK
    }
    if (probe == null) return
    const px = tickPx(series, probe)
    if (!px) return
    const res = px < CROWDED_TICK_PX ? 2 : 1
    const cols = d.columns[res]
    const h = px * res
    const both = d.sell && d.buy
    for (let i = range[0]; i <= range[1]; i += 1) {
      const col = cols[i]
      if (!col) continue
      const x = ts.timeToCoordinate(d.times[i])
      if (x == null) continue
      const left = x - w / 2
      for (let k = 0; k < col.p.length; k += 1) {
        const centre = col.p[k] * TICK + ((res - 1) * TICK) / 2
        const y = series.priceToCoordinate(centre)
        if (y == null) continue
        const top = (y - h / 2) * vr
        const hh = Math.max(1, h * vr)
        if (d.sell && col.sa[k] > 0) {
          ctx.fillStyle = `rgba(${SELL_RGB},${col.sa[k]})`
          ctx.fillRect(Math.round(left * hr), top, Math.max(1, Math.round((both ? w / 2 : w) * hr)), hh)
        }
        if (d.buy && col.ba[k] > 0) {
          ctx.fillStyle = `rgba(${BUY_RGB},${col.ba[k]})`
          const bx = both ? left + w / 2 : left
          ctx.fillRect(Math.round(bx * hr), top, Math.max(1, Math.round((both ? w / 2 : w) * hr)), hh)
        }
      }
    }
  }
}

// Absorption marks: an up triangle below the cell ("sellers absorbed") or a down triangle above it
// ("buyers absorbed"), plus a slate line at the price to the break bar or now / `until`.
// data: { marks (flowV3.chartMarks), times }. Hover: `onHover(mark | null, point)`.
export class AbsorbMarksPrimitive extends BasePrimitive {
  constructor(onHover) {
    super()
    this._onHover = onHover
    this._hit = []
    this._move = (param) => {
      const pt = param?.point
      if (!pt || !this._hit.length) {
        this._onHover?.(null, null)
        return
      }
      let best = null
      let bestD = 81
      for (const h of this._hit) {
        const dd = (h.x - pt.x) ** 2 + (h.y - pt.y) ** 2
        if (dd < bestD) {
          bestD = dd
          best = h
        }
      }
      this._onHover?.(best ? best.mark : null, best ? { x: best.x, y: best.y } : null)
    }
  }

  attached(param) {
    super.attached(param)
    param.chart.subscribeCrosshairMove(this._move)
  }

  detached() {
    this._chart?.unsubscribeCrosshairMove(this._move)
    this._onHover?.(null, null)
    super.detached()
  }

  zOrder() {
    return 'top'
  }

  paint({ context: ctx, horizontalPixelRatio: hr, verticalPixelRatio: vr, bitmapSize }, chart, series, d) {
    this._hit = []
    const range = visibleIndexRange(chart, d.times)
    if (!range || !d.marks?.length) return
    const ts = chart.timeScale()
    const rightEdge = bitmapSize.width / hr
    ctx.lineWidth = Math.max(1, Math.round(hr))
    for (const mk of d.marks) {
      if (mk.endIdx < range[0] || mk.idx > range[1]) continue
      const y = series.priceToCoordinate(mk.price)
      if (y == null) continue
      const x0 = ts.timeToCoordinate(d.times[mk.idx])
      const x1raw = ts.timeToCoordinate(d.times[mk.endIdx])
      const x1 = x1raw == null ? rightEdge : x1raw
      const xs = x0 == null ? 0 : x0
      ctx.strokeStyle = 'rgba(148,163,184,0.85)'
      ctx.beginPath()
      ctx.moveTo(Math.round(xs * hr), Math.round(y * vr) + 0.5)
      ctx.lineTo(Math.round(x1 * hr), Math.round(y * vr) + 0.5)
      ctx.stroke()
      if (x0 == null) continue
      const up = mk.kind === 'sellers_absorbed'
      const px = tickPx(series, mk.price) || 2
      const ty = up ? y + px / 2 + 7 : y - px / 2 - 7
      const s = 5
      ctx.fillStyle = SLATE
      ctx.beginPath()
      if (up) {
        ctx.moveTo(x0 * hr, (ty - s) * vr)
        ctx.lineTo((x0 - s) * hr, (ty + s) * vr)
        ctx.lineTo((x0 + s) * hr, (ty + s) * vr)
      } else {
        ctx.moveTo(x0 * hr, (ty + s) * vr)
        ctx.lineTo((x0 - s) * hr, (ty - s) * vr)
        ctx.lineTo((x0 + s) * hr, (ty - s) * vr)
      }
      ctx.closePath()
      ctx.fill()
      this._hit.push({ x: x0, y: ty, mark: mk })
    }
  }
}

// Volume profile at the right edge of the plot (DC16): one stacked bar per price row, sell (grey) to the
// left of buy (blue), width up to 18 % of the plot, scaled to the largest row. data: { rows: {1, 2}, max: {1, 2} }.
export class ProfilePrimitive extends BasePrimitive {
  zOrder() {
    return 'normal'
  }

  paint({ context: ctx, horizontalPixelRatio: hr, verticalPixelRatio: vr, bitmapSize }, chart, series, d) {
    const r1 = d.rows?.[1]
    if (!r1?.length) return
    const px = tickPx(series, r1[0].p * TICK)
    if (!px) return
    const res = px < CROWDED_TICK_PX ? 2 : 1
    const rows = d.rows[res]
    const max = d.max[res]
    if (!(max > 0)) return
    const plotW = bitmapSize.width / hr
    const plotH = bitmapSize.height / vr
    const maxW = plotW * 0.18
    const right = plotW - 2
    const h = Math.max(1, px * res - (px * res > 3 ? 1 : 0))
    for (const r of rows) {
      const centre = r.p * TICK + ((res - 1) * TICK) / 2
      const y = series.priceToCoordinate(centre)
      if (y == null || y < -h || y > plotH + h) continue
      const wb = (r.buy / max) * maxW
      const ws = (r.sell / max) * maxW
      const top = (y - h / 2) * vr
      if (wb > 0) {
        ctx.fillStyle = `rgba(${BUY_RGB},0.55)`
        ctx.fillRect(Math.round((right - wb) * hr), top, Math.max(1, Math.round(wb * hr)), Math.max(1, h * vr))
      }
      if (ws > 0) {
        ctx.fillStyle = `rgba(${SELL_RGB},0.45)`
        ctx.fillRect(Math.round((right - wb - ws) * hr), top, Math.max(1, Math.round(ws * hr)), Math.max(1, h * vr))
      }
    }
  }
}
