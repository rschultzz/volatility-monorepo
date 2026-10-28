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

// CR-DF — absorption levels and the edge lines, in magenta (C1 deviation approved by Ryan in the kickoff:
// distinct from candles, heat and GEX lines; still no red / green).
const MAGENTA_RGB = '232,121,249' // #e879f9
// CR-DG — "sellers absorbed" (floor) levels in orange (C1 deviation approved by Ryan, chat 2026-10-28; still no
// red / green). "Buyers absorbed" (ceiling) levels and the edge lines stay magenta.
export const ORANGE_RGB = '251,146,60' // #fb923c
const levelRgb = (kind) => (kind === 'sellers_absorbed' ? ORANGE_RGB : MAGENTA_RGB)

// Absorption levels: one horizontal line per level from its start minute to its break bar or now, width from
// contracts absorbed, opacity from age, with its label at the right end. data: { levels (flowV3.absorptionLevels),
// times }. Hover: `onHover(level | null, point)` within 5 px of a line.
export class AbsorptionLevelsPrimitive extends BasePrimitive {
  constructor(onHover) {
    super()
    this._onHover = onHover
    this._hit = []
    this._move = (param) => {
      const pt = param?.point
      let best = null
      let bestD = 6
      if (pt) {
        for (const h of this._hit) {
          if (pt.x < h.x0 - 3 || pt.x > h.x1 + 3) continue
          const dd = Math.abs(pt.y - h.y)
          if (dd < bestD) {
            bestD = dd
            best = h
          }
        }
      }
      this._onHover?.(best ? best.level : null, best ? { x: pt.x, y: best.y } : null)
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
    if (!range || !d.levels?.length) return
    const ts = chart.timeScale()
    const plotW = bitmapSize.width / hr
    ctx.font = `${Math.round(10 * vr)}px sans-serif`
    ctx.textBaseline = 'bottom'
    const placed = [] // label boxes (css px): a label that would overlap one is nudged up / down, else not drawn
    const free = (r) => placed.every((q) => r.x1 < q.x0 || r.x0 > q.x1 || r.y1 < q.y0 || r.y0 > q.y1)
    for (const lv of d.levels) {
      if (lv.endIdx < range[0] || lv.idx > range[1]) continue
      const y = series.priceToCoordinate(lv.price)
      if (y == null) continue
      const x0raw = ts.timeToCoordinate(d.times[lv.idx])
      const x1raw = ts.timeToCoordinate(d.times[lv.endIdx])
      const x0 = x0raw == null ? 0 : x0raw
      const x1 = x1raw == null ? plotW : x1raw
      const lw = Math.max(1, Math.round(lv.width * vr))
      const yy = Math.round(y * vr) + (lw % 2 ? 0.5 : 0)
      const rgb = levelRgb(lv.kind)
      ctx.strokeStyle = `rgba(${rgb},${lv.alpha})`
      ctx.lineWidth = lw
      ctx.beginPath()
      ctx.moveTo(Math.round(x0 * hr), yy)
      ctx.lineTo(Math.round(Math.max(x1, x0 + 2) * hr), yy)
      ctx.stroke()
      ctx.fillStyle = `rgba(${rgb},${Math.max(0.55, lv.alpha)})`
      const tw = ctx.measureText(lv.label).width / hr
      const lx = Math.min(x1 + 4, plotW - tw - 2)
      const base = y - lv.width / 2 - 2
      for (const dy of [0, -11, 11 + lv.width + 2, -22]) {
        const box = { x0: lx, x1: lx + tw, y0: base + dy - 11, y1: base + dy }
        if (!free(box)) continue
        placed.push(box)
        ctx.fillText(lv.label, Math.round(lx * hr), Math.round((base + dy) * vr))
        break
      }
      this._hit.push({ x0, x1, y, level: lv })
    }
  }
}

// Edge lines (DF4): thin dashed magenta lines at each box's top and bottom. CR-DI amendment 9: frozen boxes (broken)
// are drawn at lower opacity from their start to the break minute; the current box from its start to the last bar.
// data: { boxes: [{ from, to (shifted epochs), top, bottom (prices), frozen, box }] }. Hover: `onHover(box | null, pt)`
// within 5 px of a frozen box's line (the current box is the header's).
export const FROZEN_EDGE_ALPHA = 0.35
export class EdgeLinesPrimitive extends BasePrimitive {
  constructor(onHover) {
    super()
    this._onHover = onHover
    this._hit = []
    this._move = (param) => {
      const pt = param?.point
      let best = null
      let bestD = 6
      if (pt) {
        for (const h of this._hit) {
          if (pt.x < h.x0 - 3 || pt.x > h.x1 + 3) continue
          const dd = Math.abs(pt.y - h.y)
          if (dd < bestD) {
            bestD = dd
            best = h
          }
        }
      }
      this._onHover?.(best ? best.box : null, best ? { x: pt.x, y: best.y } : null)
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
    const ts = chart.timeScale()
    const plotW = bitmapSize.width / hr
    const vrange = ts.getVisibleRange?.()
    if (!vrange) return
    ctx.save()
    ctx.lineWidth = Math.max(1, Math.round(hr))
    ctx.setLineDash([Math.round(4 * hr), Math.round(3 * hr)])
    for (const b of d.boxes || []) {
      if (b.to < Number(vrange.from) || b.from > Number(vrange.to)) continue
      const xa = ts.timeToCoordinate(b.from)
      const xb = ts.timeToCoordinate(b.to)
      const x0 = xa == null ? 0 : xa
      const x1 = xb == null ? plotW : xb
      ctx.strokeStyle = `rgba(${MAGENTA_RGB},${b.frozen ? FROZEN_EDGE_ALPHA : 0.8})`
      for (const p of [b.top, b.bottom]) {
        const y = series.priceToCoordinate(p)
        if (y == null) continue
        const yy = Math.round(y * vr) + 0.5
        ctx.beginPath()
        ctx.moveTo(Math.round(x0 * hr), yy)
        ctx.lineTo(Math.round(Math.max(x1, x0 + 2) * hr), yy)
        ctx.stroke()
        if (b.frozen) this._hit.push({ x0, x1, y, box: b.box })
      }
    }
    ctx.restore()
  }
}

// CR-DI — Edge effort strip, drawn in its own pane of the Flow v3 chart (on a transparent carrier series): per
// candle, the top edge upward from the middle (aggressive buys = main, blue; sells = thinner, lighter grey beside
// it) and the bottom edge downward (sells = main, grey; buys = thinner, lighter blue). Palette only (C1).
// data: { times (shifted candle times), cols: Map idx -> [top buy, top sell, bottom sell, bottom buy], max }.
export const EDGE_STRIP_LABEL_PX = 11
export class EdgeEffortPrimitive extends BasePrimitive {
  zOrder() {
    return 'normal'
  }

  paint({ context: ctx, horizontalPixelRatio: hr, verticalPixelRatio: vr, bitmapSize }, chart, series, d) {
    const H = bitmapSize.height / vr
    const mid = H / 2
    ctx.fillStyle = 'rgba(148,163,184,0.25)'
    ctx.fillRect(0, Math.round(mid * vr), bitmapSize.width, Math.max(1, Math.round(vr)))
    const range = visibleIndexRange(chart, d.times)
    if (!range || !(d.max > 0)) return
    const ts = chart.timeScale()
    const spacing = Number(ts.options().barSpacing) || 6
    const w = Math.max(1.5, spacing * 0.86)
    const wMain = Math.max(1, w * 0.62)
    const wOpp = Math.max(1, w - wMain)
    const half = Math.max(4, mid - EDGE_STRIP_LABEL_PX - 1)
    const hpx = (v) => (v > 0 ? Math.max(1, (v / d.max) * half) : 0)
    const bar = (rgba, x, wd, y0, h) => {
      if (!(h > 0)) return
      ctx.fillStyle = rgba
      ctx.fillRect(Math.round(x * hr), Math.round(y0 * vr), Math.max(1, Math.round(wd * hr)), Math.max(1, Math.round(h * vr)))
    }
    for (let i = range[0]; i <= range[1]; i += 1) {
      const c = d.cols.get(i)
      if (!c) continue
      const x = ts.timeToCoordinate(d.times[i])
      if (x == null) continue
      const left = x - w / 2
      const [tb, tsl, bs, bb] = c
      let h = hpx(tb)
      bar(`rgba(${BUY_RGB},0.85)`, left, wMain, mid - h, h)
      h = hpx(tsl)
      bar(`rgba(${SELL_RGB},0.38)`, left + wMain, wOpp, mid - h, h)
      bar(`rgba(${SELL_RGB},0.72)`, left, wMain, mid + 1, hpx(bs))
      bar(`rgba(${BUY_RGB},0.38)`, left + wMain, wOpp, mid + 1, hpx(bb))
    }
    // Visit labels "main / opposite", centred on the visit: above the top bars, below the bottom ones. Larger visits
    // are placed first; a label that would overlap one already placed on its edge is dropped (the hover has it).
    if (!d.labels) return
    ctx.font = `${Math.round(10 * vr)}px sans-serif`
    ctx.fillStyle = 'rgba(203,213,225,0.85)'
    ctx.textBaseline = 'alphabetic'
    const plotW = bitmapSize.width / hr
    for (const [edge, y] of [['top', EDGE_STRIP_LABEL_PX - 2], ['bottom', H - 2]]) {
      const placed = []
      const list = [...(d.labels[edge] || [])].sort((a, b) => b.main - a.main)
      for (const v of list) {
        if (v.i1 < range[0] || v.i0 > range[1]) continue
        const xa = ts.timeToCoordinate(d.times[v.i0])
        const xb = ts.timeToCoordinate(d.times[v.i1])
        if (xa == null && xb == null) continue
        const cx = ((xa ?? xb) + (xb ?? xa)) / 2
        const tw = ctx.measureText(v.text).width / hr
        const x0 = Math.min(Math.max(0, cx - tw / 2), plotW - tw)
        const box = [x0 - 3, x0 + tw + 3]
        if (placed.some((q) => !(box[1] < q[0] || box[0] > q[1]))) continue
        placed.push(box)
        ctx.fillText(v.text, Math.round(x0 * hr), Math.round(y * vr))
      }
    }
  }
}
