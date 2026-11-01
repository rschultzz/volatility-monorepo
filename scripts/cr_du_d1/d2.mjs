// CR-DU diagnosis D2 items 1–5 (see d2.py): per CR-CZ card at its decision minute, card side. Read-only, no outcomes.
//   node scripts/cr_du_d1/d2.mjs <job.json> <out.json>
import fs from 'node:fs'
import { LEVELS_MAX, absorptionLevels, anchorReadout, boxPercentiles } from '../../react_price_preview/src/flowV3.js'
import { prepareSession } from '../../react_price_preview/src/v1Engine.js'

const [jobPath, outPath] = process.argv.slice(2)
const job = JSON.parse(fs.readFileSync(jobPath, 'utf8'))
const cache = new Map()
const out = []
for (const c of job) {
  if (!cache.has(c.file)) cache.set(c.file, JSON.parse(fs.readFileSync(c.file, 'utf8')))
  const ses = cache.get(c.file)
  const ctx = prepareSession({ payload: ses.payload, inputs: ses.inputs, tradeDate: ses.trade_date })
  const i = ctx.bars.t.indexOf(c.t)
  const j0 = ctx.bars.t.indexOf(c.r0)
  const want = c.side === 'short' ? 1 : -1 // card side: buying for a short, selling for a long
  const pctOf = (fromUtc) => {
    const ro = anchorReadout(ctx.rows, ctx.closes, ctx.shift(fromUtc), ctx.times[i])
    const pc = boxPercentiles(ro, ctx.ptMin(fromUtc), ctx.pay.segments, ctx.baseline)
    const share = ro && ro.vol > 0 ? ro.delta / ro.vol : null
    return { share, p: pc?.share?.p ?? null, minutes: ro?.minutes ?? null, cardSide: share == null ? null : Math.sign(share) === want }
  }
  const rec = { id: c.id, label: c.label, side: c.side, warm: c.warm }
  if (i < 0 || j0 < 0) { out.push({ ...rec, error: 'bar' }); continue }
  // item 2: (b) the card's box, (c) the last 10 closed minutes (anchor bar i - 10: bars i-9 .. i summed)
  rec.cardBox = pctOf(c.r0)
  rec.last10 = i >= 10 ? pctOf(ctx.bars.t[i - 10]) : null
  // item 3: cumulative delta over the card's box (bars r0 .. t, from 0 at r0) vs minute: OLS slope and its t-stat
  const ys = []
  let acc = 0
  for (let j = j0; j <= i; j += 1) { acc += j === j0 ? 0 : ctx.bars.buy[j] - ctx.bars.sell[j]; ys.push(acc) }
  const n = ys.length
  const xs = ys.map((_, k) => k)
  const mx = (n - 1) / 2
  const my = ys.reduce((a, v) => a + v, 0) / n
  let sxx = 0, sxy = 0
  for (let k = 0; k < n; k += 1) { sxx += (xs[k] - mx) ** 2; sxy += (xs[k] - mx) * (ys[k] - my) }
  const slope = sxy / sxx
  let sse = 0
  for (let k = 0; k < n; k += 1) sse += (ys[k] - (my + slope * (xs[k] - mx))) ** 2
  const se = Math.sqrt(sse / Math.max(1, n - 2) / sxx)
  rec.cvd = { n, slope, t: se > 0 ? slope / se : null, cardSideT: se > 0 ? want * slope / se : null }
  // item 4: card-side marks inside the card's box window (mark minute ≥ r0, earlier than and confirmed by the decision)
  let hi = -Infinity, lo = Infinity
  for (let j = j0; j <= i; j += 1) { hi = Math.max(hi, ctx.bars.h[j]); lo = Math.min(lo, ctx.bars.l[j]) }
  const kind = c.side === 'short' ? 'buyers_absorbed' : 'sellers_absorbed'
  const edge = c.side === 'short' ? c.top : c.bottom
  const marks = (ctx.pay.segments[0].marks || []).filter((m) => m.kind === kind && m.time < c.t && m.confirm_time <= c.t + 60)
  const inWin = marks.filter((m) => m.time >= c.r0)
  rec.win = {
    any: inWin.length, // anywhere in price
    inRange: inWin.filter((m) => m.price <= hi + 2 + 1e-9 && m.price >= lo - 2 - 1e-9).length, // box high–low ± 2 (as boxLevels)
    atEdge: inWin.filter((m) => Math.abs(m.price - edge) <= 2 + 1e-9).length,
  }
  // the chart's top 8 as of the decision close (what the verdict reads): is a card-side in-window mark among them?
  const top8 = absorptionLevels(ctx.pay, ctx.times, ctx.shift, LEVELS_MAX, c.t + 60)
  rec.win.inTop8 = top8.filter((l) => l.kind === kind && l.start >= c.r0 && !l.broken).length
  // item 5: at the faded edge (≤ 2 pts), any earlier mark, unbroken by closes with a 1-tick vs a 2-pt threshold
  const unbroken = (m, pad) => !ctx.bars.t.some((t, j) => t > m.time && t <= c.t
    && (c.side === 'short' ? ctx.bars.c[j] > m.price + pad + 1e-9 : ctx.bars.c[j] < m.price - pad - 1e-9))
  const atEdge = marks.filter((m) => Math.abs(m.price - edge) <= 2 + 1e-9)
  const pack = (pad) => atEdge.filter((m) => unbroken(m, pad)).map((m) => ({ dist: Math.abs(m.price - edge), before: Math.round((c.t + 60 - m.time) / 60), inBox: m.time >= c.r0 }))
  rec.edge = { tick: pack(0.25), pts2: pack(2) }
  out.push(rec)
}
fs.writeFileSync(outPath, JSON.stringify(out))
