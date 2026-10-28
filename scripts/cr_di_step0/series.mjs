// CR-DI Step 0b: header edge counts (shipped edgeReadout) vs a per-minute series by the same rule, on dumped
// cells payloads; per-visit totals; and chat's fixed-band version. Times printed PT.
import fs from 'node:fs'
import { edgeReadout, flattenBars, kfmt, TICK, EDGE_NEAR_PTS } from '../../react_price_preview/src/flowV3.js'
const pt = (t) => new Date(t * 1000).toLocaleTimeString('en-GB', { timeZone: 'America/Los_Angeles', hour: '2-digit', minute: '2-digit' })
const id = (t) => t
const [anchorUtc, ...files] = process.argv.slice(2)
const GAP = 2
function visits(mins, main, opp) {
  const out = []
  let cur = null
  let gap = 0
  for (let i = 0; i < mins.length; i += 1) {
    if (main[i] > 0) {
      if (cur && gap <= GAP) { cur.end = mins[i]; cur.main += main[i]; cur.opp += opp[i] }
      else { if (cur) out.push(cur); cur = { start: mins[i], end: mins[i], main: main[i], opp: opp[i] } }
      gap = 0
    } else if (cur) {
      gap += 1
      if (gap <= GAP) cur.opp += 0
    }
  }
  if (cur) out.push(cur)
  return out
}
for (const f of files) {
  const p = JSON.parse(fs.readFileSync(f, 'utf8'))
  const bars = flattenBars(p)
  const r = edgeReadout(p, bars, Number(anchorUtc), id)
  const near = EDGE_NEAR_PTS / TICK
  const topT = Math.round(r.top.price / TICK)
  const botT = Math.round(r.bottom.price / TICK)
  const mins = bars.t.filter((t) => t >= r.from && t <= r.to)
  const ix = new Map(mins.map((t, i) => [t, i]))
  const z = () => new Array(mins.length).fill(0)
  const tb = z(), ts = z(), bs = z(), bb = z(), allB = z(), allS = z()
  let fixTop = 0, fixBot = 0, fixTopS = 0, fixBotB = 0
  for (const s of p.segments) {
    const c = s.cells
    for (let i = 0; i < c.t.length; i += 1) {
      const k = ix.get(Number(c.t[i]))
      if (k == null) continue
      const pp = Number(c.p[i]); const b = Number(c.buy[i]) || 0; const se = Number(c.sell[i]) || 0
      allB[k] += b; allS[k] += se
      if (Math.abs(pp - topT) <= near) { tb[k] += b; ts[k] += se }
      if (Math.abs(pp - botT) <= near) { bs[k] += se; bb[k] += b }
      if (pp * TICK >= 7768.25) { fixTop += b; fixTopS += se }
      if (pp * TICK <= 7763) { fixBot += se; fixBotB += b }
    }
  }
  const sum = (a) => a.reduce((x, y) => x + y, 0)
  console.log(`\n== ${f}\nanchor ${pt(r.from)} → last bar ${pt(r.to)} PT · exact ${r.exact}`)
  console.log(`header (edgeReadout): Top ${r.top.price.toFixed(2)} · ${r.top.contracts} buyers (${kfmt(r.top.contracts)}) · breaks ${r.top.breaks} last ${r.top.brokeAt ? pt(r.top.brokeAt) : '-'} | Bottom ${r.bottom.price.toFixed(2)} · ${r.bottom.contracts} sellers (${kfmt(r.bottom.contracts)}) · breaks ${r.bottom.breaks} last ${r.bottom.brokeAt ? pt(r.bottom.brokeAt) : '-'}`)
  console.log(`per-minute sums: top buy ${sum(tb)} sell ${sum(ts)} | bottom sell ${sum(bs)} buy ${sum(bb)} · main equal header: ${sum(tb) === r.top.contracts && sum(bs) === r.bottom.contracts}`)
  console.log(`fixed bands (>= 7768.25 / <= 7763, same minutes): top buy ${fixTop} (${kfmt(fixTop)}) sell ${fixTopS} | bottom sell ${fixBot} (${kfmt(fixBot)}) buy ${fixBotB}`)
  if (process.env.SERIES) {
    console.log('minute  topBuy topSell | botSell botBuy | allBuy allSell')
    mins.forEach((t, i) => { if (tb[i] || ts[i] || bs[i] || bb[i]) console.log(`${pt(t)}  ${String(tb[i]).padStart(6)} ${String(ts[i]).padStart(7)} | ${String(bs[i]).padStart(7)} ${String(bb[i]).padStart(6)} | ${String(allB[i]).padStart(6)} ${String(allS[i]).padStart(7)}`) })
  }
  for (const [name, main, opp, a, b] of [['TOP', tb, ts, 'buy', 'sell'], ['BOTTOM', bs, bb, 'sell', 'buy']]) {
    for (const v of visits(mins, main, opp)) console.log(`  ${name} visit ${pt(v.start)}–${pt(v.end)}: ${a} ${v.main} / ${b} ${v.opp}  label "${kfmt(v.main)} / ${kfmt(v.opp)}"  net ${name === 'TOP' ? v.main - v.opp : v.opp - v.main}`)
  }
}
