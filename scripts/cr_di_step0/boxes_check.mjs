// CR-DI amendment 9: shipped edgeReadout boxes on a dumped payload (anchor = UTC epoch, identity shift).
import fs from 'node:fs'
import { edgeEffortSeries, edgeReadout, edgeText, edgeVisits, flattenBars, kfmt } from '../../react_price_preview/src/flowV3.js'
const pt = (t) => new Date(t * 1000).toLocaleTimeString('en-GB', { timeZone: 'America/Los_Angeles', hour: '2-digit', minute: '2-digit' })
const [anchor, f, hold = '15'] = process.argv.slice(2)
const p = JSON.parse(fs.readFileSync(f, 'utf8'))
const r = edgeReadout(p, flattenBars(p), Number(anchor), (t) => t, Number(hold))
const sum = (a) => a.reduce((x, y) => x + y, 0)
for (const b of r.frozen) console.log(`frozen ${pt(b.from)}–${pt(b.breakAt)} ${b.bottom.price}–${b.top.price} broke ${pt(b.breakAt)} ${b.dir} · top ${b.top.contracts} buy / ${b.top.opposite} sell · bottom ${b.bottom.contracts} sell / ${b.bottom.opposite} buy · held ${b.heldMin} min`)
console.log(`current ${pt(r.from)}–${pt(r.to)} ${r.bottom.price}–${r.top.price} · header: ${edgeText(r, pt)}`)
const s = edgeEffortSeries(r)
const curK = s.boxes.length - 1
const ci = s.box.map((k, i) => (k === curK ? i : -1)).filter((i) => i >= 0)
console.log(`strip: ${s.t.length} minutes in ${s.boxes.length} drawn boxes (dropped ${s.dropped}); current-box Σ topBuy ${sum(ci.map((i) => s.topBuy[i]))} Σ bottomSell ${sum(ci.map((i) => s.bottomSell[i]))} vs header ${r.top.contracts} / ${r.bottom.contracts}`)
const v = edgeVisits(s)
for (const e of ['top', 'bottom']) console.log(`  ${e} visits: ` + v[e].map((x) => `${pt(x.start)}–${pt(x.end)} ${kfmt(x.main)} / ${kfmt(x.opp)}`).join(' | '))
