// CR-DI: shipped edgeReadout(...).minutes + edgeVisits on dumped payloads (anchor = UTC epoch, identity shift).
import fs from 'node:fs'
import { edgeReadout, edgeVisits, flattenBars, kfmt } from '../../react_price_preview/src/flowV3.js'
const pt = (t) => new Date(t * 1000).toLocaleTimeString('en-GB', { timeZone: 'America/Los_Angeles', hour: '2-digit', minute: '2-digit' })
const [anchor, ...files] = process.argv.slice(2)
for (const f of files) {
  const p = JSON.parse(fs.readFileSync(f, 'utf8'))
  const r = edgeReadout(p, flattenBars(p), Number(anchor), (t) => t)
  const v = edgeVisits(r.minutes)
  const sum = (a) => a.reduce((x, y) => x + y, 0)
  console.log(`${f.split('/').pop()} last ${pt(r.to)}: header ${r.top.contracts} / ${r.bottom.contracts}; Σ topBuy ${sum(r.minutes.topBuy)} Σ bottomSell ${sum(r.minutes.bottomSell)}`)
  for (const e of ['top', 'bottom']) for (const x of v[e]) console.log(`  ${e} ${pt(x.start)}–${pt(x.end)} ${x.main} / ${x.opp}  "${kfmt(x.main)} / ${kfmt(x.opp)}"`)
}
