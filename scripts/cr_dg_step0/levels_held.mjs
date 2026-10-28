// CR-DG Step 0: run the shipped absorptionLevels on dumped cells payloads; print held vs payload "now".
import fs from 'node:fs'
import { absorptionLevels, payloadNow } from '../../react_price_preview/src/flowV3.js'
const pt = (t) => new Date(t * 1000).toLocaleTimeString('en-GB', { timeZone: 'America/Los_Angeles', hour: '2-digit', minute: '2-digit' })
for (const f of process.argv.slice(2)) {
  const p = JSON.parse(fs.readFileSync(f, 'utf8'))
  const times = p.segments.flatMap((s) => s.bars.t)
  const now = payloadNow(p)
  console.log(`\n${f}\n  payloadNow ${pt(now)} PT · rth seg end ${pt(p.segments.at(-1).end)} PT`)
  for (const lv of absorptionLevels(p, times, (t) => t)) {
    const mk = p.segments.flatMap((s) => s.marks || []).find((m) => m.time === lv.start && m.kind === lv.kind)
    console.log(`  ${lv.kind.padEnd(17)} ${lv.price.toFixed(2)} from ${pt(lv.start)} end ${pt(lv.end)} broken=${lv.broken} level.heldMin ${lv.heldMin} · server held_min ${mk?.held_min}`)
  }
}
