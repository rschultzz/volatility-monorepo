// CR-DI amendment 9 check: closes since the anchor and what each candidate restart rule does on a dumped payload.
import fs from 'node:fs'
import { flattenBars } from '../../react_price_preview/src/flowV3.js'
const pt = (t) => new Date(t * 1000).toLocaleTimeString('en-GB', { timeZone: 'America/Los_Angeles', hour: '2-digit', minute: '2-digit' })
const [anchor, f] = process.argv.slice(2)
const b = flattenBars(JSON.parse(fs.readFileSync(f, 'utf8')))
const i0 = b.t.findIndex((t) => t >= Number(anchor))
const BRK = 2, NEAR = 1
// today: edges ratchet on > 2 pt closes, never restart
function today() {
  let top = b.c[i0], bot = b.c[i0]; const ev = []
  for (let i = i0 + 1; i < b.t.length; i++) {
    if (b.c[i] > top + BRK + 1e-9) { ev.push(`${pt(b.t[i])} top ${top}→${b.c[i]}`); top = b.c[i] }
    if (b.c[i] < bot - BRK - 1e-9) { ev.push(`${pt(b.t[i])} bottom ${bot}→${b.c[i]}`); bot = b.c[i] }
  }
  return ev
}
// literal amendment: every > 2 pt close freezes the box; new box top = bottom = that close
// tested: a > 2 pt close beyond an edge only freezes the box if that edge was touched (high/low within 1 pt) on a
// later bar after it was set; otherwise the edge grows to the close, as today
function restart(needTested) {
  const boxes = []
  let s = i0, top = b.c[i0], bot = b.c[i0], topSet = i0, botSet = i0, topT = false, botT = false
  for (let i = i0 + 1; i < b.t.length; i++) {
    const c = b.c[i]
    const up = c > top + BRK + 1e-9, dn = c < bot - BRK - 1e-9
    if ((up && (!needTested || topT)) || (dn && (!needTested || botT))) {
      boxes.push({ from: pt(b.t[s]), to: pt(b.t[i]), top, bot, dir: up ? '↑' : '↓' })
      s = i; top = bot = c; topSet = botSet = i; topT = botT = false
      continue
    }
    if (up) { top = c; topSet = i; topT = false }
    if (dn) { bot = c; botSet = i; botT = false }
    if (i > topSet && b.h[i] >= top - NEAR - 1e-9) topT = true
    if (i > botSet && b.l[i] <= bot + NEAR + 1e-9) botT = true
  }
  boxes.push({ from: pt(b.t[s]), to: 'now', top, bot, current: true })
  return boxes
}
console.log('last bar', pt(b.t.at(-1)), '\ncloses from anchor:', b.t.slice(i0).map((t, k) => `${pt(t)} ${b.c[i0 + k]}`).join(' | '))
console.log('\nTODAY (ratchet):', today().join('; '))
const lit = restart(false)
console.log(`\nLITERAL restart: ${lit.length} boxes; first 8:`, lit.slice(0, 8).map((x) => `${x.from}–${x.to} ${x.bot}–${x.top} ${x.dir || ''}`).join(' | '))
console.log('\nTESTED-edge restart:', restart(true).map((x) => `${x.from}–${x.to} ${x.bot}–${x.top} ${x.dir || '(current)'}`).join(' | '))
// held-N: a > 2 pt close beyond an edge freezes the box only once that edge has stood N minutes since it was set;
// before that the edge grows to the close (today's ratchet)
function heldN(N) {
  const boxes = []
  let s = i0, top = b.c[i0], bot = b.c[i0], topSet = i0, botSet = i0
  for (let i = i0 + 1; i < b.t.length; i++) {
    const c = b.c[i]
    const up = c > top + BRK + 1e-9, dn = c < bot - BRK - 1e-9
    if ((up && i - topSet >= N) || (dn && i - botSet >= N)) {
      boxes.push(`${pt(b.t[s])}–${pt(b.t[i])} ${bot}–${top} ${up ? '↑' : '↓'}`)
      s = i; top = bot = c; topSet = botSet = i
      continue
    }
    if (up) { top = c; topSet = i }
    if (dn) { bot = c; botSet = i }
  }
  boxes.push(`${pt(b.t[s])}–now ${bot}–${top} (current)`)
  return boxes
}
for (const N of [5, 10, 15, 30]) console.log(`\nHELD-${N}:`, heldN(N).join(' | '))
