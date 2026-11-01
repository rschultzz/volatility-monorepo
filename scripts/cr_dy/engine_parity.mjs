// CR-DY — engine frozen check and box parity, counts and hashes only (set-2 sessions: no per-trade output).
//   node scripts/cr_dy/engine_parity.mjs <tag src dir> <sessions dir>
// (1) simulateSession on every session: this checkout vs the tag's source, sha256 over trades + minutes.
// (2) For every trade: decisionT / box present; the panel's window path (edgeReadout manual, box.from → decisionT) gives
//     the engine's edges; panelEngine at the decision minute gives the trade's side; wick vs close heights.
import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import { prepareSession, simulateSession, panelEngine, V1_DEFAULTS, V1_ENGINE_VERSION } from '../../react_price_preview/src/v1Engine.js'
import { edgeReadout, flattenBars } from '../../react_price_preview/src/flowV3.js'

const [tagSrc, dir] = process.argv.slice(2)
const tag = await import(pathToFileURL(path.join(tagSrc, 'v1Engine.js')).href)
const files = fs.readdirSync(dir).filter((f) => f.endsWith('.json')).sort()
const hash = (h, sim) => h.update(JSON.stringify({ t: sim.trades, o: sim.open, m: sim.minutes, s: sim.state }))
const hNew = crypto.createHash('sha256')
const hTag = crypto.createHash('sha256')
const c = { sessions: 0, minutes: 0, trades: 0, withBox: 0, edgesEqual: 0, verdictEqual: 0, entryAtEdge: 0, heightsDiffer: 0, verdictOff: [] }
const diffs = []
for (const f of files) {
  const ses = JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8'))
  const arg = { payload: ses.payload, inputs: ses.inputs, tradeDate: ses.trade_date }
  const ctx = prepareSession(arg)
  const ctxT = tag.prepareSession(arg)
  if (!ctx) continue
  const sim = simulateSession(ctx)
  hash(hNew, sim)
  hash(hTag, tag.simulateSession(ctxT))
  c.sessions += 1
  c.minutes += sim.minutes.length
  const bars = flattenBars(ctx.pay)
  for (const tr of [...sim.trades, ...(sim.open ? [sim.open] : [])]) {
    c.trades += 1
    if (tr.decisionT != null && tr.box?.from != null && tr.box.top != null && tr.box.bottom != null) c.withBox += 1
    const r = edgeReadout(ctx.pay, bars, ctx.shift(tr.box.from), ctx.shift, ctx.shift(tr.decisionT), V1_DEFAULTS.breakHoldMin,
      { counts: false, qualifyMin: V1_DEFAULTS.boxMin, qualifyPts: V1_DEFAULTS.minHeightPts, manual: true })
    if (r.from === tr.box.from && r.top.price === tr.box.top && r.bottom.price === tr.box.bottom && r.to === tr.decisionT && !r.broke) c.edgesEqual += 1
    const pe = panelEngine(ctx, { endShifted: ctx.shift(tr.decisionT) })
    if (pe.ev.verdict === tr.side.toUpperCase() && pe.ev.box?.from === tr.box.from) c.verdictEqual += 1
    if (tr.entry === (tr.side === 'short' ? tr.box.top : tr.box.bottom)) c.entryAtEdge += 1
    const wick = tr.box.high - tr.box.low
    const close = tr.box.top - tr.box.bottom
    diffs.push(wick - close)
    if (wick > close + 1e-9) c.heightsDiffer += 1
  }
}
diffs.sort((a, b) => a - b)
delete c.verdictOff
console.log(JSON.stringify({ engine: V1_ENGINE_VERSION, tag: tag.V1_ENGINE_VERSION, ...c, wickMinusCloseHeight: { p50: diffs[diffs.length >> 1], p90: diffs[Math.floor(diffs.length * 0.9)], max: diffs[diffs.length - 1] },
  hashNew: hNew.digest('hex').slice(0, 16), hashTag: hTag.digest('hex').slice(0, 16) }, null, 1))
