// CR-DU D1 — engine box and failed rules at each CR-CZ card's decision minute (see d1.py). Usage:
//   node scripts/cr_du_d1/d1.mjs <job.json> <out.json> <engine dir: a copy of src/ whose evaluateAt takes forceSide>
import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const [jobPath, outPath, engineDir] = process.argv.slice(2)
const E = await import(pathToFileURL(path.join(engineDir, 'v1Engine.js')).href)
const job = JSON.parse(fs.readFileSync(jobPath, 'utf8'))
const cache = new Map()
const out = []
const geom = (ev, card, i, ctx) => {
  const b = ev.box
  if (!b) return { formed: false, from: null, top: null, bottom: null, minutes: null, overlap_pts: 0, overlap_min: 0 }
  const formed = !ev.failed.some((f) => f.code === 'box_forming' || f.code === 'no_box')
  const ovp = Math.max(0, Math.min(b.top, card.top) - Math.max(b.bottom, card.bottom))
  const ovm = Math.max(0, Math.round((ctx.bars.t[i] - Math.max(b.from, card.r0)) / 60))
  return { formed, from: b.from, top: b.top, bottom: b.bottom, high: b.high, low: b.low, minutes: b.minutes, overlap_pts: ovp, overlap_min: ovm }
}
for (const c of job) {
  if (!cache.has(c.file)) cache.set(c.file, JSON.parse(fs.readFileSync(c.file, 'utf8')))
  const ses = cache.get(c.file)
  const ctx = E.prepareSession({ payload: ses.payload, inputs: ses.inputs, tradeDate: ses.trade_date })
  const i = ctx ? ctx.bars.t.indexOf(c.t) : -1
  const rec = { id: c.id, date: c.date, label: c.label, side: c.side, warm: c.warm, r0: c.r0, top: c.top, bottom: c.bottom,
    card_h: c.top - c.bottom, card_min: Math.round((c.t - c.r0) / 60) }
  if (i < 0) { out.push({ ...rec, b: { formed: false, overlap_pts: 0, overlap_min: 0 }, c: { failed: [{ code: 'no_bar' }] }, error: 'no bar' }); continue }
  const sim = E.simulateSession(ctx, { keepMinutes: false, lastI: i })
  const b = E.evaluateAt(ctx, i, sim.state, { forceSide: c.side })
  const anc = ctx.shift(c.r0)
  const cc = E.evaluateAt(ctx, i, sim.state, { forceSide: c.side, anchorShifted: anc })
  const slim = (ev) => ({ verdict: ev.verdict, side: ev.side, failed: ev.failed.map(({ rule, code, text }) => ({ rule, code, text })),
    read: { share: ev.evidence?.share ?? null, sharePct: ev.evidence?.sharePct ?? null, cppPct: ev.evidence?.cppPct ?? null, noExtend: ev.evidence?.noExtend ?? null,
      kind: ev.v?.kind ?? null, inBoxAbs: (ev.v?.checks?.levels || []).map((l) => `${l.side}@${l.where}`) } })
  out.push({ ...rec, b: { ...geom(b, c, i, ctx), ...slim(b) }, c: { ...geom(cc, c, i, ctx), ...slim(cc) } })
}
fs.writeFileSync(outPath, JSON.stringify(out))
