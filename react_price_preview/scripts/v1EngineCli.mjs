// CR-DU — run the v1 engine (src/v1Engine.js) on session files for scripts/cr_du_backtest_v1.py. Usage:
//   node react_price_preview/scripts/v1EngineCli.mjs <job.json> <out.json>
// job: { variants: { name: settings }, sessions: [ "<session json path>" ... ], minutes: bool, at: optional
//        [{ id, file, t (UTC bar start) }] }
// session json: { trade_date, payload (cells payload), inputs (v1-engine inputs) }.
// out: { version, sessions: { date: { variant: { trades, minutes? } } }, at: [{ id, verdict, side, failed, evidence }] }
import fs from 'node:fs'
import { V1_ENGINE_VERSION, panelEngine, prepareSession, simulateSession } from '../src/v1Engine.js'

const [jobPath, outPath] = process.argv.slice(2)
const job = JSON.parse(fs.readFileSync(jobPath, 'utf8'))
const out = { version: V1_ENGINE_VERSION, sessions: {}, at: [] }
const slim = (m) => ({ t: m.t, verdict: m.verdict, side: m.side, failed: m.failed.map((f) => f.code), box: m.box?.from ?? null })
for (const file of job.sessions || []) {
  const ses = JSON.parse(fs.readFileSync(file, 'utf8'))
  const res = {}
  for (const [name, settings] of Object.entries(job.variants || { primary: {} })) {
    const ctx = prepareSession({ payload: ses.payload, inputs: ses.inputs, tradeDate: ses.trade_date, settings })
    if (!ctx) { res[name] = { trades: [], minutes: [], error: 'no RTH bars' }; continue }
    const sim = simulateSession(ctx, { keepMinutes: Boolean(job.minutes) })
    res[name] = { trades: sim.trades, minutes: job.minutes ? sim.minutes.map(slim) : undefined, open: sim.open }
  }
  out.sessions[ses.trade_date] = res
}
// item 5 / CR-CZ overlap: the verdict at given decision minutes (06:30 anchor; state = the engine's own trades that day
// through that minute, as the panel)
const cache = new Map()
for (const q of job.at || []) {
  if (!cache.has(q.file)) cache.set(q.file, JSON.parse(fs.readFileSync(q.file, 'utf8')))
  const ses = cache.get(q.file)
  const ctx = prepareSession({ payload: ses.payload, inputs: ses.inputs, tradeDate: ses.trade_date, settings: job.atSettings || {} })
  const i = ctx ? ctx.bars.t.indexOf(Number(q.t)) : -1
  if (i < 0) { out.at.push({ id: q.id, verdict: null, failed: [{ rule: null, code: 'no_bar', text: 'decision bar not in the session' }] }); continue }
  const ev = panelEngine(ctx, { endShifted: ctx.times[i] }).ev
  out.at.push({ id: q.id, verdict: ev.verdict, side: ev.side, failed: ev.failed,
    evidence: { ...ev.evidence, box: ev.box, order: ev.order } })
}
fs.writeFileSync(outPath, JSON.stringify(out))
