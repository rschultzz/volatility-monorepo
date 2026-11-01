import test from 'node:test'
import assert from 'node:assert/strict'

import { V1_DEFAULTS, evaluateAt, newState, prepareSession, simulateSession } from '../../src/v1Engine.js'
import { boxEvidenceRows as boxEvidenceRowsSync } from '../../src/flowV3.js'

// CR-DU item 1 — the v1 engine on a synthetic RTH session (2026-03-10, PDT: 06:30 PT = 13:30 UTC).
//   bars 0–19: flat at 6000 (box A; its edges hold 20 min)
//   bar 20:   jump to 6015 (a close > 2 pts above A held 5 bars → A freezes, box B starts at bar 20)
//   bar 21:   6012.5 (B's bottom grows: B = 6012.5 → 6015 by closes, wick high 6016)
//   bars 22+: closes 6013.5 / 6014.5 with net buying (share +20 %): a SHORT read at the top once B is 8 min old
const T0 = Date.UTC(2026, 2, 10, 13, 30) / 1000
const at = (i) => T0 + 60 * i
const N = 390

function bars(over = {}) {
  const b = { t: [], o: [], h: [], l: [], c: [], buy: [], sell: [], vol: [] }
  for (let i = 0; i < N; i += 1) {
    let o, c, buy = 500, sell = 500
    if (i < 20) { o = 6000; c = i % 2 ? 6002.5 : 6000 } // Amendment 5: a first box that qualifies, so bar 20 can break it
    else if (i === 20) { o = 6000; c = 6015 }
    else if (i === 21) { o = 6015; c = 6012.5 }
    else { o = i % 2 ? 6014.5 : 6013.5; c = i % 2 ? 6013.5 : 6014.5; buy = 600; sell = 400 }
    let h = Math.max(o, c) + 0.25
    let l = Math.min(o, c) - 0.25
    if (i === 20) h = 6016
    if (i > 21) h = Math.min(h, 6015)
    b.t.push(at(i)); b.o.push(o); b.h.push(h); b.l.push(l); b.c.push(c); b.buy.push(buy); b.sell.push(sell); b.vol.push(buy + sell)
  }
  for (const [i, v] of Object.entries(over)) for (const [k, x] of Object.entries(v)) b[k][Number(i)] = x
  return b
}

const lin = (lo, hi) => Array.from({ length: 101 }, (_, k) => lo + ((hi - lo) * k) / 100)
const BUCKETS = [['5', 3, 7], ['10', 8, 12], ['15', 13, 22], ['30', 23, 45], ['60', 46, 75], ['90+', 76, 240]]
function baseline() {
  const absShare = {}
  const cpp = {}
  for (const tod of ['open', 'mid', 'late']) {
    absShare[tod] = {}
    for (const [n] of BUCKETS) absShare[tod][n] = lin(0, 0.1) // |share| 0.1 → p100
  }
  for (const [n] of BUCKETS) cpp[n] = lin(0, 1e9) // never notable effort
  return {
    meta: { buckets: BUCKETS.map(([name, min, max]) => ({ name, min, max })), tod: [{ name: 'open', fromMin: 0, toMin: 60 },
      { name: 'mid', fromMin: 60, toMin: 270 }, { name: 'late', fromMin: 270, toMin: 390 }], tick: 0.25 },
    databento: { sessions: 30, first: '2026-01-02', last: '2026-03-09', absShare, cpp },
  }
}
const contra = { source: 'databento', buckets: Array.from({ length: 13 }, (_, k) => ({ from: 30 * k, d10: -500, d20: -300, d80: 300, d90: 500, v80: 1500 })) }
const mark = (kind, price, start, extra = {}) => ({ kind, price, time: at(start), confirm_time: at(start + 4), aggressor: 500, opposite: 50,
  merged_vol: 0, merges: [], median: 100, break_time: null, held_min: 0, ...extra })

function session({ over = {}, marks = [], gex = { reason: null, schedule: [{ from: at(0), levels: [{ price: 6100, net: 9e10, side: 'call' }] }] },
  swing = 12, tradeDate = '2026-03-10', seconds = null, settings = null } = {}) {
  const payload = { segments: [{ name: 'rth', source: 'databento', exact: true, start: T0, end: T0 + N * 60, bars: bars(over),
    cells: { t: [], p: [], buy: [], sell: [] }, marks }] }
  const inputs = { trade_date: tradeDate, baseline: baseline(), contra, swing, gex, seconds }
  return prepareSession({ payload, inputs, tradeDate, settings })
}
const codes = (ev) => ev.failed.map((f) => f.code)

test('defaults: box min 8, cutoff 12:15, +10, 3 losses, 20-min re-entry, lean p70, fixed 10-pt stop', () => {
  assert.equal(V1_DEFAULTS.leanPct, 70)
  assert.equal(V1_DEFAULTS.stop, 'fixed10')
  assert.equal(V1_DEFAULTS.boxMin, 8)
  assert.equal(V1_DEFAULTS.cutoffMin, 12 * 60 + 15)
  assert.equal(V1_DEFAULTS.target, 10)
  assert.equal(V1_DEFAULTS.maxLosses, 3)
  assert.equal(V1_DEFAULTS.reentryMin, 20)
})

test('rule 6: a box qualifies at 8 min although its edges are younger than the 15-min hold', () => {
  const ctx = session()
  const e27 = evaluateAt(ctx, 27) // box B 20 → 27 = 7 min
  assert.equal(e27.box.from, at(20))
  assert.deepEqual(codes(e27), ['box_forming'])
  const e28 = evaluateAt(ctx, 28) // 8 min
  assert.equal(e28.verdict, 'SHORT')
  assert.equal(e28.box.minutes, 8)
})

test('SHORT at the top: order at the exact edge, fixed 10-pt stop (Amendment 2), target -10; wick + 2 as a row', () => {
  const ev = evaluateAt(session(), 28)
  assert.equal(ev.verdict, 'SHORT')
  assert.deepEqual(ev.failed, [])
  assert.deepEqual([ev.order.entry, ev.order.stop, ev.order.target], [6015, 6025, 6005])
  assert.equal(evaluateAt(session({ settings: { stop: 'zone' } }), 28).order.stop, 6018)
  assert.ok(ev.evidence.noExtend.up)
})

test('sensitivity rows: D4 entry 1 pt inside; wick + 2 stop', () => {
  const d4 = evaluateAt(session({ settings: { entry: 'd4' } }), 28)
  assert.deepEqual([d4.order.entry, d4.order.target, d4.order.stop], [6014, 6004, 6024])
  const z = evaluateAt(session({ settings: { stop: 'zone' } }), 28)
  assert.equal(z.order.stop, 6018)
})

test('Amendment 3: the lean leg reads at p70 (p80 a sensitivity row); the effort leg stays p80', () => {
  const p75 = {}
  for (let i = 21; i < 40; i += 1) p75[i] = { buy: 537.5, sell: 462.5 } // share +7.5 % over the box = p75 on the test grid
  const ev = evaluateAt(session({ over: p75 }), 28)
  assert.ok(ev.evidence.sharePct >= 70 && ev.evidence.sharePct < 80)
  assert.equal(ev.verdict, 'SHORT')
  assert.deepEqual(codes(evaluateAt(session({ over: p75, settings: { leanPct: 80 } }), 28)), ['no_read'])
})

test('forceSide (D1): with no read the gates on that side are listed and the verdict stays NO ACTION', () => {
  const g = { reason: null, schedule: [{ from: at(0), levels: [{ price: 6010, net: -9e10, side: 'put' }] }] }
  const flat = {}
  for (let i = 22; i < 40; i += 1) flat[i] = { buy: 500, sell: 500 }
  const ev = evaluateAt(session({ over: flat, gex: g }), 28, newState(), { forceSide: 'short' })
  assert.equal(ev.verdict, 'NO ACTION')
  assert.equal(ev.side, null)
  assert.deepEqual(codes(ev), ['no_read', 'gex_in_path'])
})

test('rules 7 / 8: a new high in the last 3 bars (no effort) is no read', () => {
  const ev = evaluateAt(session({ over: { 27: { h: 6016.5 } } }), 28)
  assert.equal(ev.verdict, 'NO ACTION')
  assert.deepEqual(codes(ev), ['no_read'])
  assert.equal(ev.evidence.noExtend.up, false)
})

test('rule 1: FOMC decision day', () => {
  const ev = evaluateAt(session({ tradeDate: '2026-03-18' }), 28)
  assert.deepEqual(codes(ev), ['fomc'])
})

test('rule 3: no entries after the cutoff', () => {
  const ev = evaluateAt(session({ settings: { cutoffMin: 6 * 60 + 55 } }), 28)
  assert.deepEqual(codes(ev), ['cutoff'])
  const off = evaluateAt(session({ settings: { cutoffMin: 6 * 60 + 55, cutoff: false } }), 28)
  assert.equal(off.verdict, 'SHORT')
})

test('rule 6: no larger move into the box', () => {
  const ev = evaluateAt(session({ swing: 20 }), 28)
  assert.deepEqual(codes(ev), ['no_larger_move'])
})

test('rule 4: strong GEX level in the path; no GEX levels', () => {
  const g = { reason: null, schedule: [{ from: at(0), levels: [{ price: 6010, net: -9e10, side: 'put' }] }] }
  assert.deepEqual(codes(evaluateAt(session({ gex: g }), 28)), ['gex_in_path'])
  const late = { reason: null, schedule: [{ from: at(0), levels: [] }, { from: at(29), levels: [{ price: 6010, net: -9e10, side: 'put' }] }] }
  assert.equal(evaluateAt(session({ gex: late }), 28).verdict, 'SHORT') // a level drawn later isn't seen at 28
  assert.deepEqual(codes(evaluateAt(session({ gex: { reason: 'no_es_snapshot', schedule: [] } }), 28)), ['no_gex'])
})

test('rule 4: earlier opposing absorption in the path; rule 5: earlier same-side absorption is location only', () => {
  const inPath = evaluateAt(session({ marks: [mark('sellers_absorbed', 6010, 3)] }), 28)
  assert.deepEqual(codes(inPath), ['absorption_in_path'])
  const sup = evaluateAt(session({ marks: [mark('buyers_absorbed', 6016, 5)] }), 28)
  assert.equal(sup.verdict, 'SHORT')
  assert.equal(sup.evidence.locationSupport.length, 1)
  // without the flow lean, earlier buyers absorbed at the top is not a trigger (rule 5)
  const flat = {}
  for (let i = 22; i < 40; i += 1) flat[i] = { buy: 500, sell: 500 }
  const noLean = evaluateAt(session({ over: flat, marks: [mark('buyers_absorbed', 6016, 5)] }), 28)
  assert.equal(noLean.verdict, 'NO ACTION')
  // the same mark inside the box is a trigger (A23)
  const inBox = evaluateAt(session({ over: flat, marks: [mark('buyers_absorbed', 6015, 22)] }), 28)
  assert.equal(inBox.verdict, 'SHORT')
})

test('rule 9: conflict and two-sided absorption', () => {
  const conflict = evaluateAt(session({ marks: [mark('sellers_absorbed', 6012.5, 22)] }), 28)
  assert.deepEqual(codes(conflict), ['conflict'])
  const flat = {}
  for (let i = 22; i < 40; i += 1) flat[i] = { buy: 500, sell: 500 }
  const two = evaluateAt(session({ over: flat, marks: [mark('sellers_absorbed', 6012.5, 22), mark('buyers_absorbed', 6015, 22)] }), 28)
  assert.deepEqual(codes(two), ['two_sided'])
})

test('rules 19 / 20: loss cap, 20-min wait, new box after a stop, one trade per box', () => {
  const ctx = session()
  const st = newState()
  st.losses = 3
  assert.deepEqual(codes(evaluateAt(ctx, 28, st)), ['loss_cap'])
  assert.equal(evaluateAt(session({ settings: { lossCap: false } }), 28, st).verdict, 'SHORT')
  const w = newState()
  w.lastStop = { at: at(25), side: 'short', entryAt: at(15) } // the stopped trade was in box A: B is a new box
  assert.deepEqual(codes(evaluateAt(ctx, 28, w)), ['reentry_wait'])
  const nb = newState()
  nb.lastStop = { at: at(0), side: 'short', entryAt: at(21) }
  assert.deepEqual(codes(evaluateAt(ctx, 28, nb)), ['reentry_box']) // waited 29 min, but B started before that entry
  const ls = newState()
  ls.lastStop = { at: at(0), side: 'long', entryAt: at(21) }
  assert.equal(evaluateAt(ctx, 28, ls).verdict, 'SHORT') // the other side needs only the wait
  const one = newState()
  one.boxesTraded = [at(20)]
  assert.deepEqual(codes(evaluateAt(ctx, 28, one)), ['one_per_box'])
})

test('item 4: a bearish contradiction bar at the top is evidence, not a gate', () => {
  const ev = evaluateAt(session({ over: { 23: { o: 6013.75, c: 6013.75, h: 6015, l: 6012.5, buy: 1600, sell: 400, vol: 2000 } } }), 28)
  assert.equal(ev.verdict, 'SHORT')
  assert.deepEqual(ev.evidence.contra.primary, [at(23)])
  assert.ok(ev.evidence.contra.secondary.includes(at(23)))
})

test('rule 12 is evidence only', () => {
  const ev = evaluateAt(session(), 28)
  assert.ok('seen' in ev.evidence.rule12)
  assert.equal(ev.verdict, 'SHORT')
})

// ------------------------------------------------------------------------------------------- simulation (Q4)
const down = (from, px) => Object.fromEntries(Array.from({ length: N - from }, (_, k) => [from + k, { o: px, c: px, h: px + 0.25, l: px - 0.25 }]))

test('fill only on a trade-through, then +10 target', () => {
  const touch = simulateSession(session({ over: { 29: { h: 6015 }, ...down(30, 6008), 31: { h: 6008.25, l: 6004.75, o: 6008, c: 6005 } } }))
  assert.equal(touch.trades.length, 0) // touched 6015, never traded through
  const sim = simulateSession(session({ over: { 29: { h: 6015.25 }, ...down(30, 6008), 31: { h: 6008.25, l: 6004.75, o: 6008, c: 6005 } } }))
  assert.equal(sim.trades.length, 1)
  const tr = sim.trades[0]
  assert.deepEqual([tr.side, tr.entry, tr.exit, tr.reason, tr.pts, tr.net], ['short', 6015, 6005, 'target', 10, 9.75])
  assert.equal(tr.decisionT, at(28))
  assert.equal(tr.exitT, at(31))
})

test('same-minute stop and target: from 1-s data, else the stop', () => {
  const over = { 29: { h: 6015.25 }, ...down(30, 6010), 32: { o: 6010, c: 6010, h: 6018.5, l: 6004.5 } }
  const noSec = simulateSession(session({ over, settings: { stop: 'zone' } }))
  assert.equal(noSec.trades[0].reason, 'stop')
  assert.equal(noSec.trades[0].pts, -3)
  const m = at(32)
  const seconds = { t: [m + 1, m + 2], o: [6010, 6005], h: [6010, 6018.5], l: [6004.5, 6005], c: [6005, 6018] }
  const sec = simulateSession(session({ over, seconds, settings: { stop: 'zone' } }))
  assert.equal(sec.trades[0].reason, 'target')
})

test('flatten at the open of the 12:59 bar', () => {
  const sim = simulateSession(session({ over: { 29: { h: 6015.25 }, ...down(30, 6014), 389: { o: 6013, c: 6013, h: 6013.25, l: 6012.75 } } }))
  const tr = sim.trades[0]
  assert.deepEqual([tr.reason, tr.exit, tr.exitT], ['flatten', 6013, at(389)])
})

test('a stop counts a loss and starts the 20-min wait', () => {
  const sim = simulateSession(session({ settings: { stop: 'zone' }, over: { 29: { h: 6015.25 }, 30: { h: 6018.25, l: 6014, o: 6015, c: 6017 }, ...down(31, 6016) } }))
  assert.equal(sim.trades[0].reason, 'stop')
  assert.equal(sim.state.losses, 1)
  assert.equal(sim.state.lastStop.at, at(30))
})

test('the 12:15 cutoff cancels a resting order', () => {
  const sim = simulateSession(session({ settings: { cutoffMin: 7 * 60 }, over: { 29: { h: 6015 }, 30: { h: 6015.25 } } }))
  assert.equal(sim.trades.length, 0)
})

test('a read lapse cancels the order', () => {
  // bar 29 sells 6,000 against the box's buying: the lean flips → the short read lapses → cancelled before bar 30 trades through
  const sim = simulateSession(session({ over: { 29: { buy: 0, sell: 6000, vol: 6000 }, 30: { h: 6015.25 } } }))
  assert.ok(!sim.trades.some((x) => x.side === 'short' && x.entryT >= at(30) && x.entryT < at(31)))
})

test('Amendment 4 (v1.2): a valid read away from the edge rests a limit at the edge; it fills only there', async () => {
  const { panelEngine, panelVerdict, V1_ENGINE_VERSION } = await import('../../src/v1Engine.js')
  assert.equal(V1_ENGINE_VERSION, 'v1.3')
  const away = { 28: { o: 6013.5, c: 6012.75, h: 6013.75, l: 6012.5 } } // closes 2.25 below the top
  const ctx = session({ over: away })
  const ev = evaluateAt(ctx, 28)
  assert.equal(ev.verdict, 'SHORT')
  assert.equal(ev.order.entry, 6015)
  const pv = panelVerdict(panelEngine(ctx, { endShifted: ctx.times[28] }))
  assert.equal(pv.pill, 'READ — limit at 6015.00')
  assert.deepEqual(boxRow(pv, 'location'), ['limit at the top', 'neutral'])
  // price climbs back and trades through the edge at bar 31: filled at 6015
  const sim = simulateSession(session({ over: { ...away, 29: { h: 6014.25 }, 30: { h: 6015 }, 31: { h: 6015.25 }, ...down(32, 6008), 33: { h: 6008.25, l: 6004.75, o: 6008, c: 6005 } } }))
  assert.deepEqual([sim.trades[0].entry, sim.trades[0].entryT, sim.trades[0].reason], [6015, at(31), 'target'])
})

test('no lookahead: the verdict at bar i is the same with later bars removed', () => {
  const full = session({ marks: [mark('buyers_absorbed', 6016, 5), mark('sellers_absorbed', 6012.5, 40)] })
  const simFull = simulateSession(full)
  for (const cut of [25, 28, 35, 50]) {
    const b = bars()
    const short = {}
    for (const k of Object.keys(b)) short[k] = b[k].slice(0, cut + 1)
    const payload = { segments: [{ name: 'rth', source: 'databento', exact: true, start: T0, end: T0 + N * 60, bars: short,
      cells: { t: [], p: [], buy: [], sell: [] }, marks: [mark('buyers_absorbed', 6016, 5), mark('sellers_absorbed', 6012.5, 40)] }] }
    const ctx = prepareSession({ payload, inputs: { trade_date: '2026-03-10', baseline: baseline(), contra, swing: 12,
      gex: { reason: null, schedule: [{ from: at(0), levels: [{ price: 6100, net: 9e10, side: 'call' }] }] } }, tradeDate: '2026-03-10' })
    const sim = simulateSession(ctx)
    assert.deepEqual(sim.minutes.map((m) => [m.verdict, m.failed.map((f) => f.code)]),
      simFull.minutes.slice(0, cut + 1).map((m) => [m.verdict, m.failed.map((f) => f.code)]), `cut ${cut}`)
  }
})

// ------------------------------------------------------------------------------------------- item 2 (panel)
const boxRow = (pv, key) => {
  const r = boxEvidenceRowsSync(pv.checks).find((x) => x.key === key)
  return [r.reads, r.tone]
}
test('panelVerdict: SHORT pill, failed rules as reason + rows, engine trades line', async () => {
  const { panelEngine, panelVerdict } = await import('../../src/v1Engine.js')
  const { boxEvidenceRows } = await import('../../src/flowV3.js')
  const ok = panelVerdict(panelEngine(session(), { endShifted: session().times[28] }))
  assert.equal(ok.kind, 'short')
  assert.equal(ok.pill, 'SHORT at top')
  assert.deepEqual(ok.extraRows, [])
  assert.equal(ok.engineLine, "Engine's trades, not yours: 0 trades")
  assert.equal(ok.engineTag, 'engine v1.3 — untested')
  assert.equal(boxEvidenceRows(ok.checks).find((r) => r.key === 'share').reads, 'net buying') // was "buyers leaning"
  const g = { reason: null, schedule: [{ from: at(0), levels: [{ price: 6010, net: -9e10, side: 'put' }] }] }
  const ctx = session({ gex: g })
  const no = panelVerdict(panelEngine(ctx, { endShifted: ctx.times[28] }))
  assert.equal(no.pill, 'NO ACTION')
  assert.match(no.reason, /^rule 4: strong GEX level 6010\.00 \(put\) in the path$/)
  assert.deepEqual(no.extraRows.map((r) => [r.check, r.tone]), [['Rule 4', 'bad']])
  const forming = panelVerdict(panelEngine(ctx, { endShifted: ctx.times[26] }))
  assert.equal(forming.kind, 'forming')
  assert.match(forming.reason, /^rule 6: box still forming/)
  assert.equal(panelVerdict(null).reason, 'rule 2: RTH only')
})

test('panelEngine counts the engine\'s own trades and the re-entry wait', async () => {
  const { panelEngine } = await import('../../src/v1Engine.js')
  const ctx = session({ settings: { stop: 'zone' }, over: { 29: { h: 6015.25 }, 30: { h: 6018.25, l: 6014, o: 6015, c: 6017 }, ...down(31, 6016) } })
  const pe = panelEngine(ctx, { endShifted: ctx.times[35] })
  assert.equal(pe.trades.length, 1)
  assert.equal(pe.losses, 1)
  assert.equal(pe.wait, 14) // stopped in bar 30; at the close of bar 35: 15 of 20 min gone (rounded up)
})

test('box minimum default 8; a stored 10 (old default) migrates once', async () => {
  const { BOX_RULES_DEFAULTS, migrateBoxRules } = await import('../../src/flowV3.js')
  assert.equal(BOX_RULES_DEFAULTS.minAgeMin, 8)
  const m = migrateBoxRules({ k: 3, boxRules: { minAgeMin: 10, nearPts: 2 } })
  assert.equal(m.boxRules.minAgeMin, 8)
  assert.equal(m.boxRulesV, 2)
  assert.equal(migrateBoxRules({ boxRulesV: 2, boxRules: { minAgeMin: 10 } }).boxRules.minAgeMin, 10) // set after CR-DU: kept
  assert.equal(migrateBoxRules({ boxRules: { minAgeMin: 12 } }).boxRules.minAgeMin, 12)
})

// ------------------------------------------------------------------------------------------- item 4b (overlay)
test('overlayMarks: teal trade marks with stop / target lines while open and the points label', async () => {
  const { overlayMarks, minuteText } = await import('../../src/v1Engine.js')
  const ctx = session({ over: { 29: { h: 6015.25 }, ...down(30, 6008), 31: { h: 6008.25, l: 6004.75, o: 6008, c: 6005 } } })
  const sim = simulateSession(ctx)
  const m = overlayMarks(sim, ctx)
  assert.equal(m.length, 1)
  assert.equal(m[0].direction, 'short')
  assert.equal(m[0].label, '+10.00')
  assert.deepEqual(m[0].lines.map((l) => [l.kind, l.price]), [['stop', 6025], ['target', 6005]])
  assert.equal(m[0].lines[0].toT, ctx.shift(at(31)))
  assert.equal(minuteText(sim.minutes[28], () => '06:58'), 'v1 engine 06:58 · SHORT')
  assert.match(minuteText(sim.minutes[26], () => '06:56'), /^v1 engine 06:56 · NO ACTION · rule 6: box still forming/)
})

test('fixed 10-pt stop: a 3-pt pop that stops the wick + 2 bracket leaves this one open', () => {
  const over = { 29: { h: 6015.25 }, 30: { h: 6018.25, l: 6014, o: 6015, c: 6017 }, ...down(31, 6008), 33: { h: 6008.25, l: 6004.75, o: 6008, c: 6005 } }
  const f = simulateSession(session({ over }))
  assert.deepEqual([f.trades[0].stop, f.trades[0].reason, f.trades[0].pts], [6025, 'target', 10])
  assert.equal(simulateSession(session({ over, settings: { stop: 'zone' } })).trades[0].reason, 'stop')
})

test('CR-DV merge: provisional marks never reach the engine (panel, overlay, backtest) — confirmed marks only', () => {
  const flat = {}
  for (let i = 22; i < 40; i += 1) flat[i] = { buy: 500, sell: 500 }
  const confirmed = mark('buyers_absorbed', 6015, 22)
  const provisional = { ...mark('buyers_absorbed', 6015, 22), provisional: true } // even with confirm_time <= t
  const conflictProv = { ...mark('sellers_absorbed', 6012.5, 22), provisional: true }
  assert.equal(evaluateAt(session({ over: flat, marks: [confirmed] }), 28).verdict, 'SHORT')
  const none = evaluateAt(session({ over: flat, marks: [] }), 28)
  const prov = evaluateAt(session({ over: flat, marks: [provisional] }), 28)
  assert.equal(prov.verdict, 'NO ACTION')
  assert.deepEqual(prov, none)
  assert.ok(session({ marks: [confirmed, provisional, conflictProv] }).pay.segments[0].marks.every((mk) => !mk.provisional))
  // a provisional opposing mark beside a confirmed one changes nothing (no conflict), minute by minute and in the simulation
  const withProv = session({ over: flat, marks: [confirmed, conflictProv] })
  const only = session({ over: flat, marks: [confirmed] })
  for (let i = 0; i < 40; i += 1) assert.deepEqual(evaluateAt(withProv, i), evaluateAt(only, i), `minute ${i}`)
  assert.deepEqual(simulateSession(withProv).trades, simulateSession(only).trades)
})

test('Amendment 5: the engine and the panel read the same height; the engine box min sets qualification', async () => {
  const { edgeReadout } = await import('../../src/flowV3.js')
  const ctx = session()
  const ev = evaluateAt(ctx, 28)
  const r = edgeReadout(ctx.pay, ctx.bars, ctx.times[0], ctx.shift, ctx.s.holdMin, ctx.times[28], ctx.s.breakHoldMin, { counts: false, qualifyMin: ctx.s.boxMin, qualifyPts: ctx.s.minHeightPts })
  assert.equal(ev.v.checks.height, r.top.price - r.bottom.price)
  assert.equal(ev.box.top - ev.box.bottom, ev.v.checks.height)
  assert.equal(ev.v.checks.height, 2.5)
})
