import test from 'node:test'
import assert from 'node:assert/strict'

import { V1_DEFAULTS, evaluateAt, newState, prepareSession, simulateSession } from '../../src/v1Engine.js'

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
    if (i < 20) { o = 6000; c = i % 2 ? 6000.5 : 6000 }
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

test('defaults: box min 8, cutoff 12:15, +10, 3 losses, 20-min re-entry', () => {
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

test('SHORT at the top: order at the exact edge, stop = wick high + 2, target -10', () => {
  const ev = evaluateAt(session(), 28)
  assert.equal(ev.verdict, 'SHORT')
  assert.deepEqual(ev.failed, [])
  assert.deepEqual([ev.order.entry, ev.order.stop, ev.order.target], [6015, 6018, 6005])
  assert.ok(ev.evidence.noExtend.up)
})

test('sensitivity rows: D4 entry 1 pt inside; fixed 10-pt stop', () => {
  const d4 = evaluateAt(session({ settings: { entry: 'd4' } }), 28)
  assert.deepEqual([d4.order.entry, d4.order.target], [6014, 6004])
  const f10 = evaluateAt(session({ settings: { stop: 'fixed10' } }), 28)
  assert.equal(f10.order.stop, 6025)
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
  const noSec = simulateSession(session({ over }))
  assert.equal(noSec.trades[0].reason, 'stop')
  assert.equal(noSec.trades[0].pts, -3)
  const m = at(32)
  const seconds = { t: [m + 1, m + 2], o: [6010, 6005], h: [6010, 6018.5], l: [6004.5, 6005], c: [6005, 6018] }
  const sec = simulateSession(session({ over, seconds }))
  assert.equal(sec.trades[0].reason, 'target')
})

test('flatten at the open of the 12:59 bar', () => {
  const sim = simulateSession(session({ over: { 29: { h: 6015.25 }, ...down(30, 6014), 389: { o: 6013, c: 6013, h: 6013.25, l: 6012.75 } } }))
  const tr = sim.trades[0]
  assert.deepEqual([tr.reason, tr.exit, tr.exitT], ['flatten', 6013, at(389)])
})

test('a stop counts a loss and starts the 20-min wait', () => {
  const sim = simulateSession(session({ over: { 29: { h: 6015.25 }, 30: { h: 6018.25, l: 6014, o: 6015, c: 6017 }, ...down(31, 6016) } }))
  assert.equal(sim.trades[0].reason, 'stop')
  assert.equal(sim.state.losses, 1)
  assert.equal(sim.state.lastStop.at, at(30))
})

test('the 12:15 cutoff cancels a resting order', () => {
  const sim = simulateSession(session({ settings: { cutoffMin: 7 * 60 }, over: { 29: { h: 6015 }, 30: { h: 6015.25 } } }))
  assert.equal(sim.trades.length, 0)
})

test('a verdict lapse cancels the order', () => {
  // bar 29 closes 2.5 below the top (not at the top) → NO ACTION → cancelled before bar 30 trades through
  const sim = simulateSession(session({ over: { 29: { o: 6013.5, c: 6012.5, h: 6013.75, l: 6012.5 }, 30: { h: 6015.25 } } }))
  assert.equal(sim.trades.length, 0)
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
