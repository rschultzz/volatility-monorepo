// CR-DQ item 7 — the verdict line (Ryan's draft rules, applied mechanically; display aid, untested).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { boxLevels, boxVerdict } from '../../src/flowV3.js'

const hhmm = (t) => `t${t}`
// box 100–110, current to t=50; readout share = delta / vol
const box = (over = {}) => ({
  top: 110, bottom: 100, to: 50, ended: { kind: 'current' }, lastClose: 109.5,
  readout: { delta: 300, vol: 10000, dPrice: 1, minutes: 20 }, ...over,
})
const pct = (share, cpp) => ({ share: { p: share }, cpp: { p: cpp } })
const lvl = (kind, price, start = 10, extra = {}) => ({ kind, price, start, end: 50, broken: false, ...extra })
const v = (args) => boxVerdict({ hhmm, tradeDate: '2026-10-30', ...args })

test('A22: FOMC decision day → NO TRADES, before anything else', () => {
  const r = v({ box: box(), pct: pct(99, 99), levels: [], tradeDate: '2026-10-28' })
  assert.equal(r.kind, 'no-trades')
  assert.equal(r.label, 'NO TRADES — event day')
  assert.deepEqual(r.rules, ['A22'])
  assert.match(r.title, /Draft rules, untested/)
})

test('A24: notable buying vs a sellers-absorbed level at the bottom → CONFLICT, listing both', () => {
  const r = v({ box: box(), pct: pct(85, 50), levels: [lvl('sellers_absorbed', 100.5)] })
  assert.equal(r.kind, 'conflict')
  assert.equal(r.evidence, 'share +3.00 % p85 (buyers leaning) vs sellers absorbed 100.50 (t10) at the bottom')
  assert.deepEqual(r.rules, ['A24'])
  // mirror
  const m = v({ box: box({ readout: { delta: -300, vol: 10000, dPrice: -1, minutes: 20 } }), pct: pct(85, 50), levels: [lvl('buyers_absorbed', 109)] })
  assert.equal(m.kind, 'conflict')
})

test('two-sided absorption, no notable lean → NO ACTION two-sided (not A24), naming both levels', () => {
  const r = v({ box: box(), pct: pct(50, 50), levels: [lvl('buyers_absorbed', 110), lvl('sellers_absorbed', 100)] })
  assert.equal(r.kind, 'two-sided')
  assert.equal(r.label, 'NO ACTION — two-sided absorption')
  assert.match(r.evidence, /buyers absorbed 110\.00 \(t10\) at the top · sellers absorbed 100\.00 \(t10\) at the bottom; no notable flow lean/)
  assert.deepEqual(r.rules, [])
})

test('A19 + A23: SHORT READ at top — buyers leaning, no advance; absorption alone also counts', () => {
  const r = v({ box: box(), pct: pct(90, 40), levels: [] })
  assert.equal(r.kind, 'short')
  assert.equal(r.evidence, 'price 0.50 below the top · share +3.00 % p90 buyers leaning · no advance +1.0 pts')
  assert.deepEqual(r.rules, ['A19'])
  const a = v({ box: box({ readout: { delta: 10, vol: 10000, dPrice: 5, minutes: 20 } }), pct: pct(20, 85), levels: [lvl('buyers_absorbed', 111)] })
  assert.equal(a.kind, 'short')
  assert.deepEqual(a.rules, ['A19', 'A23'])
  assert.match(a.evidence, /buyers absorbed 111\.00 \(t10\) at the top · c\/pt p85 \(effort\)$/)
  // advanced and easy: not a short read
  assert.notEqual(v({ box: box({ readout: { delta: 300, vol: 10000, dPrice: 5, minutes: 20 } }), pct: pct(90, 40), levels: [] }).kind, 'short')
})

test('LONG READ at bottom (mirror)', () => {
  const r = v({ box: box({ lastClose: 101, readout: { delta: -300, vol: 10000, dPrice: -1, minutes: 20 } }), pct: pct(90, 40), levels: [] })
  assert.equal(r.kind, 'long')
  assert.equal(r.label, 'LONG READ at bottom')
  assert.equal(r.evidence, 'price 1.00 above the bottom · share -3.00 % p90 sellers leaning · no decline -1.0 pts')
})

test('WATCH BREAK (idea): pushing side winning at an edge, moving easily, nothing absorbing', () => {
  const r = v({ box: box({ readout: { delta: 300, vol: 10000, dPrice: 6, minutes: 20 } }), pct: pct(90, 40), levels: [] })
  assert.equal(r.kind, 'watch-up')
  assert.equal(r.label, 'WATCH BREAK ↑ (idea)')
  assert.match(r.evidence, /wait for an accepted break$/)
  assert.deepEqual(r.rules, ['idea'])
})

test('otherwise NO ACTION — nothing notable; levels only count when standing at the box\'s last bar', () => {
  const r = v({ box: box({ lastClose: 105 }), pct: pct(50, 50), levels: [] })
  assert.equal(r.kind, 'none')
  assert.equal(r.evidence, 'price mid-box · share +3.00 % p50 · c/pt p50')
  const later = lvl('sellers_absorbed', 100, 60) // starts after the box's last bar
  const broken = lvl('sellers_absorbed', 100, 10, { broken: true, end: 40 })
  const far = lvl('sellers_absorbed', 90)
  assert.deepEqual(boxLevels([later, broken, far], box(), 50, 2), [])
  assert.equal(boxLevels([lvl('sellers_absorbed', 105)], box(), 50, 2)[0].where, 'in')
  // no percentiles (ETH / mixed): no lean; a stall alone is not a read
  assert.equal(v({ box: box(), pct: null, levels: [] }).kind, 'none')
})

test('thresholds come from the one rules object', () => {
  const r = v({ box: box(), pct: pct(85, 40), levels: [], rules: { notable: 90 } })
  assert.equal(r.kind, 'none')
  assert.match(r.title, /notable ≥ p90/)
  assert.match(v({ box: box(), pct: pct(85, 40), levels: [], rules: { nearPts: 3 } }).title, /near an edge ≤ 3 pts/)
})

test('the panel\'s box shape: edges are objects ({ price, … }) — same verdict as with numbers', () => {
  const obj = (o) => ({ ...box(o), top: { price: 110, contracts: 1 }, bottom: { price: 100, contracts: 1 } })
  assert.equal(v({ box: obj(), pct: pct(90, 40), levels: [] }).kind, 'short')
  assert.equal(v({ box: obj({ lastClose: 105 }), pct: pct(50, 50), levels: [] }).evidence, 'price mid-box · share +3.00 % p50 · c/pt p50')
  assert.equal(v({ box: obj(), pct: pct(85, 50), levels: [lvl('sellers_absorbed', 100.5)] }).kind, 'conflict')
})

// CR-DQ amendment (Ryan, chat 2026-10-31)
test('amendment 1: absorption distance is to the box\'s high–low range (wicks); edges stay close-based', () => {
  // 2026-10-30 07:49 shape: buyers absorbed 0.5 above the box high, 3.75 above the close-based top
  const b = box({ high: 110.5, low: 99 })
  const lvTop = lvl('buyers_absorbed', 111)
  assert.equal(boxLevels([lvTop], b, 50, 2)[0].where, 'top') // 0.5 from the high
  assert.deepEqual(boxLevels([lvl('buyers_absorbed', 113)], b, 50, 2), []) // 2.5 from the high
  assert.equal(boxLevels([lvl('sellers_absorbed', 97.5)], b, 50, 2)[0].where, 'bottom') // 1.5 below the low
  // without high / low the close-based edges are used
  assert.equal(boxLevels([lvl('sellers_absorbed', 98.5)], box(), 50, 2)[0].where, 'bottom')
  // a level 1.5 above the high (3.5 above the top): counted only against the high — it blocks a watch break ↑ (moving
  // easily, no stall, so no read either) and, with a stall, makes the short read A19 + A23
  const moving = { readout: { delta: 300, vol: 10000, dPrice: 6, minutes: 20 } }
  assert.equal(v({ box: box(moving), pct: pct(90, 40), levels: [lvl('buyers_absorbed', 113.5)] }).kind, 'watch-up')
  assert.equal(v({ box: box({ ...moving, high: 112 }), pct: pct(90, 40), levels: [lvl('buyers_absorbed', 113.5)] }).kind, 'none')
  const r = v({ box: box({ high: 112 }), pct: pct(90, 40), levels: [lvl('buyers_absorbed', 113.5)] })
  assert.equal(r.kind, 'short')
  assert.deepEqual(r.rules, ['A19', 'A23'])
})

test('amendment 2: box still forming until ≥ 10 min and ≥ 2 pts (close-based), checked right after NO TRADES', () => {
  const young = box({ readout: { delta: 300, vol: 10000, dPrice: 1, minutes: 9 } })
  const r = v({ box: young, pct: pct(90, 40), levels: [lvl('sellers_absorbed', 100.5)] })
  assert.equal(r.kind, 'forming')
  assert.equal(r.label, 'NO ACTION — box still forming')
  assert.equal(r.evidence, 'box 9 min (min 10) · 10.00 pts tall (min 2)')
  const flat = box({ top: 101.5, bottom: 100, lastClose: 101 })
  assert.equal(v({ box: flat, pct: pct(90, 40), levels: [] }).kind, 'forming')
  assert.equal(v({ box: box(), pct: pct(90, 40), levels: [] }).kind, 'short') // 20 min, 10 pts
  assert.equal(v({ box: young, pct: pct(90, 40), levels: [], tradeDate: '2026-10-28' }).kind, 'no-trades')
  assert.equal(v({ box: young, pct: pct(90, 40), levels: [], rules: { minAgeMin: 5 } }).kind, 'short')
  assert.match(v({ box: young, pct: null, levels: [] }).title, /box formed at ≥ 10 min and ≥ 2 pts/)
})

test('amendment 3: wording says which side of the edge', () => {
  const above = v({ box: box({ lastClose: 111.5 }), pct: pct(90, 40), levels: [] })
  assert.match(above.evidence, /^price 1\.50 above the top/)
  const at = v({ box: box({ lastClose: 110 }), pct: pct(90, 40), levels: [] })
  assert.match(at.evidence, /^price at the top/)
  const below = v({ box: box({ lastClose: 99.5, readout: { delta: -300, vol: 10000, dPrice: -1, minutes: 20 } }), pct: pct(90, 40), levels: [] })
  assert.match(below.evidence, /^price 0\.50 below the bottom/)
})
