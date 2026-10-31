// CR-DS items 1–3 — verdict pill / reason, the evidence table (boxEvidenceRows) and the "would be a read" wording.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { boxEvidenceRows, boxVerdict } from '../../src/flowV3.js'

const hhmm = (t) => `t${t}`
const box = (over = {}) => ({
  top: 110, bottom: 100, to: 50, ended: { kind: 'current' }, lastClose: 109.5,
  readout: { delta: 300, vol: 10000, dPrice: 1, minutes: 20 }, ...over,
})
const pct = (share, cpp) => ({ share: { p: share, value: 0.03 }, cpp: { p: cpp, value: 5000 } })
const lvl = (kind, price, start = 10) => ({ kind, price, start, end: 50, broken: false })
const v = (args) => boxVerdict({ hhmm, tradeDate: '2026-10-30', ...args })
const tones = (r) => Object.fromEntries(boxEvidenceRows(r.checks).map((x) => [x.key, x.tone]))

test('item 3: flow qualifies, price not at the top → NO ACTION "short read, but price not at the top" (category unchanged)', () => {
  // 2026-10-30 07:39–08:02 shape: share and c/pt notable, buyers absorbed at the top, price 1.25 below the bottom
  const r = v({ box: box({ lastClose: 98.75, high: 110.5 }), pct: pct(93, 93), levels: [lvl('buyers_absorbed', 110.25)] })
  assert.equal(r.kind, 'none')
  assert.equal(r.label, 'NO ACTION — short read, but price not at the top')
  assert.equal(r.pill, 'NO ACTION')
  assert.equal(r.reason, 'short read, but price not at the top')
  assert.deepEqual(tones(r), { event: 'good', formed: 'good', location: 'bad', share: 'good', effort: 'good', progress: 'good', absorption: 'good', edges: 'neutral' })
  assert.equal(boxEvidenceRows(r.checks).find((x) => x.key === 'location').value, '1.25 below the bottom')
  // mirror
  const m = v({ box: box({ lastClose: 110, readout: { delta: -300, vol: 10000, dPrice: -1, minutes: 20 } }), pct: pct(93, 93), levels: [] })
  assert.equal(m.label, 'NO ACTION — long read, but price not at the bottom')
})

test('item 3: "nothing notable" only when no flow check passes', () => {
  const r = v({ box: box({ lastClose: 105 }), pct: pct(50, 50), levels: [] })
  assert.equal(r.label, 'NO ACTION — nothing notable')
  assert.equal(r.checks.candidate, null)
  for (const x of boxEvidenceRows(r.checks)) assert.equal(x.tone, 'neutral', x.key)
})

test('item 3: box still forming / event day name the read they block', () => {
  const young = box({ readout: { delta: 300, vol: 10000, dPrice: 1, minutes: 9 } })
  const f = v({ box: young, pct: pct(90, 40), levels: [] })
  assert.equal(f.kind, 'forming')
  assert.equal(f.label, 'NO ACTION — short read, but box still forming')
  assert.equal(tones(f).formed, 'bad')
  const e = v({ box: box(), pct: pct(90, 40), levels: [], tradeDate: '2026-10-28' })
  assert.equal(e.kind, 'no-trades')
  assert.equal(e.label, 'NO TRADES — event day')
  assert.equal(e.pill, 'NO TRADES')
  assert.equal(e.reason, 'short read, but event day (FOMC)')
  assert.equal(tones(e).event, 'bad')
  // a flow lean opposed by absorption is a conflict, not a blocked read
  const c = v({ box: young, pct: pct(90, 40), levels: [lvl('sellers_absorbed', 105)] })
  assert.equal(c.label, 'NO ACTION — box still forming')
})

test('item 2: SHORT READ — all side rows green; reason under the pill', () => {
  const r = v({ box: box(), pct: pct(90, 85), levels: [lvl('buyers_absorbed', 110)] })
  assert.equal(r.kind, 'short')
  assert.equal(r.pill, 'SHORT READ at top')
  assert.equal(r.reason, 'buyers leaning + buyers absorbed at the top, effort')
  assert.deepEqual(tones(r), { event: 'good', formed: 'good', location: 'good', share: 'good', effort: 'good', progress: 'good', absorption: 'good', edges: 'neutral' })
})

test('item 2: CONFLICT — green and red rows together', () => {
  const r = v({ box: box({ lastClose: 104 }), pct: pct(85, 50), levels: [lvl('sellers_absorbed', 100.5)] })
  assert.equal(r.kind, 'conflict')
  assert.equal(r.checks.candidate, 'short')
  const t = tones(r)
  assert.equal(t.share, 'good')
  assert.equal(t.absorption, 'bad')
  assert.equal(t.location, 'bad')
  assert.equal(t.effort, 'neutral')
})

test('item 2: candidate from absorption when no lean; two-sided → one red row; progress > 2 red', () => {
  const a = v({ box: box({ readout: { delta: 10, vol: 10000, dPrice: 5, minutes: 20 } }), pct: pct(20, 85), levels: [lvl('buyers_absorbed', 111)] })
  assert.equal(a.checks.candidate, 'short')
  assert.equal(tones(a).progress, 'neutral') // +5 toward the buyers, but effort p85 already meets no advance
  assert.equal(tones(a).share, 'neutral')
  const two = v({ box: box(), pct: pct(50, 50), levels: [lvl('buyers_absorbed', 110), lvl('sellers_absorbed', 100)] })
  assert.equal(two.checks.candidate, null)
  const rows = boxEvidenceRows(two.checks)
  const abs = rows.find((x) => x.key === 'absorption')
  assert.equal(abs.tone, 'bad')
  assert.equal(abs.reads, 'two-sided absorption')
  assert.equal(rows.filter((x) => x.key === 'absorption').length, 1)
})

test('item 2: edge counts row from the box edge objects, informational', () => {
  const b = box({ top: { price: 110, contracts: 1800, opposite: 900 }, bottom: { price: 100, contracts: 4100, opposite: 2000 } })
  const r = v({ box: b, pct: pct(90, 40), levels: [] })
  const e = boxEvidenceRows(r.checks).find((x) => x.key === 'edges')
  assert.equal(e.value, 'Top buyers 1.8k / sellers 900 · Bottom sellers 4.1k / buyers 2.0k')
  assert.equal(e.tone, 'neutral')
  assert.equal(boxEvidenceRows(null).length, 0)
})

test('no advance = effort OR progress ≤ 2 pts: when one leg qualifies the other is slate "not needed", never red (both ways)', () => {
  const row = (r, k) => boxEvidenceRows(r.checks).find((x) => x.key === k)
  // effort qualifies, progress +3.5 toward the buyers (the 10-30 07:38 → 07:49 shape): SHORT READ, progress slate
  const e = v({ box: box({ readout: { delta: 10, vol: 10000, dPrice: 3.5, minutes: 20 } }), pct: pct(20, 82), levels: [lvl('buyers_absorbed', 110)] })
  assert.equal(e.kind, 'short')
  assert.deepEqual([row(e, 'effort').tone, row(e, 'progress').tone, row(e, 'progress').reads], ['good', 'neutral', 'not needed (effort)'])
  // progress qualifies (+1), effort p40: effort slate "not needed"
  const p = v({ box: box(), pct: pct(90, 40), levels: [] })
  assert.equal(p.kind, 'short')
  assert.deepEqual([row(p, 'progress').tone, row(p, 'effort').tone, row(p, 'effort').reads], ['good', 'neutral', 'not needed (no advance)'])
  // long mirror: effort qualifies, progress −3.5 toward the sellers
  const lm = v({ box: box({ lastClose: 101, readout: { delta: -300, vol: 10000, dPrice: -3.5, minutes: 20 } }), pct: pct(90, 85), levels: [] })
  assert.equal(lm.kind, 'long')
  assert.deepEqual([row(lm, 'progress').tone, row(lm, 'progress').reads], ['neutral', 'not needed (effort)'])
  // neither leg: progress red, effort slate "not notable"
  const n = v({ box: box({ readout: { delta: 300, vol: 10000, dPrice: 5, minutes: 20 } }), pct: pct(90, 40), levels: [] })
  assert.deepEqual([row(n, 'progress').tone, row(n, 'progress').reads, row(n, 'effort').reads], ['bad', 'buyers advancing', 'not notable'])
})
