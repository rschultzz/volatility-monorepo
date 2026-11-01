// CR-DV — provisional marks (chart only, never levels / box panel) and the feed-stall text in the Absorption row.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { absorptionLevels, boxEvidenceRows, boxVerdict, chartMarks, feedStalledSince } from '../../src/flowV3.js'

const T0 = 1_793_367_000 // 2026-10-30 13:30 UTC
const id = (t) => t
const times = Array.from({ length: 60 }, (_, i) => T0 + 60 * i)
const mk = (min, price, provisional, over = {}) => ({
  kind: 'sellers_absorbed', time: T0 + 60 * min, price, confirm_time: T0 + 60 * (min + 4), aggressor: 500, opposite: 20,
  merged_vol: 0, median: 50, break_time: null, held_min: 0, provisional, ...over,
})
const payload = (marks, extra = {}) => ({
  segments: [{ name: 'rth', exact: true, start: T0, end: T0 + 390 * 60, marks,
    bars: { t: times.slice(0, 30), h: times.slice(0, 30).map(() => 101), l: times.slice(0, 30).map(() => 99), c: times.slice(0, 30).map(() => 100) },
    ...extra }],
})

test('chartMarks keeps provisional marks for the chart (drawn outlined)', () => {
  const p = payload([mk(10, 100, false), mk(27, 99.5, true)])
  const out = chartMarks(p, times, id)
  assert.deepEqual(out.map((m) => [m.time, m.provisional]), [[T0 + 600, false], [T0 + 1620, true]])
})

test('absorptionLevels (and so the box panel) skip provisional marks', () => {
  // payload now = end of minute 29; the provisional mark's confirm_time (minute 28) is <= now, so the flag excludes it
  const p = payload([mk(10, 100, false), mk(24, 99.5, true)])
  const lv = absorptionLevels(p, times, id)
  assert.deepEqual(lv.map((l) => l.price), [100])
})

test('feedStalledSince reads the segment field; null when absent', () => {
  assert.equal(feedStalledSince(payload([], { feed_stalled_since: T0 + 300 })), T0 + 300)
  assert.equal(feedStalledSince(payload([])), null)
  assert.equal(feedStalledSince(null), null)
})

const hhmm = (t) => `t${t}`
const box = { top: 110, bottom: 100, to: 50, ended: { kind: 'current' }, lastClose: 105, readout: { delta: 0, vol: 10000, dPrice: 0, minutes: 20 } }
const pct = { share: { p: 50, value: 0 }, cpp: { p: 50, value: 5000 } }
const absRow = (levels, stall) => boxEvidenceRows(boxVerdict({ box, pct, levels, tradeDate: '2026-10-30', hhmm }).checks, { stall })
  .find((r) => r.key === 'absorption')

test('Absorption row: stall text instead of nothing; unchanged without a stall', () => {
  assert.deepEqual([absRow([], null).value, absRow([], null).reads], ['none at the box', 'none'])
  const s = absRow([], '07:52')
  assert.equal(s.value, 'flow feed stalled since 07:52 PT')
  assert.equal(s.reads, '–')
  assert.equal(s.tone, 'neutral')
  const lv = [{ kind: 'buyers_absorbed', price: 110, start: 10, end: 50, broken: false }]
  assert.match(absRow(lv, '07:52').value, / · flow feed stalled since 07:52 PT$/)
  assert.equal(absRow(lv, '07:52').reads, absRow(lv, null).reads)
})
