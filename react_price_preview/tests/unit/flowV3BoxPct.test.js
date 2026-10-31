// CR-DQ item 5 — percentile badges against the train baseline (src/data/boxBaseline.json, scripts/cr_dq_baseline.py).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import {
  BOX_RULES_DEFAULTS, baselineBucket, baselineTod, boxPercentiles, boxRules, pctBadgeTitle, pctText, pctTier, quantilePct,
} from '../../src/flowV3.js'

const baseline = JSON.parse(readFileSync(new URL('../../src/data/boxBaseline.json', import.meta.url)))
const rth = (source) => [{ name: 'eth', source: 'ironbeam' }, { name: 'rth', source }]
const ro = (delta, vol, dPrice, minutes) => ({ delta, vol, dPrice, minutes, fromSeg: 'rth', toSeg: 'rth' })

test('quantilePct = numpy.interp on the 101-point grid', () => {
  const g = Array.from({ length: 101 }, (_, i) => i * 2)
  assert.equal(quantilePct(g, -1), 0)
  assert.equal(quantilePct(g, 300), 100)
  assert.equal(quantilePct(g, 51), 25.5)
  assert.equal(quantilePct(g, 100), 50)
  assert.equal(quantilePct([], 1), null)
})

test('buckets by box minutes, time of day by start (PT minute of day)', () => {
  assert.equal(baselineBucket(baseline, 2), null)
  assert.equal(baselineBucket(baseline, 3).name, '5')
  assert.equal(baselineBucket(baseline, 22).name, '15')
  assert.equal(baselineBucket(baseline, 23).name, '30')
  assert.equal(baselineBucket(baseline, 500).name, '90+')
  assert.equal(baselineTod(baseline, 6 * 60 + 30).name, 'open')
  assert.equal(baselineTod(baseline, 7 * 60 + 30).name, 'mid')
  assert.equal(baselineTod(baseline, 11 * 60).name, 'late')
  assert.equal(baselineTod(baseline, 6 * 60), null)
})

// Step 0's three fixed train windows (spec S0.4), hand-computed with numpy.interp on the same JSON.
test('the three Step 0 windows match the hand check (same-source baseline; share split by start: 07:05 open, others mid)', () => {
  const cases = [
    ['ironbeam', ro(-1001, 85659, 4.75, 25), 7 * 60 + 5, 42.922, 69.201, 'open'],
    ['databento', ro(-1718, 92876, 4.75, 25), 7 * 60 + 5, 58.412, 69.378, 'open'],
    ['ironbeam', ro(320, 17630, 4.5, 12), 9 * 60 + 12, 35.668, 30.412],
    ['databento', ro(725, 18923, 4.5, 12), 9 * 60 + 12, 57.741, 30.327],
    ['ironbeam', ro(-131, 62331, 1.0, 55), 10 * 60 + 40, 8.354, 88.597],
    ['databento', ro(-154, 68938, 1.0, 55), 10 * 60 + 40, 7.688, 88.987],
  ]
  for (const [src, r, ptMin, share, cpp, tod = 'mid'] of cases) {
    const pc = boxPercentiles(r, ptMin, rth(src), baseline)
    assert.equal(pc.source, src)
    assert.equal(pc.tod, tod)
    assert.ok(Math.abs(pc.share.p - share) < 0.01, `${src} share ${pc.share.p} vs ${share}`)
    assert.ok(Math.abs(pc.cpp.p - cpp) < 0.01, `${src} cpp ${pc.cpp.p} vs ${cpp}`)
  }
})

test('no badge: ETH, two segments, unknown source, under 3 min; flat box uses the one-tick floor', () => {
  assert.equal(boxPercentiles({ ...ro(1, 100, 1, 10), fromSeg: 'eth', toSeg: 'eth' }, 400, rth('ironbeam'), baseline), null)
  assert.equal(boxPercentiles({ ...ro(1, 100, 1, 10), fromSeg: 'eth', toSeg: 'rth' }, 400, rth('ironbeam'), baseline), null)
  assert.equal(boxPercentiles(ro(1, 100, 1, 10), 400, [{ name: 'rth', source: 'x' }], baseline), null)
  assert.equal(boxPercentiles(ro(1, 100, 1, 2), 400, rth('ironbeam'), baseline), null)
  assert.equal(boxPercentiles(ro(10, 1000, 0, 10), 600, rth('ironbeam'), baseline).cpp.value, 4000)
})

test('tiers, text, rules clamp, badge hover', () => {
  assert.equal(pctTier(79.9), 'normal')
  assert.equal(pctTier(80), 'notable')
  assert.equal(pctTier(95), 'strong')
  assert.equal(pctTier(null), null)
  assert.equal(pctText(94.7), 'p94')
  assert.equal(pctText(100), 'p99')
  assert.deepEqual(boxRules(null), BOX_RULES_DEFAULTS)
  assert.deepEqual(boxRules({ notable: 120, strong: 10, nearPts: 'x' }), { ...BOX_RULES_DEFAULTS, notable: 99, strong: 100 })
  const pc = boxPercentiles(ro(-1001, 85659, 4.75, 25), 425, rth('ironbeam'), baseline)
  const t = pctBadgeTitle('share', pc)
  assert.match(t, /^p42: this box's \|Δ share\| is above 42 % of RTH windows of 23–45 min starting 06:30–07:30 PT, in the Ironbeam baseline \(99 train sessions 2026-01-02 → 2026-06-05;/)
  assert.match(t, /median ~9 percentile points/)
  assert.match(pctBadgeTitle('cpp', pc), /contracts per point is above 69 % of RTH windows of 23–45 min \(all day\)/)
  assert.match(t, /Display aid — untested\.$/)
})
