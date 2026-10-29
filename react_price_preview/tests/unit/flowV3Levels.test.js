// CR-DF — absorption levels and the edge readout. Run: npm --prefix react_price_preview run test:unit
import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  absorptionLevels,
  chartMarks,
  edgeReadout,
  edgeText,
  flattenBars,
  levelAlpha,
  levelWidth,
  payloadNow,
  relinkHover,
} from '../../src/flowV3.js'

const T0 = 1_777_642_200 // a UTC minute
const id = (t) => t
const minutes = (n) => Array.from({ length: n }, (_, i) => T0 + 60 * i)
const bars = (closes, { hi = 0.5, lo = 0.5, at = T0 } = {}) => ({
  t: closes.map((_, i) => at + 60 * i), h: closes.map((c) => c + hi), l: closes.map((c) => c - lo), c: closes,
})
const mark = (kind, minute, price, extra = {}) => ({
  kind, time: T0 + 60 * minute, price, confirm_time: T0 + 60 * (minute + 4), aggressor: 400, opposite: 50,
  merged_vol: 0, median: 100, break_time: null, held_min: 0, ...extra,
})
const payload = (segs) => ({ segments: segs })
const seg = (b, marks, exact = true, cells = { t: [], p: [], buy: [], sell: [] }) => ({
  name: 'rth', exact, start: T0, end: T0 + 390 * 60, bars: b, marks: exact ? marks : null, cells,
})

test('fade: 1 at 0 min, 25 % at 60 min, flat after; width log-scaled and capped', () => {
  assert.equal(levelAlpha(0), 1)
  assert.equal(levelAlpha(30), 0.625)
  assert.equal(levelAlpha(60), 0.25)
  assert.equal(levelAlpha(240), 0.25)
  assert.equal(levelWidth(10), 1)
  assert.equal(levelWidth(50), 1)
  assert.ok(Math.abs(levelWidth(1500) - 5) < 1e-9)
  assert.equal(levelWidth(1e6), 5)
  assert.ok(levelWidth(350) > 3 && levelWidth(350) < 3.5)
})

test('now = end of the last bar (live: last data; replay: payload cut at until)', () => {
  assert.equal(payloadNow(payload([seg(bars([1, 2, 3]), [])])), T0 + 180)
  assert.equal(payloadNow(payload([])), null)
})

test('level starts at the mark minute, ends at the break or now; label and held minutes', () => {
  const b = bars(new Array(30).fill(7000))
  const p = payload([seg(b, [mark('sellers_absorbed', 2, 7000, { merged_vol: 100 }), mark('buyers_absorbed', 5, 7010, { break_time: T0 + 60 * 12 })])])
  const lv = absorptionLevels(p, minutes(30), id)
  assert.equal(lv.length, 2)
  const [s, bu] = lv
  assert.equal(s.start, T0 + 120)
  assert.equal(s.end, T0 + 30 * 60) // unbroken: runs to now
  assert.equal(s.broken, false)
  assert.equal(s.contracts, 500)
  assert.equal(s.heldMin, 28)
  assert.equal(s.label, 'sellers absorbed 500 · 28 min')
  assert.equal(bu.end, T0 + 12 * 60)
  assert.equal(bu.broken, true)
  assert.equal(bu.heldMin, 7)
  assert.equal(bu.label, 'buyers absorbed 400 · 7 min')
  assert.equal(bu.idx, 5)
  assert.equal(bu.endIdx, 12)
})

test('a level is not shown before its hold window ends; a later break is not seen early (replay)', () => {
  const b = bars(new Array(5).fill(7000)) // now = T0 + 5 min
  const p = payload([seg(b, [mark('sellers_absorbed', 2, 7000), mark('sellers_absorbed', 0, 6990, { break_time: T0 + 60 * 9 })])])
  const lv = absorptionLevels(p, minutes(5), id)
  assert.equal(lv.length, 1) // the minute-2 mark confirms at minute 6 > now
  assert.equal(lv[0].broken, false)
  assert.equal(lv[0].end, T0 + 300)
})

test('fading with age and the top 8 by contracts x opacity', () => {
  const b = bars(new Array(200).fill(7000))
  const marks = []
  for (let i = 0; i < 12; i += 1) marks.push(mark('sellers_absorbed', i * 15, 6900 + i, { aggressor: 300 }))
  marks.push(mark('buyers_absorbed', 0, 7100, { aggressor: 5000 })) // old but huge: 5000 x 0.25 beats 300 x 1
  const lv = absorptionLevels(payload([seg(b, marks)]), minutes(200), id)
  assert.equal(lv.length, 8)
  assert.ok(lv.some((l) => l.price === 7100))
  const recent = lv.filter((l) => l.kind === 'sellers_absorbed').map((l) => l.price)
  assert.deepEqual(recent, [6905, 6906, 6907, 6908, 6909, 6910, 6911]) // the 7 most recent small ones
  const newest = lv.find((l) => l.price === 6911)
  assert.ok(Math.abs(newest.alpha - levelAlpha((200 - 165) )) < 1e-9)
  assert.equal(lv.find((l) => l.price === 7100).alpha, 0.25)
})

test('approximate segments give no levels', () => {
  const b = bars(new Array(30).fill(7000))
  assert.deepEqual(absorptionLevels(payload([seg(b, [mark('sellers_absorbed', 2, 7000)], false)]), minutes(30), id), [])
})

test('edge: closes within 2 pts are tests, a close > 2 pts beyond breaks and moves the edge', () => {
  const closes = [7000, 7001.5, 7002, 6999, 7004.25, 7005, 6998, 6997.75, 6995.5]
  const r = edgeReadout(payload([]), flattenBars(payload([seg(bars(closes), [])])), T0, id)
  assert.equal(r.top.price, 7004.25) // 7001.5 / 7002 are tests; 7004.25 > 7002 breaks
  assert.equal(r.top.brokeAt, T0 + 4 * 60)
  assert.equal(r.top.breaks, 1)
  // bottom: 6999 / 6998 are tests; 6997.75 is 2.25 below 7000 (break); 6995.5 is 2.25 below 6997.75 (break again)
  assert.equal(r.bottom.price, 6995.5)
  assert.equal(r.bottom.breaks, 2)
  assert.equal(r.bottom.brokeAt, T0 + 8 * 60)
})

test('edge: repeated breaks, first touch within 1 pt and held minutes', () => {
  const closes = [7000, 7003, 7006, 7005, 7004, 7005.5]
  const b = bars(closes, { hi: 0.25, lo: 0.25 })
  const r = edgeReadout(payload([]), flattenBars(payload([seg(b, [])])), T0, id)
  assert.equal(r.top.price, 7006)
  assert.equal(r.top.breaks, 2)
  assert.equal(r.top.brokeAt, T0 + 120)
  assert.equal(r.top.firstTouch, T0 + 120) // minute 1 high 7003.25 < 7005
  assert.equal(r.top.heldMin, 3)
  assert.equal(r.bottom.price, 7000)
  assert.equal(r.bottom.firstTouch, T0)
  assert.equal(r.bottom.heldMin, 5)
  assert.equal(r.bottom.brokeAt, null)
})

test('edge contracts: within 1 pt of the edge since the anchor, buyers at the top, sellers at the bottom', () => {
  const closes = [7000, 7001, 7000.5, 6999.5, 7000]
  const tick = (px) => Math.round(px / 0.25)
  const cells = {
    t: [T0 - 60, T0, T0 + 60, T0 + 60, T0 + 120, T0 + 180, T0 + 240],
    p: [tick(7000), tick(7000), tick(7001), tick(7001.25), tick(6999), tick(6998.75), tick(7000)],
    buy: [999, 10, 20, 40, 5, 7, 1],
    sell: [999, 3, 1, 2, 30, 60, 4],
  }
  const p = payload([seg(bars(closes), [], true, cells)])
  const r = edgeReadout(p, flattenBars(p), T0, id)
  assert.equal(r.top.price, 7000)
  assert.equal(r.bottom.price, 7000)
  // within [6999, 7001]: t0 (7000), t1 (7001), t2 (6999), t4 (7000); 7001.25 and 6998.75 are outside; T0-60 is before the anchor
  assert.equal(r.top.contracts, 10 + 20 + 5 + 1)
  assert.equal(r.bottom.contracts, 3 + 1 + 30 + 4)
  assert.equal(r.exact, true)
})

test('edge contracts are null ("exact data only") when the window touches an approximate segment', () => {
  const eth = { ...seg(bars([7000, 7001]), [], false), name: 'eth' }
  const rth = seg(bars([7002, 7003], { at: T0 + 120 }), [])
  const p = payload([eth, rth])
  const r = edgeReadout(p, flattenBars(p), T0, id)
  assert.equal(r.exact, false)
  assert.equal(r.top.contracts, null)
  assert.equal(edgeText(r, () => '07:00'),
    'Top 7003.00 · exact data only · held 1 m · moved 07:00 | Bottom 7000.00 · exact data only · held 3 m') // CR-DI: growth = "moved"
  const r2 = edgeReadout(p, flattenBars(p), T0 + 120, id)
  assert.equal(r2.exact, true)
})

test('edge text format', () => {
  const r = {
    top: { price: 7724.75, contracts: 1800, heldMin: 9, brokeAt: null },
    bottom: { price: 7715.25, contracts: 4100, heldMin: 6, brokeAt: T0 },
  }
  assert.equal(edgeText(r, () => '07:12'), 'Top 7724.75 · 1.8k buyers · held 9 m | Bottom 7715.25 · 4.1k sellers · held 6 m · moved 07:12') // CR-DI
  assert.equal(edgeReadout(payload([]), flattenBars(payload([])), T0, id), null)
  assert.equal(edgeReadout(payload([]), bars([1]), null, id), null)
})

// CR-DG item 2: held minutes for unbroken levels / marks run to the payload's last bar, and grow with it.
test('held minutes: unbroken run to the last bar and grow on refresh; broken stay start to break', () => {
  const marks = [mark('sellers_absorbed', 2, 7000, { held_min: 388 }), mark('buyers_absorbed', 5, 7010, { break_time: T0 + 60 * 12, held_min: 7 })]
  const at = (n) => payload([seg(bars(new Array(n).fill(7000)), marks)])
  const lv20 = absorptionLevels(at(20), minutes(30), id)
  const lv25 = absorptionLevels(at(25), minutes(30), id)
  assert.deepEqual(lv20.map((l) => l.heldMin), [18, 7])
  assert.deepEqual(lv25.map((l) => l.heldMin), [23, 7])
  assert.equal(lv25[0].label, 'sellers absorbed 400 · 23 min')
  // marks hover: client-side, not the server's held_min (which stops at the segment end, 13:00 PT live)
  const mk20 = chartMarks(at(20), minutes(30), id)
  const mk25 = chartMarks(at(25), minutes(30), id)
  assert.deepEqual(mk20.map((m) => m.heldMin), [18, 7])
  assert.deepEqual(mk25.map((m) => m.heldMin), [23, 7])
  assert.equal(mk25[0].held_min, 388) // payload field untouched
})

// CR-DG item 2 (live check deferred): the payload's last bar moves from T to T + 2 min. Unbroken held minutes go from
// n to n + 2 in the level label and the marks hover, and an open hover (pointer still) picks up the new object.
test('refresh T -> T+2 min: held n -> n+2 (label, marks hover); an open hover updates without a pointer move', () => {
  const marks = [mark('buyers_absorbed', 3, 7010)]
  const at = (n) => payload([seg(bars(new Array(n).fill(7000)), marks)])
  const lvA = absorptionLevels(at(20), minutes(30), id)[0]
  const lvB = absorptionLevels(at(22), minutes(30), id)[0]
  assert.equal(lvA.heldMin, 17)
  assert.equal(lvB.heldMin, 19)
  assert.equal(lvA.label, 'buyers absorbed 400 · 17 min')
  assert.equal(lvB.label, 'buyers absorbed 400 · 19 min')
  const mkA = chartMarks(at(20), minutes(30), id)[0]
  const mkB = chartMarks(at(22), minutes(30), id)[0]
  assert.equal(mkA.heldMin, 17)
  assert.equal(mkB.heldMin, 19)
  // open hovers captured on the first payload, then the refresh arrives with the pointer still
  const pt = { x: 100, y: 50 }
  const levelHover = relinkHover({ level: lvA, pt }, 'level', [lvB], 'start')
  assert.equal(levelHover.level, lvB)
  assert.equal(levelHover.level.heldMin, 19)
  assert.equal(levelHover.pt, pt) // same place: no pointer move needed
  const markHover = relinkHover({ mark: mkA, pt }, 'mark', [mkB], 'time')
  assert.equal(markHover.mark.heldMin, 19)
  // same object -> same hover (no re-render); gone from the new list or no list -> closed
  const same = { level: lvB, pt }
  assert.equal(relinkHover(same, 'level', [lvB], 'start'), same)
  assert.equal(relinkHover({ level: lvA, pt }, 'level', [], 'start'), null)
  assert.equal(relinkHover({ level: lvA, pt }, 'level', null, 'start'), null)
  assert.equal(relinkHover(null, 'level', [lvB], 'start'), null)
})

test('CR-DN item 5c: unbroken levels are drawn fading to 50 %, broken ones as before; the selection is unchanged', () => {
  const b = bars(new Array(200).fill(7000))
  const marks = []
  for (let i = 0; i < 12; i += 1) marks.push(mark('sellers_absorbed', i * 15, 6900 + i, { aggressor: 300 }))
  marks.push(mark('buyers_absorbed', 0, 7100, { aggressor: 5000 }))
  marks.push(mark('buyers_absorbed', 10, 7101, { aggressor: 5000, break_time: T0 + 60 * 40 }))
  const lv = absorptionLevels(payload([seg(b, marks)]), minutes(200), id)
  const old = lv.find((l) => l.price === 7100)
  assert.equal(old.alpha, 0.25) // selection score unchanged (contracts x today's alpha)
  assert.equal(old.drawAlpha, 0.5) // drawn at the unbroken floor
  const broken = lv.find((l) => l.price === 7101)
  assert.equal(broken.broken, true)
  assert.equal(broken.drawAlpha, broken.alpha) // broken: today's fade
  assert.equal(levelAlpha(30, 0.5), 0.75)
  // the same 8 levels as with today's opacity: the score uses alpha, not drawAlpha
  assert.deepEqual(lv.map((l) => l.score), lv.map((l) => l.contracts * l.alpha))
})
