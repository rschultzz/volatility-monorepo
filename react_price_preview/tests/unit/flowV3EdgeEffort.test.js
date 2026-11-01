// CR-DI — Edge effort: the per-minute series behind the edge readout, visits, strip columns, chip blocking.
// Run: npm --prefix react_price_preview run test:unit
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { EDGE_STRIP_HEIGHT, anchorReadout, boxEvidenceRows, boxVerdict, drawnBoxes, edgeEffortBlock, edgeStripDrawn, edgeEffortSeries, edgeReadout, edgeStripColumns, edgeText, edgeVisits, flattenBars } from '../../src/flowV3.js'

const T0 = 1_777_642_200 // a UTC minute
const id = (t) => t
const tick = (price) => Math.round(price / 0.25)
const bars = (closes) => ({
  t: closes.map((_, i) => T0 + 60 * i), h: closes.map((c) => c + 0.5), l: closes.map((c) => c - 0.5), c: closes,
})
// cells: [minute, price, buy, sell]
const payload = (closes, cells, exact = true) => ({
  segments: [{
    name: 'rth', exact, start: T0, end: T0 + 390 * 60, bars: bars(closes), marks: null,
    cells: { t: cells.map((c) => T0 + 60 * c[0]), p: cells.map((c) => tick(c[1])), buy: cells.map((c) => c[2]), sell: cells.map((c) => c[3]) },
  }],
})
const sum = (a) => a.reduce((x, y) => x + y, 0)

test('main sides of the series sum to the header; opposite sides use the same ±1 pt inclusive band', () => {
  // box 100 .. 110 (closes); cells at the top band edges (109, 111), beyond it (111.25), at the bottom (99, 101), middle
  const p = payload([100, 110, 105, 101, 109], [
    [1, 109, 10, 4], [1, 111, 5, 1], [1, 111.25, 99, 99], [2, 105, 50, 50],
    [3, 99, 7, 20], [3, 101, 3, 30], [3, 98.75, 99, 99], [4, 110, 8, 9],
  ])
  const r = edgeReadout(p, flattenBars(p), T0, id)
  assert.equal(r.top.price, 110)
  assert.equal(r.bottom.price, 100)
  assert.deepEqual(r.minutes.t, [T0 + 60, T0 + 180, T0 + 240])
  assert.deepEqual(r.minutes.topBuy, [15, 0, 8])
  assert.deepEqual(r.minutes.topSell, [5, 0, 9])
  assert.deepEqual(r.minutes.bottomSell, [0, 50, 0])
  assert.deepEqual(r.minutes.bottomBuy, [0, 10, 0])
  assert.equal(r.top.contracts, sum(r.minutes.topBuy))
  assert.equal(r.bottom.contracts, sum(r.minutes.bottomSell))
})

test('a break moves the edge and re-counts every minute since the anchor against the new edge', () => {
  const cells = [[1, 110, 100, 90], [3, 113, 40, 30]]
  const before = payload([100, 110, 104], cells.slice(0, 1))
  const rb = edgeReadout(before, flattenBars(before), T0, id)
  assert.equal(rb.top.contracts, 100)
  const after = payload([100, 110, 104, 113], cells) // 113 > 110 + 2: break
  const ra = edgeReadout(after, flattenBars(after), T0, id)
  assert.equal(ra.top.price, 113)
  assert.deepEqual(ra.minutes.t, [T0 + 180]) // the 110 minute is > 1 pt from 113 (and from the bottom) now
  assert.deepEqual(ra.minutes.topBuy, [40])
  assert.equal(ra.top.contracts, 40)
  assert.equal(ra.top.contracts, sum(ra.minutes.topBuy))
})

test('approximate bars give no series, and the chip says why', () => {
  const p = payload([100, 110], [[1, 110, 5, 5]], false)
  const r = edgeReadout(p, flattenBars(p), T0, id)
  assert.equal(r.exact, false)
  assert.equal(r.minutes, null)
  assert.match(edgeEffortBlock(true, T0, r), /exact data only/)
  assert.match(edgeEffortBlock(false, T0, r), /Edges on/)
  assert.match(edgeEffortBlock(true, null, null), /anchor/)
  assert.equal(edgeEffortBlock(true, T0, null), '') // cells still loading: nothing to block on yet
  const q = payload([100, 110], [[1, 110, 5, 5]])
  assert.equal(edgeEffortBlock(true, T0, edgeReadout(q, flattenBars(q), T0, id)), '')
})

test('visits join across up to 2 empty minutes; opposite side counts the quiet minutes inside a visit', () => {
  const m = (k) => T0 + 60 * k
  const minutes = {
    t: [m(0), m(1), m(4), m(5), m(9), m(10)],
    topBuy: [10, 0, 20, 5, 7, 0],
    topSell: [1, 3, 2, 2, 4, 6],
    bottomSell: [0, 0, 0, 0, 0, 0],
    bottomBuy: [0, 0, 0, 0, 0, 0],
  }
  const v = edgeVisits(minutes)
  // 0 -> 4 is a 3-minute gap (1, 2, 3 empty of main bars): split; 5 -> 9 is a 3-minute gap: split
  assert.deepEqual(v.top.map((x) => [x.start, x.end, x.main, x.opp]), [
    [m(0), m(0), 10, 1], [m(4), m(5), 25, 4], [m(9), m(9), 7, 4],
  ])
  const joined = edgeVisits({ ...minutes, t: [m(0), m(1), m(3), m(4), m(9), m(10)] }) // 0 -> 3: 2 empty minutes
  assert.deepEqual(joined.top.map((x) => [x.start, x.end, x.main, x.opp]), [[m(0), m(4), 35, 8], [m(9), m(9), 7, 4]])
  assert.deepEqual(joined.run.top.slice(0, 4), [[10, 1], [10, 4], [30, 6], [35, 8]])
  assert.deepEqual(joined.of.top, [0, 0, 0, 0, 1, -1])
  assert.deepEqual(v.bottom, [])
})

test('strip columns bucket minutes onto the candles (5-minute chart) and scale to the largest bar', () => {
  const minutes = { t: [T0, T0 + 60, T0 + 300], topBuy: [1, 2, 3], topSell: [4, 5, 6], bottomSell: [0, 0, 9], bottomBuy: [0, 1, 0] }
  const s = edgeStripColumns(minutes, [T0, T0 + 300, T0 + 600], id)
  assert.deepEqual(s.cols.get(0), [3, 9, 0, 1])
  assert.deepEqual(s.cols.get(1), [3, 6, 9, 0])
  assert.deepEqual(s.idxOf, [0, 0, 1])
  assert.equal(s.max, 9)
})

// CR-DI amendment 9 — the box restarts on a break once the broken edge has held N minutes.
const flat = (n, c) => new Array(n).fill(c)
test('box: forming edges grow with the closes; once qualified a close > 2 pts beyond breaks and a new box starts', () => {
  // CR-DU Amendment 5: minutes 0-4 at 100, 5: 103 (forming -> top 103), qualifies at minute 8 (3 pts), 6-24 at 102, 25: 106 (> 2 pts beyond -> break)
  const closes = [...flat(5, 100), 103, ...flat(19, 102), 106, 105, 106.5]
  const cells = [[5, 103, 50, 40], [10, 102.5, 7, 3], [25, 106, 9, 1], [27, 106.5, 11, 2], [3, 100, 4, 6]]
  const p = payload(closes, cells)
  // CR-DN item 2: the break must hold A bars; CR-DI's single-close break is A = 1 (105 at minute 26 is not back inside)
  const r = edgeReadout(p, flattenBars(p), T0, id, 15, null, 1)
  assert.equal(r.frozen.length, 1)
  const f = r.frozen[0]
  assert.deepEqual([f.from, f.to, f.breakAt, f.dir], [T0, T0 + 24 * 60, T0 + 25 * 60, 'up'])
  assert.deepEqual([f.bottom.price, f.top.price], [100, 103])
  assert.equal(f.top.contracts, 50 + 7) // 103 and 102.5 within 1 pt of 103, minutes 0-24 only
  assert.equal(f.top.opposite, 40 + 3)
  assert.equal(f.bottom.contracts, 6)
  assert.equal(f.heldMin, 25)
  // current box from the break bar, forming: running max / min close 106.5 / 105
  assert.deepEqual([r.from, r.top.price, r.bottom.price, r.startDir], [T0 + 25 * 60, 106.5, 105, 'up'])
  assert.equal(r.top.contracts, 9 + 11)
  assert.equal(r.top.contracts, sum(r.minutes.topBuy))
  assert.equal(r.bottom.contracts, sum(r.minutes.bottomSell))
  assert.match(edgeText(r, () => '07:25'), /^Box from 07:25 ↑ · Top 106\.50/)
  // Amendment 5: the edge hold no longer gates the break
  const one = edgeReadout(p, flattenBars(p), T0, id, 30, null, 1)
  assert.equal(one.frozen.length, 1)
  assert.equal(one.frozen[0].top.price, 103)
})

test('box: drawn boxes keep the last 4 frozen; the strip series and visits stay inside each box', () => {
  // a break every 6 minutes with N = 5: closes step up 3 pts every 6 minutes
  const closes = []
  for (let k = 0; k < 7; k += 1) closes.push(...flat(6, 100 + 3 * k))
  const cells = closes.map((c, i) => [i, c, 1, 1])
  const p = payload(closes, cells)
  const r = edgeReadout(p, flattenBars(p), T0, id, 5, null, undefined, { qualifyMin: 1, qualifyPts: 0 })
  assert.equal(r.frozen.length, 6)
  const { boxes, dropped } = drawnBoxes(r)
  assert.equal(boxes.length, 5)
  assert.equal(dropped, 2)
  assert.equal(boxes[4], r)
  const s = edgeEffortSeries(r)
  assert.equal(s.dropped, 2)
  assert.equal(s.t[0], r.frozen[2].from) // dropped boxes' minutes are not drawn
  const v = edgeVisits(s)
  // each box is 6 minutes of continuous bars; consecutive boxes touch (gap 0) but visits must not join
  assert.equal(v.top.length, 5)
  for (const x of v.top) assert.equal((x.end - x.start) / 60, 5)
})

test('edgeText uses "moved" for growth and no box prefix without a break', () => {
  const p = payload([100, 103, 103], [])
  const r = edgeReadout(p, flattenBars(p), T0, id)
  assert.equal(edgeText(r, () => '07:01'), 'Top 103.00 · 0 buyers · held 1 m · moved 07:01 | Bottom 100.00 · 0 sellers · held 2 m')
})

test('amendment 2: the strip (and the pane growth) needs the chip, Edges, an anchor and an exact readout', () => {
  const q = payload([100, 110], [[1, 110, 5, 5]])
  const r = edgeReadout(q, flattenBars(q), T0, id)
  assert.equal(EDGE_STRIP_HEIGHT, 64)
  assert.equal(edgeStripDrawn(true, true, T0, r), true)
  assert.equal(edgeStripDrawn(false, true, T0, r), false)
  assert.equal(edgeStripDrawn(true, false, T0, r), false)
  assert.equal(edgeStripDrawn(true, true, null, r), false)
  assert.equal(edgeStripDrawn(true, true, T0, null), false) // cells loading: no strip, no growth yet
  const a = payload([100, 110], [[1, 110, 5, 5]], false)
  assert.equal(edgeStripDrawn(true, true, T0, edgeReadout(a, flattenBars(a), T0, id)), false)
})

// CR-DU Amendment 5 (engine v1.3): forming edges = running max / min close (A1.1); edges freeze at qualification
const heightOf = (r) => r.top.price - r.bottom.price
test('Amendment 5: 2026-04-15 screenshot box (real closes, 09:18–09:29): 2.50 pts by frozen edges, not 0.00; true max − min 4.00', () => {
  const closes = [7037.75, 7037.75, 7039.5, 7039.5, 7039.25, 7039.75, 7039.75, 7037.25, 7038, 7038.5, 7037.5, 7035.75]
  assert.equal(Math.max(...closes) - Math.min(...closes), 4)
  const p = payload(closes, [])
  for (const qualifyMin of [7, 8]) {
    const r = edgeReadout(p, flattenBars(p), T0, id, 15, null, 5, { qualifyMin, qualifyPts: 2 })
    assert.deepEqual([r.top.price, r.bottom.price, heightOf(r), r.qualified, r.frozen.length, r.pending], [7039.75, 7037.25, 2.5, true, 0, null])
    const rows = closes.map((_, i) => ({ time: T0 + i * 60, delta: 0, vol: 1000, seg: 'rth' }))
    const v = boxVerdict({ box: { ...r, readout: anchorReadout(rows, closes, T0, T0 + 11 * 60), ended: { kind: 'current' } }, pct: null, levels: [], tradeDate: '2026-04-15', rules: { minAgeMin: qualifyMin, minHeightPts: 2 } })
    const row = boxEvidenceRows(v.checks).find((x) => x.key === 'formed')
    assert.equal(row.value, `11 min · 2.50 pts (min ${qualifyMin} · 2)`)
    assert.equal(row.reads, 'formed')
  }
})
test('Amendment 5: a forming box widens with its closes, height = max − min close at every minute, and cannot break', () => {
  const closes = [100, 100.5, 99.75, 101, 106, 106.25, 98, 100, 100]
  const p = payload(closes, [])
  for (let i = 0; i < 8; i += 1) {
    const r = edgeReadout(p, flattenBars(p), T0, id, 15, T0 + i * 60, 1)
    const seen = closes.slice(0, i + 1)
    assert.equal(heightOf(r), Math.max(...seen) - Math.min(...seen), `minute ${i}`)
    assert.deepEqual([r.frozen.length, r.pending, r.qualified], [0, null, false])
  }
  // flat under the height minimum stays forming past the box minimum
  const q = payload([...flat(12, 100), 101, 102.5, 102], [])
  const under = edgeReadout(q, flattenBars(q), T0, id, 15, T0 + 13 * 60 - 60, 1)
  assert.deepEqual([heightOf(under), under.qualified], [1, false])
  const over = edgeReadout(q, flattenBars(q), T0, id, 15, null, 1)
  assert.deepEqual([over.top.price, over.bottom.price, over.qualified, over.frozen.length], [102.5, 100, true, 0])
})
test('Amendment 5: a qualified box tested 1.5 pts through holds; its edges stay frozen; > 2 pts breaks even with a young edge', () => {
  const base = [100, 101, 102, 103, 102, 101, 102, 103, 102] // qualifies at minute 8: 103 / 100
  const test15 = payload([...base, 104.5, 102, 98.5, 101, 102], [])
  const r = edgeReadout(test15, flattenBars(test15), T0, id, 15, null, 5)
  assert.deepEqual([r.top.price, r.bottom.price, r.qualified, r.frozen.length, r.pending, r.probes.length], [103, 100, true, 0, null, 0])
  const brk = payload([...base, 105.5, 106, 106, 106, 106, 106], [])
  const b = edgeReadout(brk, flattenBars(brk), T0, id, 15, null, 5) // top set at minute 7: only 2 min old
  assert.equal(b.frozen.length, 1)
  assert.deepEqual([b.frozen[0].top.price, b.frozen[0].bottom.price, b.frozen[0].dir, b.from], [103, 100, 'up', T0 + 9 * 60])
  // a > 2-pt close back strictly inside within the break hold is a probe, the box stands
  const prb = payload([...base, 105.5, 102, 102], [])
  const pb = edgeReadout(prb, flattenBars(prb), T0, id, 15, null, 5)
  assert.deepEqual([pb.frozen.length, pb.probes.length, pb.top.price], [0, 1, 103])
})
