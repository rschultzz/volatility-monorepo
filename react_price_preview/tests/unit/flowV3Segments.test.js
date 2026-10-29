// CR-DN — Flow v3 ranges as segments: set end (item 1).
// Run: npm --prefix react_price_preview run test:unit
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { anchorReadout, anchoredCvd, boxLabelText, boxSegments, edgeReadout, edgeText, endFromPick, endWindowText, flattenBars, rowCloses, segmentCvd, segmentReadouts } from '../../src/flowV3.js'

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
const rows = (deltas) => deltas.map((d, i) => ({ time: T0 + 60 * i, delta: d, vol: Math.abs(d) * 2 }))
const hm = (t) => `${String(Math.floor(((t - T0) / 60 + 409) / 60)).padStart(2, '0')}:${String(((t - T0) / 60 + 409) % 60).padStart(2, '0')}`

test('anchoredCvd / anchorReadout with an end: nothing after the end; without one, unchanged', () => {
  const r = rows([5, 1, 2, 3, 4, 10])
  const closes = [100, 101, 102, 103, 104, 110]
  assert.deepEqual(anchoredCvd(r, T0 + 60), [null, 0, 2, 5, 9, 19])
  assert.deepEqual(anchoredCvd(r, T0 + 60, T0 + 180), [null, 0, 2, 5, null, null])
  const full = anchorReadout(r, closes, T0 + 60)
  const cut = anchorReadout(r, closes, T0 + 60, T0 + 180)
  assert.equal(full.delta, 19)
  assert.equal(full.minutes, 4)
  assert.equal(cut.delta, 5)
  assert.equal(cut.dPrice, 2)
  assert.equal(cut.minutes, 2)
  assert.equal(cut.vol, 10)
  assert.equal(cut.cpp, 5)
})

test('edgeReadout with an end: bars after it are ignored, the box ends on the end bar', () => {
  // the 110 close (minute 3) would grow the box; with the end at minute 2 it is never seen
  const p = payload([100, 103, 99, 110, 100], [[1, 103, 5, 1], [3, 110, 50, 0]])
  const b = flattenBars(p)
  const full = edgeReadout(p, b, T0, id, 15)
  const cut = edgeReadout(p, b, T0, id, 15, T0 + 120)
  assert.equal(full.top.price, 110)
  assert.equal(full.end, null)
  assert.equal(cut.top.price, 103)
  assert.equal(cut.to, T0 + 120)
  assert.equal(cut.end, T0 + 120)
  assert.equal(cut.top.contracts, 5)
})

test('endFromPick: after the anchor, not after the last bar; a 5-min pick keeps the whole candle', () => {
  const a = T0
  assert.equal(endFromPick(T0 + 480, 60, a, T0 + 600), T0 + 480)
  assert.equal(endFromPick(T0, 60, a, T0 + 600), null) // at the anchor
  assert.equal(endFromPick(T0 - 60, 60, a, T0 + 600), null) // before it
  assert.equal(endFromPick(T0 + 660, 60, a, T0 + 600), null) // after the last bar
  assert.equal(endFromPick(T0 + 300, 300, a, T0 + 900), T0 + 540) // 5-min candle 300..599 -> its last minute
  assert.equal(endFromPick(T0 + 600, 300, a, T0 + 720), T0 + 720) // last candle still forming -> the last bar
  assert.equal(endFromPick(T0 + 60, 60, null, T0 + 600), null) // no anchor
})

test('endWindowText: "06:49 → 06:57 · 8 min"', () => {
  assert.equal(endWindowText(T0, T0 + 480, hm), '06:49 → 06:57 · 8 min')
  assert.equal(endWindowText(T0, null, hm), '')
})

// Item 2 fixture: box 100..103 held 20 min, then a close 3.25 pts beyond the top (106.25) at minute 21.
const flat = (n, v) => new Array(n).fill(v)
const held = [100, ...flat(4, 101), 103, ...flat(15, 102)] // top set at minute 5 (103), held >= 15 by minute 21

test('item 2: a close strictly back inside within A minutes cancels the break (probe); the box is unchanged', () => {
  // 21: 106.25 (break, pending), 22: 104 (outside), 23: 103 (ON the edge: not inside), 24: 102.5 (inside) -> probe
  const closes = [...held, 106.25, 104, 103, 102.5, 101]
  const cells = [[21, 105, 30, 20], [21, 103, 9, 9], [22, 104, 5, 7], [22, 106.5, 1, 2]]
  const p = payload(closes, cells)
  const r = edgeReadout(p, flattenBars(p), T0, id, 15, null, 5)
  assert.equal(r.frozen.length, 0)
  assert.deepEqual([r.bottom.price, r.top.price], [100, 103])
  assert.equal(r.pending, null)
  assert.equal(r.probes.length, 1)
  const pr = r.probes[0]
  assert.deepEqual([pr.at, pr.dir, pr.edge, pr.minutesOutside, pr.back], [T0 + 21 * 60, 'up', 103, 3, T0 + 24 * 60])
  assert.equal(pr.extreme, 106.75) // highest high over minutes 21-23 (close + 0.5)
  assert.deepEqual([pr.buy, pr.sell], [30 + 5 + 1, 20 + 7 + 2]) // strictly above 103, minutes 21-23
})

test('item 2: no close strictly inside within A minutes -> held; the freeze and new box start at the break bar', () => {
  // closes on the edge (103) are a retest, not back inside: the break holds
  const closes = [...held, 106.25, 103, 104, 103, 105, 104.5, 104]
  const p = payload(closes, [])
  const r = edgeReadout(p, flattenBars(p), T0, id, 15, null, 5)
  assert.equal(r.frozen.length, 1)
  const f = r.frozen[0]
  assert.deepEqual([f.from, f.to, f.breakAt, f.dir], [T0, T0 + 20 * 60, T0 + 21 * 60, 'up'])
  assert.deepEqual([f.bottom.price, f.top.price], [100, 103])
  assert.equal(f.probes.length, 0)
  // the new box replays from the break bar with the growth rule: 106.25, then 103 is > 2 below -> bottom grows
  assert.equal(r.from, T0 + 21 * 60)
  assert.deepEqual([r.bottom.price, r.top.price], [103, 106.25])
})

test('item 2: A = 1 is the old single-close break unless the very next close is strictly inside', () => {
  const closes = [...held, 106.25, 104, 102.5, 101]
  const p = payload(closes, [])
  const one = edgeReadout(p, flattenBars(p), T0, id, 15, null, 1)
  assert.equal(one.frozen.length, 1)
  assert.equal(one.frozen[0].breakAt, T0 + 21 * 60)
  const five = edgeReadout(p, flattenBars(p), T0, id, 15, null, 5)
  assert.equal(five.frozen.length, 0)
  assert.equal(five.probes.length, 1)
})

test('item 2: unresolved at the last bar -> pending, "testing break" in the header; nothing grows meanwhile', () => {
  const closes = [...held, 106.25, 109, 104]
  const p = payload(closes, [])
  const r = edgeReadout(p, flattenBars(p), T0, id, 15, null, 5)
  assert.equal(r.frozen.length, 0)
  assert.deepEqual([r.bottom.price, r.top.price], [100, 103])
  assert.deepEqual(r.pending, { at: T0 + 21 * 60, dir: 'up', edge: 103 })
  assert.match(edgeText(r, () => '06:51'), /^testing break 06:51 ↑ · Top 103\.00/)
  // an end inside the pending window: still pending at the end, box unchanged
  const cut = edgeReadout(p, flattenBars(p), T0, id, 15, T0 + 22 * 60, 5)
  assert.equal(cut.pending.at, T0 + 21 * 60)
  assert.equal(cut.frozen.length, 0)
})

test('item 3: segmentCvd restarts at 0 per span through anchoredCvd; gap on the last row of each older span', () => {
  const r = rows([1, 2, 3, 4, 5, 6, 7])
  const spans = [{ from: T0 + 60, to: T0 + 180 }, { from: T0 + 240, to: T0 + 360 }]
  const { cvd, gapAt } = segmentCvd(r, spans)
  assert.deepEqual(cvd, [null, 0, 3, 7, 0, 6, 13]) // span 2 starts at 0 on its first bar (that bar's delta not counted)
  assert.deepEqual([...gapAt], [3])
  // the same as anchoring at each span's start with its end
  assert.deepEqual(cvd.slice(4), anchoredCvd(r, T0 + 240, T0 + 360).slice(4))
  // a shared (5-min) row: the newer span's 0 wins and is not a gap end
  const shared = segmentCvd(r, [{ from: T0, to: T0 + 120 }, { from: T0 + 120, to: T0 + 360 }])
  assert.deepEqual(shared.cvd, [0, 2, 0, 4, 9, 15, 22])
  assert.deepEqual([...shared.gapAt], [1])
})

test('item 3: boxSegments / segmentReadouts / boxLabelText on a held break', () => {
  const closes = [...held, 106.25, 103, 104, 103, 105, 104.5, 104]
  const p = payload(closes, [])
  const e = edgeReadout(p, flattenBars(p), T0, id, 15, null, 5)
  const seg = boxSegments(e, id)
  assert.deepEqual(seg.spans, [{ from: T0, to: T0 + 20 * 60 }, { from: T0 + 21 * 60, to: T0 + 27 * 60 }])
  assert.equal(seg.dropped, 0)
  const rr = closes.map((c, i) => ({ time: T0 + 60 * i, delta: i % 2 ? 100 : -40, vol: 300 }))
  const cl = rowCloses(rr, closes.map((c, i) => ({ time: T0 + 60 * i, close: c })), id)
  const ro = segmentReadouts(rr, cl, seg.spans)
  // box 1: bars 1..20 -> 10 odd (+100) and 10 even (-40) = +600; close 102 - 100 = +2; 20 min; vol 6000 / 2 pts
  assert.deepEqual([ro[0].delta, ro[0].dPrice, ro[0].minutes, ro[0].vol, ro[0].cpp], [600, 2, 20, 6000, 3000])
  assert.equal(boxLabelText(ro[0]), 'Δ +600 · +2.0 pts · 20 m · 3.0k/pt')
  assert.equal(boxLabelText({ delta: -1234, dPrice: -7.5, minutes: 45, cpp: 1800 }), 'Δ -1.2k · -7.5 pts · 45 m · 1.8k/pt')
})
