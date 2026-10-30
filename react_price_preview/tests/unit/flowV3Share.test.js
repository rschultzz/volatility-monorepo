// CR-DP item 6 — cumulative delta share (Σ delta ÷ Σ volume from a segment start), per box with Edges on.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  DEFAULT_V3_SETTINGS, SHARE_GUIDES, SHARE_SCALE, anchorReadout, anchoredShare, boxPopupLines, clipShare, readoutShare,
  segmentReadouts, segmentShare, sharePctText,
} from '../../src/flowV3.js'

const rows = [
  { time: 0, delta: 50, vol: 100 },
  { time: 60, delta: 10, vol: 100 },
  { time: 120, delta: -30, vol: 200 },
  { time: 180 }, // no data
  { time: 240, delta: 20, vol: 100 },
  { time: 300, delta: 5, vol: 0 },
]
const closes = [100, 101, 102, 102, 103, 104]

test('default: the anchored line plots delta share', () => {
  assert.equal(DEFAULT_V3_SETTINGS.anchoredLine, 'share')
  assert.equal(SHARE_SCALE, 0.10)
  assert.deepEqual(SHARE_GUIDES.slice().sort((a, b) => a - b), [-0.05, -0.02, 0, 0.02, 0.05])
})

test('anchoredShare sums the bars after the start bar (as anchorReadout); start bar and empty rows are null', () => {
  const s = anchoredShare(rows, 0)
  assert.deepEqual(s.share, [null, 0.1, -20 / 300, null, 0 / 400, 5 / 400])
  assert.deepEqual(s.sumDelta, [null, 10, -20, null, 0, 5])
  assert.deepEqual(s.sumVol, [null, 100, 300, null, 400, 400])
  // the last value equals the readout's share (same bars)
  const ro = anchorReadout(rows, closes, 0)
  assert.equal(readoutShare(ro), s.share[5])
})

test('anchoredShare respects the end and returns nulls without an anchor', () => {
  const s = anchoredShare(rows, 60, 180)
  assert.deepEqual(s.share, [null, null, -30 / 200, null, null, null])
  assert.ok(anchoredShare(rows, null).share.every((v) => v == null))
  assert.ok(anchoredShare(rows, -999).share.every((v) => v == null))
})

test('segmentShare restarts at each span; the next span\'s start bar is a gap', () => {
  const spans = [{ from: 0, to: 120 }, { from: 180, to: 300 }]
  const s = segmentShare(rows, spans)
  assert.deepEqual(s.share, [null, 0.1, -20 / 300, null, 20 / 100, 25 / 100])
  assert.deepEqual([...s.gapAt], [2]) // the line from box 1's last value to box 2's first is drawn transparent
  const ros = segmentReadouts(rows, closes, spans)
  assert.equal(readoutShare(ros[0]), s.share[2])
  assert.equal(readoutShare(ros[1]), s.share[5])
})

test('clip, format and readoutShare', () => {
  assert.equal(clipShare(0.25), 0.10)
  assert.equal(clipShare(-0.3), -0.10)
  assert.equal(clipShare(0.03), 0.03)
  assert.equal(sharePctText(0.01234), '+1.23 %')
  assert.equal(sharePctText(-0.05), '-5.00 %')
  assert.equal(sharePctText(null), '–')
  assert.equal(readoutShare({ delta: 5, vol: 0 }), null)
})

test('boxPopupLines: head, edges and states from the box fields', () => {
  const hhmm = (t) => `t${t}`
  const box = {
    from: 1, readout: { delta: 1200, vol: 60000, dPrice: 7.5, minutes: 45, cpp: 8000 },
    top: { price: 7724.75, contracts: 1800, opposite: 900, heldMin: 9, brokeAt: null },
    bottom: { price: 7715.25, contracts: null, opposite: null, heldMin: null, brokeAt: 5 },
    probes: [{ dir: 'up', at: 7 }, { dir: 'up', at: 8 }],
    pending: { dir: 'down', at: 9 },
    ended: { kind: 'end', at: 10 },
  }
  const l = boxPopupLines(box, hhmm)
  assert.equal(l.head, 'box from t1 → t10 · Δ +1.2k · share +2.00 % · +7.5 pts · 45 m · 8.0k/pt')
  assert.equal(l.top, 'Top 7724.75 · buyers 1.8k / sellers 900 · held 9 m · probe t8 (2)')
  assert.equal(l.bottom, 'Bottom 7715.25 · sellers exact data only / buyers exact data only · not touched · testing break t9 ↓')
})
