// CR-DY — pure helpers of the selectable boxes.
import test from 'node:test'
import assert from 'node:assert/strict'
import { boxPanelTitle, drawnBoxOf, engineAtLine, engineTradeBoxes, escAction, hitBox, reviewBoxRecord, snapBox, tradeText } from '../../src/selectableBoxes.js'

const fmt = (t) => `t${t}`
const trade = (o = {}) => ({ side: 'short', entry: 7655.75, stop: 7665.75, target: 7645.75, entryT: 300, exitT: 400, reason: 'target', pts: 10,
  decisionT: 120, box: { from: 60, top: 7655.75, bottom: 7650, high: 7657, low: 7649 }, ...o })

test('engineTradeBoxes reads the engine fields, ids as overlayMarks, open last', () => {
  const b = engineTradeBoxes({ trades: [trade(), trade({ decisionT: 500 })], open: trade({ decisionT: 900 }) })
  assert.deepEqual(b.map((x) => [x.id, x.k, x.toT, x.open]), [['v1-0', 0, 120, false], ['v1-1', 1, 500, false], ['v1-2', 2, 900, true]])
  assert.deepEqual([b[0].fromT, b[0].top, b[0].bottom], [60, 7655.75, 7650])
  assert.deepEqual(engineTradeBoxes(null), [])
})

test('drawnBoxOf: the readout edges, not its wicks', () => {
  const r = { from: 60, to: 120, top: { price: 10 }, bottom: { price: 5 }, high: 12, low: 4 }
  assert.deepEqual(drawnBoxOf(r), { fromT: 60, toT: 120, top: 10, bottom: 5 })
  assert.equal(drawnBoxOf(null), null)
})

test('snapBox floors to candle times; null before the first candle', () => {
  const box = { fromT: 70, toT: 130, top: 2, bottom: 1 }
  assert.deepEqual(snapBox('a', box, [60, 120, 180], (t) => t, '1,2,3'), { id: 'a', from: 60, to: 120, top: 2, bottom: 1, rgb: '1,2,3' })
  assert.equal(snapBox('a', { ...box, fromT: 10 }, [60, 120], (t) => t, 'x'), null)
})

test('hitBox: inside only, smallest wins', () => {
  const rects = [{ id: 'big', x0: 0, x1: 100, y0: 0, y1: 100 }, { id: 'small', x0: 10, x1: 20, y0: 10, y1: 20 }]
  assert.equal(hitBox(rects, { x: 15, y: 15 }), 'small')
  assert.equal(hitBox(rects, { x: 50, y: 50 }), 'big')
  assert.equal(hitBox(rects, { x: 150, y: 50 }), null)
  assert.equal(hitBox(rects, null), null)
})

test('panel titles', () => {
  const r = { from: 1, to: 2 }
  assert.equal(boxPanelTitle({ kind: 'drawn' }, r, fmt), 'Your box t1–t2')
  assert.equal(boxPanelTitle({ kind: 'drawn' }, r, fmt, { running: true }), 'Your box t1–now')
  const eb = { fromT: 5, toT: 6, k: 1, trade: { side: 'short' } }
  assert.equal(boxPanelTitle({ kind: 'engine' }, r, fmt, { engineBox: eb }), 'Engine box t5–t6 · trade 2 (short)')
  assert.equal(boxPanelTitle(null, null, fmt), '')
})

test('tradeText: closed and open', () => {
  assert.equal(tradeText({ trade: trade() }, fmt), 'short · entry 7655.75 at t300 · stop 7665.75 · TP 7645.75 · target +10.00 at t400')
  assert.equal(tradeText({ trade: trade({ reason: 'stop', pts: -10 }) }, fmt).endsWith('stop -10.00 at t400'), true)
  assert.equal(tradeText({ trade: trade(), open: true }, fmt).endsWith('TP 7645.75 · open'), true)
})

test('engineAtLine: the end minute and the first failed rule, no engine box', () => {
  const pe = { ev: { t: 9, verdict: 'NO ACTION', failed: [{ rule: 6, text: 'box still forming' }], box: { from: 9 } } }
  assert.equal(engineAtLine(pe, fmt), 'Engine at t9: NO ACTION — rule 6: box still forming')
  assert.equal(engineAtLine({ ev: { t: 9, verdict: 'SHORT', failed: [] } }, fmt), 'Engine at t9: SHORT')
})

test('escAction order: pick, selection, box', () => {
  assert.equal(escAction({ endArmed: true, selected: true, anchor: 1 }), 'end')
  assert.equal(escAction({ anchorArmed: true, selected: true, anchor: 1 }), 'start')
  assert.equal(escAction({ selected: true, anchor: 1 }), 'selection')
  assert.equal(escAction({ anchor: 1 }), 'box')
  assert.equal(escAction({}), null)
})

test('reviewBoxRecord: start, end, top, bottom; nulls without a box', () => {
  const pt = (t) => (t == null ? null : `pt${t}`)
  assert.deepEqual(reviewBoxRecord(60, { toT: 120, top: 10, bottom: 5 }, (t) => t, pt), { anchor_ts: 'pt60', box_end_ts: 'pt120', box_top: 10, box_bottom: 5 })
  assert.deepEqual(reviewBoxRecord(60, null, (t) => t, pt), { anchor_ts: 'pt60', box_end_ts: null, box_top: null, box_bottom: null })
})
