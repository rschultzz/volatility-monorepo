// CR-DQ items 2 / 3 — box panel sections (same fields as CR-DP's popup) and the panel's remembered place.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { BOX_PANEL_INSET, boxPanelOffset, boxPanelPlace, boxPanelSections, boxPopupLines } from '../../src/flowV3.js'

const hhmm = (t) => `t${t}`
const edge = (price) => ({ price, contracts: 100, opposite: 50, heldMin: 3, brokeAt: null })
const frozen = {
  from: 1, to: 4, readout: { delta: -239, vol: 10000, dPrice: -2, minutes: 4, cpp: 5000 },
  top: edge(10), bottom: edge(5), probes: [{ dir: 'up', at: 2 }], pending: null, ended: { kind: 'broke', dir: 'down', at: 5 }, dropped: 2,
}
const current = {
  from: 5, to: 9, readout: { delta: 100, vol: 5000, dPrice: 1, minutes: 4, cpp: 5000 },
  top: edge(6), bottom: edge(3), probes: [], pending: { dir: 'up', at: 9 }, ended: { kind: 'current' }, dropped: 2,
}

test('boxPanelSections: oldest first, current last; one-line summaries; stats = boxPopupLines', () => {
  const p = boxPanelSections([{ box: frozen }, { box: current }], hhmm)
  assert.equal(p.sections.length, 2)
  assert.deepEqual(p.sections.map((s) => s.current), [false, true])
  assert.equal(p.sections[0].summary, 'Box t1–t5 · broke ↓ t5 · share -2.39 %')
  assert.equal(p.sections[1].summary, 'Box t5–t9 · current')
  assert.deepEqual(p.sections[1].lines, boxPopupLines(current, hhmm))
  assert.equal(p.sections[0].probes, '1 probe')
  assert.equal(p.sections[1].probes, '0 probes · testing break t9 ↑')
  assert.equal(p.droppedText, '2 older boxes not shown')
  assert.equal(boxPanelSections([{ box: { ...current, dropped: 0 } }], hhmm).droppedText, '')
  const ended = boxPanelSections([{ box: { ...current, ended: { kind: 'end', at: 9 } } }], hhmm)
  assert.equal(ended.sections[0].summary, 'Box t5–t9 · set end t9')
  assert.equal(ended.sections[0].probes, '0 probes · testing break t9 ↑ (pending at the end)')
})

test('boxPanelPlace: default bottom-left inset; stored offset from left / bottom; clamped inside the plot', () => {
  const pane = { left: 10, top: 20, width: 500, height: 300 }
  assert.deepEqual(boxPanelPlace(pane, null, 200, 100), { left: 10 + BOX_PANEL_INSET, top: 20 + 300 - BOX_PANEL_INSET - 100 })
  assert.deepEqual(boxPanelPlace(pane, { left: 50, bottom: 40 }, 200, 100), { left: 60, top: 180 })
  assert.deepEqual(boxPanelPlace(pane, { left: 9999, bottom: 9999 }, 200, 100), { left: 310, top: 20 })
  assert.deepEqual(boxPanelPlace(pane, { left: -50, bottom: -50 }, 200, 100), { left: 10, top: 220 })
  // a shorter plot keeps the panel's distance from the bottom
  assert.deepEqual(boxPanelPlace({ ...pane, height: 250 }, { left: 50, bottom: 40 }, 200, 100), { left: 60, top: 130 })
  assert.deepEqual(boxPanelOffset(pane, 60, 180, 100), { left: 50, bottom: 40 })
  assert.equal(boxPanelPlace(null, null, 1, 1), null)
})
