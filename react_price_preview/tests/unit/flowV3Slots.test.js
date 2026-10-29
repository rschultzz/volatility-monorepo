// CR-DP item 7 — adjustable Flow v3 slots: stored fractions → bounds → margins; divider drags; 24 px minimum.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  SLOT_MIN_PX, boundsFromFracs, boundsToMargins, dragSlotFracs, marginsToBounds, paneSlotMargins, slotGapPx, slotOrder,
} from '../../src/flowV3.js'

const close = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} vs ${b}`)
const fracsOf = (bounds) => {
  const h = bounds.map((b) => b.bottom - b.top)
  const s = h.reduce((x, y) => x + y, 0)
  return h.map((x) => x / s)
}

test('the default layout written as fractions reproduces paneSlotMargins (rolling off and on)', () => {
  for (const [H, X] of [[160, 0], [300, 0], [211, 51], [400, 97]]) {
    const order = slotOrder(X > 0)
    const def = marginsToBounds(H, paneSlotMargins(H, X), order)
    const back = boundsFromFracs(H, fracsOf(def), slotGapPx(H, X))
    def.forEach((b, i) => { close(back[i].top, b.top, 1e-6); close(back[i].bottom, b.bottom, 1e-6) })
    const m = boundsToMargins(H, back, order)
    const want = paneSlotMargins(H, X)
    for (const k of order) { close(m[k].top, want[k].top, 1e-9); close(m[k].bottom, want[k].bottom, 1e-9) }
  }
})

test('a divider drag moves one boundary; both slots stay >= 24 px; the pane height is unchanged', () => {
  const H = 160
  const order = slotOrder(false)
  const G = slotGapPx(H, 0)
  const def = marginsToBounds(H, paneSlotMargins(H, 0), order)
  const fr = dragSlotFracs(def, 0, 20)
  const b = boundsFromFracs(H, fr, G)
  close(b[0].bottom - b[0].top, def[0].bottom - def[0].top + 20, 1e-6)
  close(b[1].bottom - b[1].top, def[1].bottom - def[1].top - 20, 1e-6)
  close(b[2].bottom - b[2].top, def[2].bottom - def[2].top, 1e-6)
  close(b[2].bottom + G / 2, H, 1e-6) // same bottom pad: the total height is unchanged
  // clamp at the minimum
  const fr2 = dragSlotFracs(def, 0, 1000)
  const b2 = boundsFromFracs(H, fr2, G)
  close(b2[1].bottom - b2[1].top, SLOT_MIN_PX, 1e-6)
  const fr3 = dragSlotFracs(def, 1, -1000)
  const b3 = boundsFromFracs(H, fr3, G)
  close(b3[1].bottom - b3[1].top, SLOT_MIN_PX, 1e-6)
})

test('stored fractions keep their proportions when the pane is resized; tiny slots are raised to the minimum', () => {
  const b = boundsFromFracs(300, [0.5, 0.3, 0.2], slotGapPx(300, 0))
  const h = b.map((x) => x.bottom - x.top)
  close(h[0] / h[1], 0.5 / 0.3, 1e-9)
  const tiny = boundsFromFracs(160, [0.97, 0.02, 0.01], slotGapPx(160, 0))
  for (const x of tiny) assert.ok(x.bottom - x.top >= SLOT_MIN_PX - 1e-9)
  assert.equal(boundsFromFracs(160, null, 6), null)
  assert.equal(boundsFromFracs(160, [1, 0, 1], 6), null)
})

test('zoomRange: about 0 when 0 is in range, else about the middle; 1 = unchanged', async () => {
  const { zoomRange } = await import('../../src/flowV3.js')
  assert.deepEqual(zoomRange({ minValue: 0, maxValue: 100 }, 2), { minValue: 0, maxValue: 50 })
  assert.deepEqual(zoomRange({ minValue: -0.1, maxValue: 0.1 }, 2), { minValue: -0.05, maxValue: 0.05 })
  assert.deepEqual(zoomRange({ minValue: 1000, maxValue: 3000 }, 0.5), { minValue: 0, maxValue: 4000 })
  const r = { minValue: -5, maxValue: 5 }
  assert.equal(zoomRange(r, 1), r)
})
