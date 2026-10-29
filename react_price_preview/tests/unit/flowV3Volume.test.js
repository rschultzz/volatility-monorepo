// CR-DM items 1+2 — volume per minute and the rolling slot layout. Run: npm --prefix react_price_preview run test:unit
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { DEFAULT_V3_SETTINGS, FLOW_V3_CHROME_PX, minuteFlow, paneSlotMargins, rollingExtraHeight } from '../../src/flowV3.js'

test('defaults: rolling net delta off, live effort min move 4', () => {
  assert.equal(DEFAULT_V3_SETTINGS.rollingOn, false)
  assert.equal(DEFAULT_V3_SETTINGS.liveMinMove, 4)
  assert.equal(DEFAULT_V3_SETTINGS.rollingMin, 10)
})

test('minute hover: volume, delta and the raw delta share from the same row', () => {
  assert.deepEqual(minuteFlow({ vol: 1000, delta: -250 }), { vol: 1000, delta: -250, share: -0.25 })
  assert.deepEqual(minuteFlow({ vol: 0, delta: 0 }), { vol: 0, delta: 0, share: null })
  assert.deepEqual(minuteFlow({ vol: null, delta: 5 }), { vol: null, delta: 5, share: null })
})

test('rolling off: the CR-DC margins exactly (volume takes the rolling slot)', () => {
  const m = paneSlotMargins(160, 0)
  assert.deepEqual(m, { cvd: { top: 0.04, bottom: 0.62 }, mid: { top: 0.42, bottom: 0.30 }, low: null, delta: { top: 0.74, bottom: 0.02 } })
})

test('rolling on: the pane grows by one slot and every slot keeps its pixel height', () => {
  const base = 160
  const extra = rollingExtraHeight(base + FLOW_V3_CHROME_PX)
  assert.equal(extra, 51) // round(0.32 × 160)
  const H = base + extra
  const px = (s) => [s.top * H, (1 - s.bottom) * H]
  const m = paneSlotMargins(H, extra)
  const near = (a, b) => assert.ok(Math.abs(a - b) < 0.6, `${a} vs ${b}`)
  const [c0, c1] = px(m.cvd); near(c0, 0.04 * base); near(c1, 0.38 * base)
  const [r0, r1] = px(m.mid); near(r0, 0.42 * base); near(r1, 0.70 * base)
  const [v0, v1] = px(m.low); near(v1 - v0, 0.28 * base) // volume: one slot below rolling
  near(v0 - r1, 0.04 * base)
  const [d0, d1] = px(m.delta); near(d1 - d0, 0.24 * base); near(d0 - v1, 0.04 * base); near(H - d1, 0.02 * base)
})

// CR-DM item 3 — the live effort label.
import { LIVE_EFFORT_ID, liveEffortLabel } from '../../src/flowV3.js'

test('live effort label: text, min move and previous-leg gate', () => {
  const up = { dir: 'up', pts: 6.5, end_time: 1, ratio: 1.234, prev_cpp: 900 }
  assert.equal(LIVE_EFFORT_ID, 'cr-dm-live-effort')
  assert.equal(liveEffortLabel(up, 4).text, '↑ +6.5 pts so far · 1.2× vs last ↑')
  assert.equal(liveEffortLabel(up, 4).up, true)
  assert.equal(liveEffortLabel(up, 7), null) // below M
  assert.equal(liveEffortLabel({ ...up, prev_cpp: null, ratio: null }, 4), null) // no previous leg the same way
  assert.equal(liveEffortLabel({ dir: 'down', pts: -4, end_time: 1, ratio: 0.5, prev_cpp: 10 }, 4).text, '↓ -4.0 pts so far · 0.5× vs last ↓') // exactly M
  assert.equal(liveEffortLabel({ ...up, ratio: null, prev_cpp: 0 }, 4).text, '↑ +6.5 pts so far · – vs last ↑') // previous cpp 0: no ratio
  assert.equal(liveEffortLabel(null, 4), null)
})

test('CR-DN item 6: volumeCap = the 98th percentile (nearest rank) of the positive volumes', async () => {
  const { volumeCap } = await import('../../src/flowV3.js')
  const rows = Array.from({ length: 100 }, (_, i) => ({ vol: i + 1 })) // 1..100
  assert.equal(volumeCap(rows), 98)
  assert.equal(volumeCap([...rows, { vol: 57000 }]), 99) // one huge minute doesn't move the cap much
  assert.equal(volumeCap([{ vol: 0 }, {}, { vol: null }]), null)
  assert.equal(volumeCap([{ vol: 5 }]), 5)
})
