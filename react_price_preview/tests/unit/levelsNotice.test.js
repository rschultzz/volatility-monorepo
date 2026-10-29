// CR-DK — the price chart's notice when Levels: Corrected has no lines. Run: npm --prefix react_price_preview run test:unit
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { correctedLevelsNotice, levelsReasonText } from '../../src/gexLevelsMode.js'

const corrected = (notes, extra = {}) => ({ levels: 'corrected', gex_enabled: true, trade_date: '2024-03-15', levels_notes: notes, ...extra })

test('reason text from the API reason', () => {
  assert.equal(levelsReasonText('pre_forward_era'), 'pre-forward era')
  assert.equal(levelsReasonText('no_es_snapshot'), 'no ES snapshot')
  assert.equal(levelsReasonText('no_rows'), 'no rows')
  assert.equal(levelsReasonText('error: boom'), 'error')
})

test('Corrected with a reason and no lines shows the notice', () => {
  assert.equal(
    correctedLevelsNotice(corrected([{ session_date: '2024-03-15', reason: 'pre_forward_era' }]), false),
    'No corrected GEX levels for this date (pre-forward era) — switch Levels to Current'
  )
})

test('the centre date\'s note wins', () => {
  const p = corrected([{ session_date: '2025-11-28', reason: 'no_rows' }, { session_date: '2025-12-01', reason: 'no_es_snapshot' }],
    { trade_date: '2025-12-01' })
  assert.match(correctedLevelsNotice(p, false), /\(no ES snapshot\)/)
})

test('nothing when there are lines, no note, Current, or GEX off', () => {
  const note = [{ session_date: '2024-03-15', reason: 'pre_forward_era' }]
  assert.equal(correctedLevelsNotice(corrected(note), true), null)
  assert.equal(correctedLevelsNotice(corrected([]), false), null)
  assert.equal(correctedLevelsNotice({ gex_enabled: true, trade_date: '2024-03-15' }, false), null)   // Current: no keys
  assert.equal(correctedLevelsNotice(corrected(note, { gex_enabled: false }), false), null)
  assert.equal(correctedLevelsNotice(null, false), null)
})
