// CR-DH — Levels default and v2 key. Run: npm --prefix react_price_preview run test:unit
import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  GEX_LEVELS_MODE_DEFAULT,
  GEX_LEVELS_MODE_STORAGE_KEY,
  coerceGexLevelsMode,
  resolveInitialGexLevelsMode,
} from '../../src/gexLevelsMode.js'

const store = (entries) => ({ getItem: (k) => (k in entries ? entries[k] : null) })

test('default is corrected under a v2 key', () => {
  assert.equal(GEX_LEVELS_MODE_DEFAULT, 'corrected')
  assert.equal(GEX_LEVELS_MODE_STORAGE_KEY, 'ib-react-gex-levels-mode-v2')
  assert.equal(resolveInitialGexLevelsMode(store({})), 'corrected')
})

test('the old key is ignored', () => {
  assert.equal(resolveInitialGexLevelsMode(store({ 'ib-react-gex-levels-mode': 'current' })), 'corrected')
})

test('a stored choice is kept', () => {
  assert.equal(resolveInitialGexLevelsMode(store({ 'ib-react-gex-levels-mode-v2': 'current' })), 'current')
  assert.equal(resolveInitialGexLevelsMode(store({ 'ib-react-gex-levels-mode-v2': 'corrected' })), 'corrected')
})

test('unknown values and storage errors fall back to corrected', () => {
  for (const v of [null, '', 'foo', 'CURRENT']) assert.equal(coerceGexLevelsMode(v), 'corrected', String(v))
  assert.equal(resolveInitialGexLevelsMode({ getItem() { throw new Error('blocked') } }), 'corrected')
  assert.equal(resolveInitialGexLevelsMode(() => { throw new Error('no localStorage') }), 'corrected')
  assert.equal(resolveInitialGexLevelsMode(() => store({ 'ib-react-gex-levels-mode-v2': 'current' })), 'current')
})
