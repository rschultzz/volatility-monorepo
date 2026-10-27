// CR-CQ — Min |GEX| (B) coercion and load precedence. Run: npm --prefix react_price_preview run test:unit
import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  GEX_MIN_ABS_B_DEFAULT,
  coerceGexMinAbsB,
  parseGexMinAbsB,
  resolveInitialGexMinAbsB,
} from '../../src/gexMinAbsB.js'

test('default is 40', () => {
  assert.equal(GEX_MIN_ABS_B_DEFAULT, 40)
  assert.equal(coerceGexMinAbsB(undefined), 40)
})

test('blank / missing / invalid is "no value", never 0', () => {
  for (const v of [null, undefined, '', '   ', 'abc', NaN, Infinity, '1e999']) {
    assert.equal(parseGexMinAbsB(v), null, String(v))
    assert.equal(coerceGexMinAbsB(v), 40, String(v))
    assert.equal(coerceGexMinAbsB(v, 60), 60, String(v)) // the change handler passes prev: empty keeps it
  }
})

test('numbers are rounded and clamped 0–200; explicit 0 is kept', () => {
  assert.equal(coerceGexMinAbsB('0'), 0)
  assert.equal(coerceGexMinAbsB(0, 60), 0)
  assert.equal(coerceGexMinAbsB('60'), 60)
  assert.equal(coerceGexMinAbsB(' 25 '), 25)
  assert.equal(coerceGexMinAbsB(12.6), 13)
  assert.equal(coerceGexMinAbsB(-3), 0)
  assert.equal(coerceGexMinAbsB('500'), 200)
})

test('load precedence: v2 stored → legacy > 0 → URL param → 40', () => {
  assert.equal(resolveInitialGexMinAbsB(), 40)
  assert.equal(resolveInitialGexMinAbsB({ stored: null, legacyStored: null, urlParam: null }), 40)
  assert.equal(resolveInitialGexMinAbsB({ stored: '60', legacyStored: '25', urlParam: '10' }), 60)
  assert.equal(resolveInitialGexMinAbsB({ stored: '0', legacyStored: '25', urlParam: '10' }), 0)
  assert.equal(resolveInitialGexMinAbsB({ stored: '', legacyStored: '25', urlParam: '10' }), 25)
  assert.equal(resolveInitialGexMinAbsB({ stored: 'abc', legacyStored: '25' }), 25)
  // legacy 0 was written by the old code into empty storage — not a choice
  assert.equal(resolveInitialGexMinAbsB({ stored: null, legacyStored: '0' }), 40)
  assert.equal(resolveInitialGexMinAbsB({ stored: null, legacyStored: '0', urlParam: '10' }), 10)
  assert.equal(resolveInitialGexMinAbsB({ stored: null, legacyStored: null, urlParam: '' }), 40)
  assert.equal(resolveInitialGexMinAbsB({ stored: null, legacyStored: null, urlParam: '0.0' }), 0)
})
