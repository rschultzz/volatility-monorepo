// CR-CQ: the price chart's Min |GEX| (B) filter — default, coercion and load precedence.
// Kept free of React so tests/unit can import it directly.

export const GEX_MIN_ABS_B_DEFAULT = 40
export const GEX_MIN_ABS_B_STORAGE_KEY = 'ib-react-gex-min-abs-b-v2'
// Written by the pre-CR-CQ code, which stored "0" whenever storage started empty — a 0 there is not a choice.
export const GEX_MIN_ABS_B_LEGACY_STORAGE_KEY = 'ib-react-gex-min-abs-b'

// null for null / undefined / blank / non-numeric (Number('') and Number(null) are 0, not "no value").
export function parseGexMinAbsB(value) {
  if (value == null) return null
  if (typeof value === 'string' && value.trim() === '') return null
  const num = Number(value)
  if (!Number.isFinite(num)) return null
  return Math.max(0, Math.min(200, Math.round(num)))
}

export function coerceGexMinAbsB(value, fallback = GEX_MIN_ABS_B_DEFAULT) {
  const num = parseGexMinAbsB(value)
  return num == null ? fallback : num
}

// Load order: v2 stored value (0 allowed) → legacy stored value (> 0 only) → URL param → default.
export function resolveInitialGexMinAbsB({ stored, legacyStored, urlParam } = {}) {
  const fromStored = parseGexMinAbsB(stored)
  if (fromStored != null) return fromStored
  const fromLegacy = parseGexMinAbsB(legacyStored)
  if (fromLegacy != null && fromLegacy > 0) return fromLegacy
  return coerceGexMinAbsB(urlParam, GEX_MIN_ABS_B_DEFAULT)
}
