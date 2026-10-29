// CR-DH — GEX Levels placement: 'corrected' (CR-BT, default) or 'current' (discounted_level).
// The CR-BT key ('ib-react-gex-levels-mode') was saved on every mount, so it holds 'current' in any
// browser that loaded the page before CR-DH; it is ignored, and the choice lives under a v2 key.
export const GEX_LEVELS_MODE_STORAGE_KEY = 'ib-react-gex-levels-mode-v2'
export const GEX_LEVELS_MODE_DEFAULT = 'corrected'

export function coerceGexLevelsMode(value) {
  return value === 'current' ? 'current' : 'corrected'
}

// `storage` may be a Storage or a function returning one (reading window.localStorage can throw).
export function resolveInitialGexLevelsMode(storage) {
  try {
    const s = typeof storage === 'function' ? storage() : storage
    return coerceGexLevelsMode(s.getItem(GEX_LEVELS_MODE_STORAGE_KEY))
  } catch (err) {
    return GEX_LEVELS_MODE_DEFAULT
  }
}

// CR-DK — the price chart's notice when Levels: Corrected draws no lines for the centre date. `payload` is the centre
// /api/ironbeam/bars response; its `levels_notes` ({session_date, reason}) is only present on Corrected responses.
const LEVELS_REASON_TEXT = { pre_forward_era: 'pre-forward era', no_es_snapshot: 'no ES snapshot', no_rows: 'no rows' }

export function levelsReasonText(reason) {
  const r = String(reason || '')
  return LEVELS_REASON_TEXT[r] || (r.startsWith('error') ? 'error' : r)
}

export function correctedLevelsNotice(payload, hasLines) {
  if (hasLines || payload?.levels !== 'corrected' || !payload?.gex_enabled) return null
  const notes = Array.isArray(payload.levels_notes) ? payload.levels_notes.filter((n) => n?.reason) : []
  const day = payload.effective_trade_date || payload.trade_date
  const note = notes.find((n) => n.session_date === day) || notes[0]
  if (!note) return null
  return `No corrected GEX levels for this date (${levelsReasonText(note.reason)}) — switch Levels to Current`
}
