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
