/**
 * CR-AV decision 5: the built bundle carries no direction-support badge strings.
 *
 * Run after `npm run build`:  cd react_today_setup && npm run build && npm test
 * (skips with a message when dist/ is absent).
 */
import { describe, it, expect } from 'vitest'
import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

const DIST_ASSETS = join(process.cwd(), 'dist', 'assets')
const BADGE_STRINGS = [
  'debit-to-target supported',
  'credit-fade supported',
  'low-confidence',
  'mixed pattern — no clear direction',
  'Direction signal:',
]

describe('built bundle — no badge strings (CR-AV)', () => {
  const built = existsSync(DIST_ASSETS)
  it.skipIf(!built)('dist/assets/*.js contains none of the badge phrases', () => {
    const files = readdirSync(DIST_ASSETS).filter(f => f.endsWith('.js'))
    expect(files.length).toBeGreaterThan(0)
    for (const f of files) {
      const text = readFileSync(join(DIST_ASSETS, f), 'utf8')
      for (const s of BADGE_STRINGS) {
        expect(text.includes(s), `${f} contains ${JSON.stringify(s)}`).toBe(false)
      }
    }
  })
  if (!built) console.warn('bundleBadgeStrings: dist/assets missing — run `npm run build` first')
})
