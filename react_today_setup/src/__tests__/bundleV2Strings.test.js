/**
 * CR-AW decision 8: the built Setup v2 bundle carries no edge-ratio, badge or
 * "K=" strings. The v2 page is its own Vite entry (dist/setup-v2.html); the
 * test greps every chunk that entry references (script src + modulepreload).
 *
 * Run after `npm run build`:  cd react_today_setup && npm run build && npm test
 * (skips with a message when dist/setup-v2.html is absent).
 */
import { describe, it, expect } from 'vitest'
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

const DIST = join(process.cwd(), 'dist')
const ENTRY = join(DIST, 'setup-v2.html')
// Phrases banned in every chunk the v2 entry references (app + vendor).
const BANNED_EVERYWHERE = [
  'edge ×',
  'Edge ×',
  'edge_ratio',
  'debit-to-target supported',
  'credit-fade supported',
  'low-confidence',
  'Direction signal:',
]
// Bare words banned in the v2 app chunks only — the shared vendor chunk
// (React) legitimately contains "supported" in its own messages, and "K="
// can occur as a minified assignment, so the app chunks are checked for the
// string-literal forms.
const BANNED_IN_APP = ['supported', '"K=', "'K=", '`K=']
const isAppChunk = rel => /\/setupV2-[^/]+\.js$/.test(rel)

function referencedChunks(html) {
  const out = new Set()
  const re = /(?:src|href)="\/today-setup\/(assets\/[^"]+\.js)"/g
  let m
  while ((m = re.exec(html)) !== null) out.add(m[1])
  return [...out]
}

describe('built Setup v2 bundle — banned strings (CR-AW)', () => {
  const built = existsSync(ENTRY)
  it.skipIf(!built)('every chunk dist/setup-v2.html references is free of the banned strings', () => {
    const html = readFileSync(ENTRY, 'utf8')
    const chunks = referencedChunks(html)
    expect(chunks.length).toBeGreaterThan(0)
    expect(chunks.some(isAppChunk), 'no setupV2 app chunk referenced').toBe(true)
    for (const rel of chunks) {
      const text = readFileSync(join(DIST, rel), 'utf8')
      for (const s of BANNED_EVERYWHERE) {
        expect(text.includes(s), `${rel} contains ${JSON.stringify(s)}`).toBe(false)
      }
      if (isAppChunk(rel)) {
        for (const s of BANNED_IN_APP) {
          expect(text.includes(s), `${rel} contains ${JSON.stringify(s)}`).toBe(false)
        }
      }
    }
  })
  if (!built) console.warn('bundleV2Strings: dist/setup-v2.html missing — run `npm run build` first')
})
