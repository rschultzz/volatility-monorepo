// CR-DE — review comments that save, post-reveal notes, comment box, set filter
// (specs/CR-DE-review-comments.md). Every Save / Reveal is intercepted in the browser (B8), and the
// backend's review connections are read-only, so a missed intercept fails instead of writing.
//
// Servers (started outside Playwright):
//   backend   CR_SMOKE_REVIEW_READONLY=1 apps/web/.venv/bin/python -u scripts/cr_cc_smoke_backend.py   (port 8061)
//   frontend  npm --prefix react_price_preview run dev -- --port 5172 --strictPort      (CRDE_BRANCH_URL; the smoke backend's CORS allows :517x)
// then:      npx playwright test cr-de   (from react_price_preview/)
import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const API_BASE = process.env.CRCC_API_BASE || 'http://127.0.0.1:8061'
const BRANCH_URL = process.env.CRDE_BRANCH_URL || 'http://localhost:5172'
const ASSETS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../specs/assets/cr-de')
const SET = 'cr-cz:cp-ranges'
const MOD = process.platform === 'darwin' ? 'Meta' : 'Control'

async function api(request, p) {
  const r = await request.get(`${API_BASE}/api/setup-review${p}`)
  expect(r.ok()).toBeTruthy()
  return r.json()
}

async function pickCard(request) {
  const list = await api(request, `/candidates?set=${encodeURIComponent(SET)}&status=unreviewed`)
  expect(list.candidates.length).toBeGreaterThan(0)
  return (await api(request, `/candidates/${encodeURIComponent(list.candidates[0].id)}`)).candidate
}

const isPath = (id, suffix = '') => (url) =>
  new URL(url).pathname === `/api/setup-review/candidates/${encodeURIComponent(id)}${suffix}`

// A fake review store for one card: detail GETs, review POSTs and reveal POSTs all answer from it.
async function fakeCard(page, cand, { status = 200 } = {}) {
  const st = { cand: structuredClone(cand), posts: [], reveals: 0, nextId: 900001 }
  await page.route(isPath(cand.id), (route) => route.fulfill({
    status: 200, contentType: 'application/json', body: JSON.stringify({ ok: true, candidate: st.cand }),
  }))
  await page.route(isPath(cand.id, '/review'), async (route) => {
    const body = route.request().postDataJSON()
    st.posts.push(body)
    if (status !== 200) {
      return route.fulfill({ status, contentType: 'application/json', body: JSON.stringify({ ok: false, error: 'db', message: 'forced failure' }) })
    }
    const first = st.cand.reviews.find((r) => !r.post_reveal)
    const review = {
      id: st.nextId++, label: body.label || first?.label, reasons: body.reasons || [], comment: body.comment || null,
      reviewed_at: new Date(Date.now() + st.posts.length * 60_000).toISOString(), revealed_at: null,
      post_reveal: Boolean(st.cand.revealed),
    }
    st.cand.reviews = [...st.cand.reviews, review]
    st.cand.reviewed = true
    st.cand.n_reviews = st.cand.reviews.length
    if (!first) st.cand.label = review.label
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ ok: true, review }) })
  })
  await page.route(isPath(cand.id, '/reveal'), (route) => {
    st.reveals += 1
    st.cand.revealed = true
    st.cand.outcome = { pnl_pts: 4.25, result: 'target', exit_ts_pt: `${cand.session_date}T08:12:00`, exit_price: 7740 }
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ ok: true, outcome: st.cand.outcome }) })
  })
  return st
}

async function openApp(page, viewport, { storage = 'set' } = {}) {
  await page.setViewportSize(viewport)
  await page.addInitScript((storage) => {
    try {
      window.localStorage.setItem('ib-react-gex-min-abs-b', '40')
      window.localStorage.setItem('ib-react-gex-levels-mode', 'current')
      if (storage === 'set' && !window.sessionStorage.getItem('crde-init')) {
        window.localStorage.setItem('dash.review.set', JSON.stringify({ cr: '', set: 'cr-cz:cp-ranges' }))
      }
      window.sessionStorage.setItem('crde-init', '1')
    } catch (e) { /* ignore */ }
  }, storage)
  await page.goto(`${BRANCH_URL}/?trade_date=2026-10-23&interval=1min&api_base=${encodeURIComponent(API_BASE)}`)
  await page.locator('.tv-lightweight-charts').first().waitFor()
  await page.getByRole('button', { name: 'Show setup review panel' }).click()
}

async function openCard(page, c, viewport = { width: 1440, height: 900 }) {
  await openApp(page, viewport)
  await expect(page.getByTestId('review-panel').getByLabel('Set', { exact: true })).toHaveValue(SET, { timeout: 120_000 })
  await page.locator(`[data-testid="review-row"][data-candidate-id="${c.id}"]`).click()
  await expect(page.getByTestId('review-active')).toContainText(String(c.id))
}

async function expectInView(page, locator) {
  const vp = page.viewportSize()
  const box = await locator.boundingBox()
  const panel = await page.getByTestId('review-panel').boundingBox()
  expect(box, 'element rendered').not.toBeNull()
  expect(box.y).toBeGreaterThanOrEqual(panel.y)
  expect(box.y + box.height).toBeLessThanOrEqual(Math.min(vp.height, panel.y + panel.height) + 0.5)
}

async function expectUntouched(request, c) {
  const after = await api(request, `/candidates/${encodeURIComponent(c.id)}`)
  expect(after.candidate.reviews || []).toEqual(c.reviews || []) // nothing reached the DB
}

test('pre-reveal: label + multi-line comment are both sent; confirmation; label then locks', async ({ page, request }) => {
  const c = await pickCard(request)
  const st = await fakeCard(page, c)
  await openCard(page, c)
  const comment = 'range too short\nflow against at the top'
  await page.keyboard.press('t')
  await page.getByTestId('review-comment').fill(comment)
  await page.getByTestId('review-save').click()
  await expect.poll(() => st.posts.length).toBe(1)
  expect(st.posts[0]).toEqual({ label: 'take', reasons: [], comment })
  await expect(page.getByTestId('review-notice')).toContainText('Saved ✓')
  await expect(page.getByTestId('review-saved')).toContainText('range too short')
  // locked: Save disabled, label buttons disabled, keys ignored, the panel says so
  await expect(page.getByTestId('review-save')).toBeDisabled()
  await expect(page.getByTestId('review-label-skip')).toBeDisabled()
  await expect(page.getByTestId('review-mode-hint')).toContainText('Label locked')
  await page.getByTestId('review-comment').blur()
  await page.keyboard.press('s')
  await expect(page.getByTestId('review-label-take')).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByTestId('review-label-skip')).toHaveAttribute('aria-pressed', 'false')
  expect(st.posts.length).toBe(1)
  await expectUntouched(request, c)
})

test('post-reveal: two notes send two inserts, each confirmed, history newest first', async ({ page, request }) => {
  const c = await pickCard(request)
  const st = await fakeCard(page, c)
  await openCard(page, c)
  await page.keyboard.press('s')
  await page.getByTestId('review-save').click()
  await expect(page.getByTestId('review-notice')).toContainText('Saved ✓')
  // a comment typed while locked is kept for the first note
  await page.getByTestId('review-comment').fill('first note')
  await page.getByTestId('review-reveal').click()
  await expect(page.getByTestId('review-outcome')).toBeVisible()
  await expect(page.getByTestId('review-comment')).toHaveValue('first note')
  await expect(page.getByTestId('review-save')).toHaveText('Save note')
  await expect(page.getByTestId('review-save')).toBeEnabled()
  await page.getByTestId('review-save').click()
  await expect.poll(() => st.posts.length).toBe(2)
  await expect(page.getByTestId('review-notice')).toContainText('Note saved ✓')
  await expect(page.getByTestId('review-comment')).toHaveValue('')

  // second note: a chip, a changed label, saved with Cmd/Ctrl+Enter
  await page.getByTestId('review-reasons').getByRole('button', { name: 'trend day', exact: true }).click()
  await page.getByTestId('review-label-take').click()
  await page.getByTestId('review-comment').fill('second note\nline two')
  await page.getByTestId('review-comment').press(`${MOD}+Enter`)
  await expect.poll(() => st.posts.length).toBe(3)
  expect(st.posts[0]).toEqual({ label: 'skip', reasons: [], comment: '' })
  expect(st.posts[1]).toEqual({ reasons: [], comment: 'first note' })            // R1: no label → API copies
  expect(st.posts[2]).toEqual({ label: 'take', reasons: ['trend day'], comment: 'second note\nline two' })
  expect(st.reveals).toBe(1)
  await expect(page.getByTestId('review-notice')).toContainText('Note saved ✓')

  const notes = page.getByTestId('review-note')
  await expect(notes).toHaveCount(2)
  await expect(notes.nth(0)).toContainText('second note')
  await expect(notes.nth(0)).toContainText('label → TAKE')
  await expect(notes.nth(0)).toContainText('trend day')
  await expect(notes.nth(0)).toContainText('PT')
  await expect(notes.nth(1)).toContainText('first note')
  await expect(notes.nth(1)).not.toContainText('label →')
  await expect(page.getByTestId('review-saved')).toContainText('SKIP')            // the blind label is unchanged
  fs.mkdirSync(ASSETS, { recursive: true })
  await page.getByTestId('review-panel').screenshot({ path: path.join(ASSETS, 'notes-1440x900.png') })
  await expectUntouched(request, c)
})

test('a failed save shows an error and keeps the text', async ({ page, request }) => {
  const c = await pickCard(request)
  const st = await fakeCard(page, c, { status: 500 })
  await openCard(page, c)
  await page.keyboard.press('u')
  await page.getByTestId('review-comment').fill('keep me')
  await page.getByTestId('review-save').click()
  await expect.poll(() => st.posts.length).toBe(1)
  await expect(page.getByTestId('review-action-error')).toContainText('Not saved: forced failure')
  await expect(page.getByTestId('review-notice')).toHaveCount(0)
  await expect(page.getByTestId('review-comment')).toHaveValue('keep me')
  await expect(page.getByTestId('review-save')).toBeEnabled()
  await expectUntouched(request, c)
})

test('chip list has no smoke-test reason', async ({ page, request }) => {
  const list = await api(request, '/candidates')
  expect(list.reasons).not.toContain('fixture smoke reason')
  const c = await pickCard(request)
  await fakeCard(page, c)
  await openCard(page, c)
  const chips = page.getByTestId('review-reasons').getByRole('button')
  await expect(chips).toHaveCount(list.reasons.length)
  await expect(page.getByTestId('review-reasons')).not.toContainText('fixture smoke reason')
  await expect(page.getByTestId('review-reasons')).not.toContainText('smoke')
})

test('set filter: defaults to the latest-loaded set, then remembers the choice', async ({ page, request }) => {
  const { sets } = await api(request, '/candidates')
  const latest = sets.reduce((a, b) => (b.loaded_at > a.loaded_at ? b : a))
  const other = sets.find((s) => s.set_name !== latest.set_name && !s.set_name.endsWith(':fixture'))
  await openApp(page, { width: 1440, height: 900 }, { storage: 'none' })
  const setSel = page.getByTestId('review-panel').getByLabel('Set', { exact: true })
  const crSel = page.getByTestId('review-panel').getByLabel('CR', { exact: true })
  await expect(setSel).toHaveValue(latest.set_name, { timeout: 120_000 })
  await expect(crSel).toHaveValue('')                    // CR stays "all": every set stays listed
  await expect(setSel.locator('option')).toHaveCount(sets.length + 1)

  await crSel.selectOption(other.cr_id)
  await setSel.selectOption(other.set_name)
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem('dash.review.set')))).toEqual({ cr: other.cr_id, set: other.set_name })
  await page.reload()
  await page.locator('.tv-lightweight-charts').first().waitFor()
  await page.getByRole('button', { name: 'Show setup review panel' }).click()
  await expect(setSel).toHaveValue(other.set_name, { timeout: 120_000 })
  await expect(crSel).toHaveValue(other.cr_id)
})

for (const viewport of [{ width: 1440, height: 900 }, { width: 900, height: 600 }]) {
  test(`comment box ≥ 5 lines, resizable; actions pinned at ${viewport.width}×${viewport.height}`, async ({ page, request }) => {
    const c = await pickCard(request)
    await fakeCard(page, c)
    await openCard(page, c, viewport)
    const box = page.getByTestId('review-comment')
    const m = await box.evaluate((el) => {
      const cs = getComputedStyle(el)
      return { h: el.clientHeight, lh: parseFloat(cs.lineHeight), pad: parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom), resize: cs.resize, rows: el.rows, ws: cs.whiteSpace }
    })
    expect(m.rows).toBe(5)
    expect(m.h - m.pad).toBeGreaterThanOrEqual(5 * m.lh - 1)
    expect(m.resize).toBe('vertical')
    expect(m.ws).toBe('pre-wrap')
    for (const id of ['review-save', 'review-reveal', 'review-label-take', 'review-label-skip', 'review-label-unsure', 'review-comment']) {
      await expectInView(page, page.getByTestId(id))
    }
    fs.mkdirSync(ASSETS, { recursive: true })
    await page.screenshot({ path: path.join(ASSETS, `card-${viewport.width}x${viewport.height}.png`) })
  })
}
