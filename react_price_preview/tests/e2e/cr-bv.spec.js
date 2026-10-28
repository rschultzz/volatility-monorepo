// CR-BV step 5 — landscape panel and price chart in one space (specs/CR-BV-landscape-chart-one-space.md).
//
// L4 pixel test: at two viewport sizes, for three prices (10 / 50 / 90 % of the visible range), the
//   panel's rendered total curve and the chart's series.priceToCoordinate land within 1 px.
// L5 crosshair test: hovering the chart puts the panel's line at the chart crosshair (the cursor y);
//   hovering the panel sets the chart crosshair at that price, hides the chart's vertical line and
//   floating tooltip; leaving both clears everything.
// Screenshots of 2026-10-06 and 2026-10-23, Current and Corrected, go to specs/assets/cr-bv/.
//
// The chart's lightweight-charts series is reached through React's fiber tree (test-only).
import { test, expect } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const API_BASE = process.env.CRBV_API_BASE || 'http://127.0.0.1:8060'
const ASSETS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../specs/assets/cr-bv')
const DATES = ['2026-10-06', '2026-10-23']
const SIZES = [
  { width: 1440, height: 900 },
  { width: 1920, height: 1080 },
]
const MODES = ['current', 'corrected']

async function openChart(page, { date, mode, size }) {
  await page.setViewportSize(size)
  await page.addInitScript((m) => {
    try {
      window.localStorage.setItem('ib-react-gex-levels-mode', m)
      window.localStorage.setItem('ib-react-gex-levels-mode-v2', m) // CR-DH key
    } catch (e) { /* ignore */ }
  }, mode)
  await page.goto(`/?trade_date=${date}&api_base=${encodeURIComponent(API_BASE)}&interval=1min`)
  await page.locator('.tv-lightweight-charts').first().waitFor()
  await page.getByRole('button', { name: 'Show GEX landscape panel' }).click()
  await page.waitForFunction(() => {
    const svgs = [...document.querySelectorAll('svg')]
    return svgs.some((s) => s.querySelector('polyline[stroke="#f1f5f9"]'))
  }, null, { timeout: 60_000 })
  await page.waitForTimeout(500)
  // helpers in the page
  await page.evaluate(() => {
    const chartEl = document.querySelector('.tv-lightweight-charts')
    const key = Object.keys(chartEl.parentElement).find((k) => k.startsWith('__reactFiber'))
    const find = (pred) => {
      let f = chartEl.parentElement[key]
      for (let d = 0; f && d < 30; d += 1, f = f.return) {
        for (let h = f.memoizedState; h; h = h.next) {
          const v = h.memoizedState
          const c = v && typeof v === 'object' && 'current' in v ? v.current : null
          if (c && pred(c)) return c
        }
      }
      return null
    }
    const chart = find((c) => typeof c.timeScale === 'function')
    const series = find((c) => typeof c.priceToCoordinate === 'function' && c.seriesType() === 'Candlestick')
    const calls = []
    const set = chart.setCrosshairPosition.bind(chart)
    chart.setCrosshairPosition = (p, t, s) => { calls.push(p); return set(p, t, s) }
    const landscapeProps = () => {
      const rootEl = document.getElementById('root')
      const ck = Object.keys(rootEl).find((k) => k.startsWith('__reactContainer'))
      const stack = [rootEl[ck].stateNode.current]
      while (stack.length) {
        const f = stack.pop()
        if (!f) continue
        if (f.type && f.type.name === 'GexLandscape') return f.memoizedProps
        if (f.sibling) stack.push(f.sibling)
        if (f.child) stack.push(f.child)
      }
      return null
    }
    const paneTop = () => chartEl.querySelectorAll('td')[1].getBoundingClientRect().top
    const panelSvg = () => {
      const right = chartEl.querySelectorAll('td')[1].getBoundingClientRect().right
      return [...document.querySelectorAll('svg')].find((s) => {
        const r = s.getBoundingClientRect()
        return r.width > 200 && r.left > right - 5
      })
    }
    window.__crbv = { chart, series, calls, landscapeProps, paneTop, panelSvg }
  })
  // The panel follows the chart's rAF poll one frame behind; measure only once the
  // published window equals the chart's live window on two consecutive frames.
  await page.waitForFunction(() => new Promise((resolve) => {
    const settled = () => {
      const { series, landscapeProps } = window.__crbv
      const v = landscapeProps()?.visiblePriceRange
      return v && Math.abs(v.priceTop - series.coordinateToPrice(0)) < 1e-6
    }
    requestAnimationFrame(() => {
      const a = settled()
      requestAnimationFrame(() => resolve(a && settled()))
    })
  }), null, { timeout: 30_000, polling: 250 })
}

for (const size of SIZES) {
  for (const date of DATES) {
    for (const mode of MODES) {
      test(`L4 pixel alignment ${date} ${mode} ${size.width}x${size.height}`, async ({ page }) => {
        await openChart(page, { date, mode, size })
        const res = await page.evaluate(() => {
          const { series, landscapeProps, paneTop, panelSvg } = window.__crbv
          const props = landscapeProps()
          const land = props.data.landscape
          const svg = panelSvg()
          const svgTop = svg.getBoundingClientRect().top
          const pts = svg.querySelector('polyline[stroke="#f1f5f9"]').getAttribute('points')
            .split(' ').map((t) => Number(t.split(',')[1]))
          const { priceTop, priceBot } = props.visiblePriceRange
          const lo = Math.max(priceBot, land[0].price)
          const hi = Math.min(priceTop, land[land.length - 1].price)
          return {
            levels: props.data.levels || 'current',
            // panel price-axis labels: x = PAD.left − 5, anchored end (the $gamma ticks are centred)
            ticks: [...svg.querySelectorAll('text[text-anchor="end"]')]
              .filter((t) => t.getAttribute('x') === '41' && /^\d{4}$/.test(t.textContent.trim())).length,
            rows: [0.1, 0.5, 0.9].map((q) => {
              const want = hi - q * (hi - lo)
              let i = 0
              let best = Infinity
              land.forEach((r, j) => { const dd = Math.abs(r.price - want); if (dd < best) { best = dd; i = j } })
              const price = land[i].price
              return { price, yPanel: svgTop + pts[i], yChart: paneTop() + series.priceToCoordinate(price) }
            }),
          }
        })
        console.log(`L4 ${date} ${mode} ${size.width}x${size.height} ` + res.rows
          .map((r) => `${r.price}: panel ${r.yPanel.toFixed(2)} chart ${r.yChart.toFixed(2)} err ${(r.yPanel - r.yChart).toFixed(2)}`)
          .join(' | '))
        expect(res.levels).toBe(mode)
        expect(res.ticks).toBe(0) // L7: panel price ticks hidden when synced
        for (const r of res.rows) {
          expect(Math.abs(r.yPanel - r.yChart), `price ${r.price}`).toBeLessThanOrEqual(1)
        }
        if (size.width === 1440) {
          // park the cursor off both surfaces (blank area under the panel) and let panels settle
          await page.mouse.move(size.width - 150, size.height - 20)
          await page.waitForTimeout(2500)
          await page.screenshot({ path: path.join(ASSETS, `${date}-${mode}.png`) })
        }
      })
    }
  }
}

test('L5 crosshair across chart and panel (2026-10-23, 1440x900)', async ({ page }) => {
  await openChart(page, { date: '2026-10-23', mode: 'current', size: SIZES[0] })
  const state = () => page.evaluate(() => {
    const svg = window.__crbv.panelSvg()
    const line = [...svg.querySelectorAll('line')].find((l) => l.getAttribute('stroke') === 'rgba(191, 219, 254, 0.45)')
    const tip = [...document.querySelectorAll('div')].find((d) => d.textContent.startsWith('Cursor') && d.children.length < 6)
    const readout = [...svg.querySelectorAll('text')].find((t) => t.getAttribute('fill') === '#dbeafe')
    const { calls, series, paneTop, chart } = window.__crbv
    const last = calls[calls.length - 1]
    return {
      panelLineY: line ? svg.getBoundingClientRect().top + Number(line.getAttribute('y1')) : null,
      readout: readout ? readout.textContent : null,
      tooltipShown: tip ? getComputedStyle(tip).opacity !== '0' && getComputedStyle(tip).display !== 'none' : false,
      vertVisible: chart.options().crosshair.vertLine.visible,
      lastSetPrice: last ?? null,
      lastSetY: last != null ? paneTop() + series.priceToCoordinate(last) : null,
      nCalls: calls.length,
    }
  })

  // chart → panel
  await page.mouse.move(400, 270)
  await page.waitForTimeout(300)
  let s = await state()
  expect(Math.abs(s.panelLineY - 270)).toBeLessThanOrEqual(1)
  expect(s.readout).toMatch(/^[+−]\d+\.\dB$/)
  expect(s.nCalls).toBe(0)
  console.log(`L5 chart→panel: cursor y 270, panel line y ${s.panelLineY.toFixed(2)}, readout ${s.readout}`)

  // panel → chart
  const box = await page.evaluate(() => {
    const r = window.__crbv.panelSvg().getBoundingClientRect()
    return { x: r.left + r.width / 2, y: r.top + r.height * 0.6 }
  })
  await page.mouse.move(box.x, box.y, { steps: 1 })
  await page.waitForTimeout(300)
  s = await state()
  expect(s.nCalls).toBeGreaterThan(0)
  expect(Math.abs(s.lastSetY - box.y)).toBeLessThanOrEqual(1)
  expect(Math.abs(s.panelLineY - box.y)).toBeLessThanOrEqual(1)
  expect(s.vertVisible).toBe(false)
  expect(s.tooltipShown).toBe(false)
  console.log(`L5 panel→chart: cursor y ${box.y.toFixed(2)}, chart crosshair ${s.lastSetPrice.toFixed(2)} at y ${s.lastSetY.toFixed(2)}, panel line y ${s.panelLineY.toFixed(2)}, readout ${s.readout}`)
  await page.screenshot({ path: path.join(ASSETS, 'crosshair-panel-to-chart-2026-10-23.png') })

  // leave both
  await page.mouse.move(box.x, 880)
  await page.waitForTimeout(300)
  s = await state()
  expect(s.panelLineY).toBeNull()
  expect(s.vertVisible).toBe(true)
  expect(s.tooltipShown).toBe(false)
})
