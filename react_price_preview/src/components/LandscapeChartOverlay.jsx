// CR-009 item 3 — semi-transparent cyan bands marking the landscape's
// proximate negative-GEX zones, drawn on an SVG overlay above the price
// chart canvas. A sibling overlay can't sit behind the candles (the chart
// canvas background is opaque), so the bands are a low-opacity tint above
// the canvas — the candles read clearly through it. pointer-events:none
// lets crosshair / tooltip / drag / scroll-wheel zoom pass straight to the
// chart. Band Y-positions use the same affine price→pixel transform
// PriceChart publishes for the panel's Y-axis sync (item 1).

import { PANEL_WIDTH, NEG_COLOR } from 'web-shared'

const BAND_HALF_PTS = 2 // band spans ±2pt around the zone price
const BAND_OPACITY = 0.12

export default function LandscapeChartOverlay({ negZones, visiblePriceRange }) {
  if (!visiblePriceRange || !Array.isArray(negZones) || negZones.length === 0) {
    return null
  }

  const { priceTop, priceBot, paneHeight } = visiblePriceRange
  if (
    !Number.isFinite(priceTop) ||
    !Number.isFinite(priceBot) ||
    !Number.isFinite(paneHeight) ||
    priceTop === priceBot
  ) {
    return null
  }

  const span = priceTop - priceBot
  const yOf = (price) => ((priceTop - price) / span) * paneHeight
  const halfPx = Math.max(1.5, Math.abs((BAND_HALF_PTS / span) * paneHeight))

  const bands = []
  for (let i = 0; i < negZones.length; i += 1) {
    const price = Number(negZones[i]?.price)
    if (!Number.isFinite(price)) continue
    const y = yOf(price)
    // Skip zones fully outside the visible price pane.
    if (y + halfPx < 0 || y - halfPx > paneHeight) continue
    bands.push(
      <rect
        key={`neg-band-${i}`}
        x="0"
        y={y - halfPx}
        width="100%"
        height={halfPx * 2}
        fill={NEG_COLOR}
        opacity={BAND_OPACITY}
      />,
    )
  }
  if (bands.length === 0) return null

  return (
    <svg
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: PANEL_WIDTH,
        height: paneHeight,
        zIndex: 4,
        pointerEvents: 'none',
      }}
    >
      {bands}
    </svg>
  )
}
