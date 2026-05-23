// Timezone helpers for lightweight-charts epoch-shift trick.
//
// lightweight-charts treats all time values as UTC and labels the X axis
// accordingly.  To display wall-clock times for a specific timezone (e.g.
// "06:30" PT instead of "13:30" UTC), shift the UTC epoch so that the UTC
// interpretation *reads* the same numbers as the target timezone.
//
// Same pattern as PriceChart.jsx in react_price_preview (lines 89–120).

function partsForZone(epochSec, timeZone) {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone,
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(new Date(epochSec * 1000))
  const out = {}
  for (const p of parts) {
    if (p.type !== 'literal') out[p.type] = p.value
  }
  return out
}

/**
 * Returns a fake UTC epoch that, when interpreted as UTC wall-clock time,
 * shows the same H:MM the original epoch has in `timeZone`.
 *
 * Usage:
 *   bars.map(b => ({ ...b, time: utcEpochShowingZoneTime(b.time, 'America/Los_Angeles') }))
 */
export function utcEpochShowingZoneTime(originalEpochSec, timeZone = 'America/Los_Angeles') {
  const p = partsForZone(originalEpochSec, timeZone)
  return (
    Date.UTC(
      Number(p.year),
      Number(p.month) - 1,
      Number(p.day),
      Number(p.hour),
      Number(p.minute),
      Number(p.second),
    ) / 1000
  )
}
