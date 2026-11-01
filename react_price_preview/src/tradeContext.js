// CR-DR item 3b — "What the tools showed at entry": the price chart's own Flow v3 code (edge readout, box segments,
// box panel percentiles, CR-DQ verdict, absorption levels) run on an until-replay of the trade's session cut at the
// entry minute. The backend floors `until` to the minute and returns only bars that START before it, so nothing from
// the entry minute or later is read (the entry minute itself, :00 → the fill second, is not seen either).
// Pure parts are separate from the fetch so they are unit-tested on fixed payloads.
import {
  DEFAULT_V3_SETTINGS, absorptionLevels, boxPanelOf, boxRules, boxSegments, currentBoxVerdict, edgeBoxEntries, edgeReadout,
  flattenBars, makeShifter, rowCloses, segmentReadouts,
} from './flowV3.js'
import { alignV2ToCandles, formatShiftedHHMM, normalizeShiftedCandleTimes, utcEpochShowingZoneTime } from './flowShared.js'

export const NEAR_LEVEL_PTS = 5 // absorption levels listed within this distance of the entry price
export const LAST_MIN = 5

// "2026-10-30T07:49:17" (naive PT) → "2026-10-30T07:49:00" (the until cutoff: the entry minute's start).
export function entryUntil(entryPt) {
  const m = /^(\d{4}-\d{2}-\d{2})[T ](\d{2}):(\d{2})/.exec(String(entryPt || ''))
  return m ? `${m[1]}T${m[2]}:${m[3]}:00` : null
}

// A naive PT wall-clock string as a shifted epoch (the charts' time axis).
export function ptToShifted(pt) {
  const m = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?/.exec(String(pt || ''))
  return m ? Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +(m[6] || 0)) / 1000 : null
}

export function shiftedToPt(t) {
  if (t == null || !Number.isFinite(Number(t))) return null
  return new Date(Number(t) * 1000).toISOString().slice(0, 19)
}

/**
 * Pure: the context at entry from until-cut payloads.
 *   cells  — /api/ironbeam/flow/cells?…&until (absorption marks + 1-minute bars for the edge readout)
 *   v2Bars — /api/ironbeam/flow?mode=v2&…&until `.bars` (per-minute delta / volume)
 *   bars   — /api/ironbeam/bars?interval=1min&…&until `.bars` (candles: { time (utc epoch), close, … })
 *   anchorShifted — the trade's anchor (shifted epoch) or null; entryShifted — the entry minute's start (shifted)
 */
export function entryContext({ cells, v2Bars, bars, anchorShifted, entryShifted, entryPrice, tradeDate, settings, rules, baseline }) {
  const s = { ...DEFAULT_V3_SETTINGS, ...(settings || {}) }
  const shift = makeShifter((t) => utcEpochShowingZoneTime(t, 'America/Los_Angeles'))
  const fmt = formatShiftedHHMM
  const times = normalizeShiftedCandleTimes(bars)
  const lastBar = times.length ? times[times.length - 1] : null

  // the same top-8 levels the chart draws; listed when unbroken at the cutoff and near the entry price
  const levels = cells ? absorptionLevels(cells, times, shift) : []
  const near = levels
    .filter((lv) => !lv.broken && Math.abs(lv.price - entryPrice) <= NEAR_LEVEL_PTS)
    .map((lv) => ({ kind: lv.kind, price: lv.price, dist: lv.price - entryPrice, contracts: lv.contracts, since: fmt(lv.startShifted), label: lv.label }))
    .sort((a, b) => Math.abs(a.dist) - Math.abs(b.dist))

  // last LAST_MIN complete minutes before the entry minute
  const rows = alignV2ToCandles(v2Bars || [], bars || [])
  const recent = rows.filter((r) => r.time < entryShifted && r.time >= entryShifted - LAST_MIN * 60)
  const sum = (k) => recent.reduce((a, r) => a + (Number.isFinite(Number(r[k])) ? Number(r[k]) : 0), 0)
  const last5 = { minutes: recent.length, delta: sum('delta'), vol: sum('vol'),
    from: recent.length ? fmt(recent[0].time) : null, to: recent.length ? fmt(recent[recent.length - 1].time) : null }

  let box = null
  if (anchorShifted != null && cells) {
    const edges = edgeReadout(cells, flattenBars(cells), anchorShifted, shift, s.boxHoldMin, null, s.breakHoldMin ?? DEFAULT_V3_SETTINGS.breakHoldMin, { qualifyMin: boxRules(rules).minAgeMin, qualifyPts: boxRules(rules).minHeightPts })
    const seg = boxSegments(edges, shift)
    if (edges && seg) {
      const readouts = segmentReadouts(rows, rowCloses(rows, bars || [], shift), seg.spans)
      const entries = edgeBoxEntries(edges, { ...seg, readouts }, times, shift, fmt)
      const panel = boxPanelOf(entries, shift, cells, baseline, fmt)
      const verdict = currentBoxVerdict(panel, entries, levels, tradeDate, rules, shift, fmt)
      const current = panel?.sections.find((x) => x.current) || null
      const cur = entries?.boxes.find((b) => !b.frozen) || null
      box = {
        verdict, current, panel,
        top: cur?.top ?? null, bottom: cur?.bottom ?? null,
        from: cur ? fmt(cur.from) : null, to: cur ? fmt(cur.to) : null,
        sharePct: current?.pct?.share?.p ?? null, cppPct: current?.pct?.cpp?.p ?? null,
      }
    }
  }
  return { cutoff: lastBar != null ? fmt(lastBar) : null, levels: near, last5, box }
}

// Fetch the three until-replay payloads for a trade and compute its context. settings: Flow v3 settings (k, m,
// boxHoldMin, breakHoldMin), session ('FULL' / 'RTH'), markerSettings (the v2 marker knobs App sends), rules, baseline.
export async function loadEntryContext({ apiBase, trade, session = 'FULL', settings, markerSettings = {}, rules, baseline, signal }) {
  const until = entryUntil(trade.entry_pt)
  const day = String(trade.trade_date).slice(0, 10)
  const s = { ...DEFAULT_V3_SETTINGS, ...(settings || {}) }
  const get = async (path, params) => {
    const url = new URL(`${apiBase}${path}`)
    for (const [k, v] of Object.entries(params)) if (v != null) url.searchParams.set(k, String(v))
    const r = await fetch(url.toString(), { credentials: 'include', signal, cache: 'no-store' })
    if (!r.ok) throw new Error(`${path} ${r.status}`)
    return r.json()
  }
  const [cells, v2, bars] = await Promise.all([
    get('/api/ironbeam/flow/cells', { trade_date: day, session, a_k: s.k, a_m: s.m, until }),
    get('/api/ironbeam/flow', {
      trade_date: day, session, mode: 'v2', until, effort: 1,
      m_swing: markerSettings.swing, m_push: markerSettings.push, m_div: markerSettings.div, m_fade: markerSettings.fade, m_gap: markerSettings.gap,
    }),
    get('/api/ironbeam/bars', { trade_date: day, interval: '1min', gex_enabled: 0, phase: 'center', until }),
  ])
  return entryContext({
    cells, v2Bars: v2?.bars || [], bars: bars?.bars || [],
    anchorShifted: trade.anchor_pt ? ptToShifted(trade.anchor_pt) : null,
    entryShifted: ptToShifted(until), entryPrice: Number(trade.entry_price), tradeDate: day,
    settings: s, rules, baseline,
  })
}
