// CR-DU — discretionary-rules-v1 as one mechanical engine (vault research/discretionary-rules-v1.md; spec
// specs/CR-DU-v1-rules-engine.md, Step 0 Q2 + Amendment 1). Used by the box panel (verdict at the chart's anchor), the
// "v1 trades" overlay and the train backtest (scripts/cr_du_backtest_v1.py → scripts/v1EngineCli.mjs), so all three
// run this code. Built on flowV3's box / readout / percentile / absorption / verdict functions (edgeReadout,
// anchorReadout, boxPercentiles, absorptionLevels, boxVerdict); nothing of them is copied.
//
// Inputs: the chart's cells payload (`/api/ironbeam/flow/cells`; only its RTH segment is read — rule 2) and the engine
// inputs (`/api/ironbeam/v1-engine/inputs`: prior-sessions baseline and contradiction percentiles, A7 swing, strong GEX
// levels per minute, 1-s OHLC). A decision at the close of RTH bar i reads bars starting at or before bar i only;
// absorption marks are taken as of that close (confirmed, broken and merged by then).
// Times: bars.t = UTC minute starts; "shifted" = PT wall-clock epochs (flowV3).
import {
  FOMC_2026_DECISION_DAYS, LEVELS_MAX, TICK, absorptionLevels, anchorReadout, boxPercentiles, boxVerdict, edgeReadout,
  floorIndex, makeShifter,
} from './flowV3.js'
import { utcEpochShowingZoneTime } from './flowShared.js'

// v1.1 (Amendments 2 / 3): lean ≥ p70, fixed 10-pt stop. v1.2 (Amendment 4): location moves from the read to the order
// (a valid read rests a limit at the faded edge while it holds). v1.3 (Amendment 5, a bug fix): forming box edges = running max / min close, frozen at qualification (A1.1 / rule 6).
export const V1_ENGINE_VERSION = 'v1.3'
export const V1_DEFAULTS = {
  boxMin: 8, minHeightPts: 2, // rule 6
  notable: 80, nearPts: 2, absorbPts: 2, // CR-DQ ⚙ "Verdict rules" defaults (notable = the effort leg's cut)
  leanPct: 70, // rules 7 / 8 lean: Δ share ≥ p70 on the side (Amendment 3; p80 = sensitivity row)
  holdMin: 15, breakHoldMin: 5, // rule 6 anchor: the CR-DI / CR-DN box chain at its display defaults, fixed here
  moveLookbackMin: 60, // rule 6 "after a larger move"
  noExtendBars: 3, // rules 7 / 8 "highs (lows) not extending"
  cutoff: true, cutoffMin: 12 * 60 + 15, flattenMin: 12 * 60 + 59, // rule 3; Q4 flatten (PT minutes of day)
  entry: 'edge', // rule 13: 'edge' = the exact edge by closes; 'd4' = CR-CP D4, 1 pt inside (sensitivity)
  stop: 'fixed10', stopPad: 2, // rule 15 (Amendment 2): 'fixed10' = 10 pts from entry; 'zone' = the box's wick extreme + 2 pts (sensitivity)
  target: 10, // rule 16
  lossCap: true, maxLosses: 3, // rule 19
  reentryMin: 20, // rule 20
  costPts: 0.25, // Q4: per round trip
}
export const SIDES = ['short', 'long']
const RTH_OPEN_MIN = 6 * 60 + 30

// Failure reasons (rule = the v1 rule number): { rule, code, text }.
const fail = (rule, code, text) => ({ rule, code, text })

// The engine's context for one session: RTH bars, flow rows, inputs. `fixedBaseline` = the frontend's train baseline
// (boxBaseline.json) for dates whose inputs carry none (forward / live).
export function prepareSession({ payload, inputs, tradeDate, fixedBaseline = null, settings = null }) {
  const s = { ...V1_DEFAULTS, ...(settings || {}) }
  const shift = makeShifter((t) => utcEpochShowingZoneTime(t, 'America/Los_Angeles'))
  const rth = (payload?.segments || []).find((g) => g?.name === 'rth')
  if (!rth?.bars?.t?.length) return null
  const b = rth.bars
  const n = b.t.length
  const num = (a, i) => Number(a?.[i])
  const bars = { t: [], o: [], h: [], l: [], c: [], buy: [], sell: [], vol: [], exact: [] }
  for (let i = 0; i < n; i += 1) {
    bars.t.push(Number(b.t[i]))
    bars.o.push(Number.isFinite(num(b.o, i)) ? num(b.o, i) : num(b.c, i))
    bars.h.push(num(b.h, i))
    bars.l.push(num(b.l, i))
    bars.c.push(num(b.c, i))
    bars.buy.push(num(b.buy, i) || 0)
    bars.sell.push(num(b.sell, i) || 0)
    bars.vol.push(num(b.vol, i) || 0)
    bars.exact.push(Boolean(rth.exact))
  }
  // CR-DV merge: provisional (hollow) marks are display-only — the engine reads confirmed marks only, live or backtest
  const seg = { ...rth, marks: rth.exact && Array.isArray(rth.marks) ? rth.marks.filter((mk) => !mk.provisional) : null }
  const pay = { ...payload, segments: [seg] } // rule 2: RTH only
  const times = bars.t.map(shift)
  // Q7: closes, delta and volume from the same flow bars as the box (one feed)
  const rows = bars.t.map((t, i) => ({ time: times[i], delta: bars.buy[i] - bars.sell[i], vol: bars.vol[i], seg: 'rth' }))
  const ptMin = (utc) => Math.floor(((((shift(utc) % 86400) + 86400) % 86400)) / 60)
  // 1-s OHLC grouped by minute start
  const secs = new Map()
  const sc = inputs?.seconds
  if (sc?.t?.length) {
    for (let k = 0; k < sc.t.length; k += 1) {
      const m = Math.floor(Number(sc.t[k]) / 60) * 60
      let a = secs.get(m)
      if (!a) secs.set(m, (a = []))
      a.push({ t: Number(sc.t[k]), o: Number(sc.o[k]), h: Number(sc.h[k]), l: Number(sc.l[k]) })
    }
  }
  return {
    s, shift, pay, bars, times, rows, closes: bars.c, ptMin, secs, tradeDate: String(tradeDate || inputs?.trade_date || '').slice(0, 10),
    anchor: shift(Number(rth.start)), baseline: inputs?.baseline || fixedBaseline, contra: inputs?.contra || null,
    swing: Number(inputs?.swing) || 12, gex: inputs?.gex || { reason: 'no inputs', schedule: [] },
  }
}

export const newState = () => ({ losses: 0, lastStop: null, boxesTraded: [] })

// Strong GEX levels in force at bar t (the last schedule entry from ≤ t).
function gexAt(gex, t) {
  let cur = null
  for (const e of gex?.schedule || []) if (Number(e.from) <= t) cur = e
  return cur?.levels || []
}

// Rules 7 / 8: no 1-min high (low) in the last `k` closed bars above (below) the box's wick high (low) before them.
function noExtend(bars, j0, i, k) {
  if (i - k < j0) return { up: false, down: false }
  let hi0 = -Infinity
  let lo0 = Infinity
  for (let j = j0; j <= i - k; j += 1) {
    hi0 = Math.max(hi0, bars.h[j])
    lo0 = Math.min(lo0, bars.l[j])
  }
  let hi1 = -Infinity
  let lo1 = Infinity
  for (let j = i - k + 1; j <= i; j += 1) {
    hi1 = Math.max(hi1, bars.h[j])
    lo1 = Math.min(lo1, bars.l[j])
  }
  return { up: hi1 <= hi0 + 1e-9, down: lo1 >= lo0 - 1e-9 }
}

// Rule 6: the box's first bar follows a larger move — a high or low in the previous `look` RTH minutes at least one A7
// swing from that bar's close.
function largerMove(bars, j0, look, swing) {
  const c0 = bars.c[j0]
  let best = 0
  for (let j = Math.max(0, j0 - look); j < j0; j += 1) best = Math.max(best, bars.h[j] - c0, c0 - bars.l[j])
  return { ok: best >= swing - 1e-9, move: best }
}

// Rule 12 (evidence only, Amendment 1): for a long, the box's lowest low L1, then within 10 min a higher low L2 (≥ L1 +
// 1 tick, a local low) whose push (from the last high before it) sold less than the push into L1. Mirror for a short.
function rule12(bars, j0, i, side) {
  const lowSide = side === 'long'
  const ext = (j) => (lowSide ? bars.l[j] : -bars.h[j])
  const pushVol = (j) => (lowSide ? bars.sell[j] : bars.buy[j])
  let k1 = j0
  for (let j = j0; j <= i; j += 1) if (ext(j) < ext(k1)) k1 = j
  const counter = (a, z) => { // the push's start: highest high (long) / lowest low (short) in [a, z]
    let m = a
    for (let j = a; j <= z; j += 1) if (lowSide ? bars.h[j] > bars.h[m] : bars.l[j] < bars.l[m]) m = j
    return m
  }
  for (let k2 = k1 + 2; k2 <= Math.min(i, k1 + 10); k2 += 1) {
    const local = ext(k2) <= ext(k2 - 1) && (k2 === i || ext(k2) <= ext(k2 + 1))
    if (!local || ext(k2) < ext(k1) + TICK - 1e-9) continue
    const p1 = counter(Math.max(j0, k1 - 10), k1)
    const p2 = counter(k1, k2)
    let v1 = 0
    let v2 = 0
    for (let j = p1; j <= k1; j += 1) v1 += pushVol(j)
    for (let j = p2 + 1; j <= k2; j += 1) v2 += pushVol(j)
    return { seen: true, l1: bars.t[k1], l2: bars.t[k2], push1: v1, push2: v2, lighter: v2 < v1 }
  }
  return { seen: false }
}

// Item 4 — contradiction bars inside the box on the faded edge, up to bar i: for a long (bullish), a bar whose low is
// within nearPts of the box's wick low with signed delta ≤ the bucket's p10, lower wick ≥ 50 % of its range, volume ≥ p80
// and range ≥ 2 pts; secondary (report-only): delta ≤ p20, wick ≥ 40 %, no volume floor. Short (bearish): the mirror
// (p90 / p80). Percentiles per 30-min RTH bucket, same feed, prior sessions only (inputs.contra).
export function contradictionBars(ctx, j0, i, side, box) {
  const { bars, contra, ptMin, s } = ctx
  const out = { primary: [], secondary: [], source: contra?.source ?? null }
  if (!contra?.buckets?.length) return out
  for (let j = j0; j <= i; j += 1) {
    const rg = bars.h[j] - bars.l[j]
    if (!(rg >= 2 - 1e-9)) continue
    const bk = contra.buckets[Math.min(contra.buckets.length - 1, Math.max(0, Math.floor((ptMin(bars.t[j]) - RTH_OPEN_MIN) / 30)))]
    const d = bars.buy[j] - bars.sell[j]
    const v = bars.vol[j]
    if (side === 'long') {
      if (bars.l[j] > box.low + s.nearPts + 1e-9) continue
      const wick = (Math.min(bars.o[j], bars.c[j]) - bars.l[j]) / rg
      if (d <= bk.d10 && wick >= 0.5 - 1e-9 && v >= bk.v80) out.primary.push(bars.t[j])
      if (d <= bk.d20 && wick >= 0.4 - 1e-9) out.secondary.push(bars.t[j])
    } else {
      if (bars.h[j] < box.high - s.nearPts - 1e-9) continue
      const wick = (bars.h[j] - Math.max(bars.o[j], bars.c[j])) / rg
      if (d >= bk.d90 && wick >= 0.5 - 1e-9 && v >= bk.v80) out.primary.push(bars.t[j])
      if (d >= bk.d80 && wick >= 0.4 - 1e-9) out.secondary.push(bars.t[j])
    }
  }
  return out
}

const VERDICT_RULE = {
  'no-trades': [1, 'fomc', 'FOMC decision day'],
  forming: [6, 'box_forming', 'box still forming'],
  conflict: [9, 'conflict', 'conflicting evidence'],
  'two-sided': [9, 'two_sided', 'two-sided absorption'],
  'watch-up': [7, 'no_read', 'no fade read (watch break idea)'],
  'watch-down': [8, 'no_read', 'no fade read (watch break idea)'],
  none: [7, 'no_read', 'nothing notable'],
}

/**
 * The verdict at the close of RTH bar i: { verdict 'SHORT' / 'LONG' / 'NO ACTION', side, failed[], evidence, order }.
 * `state` = the session's engine trades so far (rules 19 / 20, one trade per box); `anchorShifted` overrides the 06:30
 * anchor (the panel's chart anchor; clamped to RTH). `forceSide` (diagnosis D1): with no read, still check the gates on
 * that side (listed in `failed`; the verdict stays NO ACTION).
 */
export function evaluateAt(ctx, i, state = newState(), { anchorShifted = null, counts = false, forceSide = null } = {}) {
  const { s, shift, pay, bars, times, rows, closes, ptMin } = ctx
  const t = bars.t[i]
  const tClose = t + 60
  const anchor = anchorShifted != null ? Math.max(anchorShifted, ctx.anchor) : ctx.anchor
  const r = edgeReadout(pay, bars, anchor, shift, s.holdMin, times[i], s.breakHoldMin, { counts, qualifyMin: s.boxMin, qualifyPts: s.minHeightPts })
  const base = { t, verdict: 'NO ACTION', side: null, failed: [], evidence: {}, order: null, box: null, v: null }
  if (!r) return { ...base, failed: [fail(6, 'no_box', 'no box yet')] }
  const j0 = floorIndex(bars.t, r.from)
  const ro = anchorReadout(rows, closes, shift(r.from), times[i])
  const pct = boxPercentiles(ro, ptMin(r.from), pay.segments, ctx.baseline)
  const levels = absorptionLevels(pay, times, shift, LEVELS_MAX, tClose)
  const inBox = levels.filter((lv) => lv.start >= r.from) // rule 5: only in-box absorption is a trigger
  const earlier = levels.filter((lv) => lv.start < r.from)
  const ne = noExtend(bars, j0, i, s.noExtendBars)
  const box = { ...r, readout: ro, ended: { kind: 'current' } }
  const v = boxVerdict({
    box, pct, levels: inBox, tradeDate: ctx.tradeDate, fomc: FOMC_2026_DECISION_DAYS,
    rules: { notable: s.notable, lean: s.leanPct, nearPts: s.nearPts, absorbPts: s.absorbPts, minAgeMin: s.boxMin, minHeightPts: s.minHeightPts },
    v1: { noExtendUp: ne.up, noExtendDown: ne.down, anyLocation: true },
  })
  const boxInfo = { from: r.from, top: r.top.price, bottom: r.bottom.price, high: r.high, low: r.low, minutes: ro?.minutes ?? null }
  const evidence = { kind: v.kind, pill: v.pill, reason: v.reason, sharePct: pct?.share?.p ?? null, cppPct: pct?.cpp?.p ?? null,
    share: ro && ro.vol > 0 ? ro.delta / ro.vol : null, noExtend: ne }
  const out = { ...base, box: boxInfo, v, evidence, pct, readout: ro, edges: r }
  const side = v.kind === 'short' || v.kind === 'long' ? v.kind : null
  if (!side) {
    const [rule, code, text] = VERDICT_RULE[v.kind] || [7, 'no_read', v.kind]
    const why = v.reason && v.reason !== text ? (v.reason.startsWith(text) ? v.reason : `${text}: ${v.reason}`) : text
    out.failed.push(fail(rule, code, why))
    if (!forceSide) return out
  }
  out.side = side
  const sd = side || forceSide // D1 diagnostic: the gates on the card's side when the engine has no read
  const h = sd === 'short' ? -1 : 1
  const edge = sd === 'short' ? r.top.price : r.bottom.price
  const entry = s.entry === 'd4' ? edge + h * 1 : edge
  const target = entry + h * s.target
  const stop = s.stop === 'fixed10' ? entry - h * 10 : sd === 'short' ? r.high + s.stopPad : r.low - s.stopPad
  const order = { side: sd, boxFrom: r.from, entry, stop, target, placedT: tClose }
  if (!(sd === 'short' ? stop > entry : stop < entry)) out.failed.push(fail(14, 'stop_side', 'stop on the wrong side of the entry'))
  // rule 3: no entries after 12:15 PT (an order placed now rests from the next minute)
  const nextMin = ptMin(t) + 1
  if (s.cutoff && nextMin >= s.cutoffMin) out.failed.push(fail(3, 'cutoff', 'after 12:15 PT'))
  if (nextMin >= s.flattenMin) out.failed.push(fail(3, 'flatten', 'at the 12:59 flatten'))
  // rule 6: after a larger move
  const mv = largerMove(bars, j0, s.moveLookbackMin, ctx.swing)
  out.evidence.move = { ...mv, swing: ctx.swing }
  if (!mv.ok) out.failed.push(fail(6, 'no_larger_move', `no ${ctx.swing.toFixed(1)}-pt move into the box (max ${mv.move.toFixed(2)})`))
  // rule 4: nothing strong in the path (entry → target, exclusive)
  const inPath = (p) => (sd === 'short' ? p < entry - 1e-9 && p > target + 1e-9 : p > entry + 1e-9 && p < target - 1e-9)
  if (ctx.gex.reason) out.failed.push(fail(4, 'no_gex', `no GEX levels (${ctx.gex.reason})`))
  const gexHit = gexAt(ctx.gex, t).filter((lv) => inPath(Number(lv.price)))
  if (gexHit.length) out.failed.push(fail(4, 'gex_in_path', `strong GEX level ${gexHit.map((x) => `${Number(x.price).toFixed(2)} (${x.side})`).join(', ')} in the path`))
  const oppKind = sd === 'short' ? 'sellers_absorbed' : 'buyers_absorbed'
  const absHit = earlier.filter((lv) => lv.kind === oppKind && !lv.broken && inPath(lv.price))
  if (absHit.length) out.failed.push(fail(4, 'absorption_in_path', `earlier ${sd === 'short' ? 'sellers' : 'buyers'} absorbed ${absHit.map((x) => x.price.toFixed(2)).join(', ')} in the path`))
  out.evidence.gexInPath = gexHit
  // rule 5: earlier absorption that held at the faded edge = location support (not a trigger)
  const supKind = sd === 'short' ? 'buyers_absorbed' : 'sellers_absorbed'
  const wick = sd === 'short' ? r.high : r.low
  out.evidence.locationSupport = earlier.filter((lv) => lv.kind === supKind && !lv.broken && Math.abs(lv.price - wick) <= s.absorbPts + 1e-9)
    .map((lv) => ({ price: lv.price, start: lv.start }))
  // rules 19 / 20 and one trade per box
  if (s.lossCap && state.losses >= s.maxLosses) out.failed.push(fail(19, 'loss_cap', `${state.losses} losing trades today`))
  if (state.lastStop) {
    const waited = (tClose - state.lastStop.at) / 60
    if (waited < s.reentryMin - 1e-9) out.failed.push(fail(20, 'reentry_wait', `${Math.ceil(s.reentryMin - waited)} min left of the 20-min wait`))
    if (state.lastStop.side === sd && !(r.from > state.lastStop.entryAt)) out.failed.push(fail(20, 'reentry_box', 'same side needs a new box after the stop'))
  }
  if (state.boxesTraded.includes(r.from)) out.failed.push(fail(20, 'one_per_box', 'already traded this box'))
  out.evidence.rule12 = rule12(bars, j0, i, sd)
  out.evidence.contra = contradictionBars(ctx, j0, i, sd, r)
  out.order = order
  if (!out.failed.length && side) out.verdict = side.toUpperCase()
  return out
}

// Same-minute resolution from 1-s OHLC: walks the seconds of `minute` from `fromT` (inclusive); returns 'stop' /
// 'target' / null and the price (a stop that opens beyond fills at that second's open). `adverseOnlyAt` = the fill
// second: its adverse range counts, its favourable range doesn't (the order within it is unknown).
function resolveSeconds(ctx, minute, pos, fromT, adverseOnlyAt = null) {
  const list = ctx.secs.get(minute)
  if (!list) return null
  const short = pos.side === 'short'
  for (const x of list) {
    if (x.t < fromT) continue
    const hitStop = short ? x.h >= pos.stop - 1e-9 : x.l <= pos.stop + 1e-9
    const hitTgt = x.t !== adverseOnlyAt && (short ? x.l <= pos.target + 1e-9 : x.h >= pos.target - 1e-9)
    if (hitStop) return { kind: 'stop', t: x.t, px: short ? Math.max(pos.stop, x.o) : Math.min(pos.stop, x.o) }
    if (hitTgt) return { kind: 'target', t: x.t, px: pos.target }
  }
  return { kind: null }
}

/**
 * Simulate the session minute by minute (no lookahead): orders placed at a minute's close rest from the next minute,
 * fill only when price trades ≥ 1 tick through (Q4), follow the faded edge while the read holds (v1.2), are cancelled
 * on a lapse / box break / 12:15; brackets resolve on
 * 1-min bars, same-minute stop + target from 1-s data (else the stop); flatten at the open of the 12:59 bar.
 * Returns { trades, minutes: [{ t, verdict, side, failed, box }], state, open }.
 */
export function simulateSession(ctx, { keepMinutes = true, lastI = null } = {}) {
  const { s, bars, ptMin } = ctx
  const state = newState()
  const trades = []
  const minutes = []
  let order = null
  let pos = null
  const short = (p) => p.side === 'short'
  const close = (exit, i) => {
    const pts = short(pos) ? pos.entry - exit.px : exit.px - pos.entry
    let mfe = 0
    for (let j = pos.fillI; j <= i; j += 1) mfe = Math.max(mfe, short(pos) ? pos.entry - bars.l[j] : bars.h[j] - pos.entry)
    const tr = { ...pos, exitT: exit.t, exit: exit.px, reason: exit.kind, pts, net: pts - s.costPts, mfe }
    trades.push(tr)
    if (exit.kind === 'stop') {
      state.losses += 1
      state.lastStop = { at: exit.t, side: pos.side, entryAt: pos.entryT }
    }
    pos = null
  }
  const last = lastI == null ? bars.t.length - 1 : Math.min(lastI, bars.t.length - 1)
  for (let i = 0; i <= last; i += 1) {
    const t = bars.t[i]
    const pm = ptMin(t)
    // during bar i: the open position, else the resting order
    if (pos) {
      if (pm >= s.flattenMin) close({ kind: 'flatten', t, px: bars.o[i] }, i)
      else {
        const hitStop = short(pos) ? bars.h[i] >= pos.stop - 1e-9 : bars.l[i] <= pos.stop + 1e-9
        const hitTgt = short(pos) ? bars.l[i] <= pos.target + 1e-9 : bars.h[i] >= pos.target - 1e-9
        if (hitStop && hitTgt) {
          const res = resolveSeconds(ctx, t, pos, t)
          close(res?.kind ? res : { kind: 'stop', t, px: pos.stop }, i)
        } else if (hitStop) {
          const o = bars.o[i]
          close({ kind: 'stop', t, px: short(pos) ? Math.max(pos.stop, o) : Math.min(pos.stop, o) }, i)
        } else if (hitTgt) close({ kind: 'target', t, px: pos.target }, i)
      }
    } else if (order) {
      if ((s.cutoff && pm >= s.cutoffMin) || pm >= s.flattenMin) {
        order = null
      } else {
        const through = short(order) ? bars.h[i] >= order.entry + TICK - 1e-9 : bars.l[i] <= order.entry - TICK + 1e-9
        if (through) {
          const list = ctx.secs.get(t)
          const fs = list?.find((x) => (short(order) ? x.h >= order.entry + TICK - 1e-9 : x.l <= order.entry - TICK + 1e-9))
          pos = { ...order, entryT: fs ? fs.t : t, fillI: i, contra: order.contra }
          state.boxesTraded.push(order.boxFrom)
          order = null
          if (fs) {
            const res = resolveSeconds(ctx, t, pos, fs.t, fs.t)
            if (res?.kind) close(res, i)
          } else if (short(pos) ? bars.h[i] >= pos.stop - 1e-9 : bars.l[i] <= pos.stop + 1e-9) {
            close({ kind: 'stop', t, px: pos.stop }, i)
          }
        }
      }
    }
    // at the close of bar i
    const ev = evaluateAt(ctx, i, state)
    if (keepMinutes) minutes.push({ t, verdict: ev.verdict, side: ev.side, failed: ev.failed, box: ev.box })
    if (order && (ev.verdict !== order.side.toUpperCase() || ev.box?.from !== order.boxFrom)) order = null
    else if (order) { // Amendment 4: the limit stays at the faded edge (by closes) while the read holds — re-pegged
      order = { ...order, entry: ev.order.entry, stop: ev.order.stop, target: ev.order.target }
    }
    if (!pos && !order && (ev.verdict === 'SHORT' || ev.verdict === 'LONG')) {
      order = { ...ev.order, decisionT: t, box: ev.box, contra: { primary: ev.evidence.contra.primary.length, secondary: ev.evidence.contra.secondary.length },
        rule12: ev.evidence.rule12, locationSupport: ev.evidence.locationSupport.length }
    }
  }
  return { trades, minutes, state, open: pos }
}

// The engine settings the box panel's ⚙ "Verdict rules" set (boxRules: box min, height, near, absorption, notable).
export function engineSettingsFromRules(r) {
  if (!r) return {}
  return { boxMin: r.minAgeMin, minHeightPts: r.minHeightPts, nearPts: r.nearPts, absorbPts: r.absorbPts, notable: r.notable }
}

// Item 2 / Q6 — the box panel's verdict from the engine: the engine's own trades from 06:30 through bar i (rules 19 /
// 20 state, "engine's trades, not yours"), then the verdict at bar i at the chart's anchor. `endShifted` = the set end
// (or null = the last bar). Null without RTH bars at or before the end.
export function panelEngine(ctx, { anchorShifted = null, endShifted = null } = {}) {
  if (!ctx) return null
  let i = ctx.bars.t.length - 1
  if (endShifted != null) while (i >= 0 && ctx.times[i] > endShifted) i -= 1
  if (i < 0) return null
  const sim = simulateSession(ctx, { keepMinutes: false, lastI: i })
  const ev = evaluateAt(ctx, i, sim.state, { anchorShifted, counts: false })
  const tClose = ctx.bars.t[i] + 60
  const wait = sim.state.lastStop ? Math.max(0, Math.ceil(ctx.s.reentryMin - (tClose - sim.state.lastStop.at) / 60)) : 0
  return { ev, trades: sim.trades, open: sim.open, losses: sim.state.losses, wait, i }
}

const ruleRow = (f) => ({ key: `rule-${f.rule}-${f.code}`, check: `Rule ${f.rule}`, value: f.text, reads: 'fails', tone: 'bad' })
// "Engine's trades, not yours: 2 trades (1 stop) · re-entry wait 12 min · open short 7015.00"
export function engineTradesText(pe, fmt = (t) => String(t)) {
  if (!pe) return ''
  const n = pe.trades.length
  const parts = [`${n} trade${n === 1 ? '' : 's'}${pe.losses ? ` (${pe.losses} stop${pe.losses === 1 ? '' : 's'})` : ''}`]
  if (pe.wait > 0) parts.push(`re-entry wait ${pe.wait} min`)
  if (pe.open) parts.push(`open ${pe.open.side} ${pe.open.entry.toFixed(2)} since ${fmt(pe.open.entryT)}`)
  return `Engine's trades, not yours: ${parts.join(' · ')}`
}

// The panel's verdict object (BoxPanel's Verdict: kind / pill / reason / checks / title) from an engine evaluation:
// SHORT / LONG ("READ — limit at <price>" while price is not at the edge, Amendment 4) when every v1 rule passes, else NO ACTION (NO TRADES on FOMC days, CONFLICT kept) with the failed rules
// as the reason and as extra evidence rows (existing row style, tone 'bad').
export function panelVerdict(pe, fmt = (t) => String(t)) {
  const title = 'v1 rules engine (discretionary-rules-v1, applied mechanically; CR-DU). Rules 1–20: FOMC, RTH, 12:15 cutoff, '
    + 'box ≥ 8 min after a larger move, Δ share / effort / highs not extending, absorption inside the box, nothing strong in '
    + 'the +10 path, ≤ 3 losses, 20-min re-entry. Train backtest in-sample only — not a tested edge.'
  const engineTag = `engine ${V1_ENGINE_VERSION} — untested` // shown beside the pill (approved visible change)
  if (!pe) return { kind: 'none', pill: 'NO ACTION', reason: 'rule 2: RTH only', checks: null, title, evidence: '', extraRows: [], engineLine: '', engineTag }
  const { ev } = pe
  const v = ev.v
  const ok = ev.verdict === 'SHORT' || ev.verdict === 'LONG'
  const kind = ok ? ev.side : !v ? 'forming' : v.kind === 'short' || v.kind === 'long' ? 'none' : v.kind
  // Amendment 4: a valid read away from the edge shows its resting limit
  const atEdge = ok && (ev.side === 'short' ? v?.checks?.nearTop : v?.checks?.nearBottom)
  const pill = ok ? (atEdge ? `${ev.verdict} at ${ev.side === 'short' ? 'top' : 'bottom'}` : `READ — limit at ${ev.order.entry.toFixed(2)}`)
    : kind === 'no-trades' ? 'NO TRADES' : kind === 'conflict' ? 'CONFLICT → NO ACTION' : 'NO ACTION'
  const reason = ok ? (v?.reason || '') : ev.failed.map((f) => `rule ${f.rule}: ${f.text}`).join(' · ')
  return {
    kind, pill, reason, checks: v?.checks || null, title, evidence: v?.evidence || '', engine: { verdict: ev.verdict, failed: ev.failed },
    extraRows: ok ? [] : ev.failed.filter((f) => !['box_forming', 'no_read', 'conflict', 'two_sided', 'fomc'].includes(f.code)).map(ruleRow),
    engineLine: engineTradesText(pe, fmt),
    engineTag,
  }
}

// Item 4b — the overlay's marks (TradeMarksPrimitive data, shifted times): entry ▲ / ▼, exit ×, stop / +10 target lines
// while open, the trade's points beside the exit; an open trade runs its lines to the last bar and has no exit marker.
export const V1_TRADES_RGB = '45,212,191' // teal (C1 deviation approved by Ryan 2026-10-31)
export function overlayMarks(sim, ctx) {
  if (!sim || !ctx) return []
  const sh = ctx.shift
  const lastT = sh(ctx.bars.t[ctx.bars.t.length - 1] + 60)
  const mk = (tr, k, open) => {
    const toT = open ? lastT : sh(tr.exitT)
    return {
      id: `v1-${k}`, direction: tr.side, entryT: sh(tr.entryT), entryPrice: tr.entry,
      exitT: open ? null : sh(tr.exitT), exitPrice: open ? null : tr.exit,
      lines: [{ kind: 'stop', price: tr.stop, fromT: sh(tr.entryT), toT }, { kind: 'target', price: tr.target, fromT: sh(tr.entryT), toT }],
      label: open ? '' : `${tr.pts >= 0 ? '+' : ''}${tr.pts.toFixed(2)}`,
    }
  }
  const out = sim.trades.map((tr, k) => mk(tr, k, false))
  if (sim.open) out.push(mk(sim.open, out.length, true))
  return out
}

// The overlay's hover text for one simulated minute: "07:49 SHORT" / "07:49 NO ACTION · rule 6: box still forming".
export function minuteText(m, fmt = (t) => String(t)) {
  if (!m) return ''
  const head = `v1 engine ${fmt(m.t)} · ${m.verdict}`
  return m.failed?.length ? `${head} · ${m.failed.map((f) => `rule ${f.rule}: ${f.text}`).join(' · ')}` : head
}

// Item 4 — the optional chart markers (⚙, off by default): primary contradiction bars of the panel's current engine box,
// both edges (bullish at the bottom, bearish at the top), as of the panel's minute. [{ t (UTC bar start), side }].
export function contraMarkers(ctx, pe) {
  const r = pe?.ev?.edges
  if (!ctx || !r) return []
  const j0 = floorIndex(ctx.bars.t, r.from)
  const out = []
  for (const side of SIDES) for (const t of contradictionBars(ctx, j0, pe.i, side, r).primary) out.push({ t, side })
  return out.sort((a, b) => a.t - b.t)
}
