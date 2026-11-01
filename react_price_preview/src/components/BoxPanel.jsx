// CR-DQ items 2 / 3 — the box panel on the price chart (replaces CR-DP's popup). One section per drawn box, oldest at the
// top, the current box last and highlighted; frozen boxes are one line, each section expands to the stats (current
// collapsed by default too). Text comes from flowV3.boxPanelSections (CR-DP's boxPopupLines), so the numbers are the
// popup's and the old box hover's. Dragged by its header; the caller owns the place (remembered per browser), the
// double-click reset and the ×. Palette: the chart's tag background, magenta border, slate / light text; CR-DS item 6
// (C1 deviation approved 2026-10-31): muted green / red / amber status colours inside this panel only.
import { Fragment, useState } from 'react'
import { BOX_RULES_DEFAULTS, boxEvidenceRows, pctBadgeTitle, pctText, pctTier } from '../flowV3.js'
import { TAG_BG } from './flowV3Primitives.js'

const MAGENTA = '232,121,249'
const TEXT = '#e2e8f0'
const MUTED = '#94a3b8'

// CR-DQ item 5b: tiers by brightness only (C1) — normal text below the notable cut-off, brighter from notable, brightest
// and bold from strong. Hover: pctBadgeTitle.
const TIER_STYLE = {
  normal: {},
  notable: { color: '#ffffff', fontWeight: 600 },
  strong: { color: '#ffffff', fontWeight: 800, background: 'rgba(255,255,255,0.12)', borderRadius: 3, padding: '0 3px' },
}
export function PctBadge({ kind, pct, rules }) {
  const p = kind === 'share' ? pct?.share?.p : pct?.cpp?.p
  if (p == null) return null
  const tier = pctTier(p, rules)
  return (
    <span data-testid={`flow-v3-box-pct-${kind}`} data-tier={tier} data-p={p.toFixed(2)} title={pctBadgeTitle(kind, pct, rules)}
      style={{ cursor: 'help', ...TIER_STYLE[tier] }}>
      {pctText(p)}
    </span>
  )
}

// The head line ("box from … · Δ · share … · pts · m · …/pt") with a badge after the share and after contracts/pt.
function Head({ s, rules }) {
  const parts = s.lines.parts || [s.lines.head]
  const badgeAt = { 2: 'share', 5: 'cpp' }
  return parts.map((t, i) => (
    <Fragment key={i}>
      {i > 0 && ' · '}
      {t}
      {s.pct && badgeAt[i] && <>{' · '}<PctBadge kind={badgeAt[i]} pct={s.pct} rules={rules} /></>}
    </Fragment>
  ))
}

function Stats({ s, testid, rules }) {
  const l = s.lines
  return (
    <div data-testid={testid} style={{ paddingLeft: 10 }}>
      <div data-testid={s.current ? 'flow-v3-box-popup-head' : undefined}><Head s={s} rules={rules} /></div>
      <div data-testid={s.current ? 'flow-v3-box-popup-top' : undefined}>{l.top}</div>
      <div data-testid={s.current ? 'flow-v3-box-popup-bottom' : undefined}>{l.bottom}</div>
      <div style={{ color: MUTED }}>{s.probes}</div>
      {l.mixed && <div data-testid={s.current ? 'flow-v3-box-popup-mixed' : undefined} style={{ color: '#fbbf24' }}>{l.mixed}</div>}
    </div>
  )
}

// CR-DQ item 7 / CR-DS item 1 — the verdict pill (its head: "SHORT READ at top", "NO ACTION", …) with one short reason
// under it, then the evidence table (item 2). Hover on the pill: rule IDs, thresholds, "draft rules, untested". The old
// evidence line stays on the element as data-evidence (no longer drawn: the table shows the same checks).
// Tone per verdict kind (item 1): reads 'good', conflict 'bad', watch 'idea', NO ACTION / NO TRADES 'neutral'.
export const VERDICT_TONE = {
  short: 'good', long: 'good', conflict: 'bad', 'watch-up': 'idea', 'watch-down': 'idea',
  'no-trades': 'neutral', forming: 'neutral', 'two-sided': 'neutral', none: 'neutral',
}
// CR-DS item 6 (C1 deviation approved by Ryan 2026-10-31, this panel only): flat, muted status colours — green =
// agreement, red = contradiction, amber = idea, slate = neutral. Drawn only as a 3 px bar / border; text stays light
// (Step 0 S0.3: bars ≥ 3.3:1 against the panel, light text 12:1).
export const TONE_COLOR = { good: '#6f8f72', bad: '#a86464', idea: '#b08a4a', neutral: '#64748b' }

function EvidenceTable({ rows, testid }) {
  if (!rows.length) return null
  return (
    <table data-testid={testid} style={{ borderCollapse: 'separate', borderSpacing: '0 1px', width: '100%', margin: '3px 0 1px', lineHeight: 1.45 }}>
      <tbody>
        {rows.map((r) => (
          <tr key={r.key} data-testid="flow-v3-box-evidence-row" data-key={r.key} data-tone={r.tone}>
            <td style={{ width: 3, padding: 0, background: TONE_COLOR[r.tone], borderRadius: 1 }} />
            <td style={{ padding: '0 6px 0 5px', color: MUTED, whiteSpace: 'nowrap', verticalAlign: 'top' }}>{r.check}</td>
            <td data-testid="flow-v3-box-evidence-value" style={{ padding: '0 6px 0 0', verticalAlign: 'top' }}>{r.value}</td>
            <td data-testid="flow-v3-box-evidence-reads" style={{ padding: 0, color: '#cbd5e1', whiteSpace: 'nowrap', verticalAlign: 'top' }}>{r.reads}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// CR-DS item 4 — "Read live: 07:49–07:58 (short)": the minutes in which a read verdict held, replayed without later bars.
function History({ s }) {
  if (!s.historyText) return null
  return (
    <div data-testid={s.current ? 'flow-v3-box-history' : 'flow-v3-box-history-frozen'} data-n={s.history?.length ?? 0}
      title="Minutes in which the verdict was a SHORT / LONG READ, replayed minute by minute from the anchor using only bars up to each minute. Times are candle times (bar start): a read at 07:49 was showing once the 07:49 bar closed. Draft rules, untested."
      style={{ color: s.history?.length ? TEXT : MUTED, cursor: 'help' }}>
      {s.historyText}
    </div>
  )
}

// CR-DS item 5 — a frozen box's compact row: time span · how it ended · share pNN · c/pt pNN · any read-live window.
function FrozenRow({ s, rules }) {
  const badge = (kind) => (s.pct ? <PctBadge kind={kind} pct={s.pct} rules={rules} /> : '–')
  return (
    <span data-testid="flow-v3-box-frozen-row">
      {s.span} · {s.state} · share {badge('share')} · c/pt {badge('cpp')}
      {s.historyShort && <span data-testid="flow-v3-box-frozen-read" style={{ color: TEXT }}>{' · '}{s.historyShort}</span>}
    </span>
  )
}

function Verdict({ v, testid = 'flow-v3-box-verdict' }) {
  const tone = VERDICT_TONE[v.kind] || 'neutral'
  const c = TONE_COLOR[tone]
  const rows = [...boxEvidenceRows(v.checks), ...(v.extraRows || [])] // CR-DU: the v1 engine's failed rules
  return (
    <div data-testid={testid} data-kind={v.kind} data-tone={tone} data-evidence={v.evidence} style={{ marginBottom: 2 }}>
      <span data-testid="flow-v3-box-verdict-pill" title={v.title}
        style={{ display: 'inline-block', cursor: 'help', borderRadius: 3, padding: '0 6px 0 5px', color: TEXT, fontWeight: tone === 'good' ? 700 : 600,
          border: `1px solid ${c}`, borderLeft: `3px solid ${c}`, background: 'rgba(255,255,255,0.03)' }}>
        {v.pill}
      </span>
      {v.reason && <div data-testid="flow-v3-box-verdict-reason" style={{ color: '#cbd5e1' }}>{v.reason}</div>}
      <EvidenceTable rows={rows} testid={`${testid}-table`} />
      {v.engineLine && <div data-testid="flow-v3-box-engine-line" style={{ color: MUTED }}>{v.engineLine}</div>}
    </div>
  )
}

export default function BoxPanel({
  panel, place, panelRef, maxHeight, dragged, onClose, onHeaderPointerDown, onHeaderPointerMove, onHeaderPointerUp,
  onHeaderDoubleClick, rules = BOX_RULES_DEFAULTS, verdict = null,
}) {
  const [open, setOpen] = useState({}) // section key -> expanded
  const toggle = (k) => setOpen((o) => ({ ...o, [k]: !o[k] }))
  return (
    <div
      ref={panelRef}
      data-testid="flow-v3-box-popup"
      data-dragged={dragged ? '1' : '0'}
      data-history-ms={panel.historyMs ?? ''}
      style={{
        position: 'absolute', left: place?.left ?? 0, top: place?.top ?? 0, visibility: place ? 'visible' : 'hidden',
        zIndex: 5, background: TAG_BG(0.92), border: `1px solid rgba(${MAGENTA},0.35)`, borderRadius: 4,
        fontSize: '11px', fontFamily: 'sans-serif', color: TEXT, lineHeight: 1.6, maxWidth: 460, minWidth: 220,
        display: 'flex', flexDirection: 'column', maxHeight: maxHeight || undefined,
      }}
    >
      <div
        data-testid="flow-v3-box-panel-header"
        title="Drag to move · double-click to return bottom-left"
        onPointerDown={onHeaderPointerDown}
        onPointerMove={onHeaderPointerMove}
        onPointerUp={onHeaderPointerUp}
        onPointerCancel={onHeaderPointerUp}
        onDoubleClick={onHeaderDoubleClick}
        style={{ cursor: 'move', userSelect: 'none', touchAction: 'none', padding: '2px 20px 2px 8px', color: MUTED,
          borderBottom: `1px solid rgba(${MAGENTA},0.2)`, flex: '0 0 auto' }}
      >
        Boxes · visual aid — untested
        <button
          type="button"
          data-popup-close="1"
          data-testid="flow-v3-box-popup-close"
          aria-label="Hide box stats until the next anchor"
          title="Hide until the next anchor"
          onPointerDown={(e) => e.stopPropagation()}
          onClick={onClose}
          style={{ position: 'absolute', top: 2, right: 3, width: 14, height: 14, padding: 0, border: 'none', background: 'transparent', color: MUTED, cursor: 'pointer', fontSize: '12px', lineHeight: '14px' }}
        >
          ×
        </button>
      </div>
      <div data-testid="flow-v3-box-panel-body" style={{ overflowY: 'auto', padding: '4px 8px 6px', flex: '1 1 auto' }}>
        {panel.droppedText && <div data-testid="flow-v3-box-panel-dropped" style={{ color: MUTED }}>{panel.droppedText}</div>}
        {panel.sections.map((s) => {
          const isOpen = Boolean(open[s.key])
          return (
            <div
              key={s.key}
              data-testid={s.current ? 'flow-v3-box-panel-current' : 'flow-v3-box-panel-frozen'}
              style={s.current
                ? { marginTop: 4, padding: '3px 6px', border: `1px solid rgba(${MAGENTA},0.8)`, borderRadius: 3 }
                : { marginTop: 2, padding: '0 6px', color: '#cbd5e1' }}
            >
              {s.current && verdict && <Verdict v={verdict} />}
              {s.current && <History s={s} />}
              <div
                role="button"
                tabIndex={0}
                data-testid="flow-v3-box-panel-summary"
                aria-expanded={isOpen}
                onClick={() => toggle(s.key)}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(s.key) } }}
                style={{ cursor: 'pointer', color: s.current ? '#f0abfc' : undefined, fontWeight: s.current ? 600 : 400 }}
              >
                <span style={{ color: MUTED, display: 'inline-block', width: 10 }}>{isOpen ? '▾' : '▸'}</span>
                {s.current ? s.summary : <FrozenRow s={s} rules={rules} />}
              </div>
              <div style={{ display: isOpen ? 'block' : 'none' }}>
                {!s.current && s.verdict && <Verdict v={s.verdict} testid="flow-v3-box-frozen-verdict" />}
                {!s.current && <History s={s} />}
                <Stats s={s} rules={rules} testid={s.current ? 'flow-v3-box-panel-current-stats' : 'flow-v3-box-panel-frozen-stats'} />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
