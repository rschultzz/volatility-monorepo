// SetupReviewPanel — right-docked REVIEW panel for the CR-CC setup review queue.
//
// Lists candidate setups from /api/setup-review (filter by CR / set / cell /
// status). Activating a candidate hands its detail to App.jsx, which replays
// that session up to entry_ts_pt (bars with start < entry, Corrected levels)
// and PriceChart draws the candidate's lines and markers. The panel labels it
// Take / Skip / Unsure with reason chips and a comment; Save is required
// before Reveal, which re-fetches the full session and shows the outcome.
//
// The outcome is rendered only once the candidate is revealed, even though the
// detail route returns it as soon as a review exists.
//
// A Summary tab shows, per CR / set, counts by label, the mean outcome of revealed
// Take / Skip / Unsure candidates with n, and the reasons tally on Skips (L9).
//
// Keys (when focus is not in a text field): N / P next / previous,
// T / S / U label, R reveal.
//
// CR-DB layout (queue tab, a card active): the card header (direction, date /
// time, called edge) stays at the top; the key facts, the collapsed "details"
// (every other field), the saved review and the outcome scroll in their own
// region above the queue; the actions (Take / Skip / Unsure, reason chips,
// comment, Save / Reveal) are a footer pinned to the bottom of the panel.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ReviewMeta from './ReviewMeta'
import { CardHeader, DIRECTION_COLOR, KEY_FACT_META_KEYS, KeyFacts } from './ReviewCard'

export const PANEL_WIDTH = 340

// CR-DB / C1: Dash palette — Take blue, Skip slate, Unsure light grey; `ink` is the text on the filled button.
const LABELS = [
  { key: 'take', text: 'Take', hotkey: 'T', color: '#60a5fa', ink: '#020617' },
  { key: 'skip', text: 'Skip', hotkey: 'S', color: '#64748b', ink: '#f8fafc' },
  { key: 'unsure', text: 'Unsure', hotkey: 'U', color: '#cbd5e1', ink: '#020617' },
]
const LABEL_COLOR = Object.fromEntries(LABELS.map((l) => [l.key, l.color]))

const selectStyle = {
  background: '#0f172a',
  color: '#cbd5e1',
  border: '1px solid #334155',
  borderRadius: 4,
  fontSize: 10,
  padding: '2px 4px',
  fontFamily: 'inherit',
  minWidth: 0,
}

const buttonBase = {
  background: '#0f172a',
  color: '#cbd5e1',
  border: '1px solid #334155',
  borderRadius: 6,
  fontSize: 11,
  fontWeight: 700,
  padding: '5px 10px',
  cursor: 'pointer',
  fontFamily: 'inherit',
}

function fmtNum(v, digits = 2) {
  if (v == null || v === '') return '—'
  const n = Number(v)
  if (!Number.isFinite(n)) return String(v)
  return Number.isInteger(n) ? String(n) : n.toFixed(digits)
}

function fmtSigned(v, digits = 2) {
  if (v == null) return '—'
  const n = Number(v)
  if (!Number.isFinite(n)) return '—'
  return `${n > 0 ? '+' : ''}${n.toFixed(digits)}`
}

function hhmm(ts) {
  return ts ? String(ts).slice(11, 16) : '—'
}

function isTypingTarget(el) {
  if (!el) return false
  const tag = (el.tagName || '').toLowerCase()
  return tag === 'input' || tag === 'textarea' || tag === 'select' || el.isContentEditable
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, { credentials: 'include', cache: 'no-store', ...options })
  let body = null
  try {
    body = await res.json()
  } catch {
    body = null
  }
  if (!res.ok || !body || body.ok === false) {
    const msg = body?.message || body?.error || `${res.status} ${res.statusText || ''}`.trim()
    const err = new Error(msg)
    err.code = body?.error
    throw err
  }
  return body
}

function Row({ cand, active, onClick }) {
  const arrow = cand.direction === 'long' ? '▲' : '▼'
  const arrowColor = DIRECTION_COLOR[cand.direction] || '#94a3b8'
  return (
    <button
      type="button"
      data-testid="review-row"
      data-candidate-id={cand.id}
      onClick={onClick}
      style={{
        display: 'grid',
        gridTemplateColumns: '78px 40px 14px 1fr 54px',
        gap: 6,
        alignItems: 'center',
        padding: '6px 10px',
        background: active ? '#1e293b' : 'transparent',
        border: 'none',
        borderBottom: '1px solid #1e293b',
        cursor: 'pointer',
        textAlign: 'left',
        color: '#e2e8f0',
        fontFamily: 'inherit',
        width: '100%',
      }}
    >
      <span style={{ fontSize: 11, fontWeight: 600 }}>{cand.session_date}</span>
      <span style={{ fontSize: 10, color: '#94a3b8', fontVariantNumeric: 'tabular-nums' }}>{hhmm(cand.entry_ts_pt)}</span>
      <span style={{ fontSize: 11, color: arrowColor }}>{arrow}</span>
      <span style={{ fontSize: 10, color: '#cbd5e1', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {cand.cell || '—'}
      </span>
      <span style={{ fontSize: 10, fontWeight: 700, textAlign: 'right', color: LABEL_COLOR[cand.label] || '#475569' }}>
        {cand.label ? cand.label.toUpperCase() : '·'}
        {cand.revealed ? ' ✓' : ''}
      </span>
    </button>
  )
}

function KV({ k, v }) {
  return (
    <>
      <span style={{ color: '#64748b' }}>{k}</span>
      <span style={{ color: '#e2e8f0', fontVariantNumeric: 'tabular-nums', overflow: 'hidden', textOverflow: 'ellipsis' }}>{v}</span>
    </>
  )
}

function SummaryView({ groups, loading, error }) {
  if (loading) return <div style={{ padding: 12, fontSize: 11, color: '#64748b' }}>Loading…</div>
  if (error) return <div style={{ padding: 12, fontSize: 11, color: '#cbd5e1' }}>{error}</div>
  if (!groups.length) return <div style={{ padding: 12, fontSize: 11, color: '#64748b' }}>No candidates.</div>
  return (
    <div data-testid="review-summary" style={{ padding: '8px 10px', display: 'flex', flexDirection: 'column', gap: 12 }}>
      {groups.map((g) => (
        <div key={`${g.cr_id}:${g.set_name}`}>
          <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 4 }}>
            {g.cr_id} · {g.set_name}
          </div>
          <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 6 }}>
            {g.n_candidates} candidates · {g.n_reviewed} reviewed · {g.n_revealed} revealed · {g.unreviewed} unreviewed
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 10 }}>
            <thead>
              <tr style={{ color: '#64748b', textAlign: 'right' }}>
                <th style={{ textAlign: 'left', fontWeight: 600 }}>label</th>
                <th style={{ fontWeight: 600 }}>n</th>
                <th style={{ fontWeight: 600 }}>revealed</th>
                <th style={{ fontWeight: 600 }}>mean pts</th>
                <th style={{ fontWeight: 600 }}>wins</th>
              </tr>
            </thead>
            <tbody>
              {LABELS.map(({ key, text, color }) => {
                const s = g.by_label?.[key]
                return (
                  <tr key={key} style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                    <td style={{ textAlign: 'left', color, fontWeight: 700 }}>{text}</td>
                    <td>{s?.n ?? 0}</td>
                    <td>{s?.n_revealed ?? 0}</td>
                    <td>{s?.n_outcome ? fmtSigned(s.mean_pnl_pts) : '—'}</td>
                    <td>{s?.n_outcome ? `${s.n_win}/${s.n_outcome}` : '—'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          {g.skip_reasons?.length > 0 && (
            <div style={{ marginTop: 6, fontSize: 10, color: '#cbd5e1' }}>
              <span style={{ color: '#64748b' }}>Skip reasons: </span>
              {g.skip_reasons.map((r) => `${r.reason} (${r.n})`).join(' · ')}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

export default function SetupReviewPanel({
  apiBase = '',
  activeCandidate = null,
  onActivate,
  onClose,
  rightOffset = 0,
}) {
  const [filters, setFilters] = useState({ cr: '', set: '', cell: '', status: 'all' })
  const [candidates, setCandidates] = useState([])
  const [sets, setSets] = useState([])
  const [reasonOptions, setReasonOptions] = useState([])
  const [listError, setListError] = useState('')
  const [listLoading, setListLoading] = useState(false)
  const [tab, setTab] = useState('queue')
  const [summary, setSummary] = useState({ groups: [], loading: false, error: '' })

  const [label, setLabel] = useState(null)
  const [chosenReasons, setChosenReasons] = useState([])
  const [newReason, setNewReason] = useState('')
  const [comment, setComment] = useState('')
  const [busy, setBusy] = useState(false)
  const [actionError, setActionError] = useState('')
  const [detailsOpen, setDetailsOpen] = useState(false)

  const base = `${apiBase}/api/setup-review`
  const activeId = activeCandidate?.id || null
  const revealed = Boolean(activeCandidate?.revealed)
  const savedPreReveal = useMemo(
    () => (activeCandidate?.reviews || []).filter((r) => !r.post_reveal),
    [activeCandidate]
  )
  const canReveal = Boolean(activeCandidate) && !revealed && savedPreReveal.length > 0

  const loadList = useCallback(async () => {
    setListLoading(true)
    setListError('')
    try {
      const url = new URL(`${base}/candidates`, window.location.origin)
      if (filters.cr) url.searchParams.set('cr', filters.cr)
      if (filters.set) url.searchParams.set('set', filters.set)
      if (filters.cell) url.searchParams.set('cell', filters.cell)
      url.searchParams.set('status', filters.status)
      const body = await fetchJson(url.toString())
      setCandidates(body.candidates || [])
      setSets(body.sets || [])
      setReasonOptions(body.reasons || [])
    } catch (err) {
      setCandidates([])
      setListError(err.message)
    } finally {
      setListLoading(false)
    }
  }, [base, filters])

  useEffect(() => {
    loadList()
  }, [loadList])

  const loadSummary = useCallback(async () => {
    setSummary((s) => ({ ...s, loading: true, error: '' }))
    try {
      const url = new URL(`${base}/summary`, window.location.origin)
      if (filters.cr) url.searchParams.set('cr', filters.cr)
      if (filters.set) url.searchParams.set('set', filters.set)
      const body = await fetchJson(url.toString())
      setSummary({ groups: body.groups || [], loading: false, error: '' })
    } catch (err) {
      setSummary({ groups: [], loading: false, error: err.message })
    }
  }, [base, filters.cr, filters.set])

  useEffect(() => {
    if (tab === 'summary') loadSummary()
  }, [tab, loadSummary])

  const resetForm = useCallback(() => {
    setLabel(null)
    setChosenReasons([])
    setNewReason('')
    setComment('')
    setActionError('')
  }, [])

  const activate = useCallback(async (id) => {
    if (!id) return
    setBusy(true)
    setActionError('')
    try {
      const body = await fetchJson(`${base}/candidates/${encodeURIComponent(id)}`)
      resetForm()
      onActivate?.(body.candidate)
    } catch (err) {
      setActionError(err.message)
    } finally {
      setBusy(false)
    }
  }, [base, onActivate, resetForm])

  const refreshActive = useCallback(async () => {
    if (!activeId) return
    const body = await fetchJson(`${base}/candidates/${encodeURIComponent(activeId)}`)
    onActivate?.(body.candidate)
  }, [activeId, base, onActivate])

  const step = useCallback((delta) => {
    if (!candidates.length) return
    const idx = candidates.findIndex((c) => c.id === activeId)
    const next = idx === -1 ? (delta > 0 ? 0 : candidates.length - 1) : idx + delta
    if (next < 0 || next >= candidates.length) return
    activate(candidates[next].id)
  }, [activate, activeId, candidates])

  const save = useCallback(async () => {
    if (!activeId || !label || busy) return
    setBusy(true)
    setActionError('')
    try {
      await fetchJson(`${base}/candidates/${encodeURIComponent(activeId)}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label, reasons: chosenReasons, comment }),
      })
      await refreshActive()
      await loadList()
    } catch (err) {
      setActionError(err.message)
    } finally {
      setBusy(false)
    }
  }, [activeId, base, busy, chosenReasons, comment, label, loadList, refreshActive])

  const reveal = useCallback(async () => {
    if (!canReveal || busy) return
    setBusy(true)
    setActionError('')
    try {
      await fetchJson(`${base}/candidates/${encodeURIComponent(activeId)}/reveal`, { method: 'POST' })
      resetForm()
      await refreshActive()
      await loadList()
    } catch (err) {
      setActionError(err.message)
    } finally {
      setBusy(false)
    }
  }, [activeId, base, busy, canReveal, loadList, refreshActive, resetForm])

  // Keyboard: N/P, T/S/U, R. Ignored while typing in a field.
  const keyStateRef = useRef({})
  keyStateRef.current = { step, reveal, activeId }
  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey || isTypingTarget(e.target)) return
      const k = e.key.toLowerCase()
      const st = keyStateRef.current
      if (k === 'n') st.step(1)
      else if (k === 'p') st.step(-1)
      else if (!st.activeId) return
      else if (k === 't') setLabel('take')
      else if (k === 's') setLabel('skip')
      else if (k === 'u') setLabel('unsure')
      else if (k === 'r') st.reveal()
      else return
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const toggleReason = (r) => {
    setChosenReasons((prev) => (prev.includes(r) ? prev.filter((x) => x !== r) : [...prev, r]))
  }
  const addReason = () => {
    const r = newReason.trim()
    if (!r) return
    if (!reasonOptions.includes(r)) setReasonOptions((prev) => [...prev, r])
    if (!chosenReasons.includes(r)) setChosenReasons((prev) => [...prev, r])
    setNewReason('')
  }

  const crOptions = useMemo(() => [...new Set(sets.map((s) => s.cr_id))], [sets])
  const setOptions = useMemo(
    () => sets.filter((s) => !filters.cr || s.cr_id === filters.cr).map((s) => s.set_name),
    [sets, filters.cr]
  )
  const cellOptions = useMemo(() => {
    const out = new Set()
    sets
      .filter((s) => (!filters.cr || s.cr_id === filters.cr) && (!filters.set || s.set_name === filters.set))
      .forEach((s) => (s.cells || []).forEach((c) => out.add(c)))
    return [...out].sort()
  }, [sets, filters.cr, filters.set])

  const outcome = revealed ? activeCandidate?.outcome : null

  return (
    <div
      data-testid="review-panel"
      onWheel={(e) => e.stopPropagation()}
      style={{
        position: 'absolute',
        right: rightOffset,
        top: 0,
        bottom: 0,
        width: PANEL_WIDTH,
        background: '#020617',
        borderLeft: '1px solid #1e293b',
        color: '#e2e8f0',
        display: 'flex',
        flexDirection: 'column',
        fontFamily: 'inherit',
        zIndex: 6,
      }}
    >
      {/* Header */}
      <div style={{ padding: '10px 12px', borderBottom: '1px solid #1e293b', display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.08em' }}>REVIEW</span>
        <span style={{ display: 'flex', gap: 4, flex: 1 }}>
          {['queue', 'summary'].map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTab(t)}
              style={{
                ...buttonBase,
                padding: '2px 8px',
                fontSize: 10,
                color: tab === t ? '#e2e8f0' : '#64748b',
                background: tab === t ? '#1e293b' : 'transparent',
              }}
            >
              {t === 'queue' ? 'Queue' : 'Summary'}
            </button>
          ))}
        </span>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close review"
          style={{ background: 'transparent', color: '#94a3b8', border: 'none', fontSize: 16, cursor: 'pointer', padding: 0, lineHeight: 1 }}
        >
          ×
        </button>
      </div>

      {/* Filters */}
      <div style={{ padding: '6px 10px', borderBottom: '1px solid #1e293b', display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 4 }}>
        <select aria-label="CR" value={filters.cr} style={selectStyle}
          onChange={(e) => setFilters((f) => ({ ...f, cr: e.target.value, set: '', cell: '' }))}>
          <option value="">all CRs</option>
          {crOptions.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <select aria-label="Set" value={filters.set} style={selectStyle}
          onChange={(e) => setFilters((f) => ({ ...f, set: e.target.value, cell: '' }))}>
          <option value="">all sets</option>
          {setOptions.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <select aria-label="Cell" value={filters.cell} style={selectStyle}
          onChange={(e) => setFilters((f) => ({ ...f, cell: e.target.value }))}>
          <option value="">all cells</option>
          {cellOptions.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <select aria-label="Status" value={filters.status} style={selectStyle}
          onChange={(e) => setFilters((f) => ({ ...f, status: e.target.value }))}>
          <option value="all">all</option>
          <option value="unreviewed">unreviewed</option>
          <option value="reviewed">reviewed</option>
        </select>
      </div>

      {tab === 'summary' ? (
        <div style={{ flex: 1, overflowY: 'auto' }}>
          <SummaryView groups={summary.groups} loading={summary.loading} error={summary.error} />
        </div>
      ) : (
        <>
          {/* Active candidate: header + scrolling body (CR-DB) */}
          {activeCandidate && (
            <div
              data-testid="review-active"
              style={{ flex: '0 1 auto', minHeight: 0, maxHeight: '55%', display: 'flex', flexDirection: 'column', borderBottom: '1px solid #1e293b' }}
            >
              <div style={{ padding: '8px 10px 6px', borderBottom: '1px solid #0f172a' }}>
                <CardHeader cand={activeCandidate} />
              </div>
              <div data-testid="review-card-body" style={{ overflowY: 'auto', minHeight: 0, padding: '6px 10px 8px', display: 'flex', flexDirection: 'column', gap: 6 }}>
                {savedPreReveal.length > 0 && (
                  <div style={{ fontSize: 10, color: '#94a3b8' }}>
                    Saved:{' '}
                    {(() => {
                      const r = savedPreReveal[savedPreReveal.length - 1]
                      return (
                        <span>
                          <b style={{ color: LABEL_COLOR[r.label] }}>{r.label.toUpperCase()}</b>
                          {r.reasons?.length ? ` · ${r.reasons.join(', ')}` : ''}
                          {r.comment ? ` · “${r.comment}”` : ''}
                        </span>
                      )
                    })()}
                  </div>
                )}

                {outcome && (
                  <div data-testid="review-outcome" style={{ fontSize: 11, padding: '6px 8px', borderRadius: 6, background: '#1e1b4b', color: '#e0e7ff' }}>
                    <b>Outcome:</b>{' '}
                    {outcome.pnl_pts != null ? `${fmtSigned(outcome.pnl_pts, 2)} pts` : '—'}
                    {outcome.result ? ` · ${outcome.result}` : ''}
                    {outcome.exit_ts_pt ? ` · exit ${hhmm(outcome.exit_ts_pt)}` : ''}
                    {outcome.exit_price != null ? ` @ ${fmtNum(outcome.exit_price)}` : ''}
                    {(activeCandidate.reviews || []).some((r) => r.post_reveal) && (
                      <div style={{ marginTop: 2, color: '#a5b4fc', fontSize: 10 }}>
                        post-reveal reviews: {(activeCandidate.reviews || []).filter((r) => r.post_reveal).map((r) => r.label).join(', ')}
                      </div>
                    )}
                  </div>
                )}

                <KeyFacts cand={activeCandidate} />

                <button
                  type="button"
                  data-testid="review-details-toggle"
                  aria-expanded={detailsOpen}
                  onClick={() => setDetailsOpen((o) => !o)}
                  style={{ alignSelf: 'flex-start', background: 'transparent', border: 'none', color: '#64748b', fontSize: 10, cursor: 'pointer', padding: 0, fontFamily: 'inherit' }}
                >
                  {detailsOpen ? '▾' : '▸'} details
                </button>
                {detailsOpen && (
                  <div data-testid="review-details" style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '2px 8px', fontSize: 10 }}>
                    <KV k="cell" v={activeCandidate.cell || '—'} />
                    <KV k="entry" v={fmtNum(activeCandidate.entry_price)} />
                    <KV k="extreme" v={fmtNum(activeCandidate.extreme_price)} />
                    <KV k="frame" v={activeCandidate.meta?.level_frame || '—'} />
                    {/* CR-CC A3.1: meta as readable rows; raw JSON only behind "show raw" */}
                    <ReviewMeta meta={activeCandidate.meta} candidate={activeCandidate} omit={KEY_FACT_META_KEYS} />
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Queue */}
          <div style={{ padding: '4px 10px', fontSize: 10, color: '#64748b', borderBottom: '1px solid #1e293b' }}>
            {listLoading ? 'Loading…' : `${candidates.length} candidates · N / P to step`}
          </div>
          <div style={{ flex: '1 1 0', minHeight: 40, overflowY: 'auto' }}>
            {listError ? (
              <div data-testid="review-error" style={{ padding: 12, fontSize: 11, color: '#cbd5e1' }}>{listError}</div>
            ) : (
              candidates.map((c) => (
                <Row key={c.id} cand={c} active={c.id === activeId} onClick={() => activate(c.id)} />
              ))
            )}
          </div>

          {/* Actions: pinned to the bottom of the panel (CR-DB) */}
          {activeCandidate && (
            <div
              data-testid="review-actions"
              style={{ flex: 'none', borderTop: '1px solid #334155', background: '#020617', padding: '8px 10px', display: 'flex', flexDirection: 'column', gap: 6 }}
            >
              <div style={{ display: 'flex', gap: 6 }}>
                {LABELS.map((l) => (
                  <button
                    key={l.key}
                    type="button"
                    data-testid={`review-label-${l.key}`}
                    aria-pressed={label === l.key}
                    onClick={() => setLabel(l.key)}
                    style={{
                      ...buttonBase,
                      flex: 1,
                      color: label === l.key ? l.ink : l.color,
                      background: label === l.key ? l.color : '#0f172a',
                      borderColor: l.color,
                    }}
                  >
                    {l.text} <span style={{ opacity: 0.6, fontWeight: 500 }}>({l.hotkey})</span>
                  </button>
                ))}
              </div>

              <div data-testid="review-reasons" style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {reasonOptions.map((r) => {
                  const on = chosenReasons.includes(r)
                  return (
                    <button
                      key={r}
                      type="button"
                      aria-pressed={on}
                      onClick={() => toggleReason(r)}
                      style={{
                        ...buttonBase,
                        padding: '2px 7px',
                        fontSize: 10,
                        fontWeight: 600,
                        borderRadius: 10,
                        color: on ? '#020617' : '#94a3b8',
                        background: on ? '#93c5fd' : 'transparent',
                        borderColor: on ? '#93c5fd' : '#334155',
                      }}
                    >
                      {r}
                    </button>
                  )
                })}
              </div>
              <div style={{ display: 'flex', gap: 4 }}>
                <input
                  aria-label="Add reason"
                  value={newReason}
                  placeholder="add reason"
                  onChange={(e) => setNewReason(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addReason() } }}
                  style={{ ...selectStyle, flex: 1, fontSize: 11, padding: '4px 6px' }}
                />
                <button type="button" onClick={addReason} style={{ ...buttonBase, padding: '3px 8px' }}>+</button>
              </div>
              <textarea
                aria-label="Comment"
                value={comment}
                placeholder="comment"
                onChange={(e) => setComment(e.target.value)}
                rows={1}
                style={{ ...selectStyle, fontSize: 11, padding: '4px 6px', resize: 'vertical' }}
              />

              <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                <button
                  type="button"
                  data-testid="review-save"
                  onClick={save}
                  disabled={!label || busy}
                  style={{ ...buttonBase, flex: 1, opacity: !label || busy ? 0.5 : 1, borderColor: '#60a5fa', color: '#e2e8f0' }}
                >
                  {revealed ? 'Save post-reveal review' : 'Save'}
                </button>
                <button
                  type="button"
                  data-testid="review-reveal"
                  onClick={reveal}
                  disabled={!canReveal || busy}
                  title={revealed ? 'Already revealed' : canReveal ? 'Reveal the outcome (R)' : 'Save a review first'}
                  style={{ ...buttonBase, flex: 1, opacity: !canReveal || busy ? 0.5 : 1, borderColor: '#a78bfa', color: '#c4b5fd' }}
                >
                  Reveal (R)
                </button>
              </div>

              {actionError && <div style={{ fontSize: 10, color: '#e2e8f0', background: '#1e293b', borderRadius: 4, padding: '3px 6px' }}>{actionError}</div>}
            </div>
          )}
        </>
      )}
    </div>
  )
}
