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
//
// CR-DE: the card's label is its FIRST pre-reveal review and locks once saved
// (the API answers 409 label_locked to a second one). After Reveal every Save
// inserts a post-reveal note (comment, reasons, optional changed label; with
// no label picked the API copies the blind label); notes list under the
// outcome, newest first. Every save shows a confirmation or a visible error.
// CR-DK: on a revealed card the blind label's comment heads the notes, tagged "before reveal", and is counted;
// the "Saved:" line then shows the label and reasons only.
// The comment box is 5 lines and resizable; Cmd / Ctrl+Enter saves. The CR / set
// filter is remembered in localStorage, defaulting to the latest-loaded set.
//
// CR-DK: reasons are shown by display label (REASON_DISPLAY); the stored and sent string is unchanged. The
// retired "take fade (called side)" is no longer a chip (the API leaves it out) but still shows on saved reviews.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ReviewMeta from './ReviewMeta'
import { CardHeader, DIRECTION_COLOR, KEY_FACT_META_KEYS, KeyFacts } from './ReviewCard'

export const PANEL_WIDTH = 340

// CR-DB / C1: Dash palette — Take blue, Skip slate, Unsure light grey; `ink` is the text on the filled button.
const LABELS = [
  { key: 'take', text: 'Take', hotkey: 'T', color: '#60a5fa', ink: '#020617' },
  { key: 'skip', text: 'Skip', hotkey: 'S', color: '#94a3b8', ink: '#020617' },
  { key: 'unsure', text: 'Unsure', hotkey: 'U', color: '#cbd5e1', ink: '#020617' },
]
const LABEL_COLOR = Object.fromEntries(LABELS.map((l) => [l.key, l.color]))

// CR-DK: display label and tooltip per stored reason string. Only what is shown changes; the API stores, sends and
// counts the stored string.
const REASON_DISPLAY = {
  'take continuation (other edge)': {
    text: 'take opposite',
    title: "You'd take the opposite trade (approach direction, at the other edge of the box).",
  },
}
const reasonText = (r) => REASON_DISPLAY[r]?.text || r
const reasonList = (rs) => rs.map(reasonText).join(', ')

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

// reviewed_at is a UTC timestamp; notes show it in PT
function fmtPt(ts) {
  if (!ts) return '—'
  const d = new Date(ts)
  if (Number.isNaN(d.getTime())) return String(ts)
  return `${d.toLocaleString('en-US', {
    timeZone: 'America/Los_Angeles', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false,
  })} PT`
}

const SET_STORAGE_KEY = 'dash.review.set'

function readStoredSet() {
  try {
    const v = JSON.parse(window.localStorage.getItem(SET_STORAGE_KEY) || 'null')
    return v && typeof v.cr === 'string' && typeof v.set === 'string' ? v : null
  } catch {
    return null
  }
}

function storeSet(cr, set) {
  try {
    window.localStorage.setItem(SET_STORAGE_KEY, JSON.stringify({ cr, set }))
  } catch { /* storage unavailable: the choice just isn't remembered */ }
}

function latestSet(sets) {
  let best = null
  for (const s of sets) if (s.loaded_at && (!best || s.loaded_at > best.loaded_at)) best = s
  return best
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

function NoteHistory({ notes, blindReview }) {
  const blindLabel = blindReview?.label
  // CR-DK: the pre-reveal comment is the first entry (display only; the label stays locked)
  const entries = blindReview?.comment ? [{ ...blindReview, beforeReveal: true }, ...notes] : notes
  if (!entries.length) return null
  return (
    <div data-testid="review-notes" style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ fontSize: 10, color: '#64748b' }}>Notes ({entries.length})</div>
      {entries.map((n) => (
        <div
          key={n.id}
          data-testid="review-note"
          data-before-reveal={n.beforeReveal ? 'true' : undefined}
          style={{ fontSize: 11, padding: '4px 6px', borderRadius: 4, background: '#0f172a', border: '1px solid #1e293b' }}
        >
          <div style={{ fontSize: 10, color: '#94a3b8' }}>
            {fmtPt(n.reviewed_at)}
            {n.beforeReveal && <span data-testid="review-note-before-reveal"> · before reveal</span>}
            {!n.beforeReveal && n.label && n.label !== blindLabel && (
              <span> · label → <b style={{ color: LABEL_COLOR[n.label] }}>{n.label.toUpperCase()}</b></span>
            )}
            {!n.beforeReveal && n.reasons?.length ? ` · ${reasonList(n.reasons)}` : ''}
          </div>
          {n.comment && <div style={{ color: '#e2e8f0', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{n.comment}</div>}
        </div>
      ))}
    </div>
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
              {g.skip_reasons.map((r) => `${reasonText(r.reason)} (${r.n})`).join(' · ')}
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
  anchorRecord = null, // CR-DU amendment 7b: { anchor_ts, box_min_age_min, box_min_height_pts }, sent with every save
  onActivate,
  onClose,
  rightOffset = 0,
}) {
  const [filters, setFilters] = useState(() => {
    const stored = readStoredSet()
    return { cr: stored?.cr || '', set: stored?.set || '', cell: '', status: 'all' }
  })
  const setDefaultedRef = useRef(false)
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
  const [notice, setNotice] = useState('')
  const [detailsOpen, setDetailsOpen] = useState(false)

  const base = `${apiBase}/api/setup-review`
  const activeId = activeCandidate?.id || null
  const revealed = Boolean(activeCandidate?.revealed)
  const savedPreReveal = useMemo(
    () => (activeCandidate?.reviews || []).filter((r) => !r.post_reveal),
    [activeCandidate]
  )
  const blindReview = savedPreReveal[0] || null
  const locked = !revealed && Boolean(blindReview)
  const notes = useMemo(
    () => (activeCandidate?.reviews || []).filter((r) => r.post_reveal).slice().reverse(),
    [activeCandidate]
  )
  const canReveal = Boolean(activeCandidate) && !revealed && savedPreReveal.length > 0
  const canSave = Boolean(activeId) && !busy && (
    revealed
      ? Boolean(comment.trim() || chosenReasons.length || (label && label !== blindReview?.label))
      : Boolean(label) && !locked
  )

  // Filters can change while a list request is in flight (e.g. the first-load set default);
  // only the latest request's response is applied
  const listSeqRef = useRef(0)
  const loadList = useCallback(async () => {
    const seq = ++listSeqRef.current
    setListLoading(true)
    setListError('')
    try {
      const url = new URL(`${base}/candidates`, window.location.origin)
      if (filters.cr) url.searchParams.set('cr', filters.cr)
      if (filters.set) url.searchParams.set('set', filters.set)
      if (filters.cell) url.searchParams.set('cell', filters.cell)
      url.searchParams.set('status', filters.status)
      const body = await fetchJson(url.toString())
      if (seq !== listSeqRef.current) return
      setCandidates(body.candidates || [])
      setSets(body.sets || [])
      setReasonOptions(body.reasons || [])
      // First load: keep a stored set that still exists, else default to the latest-loaded set
      if (!setDefaultedRef.current) {
        setDefaultedRef.current = true
        const all = body.sets || []
        const stored = readStoredSet()
        const valid = stored && (stored.cr === '' || all.some((x) => x.cr_id === stored.cr)) &&
          (stored.set === '' || all.some((x) => x.set_name === stored.set && (stored.cr === '' || x.cr_id === stored.cr)))
        if (!valid) {
          const latest = latestSet(all)
          // only the set: CR stays on "all CRs" so every set remains in the Set list
          if (latest && latest.set_name !== filters.set) {
            setFilters((f) => ({ ...f, cr: '', set: latest.set_name, cell: '' }))
          }
        }
      }
    } catch (err) {
      if (seq !== listSeqRef.current) return
      setCandidates([])
      setListError(err.message)
    } finally {
      if (seq === listSeqRef.current) setListLoading(false)
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
    setNotice('')
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

  // Saves may follow each other faster than the card refreshes; only the latest refresh is applied
  const refreshSeqRef = useRef(0)
  const refreshActive = useCallback(async () => {
    if (!activeId) return
    const seq = ++refreshSeqRef.current
    const body = await fetchJson(`${base}/candidates/${encodeURIComponent(activeId)}`)
    if (seq === refreshSeqRef.current) onActivate?.(body.candidate)
  }, [activeId, base, onActivate])

  const step = useCallback((delta) => {
    if (!candidates.length) return
    const idx = candidates.findIndex((c) => c.id === activeId)
    const next = idx === -1 ? (delta > 0 ? 0 : candidates.length - 1) : idx + delta
    if (next < 0 || next >= candidates.length) return
    activate(candidates[next].id)
  }, [activate, activeId, candidates])

  const save = useCallback(async () => {
    if (!canSave) return
    setBusy(true)
    setActionError('')
    setNotice('')
    // After reveal the label is optional: omitted, the API copies the blind label (R1)
    const payload = revealed
      ? { ...(label ? { label } : {}), reasons: chosenReasons, comment, ...anchorRecord }
      : { label, reasons: chosenReasons, comment, ...anchorRecord }
    try {
      await fetchJson(`${base}/candidates/${encodeURIComponent(activeId)}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
    } catch (err) {
      setActionError(`Not saved: ${err.message}`)
      setBusy(false)
      return
    }
    setNotice(revealed ? 'Note saved ✓' : 'Saved ✓ — label locked')
    setComment('')
    setChosenReasons([])
    setNewReason('')
    if (revealed) setLabel(null)
    // After reveal the next note can be written while the card refreshes; before reveal Save
    // stays busy until the refreshed card shows the locked label
    if (revealed) setBusy(false)
    try {
      await refreshActive()
    } catch (err) {
      setActionError(`Saved, but the card did not refresh: ${err.message}`)
    } finally {
      if (!revealed) setBusy(false)
    }
    loadList()   // the queue's label column; not awaited, so the card is usable meanwhile
  }, [activeId, anchorRecord, base, canSave, chosenReasons, comment, label, loadList, refreshActive, revealed])

  const reveal = useCallback(async () => {
    if (!canReveal || busy) return
    setBusy(true)
    setActionError('')
    try {
      await fetchJson(`${base}/candidates/${encodeURIComponent(activeId)}/reveal`, { method: 'POST' })
      // A comment typed while the label was locked stays in the box, to be saved as a note
      setLabel(null)
      setChosenReasons([])
      setNotice('')
      await refreshActive()
    } catch (err) {
      setActionError(`Reveal failed: ${err.message}`)
      setBusy(false)
      return
    }
    setBusy(false)
    loadList()
  }, [activeId, base, busy, canReveal, loadList, refreshActive])

  // Keyboard: N/P, T/S/U, R. Ignored while typing in a field.
  const keyStateRef = useRef({})
  keyStateRef.current = { step, reveal, activeId, locked }
  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey || isTypingTarget(e.target)) return
      const k = e.key.toLowerCase()
      const st = keyStateRef.current
      if (k === 'n') st.step(1)
      else if (k === 'p') st.step(-1)
      else if (!st.activeId) return
      else if (k === 'r') st.reveal()
      else if (st.locked && (k === 't' || k === 's' || k === 'u')) return
      else if (k === 't') setLabel('take')
      else if (k === 's') setLabel('skip')
      else if (k === 'u') setLabel('unsure')
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
  const shownLabel = locked ? blindReview.label : label

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
          onChange={(e) => { storeSet(e.target.value, ''); setFilters((f) => ({ ...f, cr: e.target.value, set: '', cell: '' })) }}>
          <option value="">all CRs</option>
          {crOptions.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <select aria-label="Set" value={filters.set} style={selectStyle}
          onChange={(e) => { storeSet(filters.cr, e.target.value); setFilters((f) => ({ ...f, set: e.target.value, cell: '' })) }}>
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
                {blindReview && (
                  <div data-testid="review-saved" style={{ fontSize: 10, color: '#94a3b8', overflowWrap: 'anywhere' }}>
                    Saved:{' '}
                    <b style={{ color: LABEL_COLOR[blindReview.label] }}>{blindReview.label.toUpperCase()}</b>
                    {blindReview.reasons?.length ? ` · ${reasonList(blindReview.reasons)}` : ''}
                    {/* CR-DK: once revealed the comment is the first note, so it isn't repeated here */}
                    {blindReview.comment && !revealed && <span style={{ whiteSpace: 'pre-wrap' }}>{` · “${blindReview.comment}”`}</span>}
                  </div>
                )}

                {outcome && (
                  <div data-testid="review-outcome" style={{ fontSize: 11, padding: '6px 8px', borderRadius: 6, background: '#1e1b4b', color: '#e0e7ff' }}>
                    <b>Outcome:</b>{' '}
                    {outcome.pnl_pts != null ? `${fmtSigned(outcome.pnl_pts, 2)} pts` : '—'}
                    {outcome.result ? ` · ${outcome.result}` : ''}
                    {outcome.exit_ts_pt ? ` · exit ${hhmm(outcome.exit_ts_pt)}` : ''}
                    {outcome.exit_price != null ? ` @ ${fmtNum(outcome.exit_price)}` : ''}
                  </div>
                )}

                {revealed && <NoteHistory notes={notes} blindReview={blindReview} />}

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

          {/* Actions: pinned to the bottom of the panel (CR-DB). CR-DE: the chips and the comment box
              scroll in a capped region, so the label row and Save / Reveal stay in view at 900 × 600. */}
          {activeCandidate && (
            <div
              data-testid="review-actions"
              style={{ flex: 'none', borderTop: '1px solid #334155', background: '#020617', padding: '8px 10px', display: 'flex', flexDirection: 'column', gap: 6 }}
            >
              <div style={{ display: 'flex', gap: 6 }}>
                {LABELS.map((l) => {
                  const on = shownLabel === l.key
                  return (
                    <button
                      key={l.key}
                      type="button"
                      data-testid={`review-label-${l.key}`}
                      aria-pressed={on}
                      disabled={locked}
                      onClick={() => setLabel(revealed && label === l.key ? null : l.key)}
                      title={revealed ? 'Optional: a changed label for this note' : undefined}
                      style={{
                        ...buttonBase,
                        flex: 1,
                        color: on ? l.ink : l.color,
                        background: on ? l.color : '#0f172a',
                        borderColor: l.color,
                        cursor: locked ? 'default' : 'pointer',
                        opacity: locked && !on ? 0.5 : 1,
                      }}
                    >
                      {l.text} <span style={{ opacity: 0.6, fontWeight: 500 }}>({l.hotkey})</span>
                    </button>
                  )
                })}
              </div>
              {(locked || revealed) && (
                <div data-testid="review-mode-hint" style={{ fontSize: 10, color: '#94a3b8' }}>
                  {locked
                    ? 'Label locked after Save. Reveal (R) to add notes; anything typed below is kept for the first note.'
                    : 'Notes: each Save adds a note. Label optional (a changed label); none picked keeps the saved label.'}
                </div>
              )}

              <div data-testid="review-form-scroll" style={{ overflowY: 'auto', maxHeight: '32vh', minHeight: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
                <div data-testid="review-reasons" style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {reasonOptions.map((r) => {
                    const on = chosenReasons.includes(r)
                    return (
                      <button
                        key={r}
                        type="button"
                        title={REASON_DISPLAY[r]?.title}
                        data-reason={r}
                        aria-pressed={on}
                        disabled={locked}
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
                          opacity: locked ? 0.5 : 1,
                          cursor: locked ? 'default' : 'pointer',
                        }}
                      >
                        {reasonText(r)}
                      </button>
                    )
                  })}
                </div>
                <div style={{ display: 'flex', gap: 4 }}>
                  <input
                    aria-label="Add reason"
                    value={newReason}
                    placeholder="add reason"
                    disabled={locked}
                    onChange={(e) => setNewReason(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addReason() } }}
                    style={{ ...selectStyle, flex: 1, fontSize: 11, padding: '4px 6px' }}
                  />
                  <button type="button" onClick={addReason} disabled={locked} style={{ ...buttonBase, padding: '3px 8px' }}>+</button>
                </div>
                <textarea
                  aria-label="Comment"
                  data-testid="review-comment"
                  value={comment}
                  placeholder={revealed ? 'note (Cmd/Ctrl+Enter saves)' : 'comment (Cmd/Ctrl+Enter saves)'}
                  onChange={(e) => setComment(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); save() }
                  }}
                  rows={5}
                  style={{ ...selectStyle, fontSize: 11, lineHeight: 1.35, padding: '4px 6px', resize: 'vertical', minHeight: 72, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', flex: 'none' }}
                />
              </div>

              <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                <button
                  type="button"
                  data-testid="review-save"
                  onClick={save}
                  disabled={!canSave}
                  title={locked ? 'Label locked — Reveal to add notes' : revealed ? 'Save a note (Cmd/Ctrl+Enter)' : 'Save the label (Cmd/Ctrl+Enter)'}
                  style={{ ...buttonBase, flex: 1, opacity: canSave ? 1 : 0.5, borderColor: '#60a5fa', color: '#e2e8f0' }}
                >
                  {revealed ? 'Save note' : locked ? 'Saved' : 'Save'}
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

              {notice && !actionError && (
                <div data-testid="review-notice" role="status" style={{ fontSize: 11, fontWeight: 700, color: '#020617', background: '#93c5fd', borderRadius: 4, padding: '3px 6px' }}>
                  {notice}
                </div>
              )}
              {actionError && (
                <div data-testid="review-action-error" role="alert" style={{ fontSize: 11, fontWeight: 700, color: '#020617', background: '#cbd5e1', border: '1px solid #f8fafc', borderRadius: 4, padding: '3px 6px' }}>
                  ⚠ {actionError}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}
