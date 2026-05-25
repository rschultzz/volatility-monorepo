import { useState, useEffect, useCallback, useRef } from 'react';
import ContextStrip from './components/ContextStrip';
import ProposalCard from './components/ProposalCard';
import DayView from './components/DayView';
import DayListAnalogue from './components/DayListAnalogue';
import DayListBrowse from './components/DayListBrowse';
import StructuralProbabilityBlock from './components/StructuralProbabilityBlock';

function mostRecentTradingDay() {
  const d = new Date();
  const day = d.getDay();
  if (day === 6) d.setDate(d.getDate() - 1);
  if (day === 0) d.setDate(d.getDate() - 2);
  return d.toISOString().slice(0, 10);
}

const API_BASE = import.meta.env.VITE_API_BASE || '';

function parseQS() {
  const q = new URLSearchParams(window.location.search);
  return {
    date: q.get('date') || mostRecentTradingDay(),
    ticker: q.get('ticker') || 'SPX',
    mode: q.get('mode') || 'analogue',
  };
}

// ── API helpers ──────────────────────────────────────────────────────────────

async function fetchProposals(date, ticker, signal) {
  const params = new URLSearchParams({ date, ticker });
  const r = await fetch(`${API_BASE}/api/setup/proposals?${params}`, { signal });
  if (r.status === 404) {
    const body = await r.json().catch(() => ({}));
    throw Object.assign(new Error(body.error || `No data for ${date}`), { is404: true });
  }
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

async function fetchAnalogues(date, ticker, signal, k = 5) {
  const params = new URLSearchParams({ date, ticker, k });
  const r = await fetch(`${API_BASE}/api/analogues?${params}`, { signal });
  if (!r.ok) return null;
  return r.json();
}

async function fetchLandscape(date, ticker, signal) {
  const params = new URLSearchParams({ date, ticker });
  const r = await fetch(`${API_BASE}/api/gex-landscape?${params}`, { signal });
  if (!r.ok) return null;
  return r.json();
}

async function fetchFlags(date, ticker, signal) {
  const params = new URLSearchParams({ date, ticker });
  const r = await fetch(`${API_BASE}/api/audit-flags?${params}`, { signal });
  if (!r.ok) return { flags: [] };
  return r.json();
}

async function fetchBrowse(regime, ticker, dateFrom, dateTo) {
  const params = new URLSearchParams({ regime, ticker });
  if (dateFrom) params.set('from', dateFrom);
  if (dateTo) params.set('to', dateTo);
  const r = await fetch(`${API_BASE}/api/days?${params}`);
  if (!r.ok) return { days: [] };
  return r.json();
}

async function postFlag(body) {
  const r = await fetch(`${API_BASE}/api/audit-flags`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return r.json();
}

async function deleteFlag(flagId) {
  const r = await fetch(`${API_BASE}/api/audit-flags/${flagId}`, { method: 'DELETE' });
  return r.json();
}

async function promoteFlag(flagId) {
  const r = await fetch(`${API_BASE}/api/audit-flags/${flagId}/promote`, { method: 'POST' });
  return r.json();
}

async function demoteFlag(flagId) {
  const r = await fetch(`${API_BASE}/api/audit-flags/${flagId}/demote`, { method: 'POST' });
  return r.json();
}

// ── Main component ───────────────────────────────────────────────────────────

export default function App() {
  const qs = parseQS();
  const [mode, setMode] = useState(qs.mode); // 'analogue' | 'browse'
  const [date, setDate] = useState(qs.date);
  const [ticker] = useState(qs.ticker);

  // Browse-mode filters
  const [browseRegime, setBrowseRegime] = useState('magnetic-pin');
  const [browseFrom, setBrowseFrom] = useState('');
  const [browseTo, setBrowseTo] = useState('');

  // Data state
  const [proposals, setProposals] = useState(null);     // { ok, context, proposals }
  const [analogues, setAnalogues] = useState(null);     // { ok, anchor, analogues }
  const [anchorLandscape, setAnchorLandscape] = useState(null);
  const [browseDays, setBrowseDays] = useState([]);
  const [flags, setFlags] = useState([]);               // bt_audit_flags for current date

  // UI state
  const [selectedDate, setSelectedDate] = useState(null);
  const [selectedLandscape, setSelectedLandscape] = useState(null);
  const [selectedAnalogue, setSelectedAnalogue] = useState(null); // full analogue object
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const listRef = useRef(null);
  const anchorAbortRef = useRef(null);

  // ── Anchor / proposals fetch ─────────────────────────────────────────────
  const loadAnchor = useCallback(async () => {
    if (!date) return;

    // Cancel any in-flight request from a prior date change.
    anchorAbortRef.current?.abort();
    const controller = new AbortController();
    anchorAbortRef.current = controller;
    const { signal } = controller;

    setLoading(true);
    setError(null);
    setProposals(null);
    setAnalogues(null);
    setSelectedDate(null);
    setSelectedLandscape(null);
    setSelectedAnalogue(null);
    setFlags([]);

    try {
      const results = await Promise.allSettled([
        fetchProposals(date, ticker, signal),
        fetchAnalogues(date, ticker, signal),
        fetchLandscape(date, ticker, signal),
        fetchFlags(date, ticker, signal),
      ]);

      // This request was superseded — discard results.
      if (signal.aborted) return;

      const [propR, analogR, landscapeR, flagsR] = results;

      // Proposals — surface error but don't block the other panels.
      if (propR.status === 'fulfilled' && propR.value?.ok) {
        setProposals(propR.value);
        setError(null);  // clear any stale error from a prior slow request
      } else {
        const msg = propR.status === 'rejected'
          ? propR.reason?.message || 'Proposals error'
          : (propR.value?.error || 'Proposals error');
        setError(msg);
      }

      // Analogues, landscape, flags — independent; use whatever succeeded.
      if (analogR.status === 'fulfilled' && analogR.value?.ok) setAnalogues(analogR.value);
      if (landscapeR.status === 'fulfilled' && landscapeR.value?.ok !== false) setAnchorLandscape(landscapeR.value);
      if (flagsR.status === 'fulfilled' && flagsR.value?.flags) setFlags(flagsR.value.flags);
    } finally {
      if (!signal.aborted) setLoading(false);
    }
  }, [date, ticker]);

  useEffect(() => {
    if (mode === 'analogue') loadAnchor();
  }, [mode, loadAnchor]);

  // ── Browse fetch ──────────────────────────────────────────────────────────
  const loadBrowse = useCallback(async () => {
    if (!browseRegime) return;
    setLoading(true);
    setError(null);
    setBrowseDays([]);
    setSelectedDate(null);
    setSelectedLandscape(null);
    setFlags([]);
    try {
      const res = await fetchBrowse(browseRegime, ticker, browseFrom, browseTo);
      setBrowseDays(res.days || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [browseRegime, ticker, browseFrom, browseTo]);

  useEffect(() => {
    if (mode === 'browse') loadBrowse();
  }, [mode, loadBrowse]);

  // ── Selected-day landscape fetch ─────────────────────────────────────────
  const loadSelectedLandscape = useCallback(async (d) => {
    if (!d) { setSelectedLandscape(null); return; }
    const res = await fetchLandscape(d, ticker);
    if (res?.ok !== false) setSelectedLandscape(res);
    else setSelectedLandscape(null);
  }, [ticker]);

  const loadSelectedFlags = useCallback(async (d) => {
    if (!d || mode !== 'browse') return;
    const res = await fetchFlags(d, ticker);
    if (res?.flags) setFlags(res.flags);
  }, [ticker, mode]);

  // ── Arrow-key navigation ─────────────────────────────────────────────────
  const list = mode === 'analogue'
    ? (analogues?.analogues || []).map(a => a.trade_date)
    : browseDays.map(d => d.trade_date);

  useEffect(() => {
    function onKey(e) {
      if (!['ArrowDown', 'ArrowUp'].includes(e.key)) return;
      e.preventDefault();
      const idx = list.indexOf(selectedDate);
      let next = -1;
      if (e.key === 'ArrowDown') next = idx < list.length - 1 ? idx + 1 : idx;
      else next = idx > 0 ? idx - 1 : 0;
      if (next >= 0 && list[next] !== selectedDate) {
        handleSelectDay(list[next]);
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [list, selectedDate]) // eslint-disable-line react-hooks/exhaustive-deps

  function handleSelectDay(d) {
    setSelectedDate(d);
    if (mode === 'analogue') {
      const a = (analogues?.analogues || []).find(x => x.trade_date === d);
      setSelectedAnalogue(a || null);
    }
    loadSelectedLandscape(d);
    if (mode === 'browse') loadSelectedFlags(d);
  }

  // ── Flag operations ───────────────────────────────────────────────────────
  function flagsForDate(d) {
    return flags.filter(f =>
      f.flag_type === 'regime_wrong'
        ? f.trade_date === d
        : f.trade_date === d || f.analogue_date === d
    );
  }

  function regimeFlagForDate(d) {
    return flags.find(f => f.flag_type === 'regime_wrong' && f.trade_date === d) || null;
  }

  async function handleRegimeFlag(d, corrected) {
    const autoRegime = d === date
      ? proposals?.context?.regime
      : (selectedAnalogue?.auto_regime || selectedAnalogue?.regime);
    await postFlag({
      flag_type: 'regime_wrong',
      ticker,
      trade_date: d,
      corrected_regime: corrected,
      auto_regime: autoRegime,
    });
    const res = await fetchFlags(d === date ? date : d, ticker);
    if (res?.flags) setFlags(prev => {
      const other = prev.filter(f => f.trade_date !== d || f.flag_type !== 'regime_wrong');
      return [...other, ...res.flags.filter(f => f.flag_type === 'regime_wrong' && f.trade_date === d)];
    });
  }

  async function handleDeleteFlag(flagId) {
    await deleteFlag(flagId);
    setFlags(prev => prev.filter(f => f.flag_id !== flagId));
  }

  async function handlePromote(flagId) {
    const res = await promoteFlag(flagId);
    if (res.flag) setFlags(prev => prev.map(f => f.flag_id === flagId ? res.flag : f));
  }

  async function handleDemote(flagId) {
    const res = await demoteFlag(flagId);
    if (res.flag) setFlags(prev => prev.map(f => f.flag_id === flagId ? res.flag : f));
  }

  async function handlePairFlag(anchorDate, analogueDate) {
    await postFlag({
      flag_type: 'not_a_true_analogue',
      ticker,
      trade_date: anchorDate,
      analogue_date: analogueDate,
    });
    // Remove from the current analogue list
    setAnalogues(prev => {
      if (!prev) return prev;
      return { ...prev, analogues: prev.analogues.filter(a => a.trade_date !== analogueDate) };
    });
    setSelectedDate(null);
    setSelectedLandscape(null);
    setSelectedAnalogue(null);
  }

  // ── Derived values ────────────────────────────────────────────────────────
  const context = proposals?.context;
  const structuralProb = proposals?.structural_probability;
  const anchorRegime = context?.regime || null;
  const anchorFlag = regimeFlagForDate(date);
  const selectedFlag = selectedDate ? regimeFlagForDate(selectedDate) : null;
  const selectedRegime = selectedAnalogue?.regime
    || (selectedDate && browseDays.find(d => d.trade_date === selectedDate)?.regime)
    || null;

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="page">
      {/* ── Top bar Row 1: title + Home link ── */}
      <div className="top-bar">
        <span className="top-bar-title">Surface Dynamics</span>
        <a href="https://blog.surfacedynamics.io" className="top-bar-home">Home</a>
      </div>

      {/* ── Top bar Row 2: four-pill nav ── */}
      <div className="top-nav">
        <a href="/" className="top-nav-pill">Dashboard</a>
        <a href="/?tab=price-chart" className="top-nav-pill">Price Chart</a>
        <a href="/?tab=backtests" className="top-nav-pill">Backtests</a>
        <a href="/today-setup" className="top-nav-pill selected">Today's Setup</a>
      </div>

      {/* ── Header ── */}
      <div className="header">
        <h1>Day Setup</h1>
        <span className="ticker-pill">{ticker}</span>

        {/* Mode toggle */}
        <div style={{ display: 'inline-flex', borderRadius: 6, overflow: 'hidden', border: '1px solid #334155', marginLeft: 8 }}>
          {['analogue', 'browse'].map(m => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              style={{
                padding: '4px 12px',
                fontSize: 11,
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                border: 'none',
                cursor: 'pointer',
                background: mode === m ? '#1d4ed8' : '#0f172a',
                color: mode === m ? '#dbeafe' : '#64748b',
                fontFamily: 'inherit',
              }}
            >
              {m}
            </button>
          ))}
        </div>

        {/* Mode-specific controls */}
        {mode === 'analogue' ? (
          <input
            className="date-picker"
            type="date"
            value={date}
            onChange={e => setDate(e.target.value)}
          />
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginLeft: 8 }}>
            <select
              value={browseRegime}
              onChange={e => setBrowseRegime(e.target.value)}
              style={selectStyle}
            >
              {['magnetic-pin','magnet-above','magnet-below','bounded','amplification','untethered','broken-magnet'].map(r => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
            <input type="date" value={browseFrom} onChange={e => setBrowseFrom(e.target.value)} style={dateInputStyle} placeholder="from" />
            <input type="date" value={browseTo} onChange={e => setBrowseTo(e.target.value)} style={dateInputStyle} placeholder="to" />
            <button
              type="button"
              onClick={loadBrowse}
              style={{ padding: '3px 10px', fontSize: 11, background: '#1d4ed8', color: '#dbeafe', border: 'none', borderRadius: 5, cursor: 'pointer', fontFamily: 'inherit' }}
            >
              Filter
            </button>
          </div>
        )}

        {/* Spot / IM from resolved context */}
        {context && (
          <>
            <span className="spot-readout">
              Spot: <span className="spot-value">{Number(context.spot).toFixed(1)}</span>
              {context.spot_source && context.spot_source !== 'param' && (
                <span style={{ fontSize: 9, color: '#64748b', marginLeft: 3 }}>({context.spot_source})</span>
              )}
            </span>
            {context.implied_move > 0 && (
              <span className="spot-readout">
                ±1σ: <span className="spot-value">{Number(context.implied_move).toFixed(1)}pt</span>
              </span>
            )}
          </>
        )}

        {mode === 'browse' && browseDays.length > 0 && (
          <span style={{ fontSize: 11, color: '#64748b' }}>{browseDays.length} days</span>
        )}

      </div>

      {/* ── Context strip (analogue mode only) ── */}
      {mode === 'analogue' && context && <ContextStrip context={context} />}

      {/* ── Status ── */}
      {loading && <div className="loading-msg">Loading…</div>}
      {error && <div className="error-msg">{error}</div>}

      {/* ── Main two-pane layout ── */}
      {!loading && (
        <div style={{ display: 'flex', gap: 16, marginTop: 16, alignItems: 'flex-start' }}>

          {/* Left: day list */}
          <div style={{ width: 280, flexShrink: 0, overflowY: 'auto', maxHeight: 'calc(100vh - 180px)' }} ref={listRef}>
            {mode === 'analogue' ? (
              <DayListAnalogue
                analogues={analogues?.analogues || []}
                selectedDate={selectedDate}
                onSelect={handleSelectDay}
              />
            ) : (
              <DayListBrowse
                days={browseDays}
                selectedDate={selectedDate}
                onSelect={handleSelectDay}
              />
            )}
          </div>

          {/* Right: dual view — anchor on top, selected beneath (vertical stack) */}
          <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 16 }}>

            {/* Anchor view (analogue mode) */}
            {mode === 'analogue' && (
              <div style={{ width: '100%' }}>
                <DayView
                  label="Anchor"
                  date={date}
                  ticker={ticker}
                  apiBase={API_BASE}
                  landscapeData={anchorLandscape}
                  regime={anchorRegime}
                  autoRegime={anchorRegime}
                  flag={anchorFlag}
                  allowPairFlag={false}
                  onRegimeFlag={corrected => handleRegimeFlag(date, corrected)}
                  onPromote={() => anchorFlag && handlePromote(anchorFlag.flag_id)}
                  onDemote={() => anchorFlag && handleDemote(anchorFlag.flag_id)}
                  onDeleteFlag={() => anchorFlag && handleDeleteFlag(anchorFlag.flag_id)}
                />
              </div>
            )}

            {/* Selected-day view */}
            <div style={{ width: '100%' }}>
              <DayView
                label="Selected"
                date={selectedDate}
                ticker={ticker}
                apiBase={API_BASE}
                landscapeData={selectedLandscape}
                regime={selectedRegime}
                autoRegime={selectedAnalogue?.auto_regime || selectedRegime}
                flag={selectedFlag}
                allowPairFlag={mode === 'analogue' && selectedDate != null}
                onRegimeFlag={corrected => selectedDate && handleRegimeFlag(selectedDate, corrected)}
                onPromote={() => selectedFlag && handlePromote(selectedFlag.flag_id)}
                onDemote={() => selectedFlag && handleDemote(selectedFlag.flag_id)}
                onDeleteFlag={() => selectedFlag && handleDeleteFlag(selectedFlag.flag_id)}
                onPairFlag={() => selectedDate && handlePairFlag(date, selectedDate)}
              />
            </div>
          </div>
        </div>
      )}

      {/* ── Structural probability (analogue mode only) ── */}
      {!loading && mode === 'analogue' && proposals && (
        <StructuralProbabilityBlock sp={structuralProb} />
      )}

      {/* ── Proposals (analogue mode only) ── */}
      {!loading && !error && mode === 'analogue' && proposals?.proposals?.length > 0 && (
        <>
          <p className="section-heading" style={{ marginTop: 24 }}>Proposals</p>
          <div className="proposals-grid">
            {(proposals.proposals || []).map((p, i) => (
              <ProposalCard key={`${p.template_id}-${i}`} proposal={p} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

const selectStyle = {
  background: '#0f172a', color: '#e2e8f0', border: '1px solid #334155',
  borderRadius: 5, padding: '3px 8px', fontSize: 11, fontFamily: 'inherit',
};

const dateInputStyle = {
  background: '#0f172a', color: '#94a3b8', border: '1px solid #334155',
  borderRadius: 5, padding: '3px 8px', fontSize: 11, fontFamily: 'inherit', width: 120,
};
