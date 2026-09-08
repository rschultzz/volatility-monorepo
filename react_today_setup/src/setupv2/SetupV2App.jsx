import { useEffect, useState } from 'react';
import TradeTab from './components/TradeTab';
import KnnFactorsTab from './components/KnnFactorsTab';
import { longDate, pts } from './format';

const API_BASE = import.meta.env.VITE_API_BASE || '';
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const DATE_KEY = 'setup-v2-date';

function mostRecentTradingDay() {
  const d = new Date();
  const day = d.getDay();
  if (day === 6) d.setDate(d.getDate() - 1);
  if (day === 0) d.setDate(d.getDate() - 2);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${dd}`;
}

function initialDate() {
  const q = new URLSearchParams(window.location.search);
  const urlDate = q.get('date');
  if (urlDate && DATE_RE.test(urlDate)) return urlDate;
  try {
    const stored = sessionStorage.getItem(DATE_KEY);
    if (stored && DATE_RE.test(stored)) return stored;
  } catch { /* ignore */ }
  return mostRecentTradingDay();
}

export async function fetchCard(date, ticker, signal) {
  const params = new URLSearchParams({ date, ticker });
  const r = await fetch(`${API_BASE}/api/setup-v2/card?${params}`, { signal });
  const body = await r.json().catch(() => ({}));
  if (!r.ok || body.ok === false) {
    throw new Error(body.error || `HTTP ${r.status}`);
  }
  return body;
}

export function TopNav({ selected }) {
  return (
    <>
      <div className="top-bar">
        <span className="top-bar-title">Surface Dynamics</span>
        <a href="https://blog.surfacedynamics.io" className="top-bar-home">Home</a>
      </div>
      <div className="top-nav">
        <a href="/" className="top-nav-pill">Dashboard</a>
        <a href="/?tab=price-chart" className="top-nav-pill">Price Chart</a>
        <a href="/?tab=backtests" className="top-nav-pill">Backtests</a>
        <a href="/today-setup" className={`top-nav-pill${selected === 'v1' ? ' selected' : ''}`}>Today's Setup</a>
        <a href="/setup-v2" className={`top-nav-pill${selected === 'v2' ? ' selected' : ''}`}>Setup v2</a>
      </div>
    </>
  );
}

/** The two-tab card (Trade · KNN factors). Pure render of one payload. */
export function SetupV2Card({ card, apiBase = API_BASE }) {
  const [tab, setTab] = useState(0);
  return (
    <div className="card" data-testid="setup-v2-card">
      <div className="tabs" role="tablist">
        <button type="button" role="tab" aria-selected={tab === 0} className={`tab${tab === 0 ? ' on' : ''}`} onClick={() => setTab(0)}>Trade</button>
        <button type="button" role="tab" aria-selected={tab === 1} className={`tab${tab === 1 ? ' on' : ''}`} onClick={() => setTab(1)}>KNN factors — where today sits</button>
      </div>
      {tab === 0 ? <TradeTab card={card} apiBase={apiBase} /> : <KnnFactorsTab card={card} />}
    </div>
  );
}

export default function SetupV2App() {
  const [date, setDate] = useState(initialDate);
  const [ticker] = useState(() => new URLSearchParams(window.location.search).get('ticker') || 'SPX');
  const [card, setCard] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    try { sessionStorage.setItem(DATE_KEY, date); } catch { /* ignore */ }
  }, [date]);

  useEffect(() => {
    if (!date || !DATE_RE.test(date)) return undefined;
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    setCard(null);
    fetchCard(date, ticker, controller.signal)
      .then(c => { if (!controller.signal.aborted) setCard(c); })
      .catch(e => { if (!controller.signal.aborted) setError(e.message); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [date, ticker]);

  const ctx = card?.context;
  return (
    <div className="v2-page">
      <TopNav selected="v2" />
      <div className="v2-head">
        <h1>Proposal card — {longDate(date)}</h1>
        <input className="date-picker" type="date" value={date} onChange={e => setDate(e.target.value)} aria-label="Trade date" />
        {ctx && (
          <span className="readout">
            {ticker} · spot <b>{pts(ctx.spot, 1)}</b>{ctx.spot_source && ctx.spot_source !== 'param' ? ` (${ctx.spot_source})` : ''} · regime <b>{ctx.regime || '—'}</b>
          </span>
        )}
      </div>
      <p className="intro">
        Led by the one number: the most this spread is worth paying, from what it was worth at the close on the analogue days.
        Today's quote sits on that gauge. Under, enter. Over, skip.
      </p>
      {loading && <div className="v2-status">Loading…</div>}
      {error && <div className="v2-error" data-testid="v2-error">{error}</div>}
      {!loading && card && <SetupV2Card card={card} />}
    </div>
  );
}
