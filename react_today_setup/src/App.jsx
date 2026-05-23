import { useState, useEffect, useCallback } from 'react';
import ContextStrip from './components/ContextStrip';
import ProposalCard from './components/ProposalCard';

function mostRecentTradingDay() {
  const d = new Date();
  const day = d.getDay(); // 0=Sun, 6=Sat
  if (day === 6) d.setDate(d.getDate() - 1); // Sat → Fri
  if (day === 0) d.setDate(d.getDate() - 2); // Sun → Fri
  return d.toISOString().slice(0, 10);
}

// In production the app is served at /today-setup/ by Flask; in dev
// we hit the Flask server directly. Vite's proxy config handles this if set,
// otherwise the Flask URL must be accessible.
const API_BASE = import.meta.env.VITE_API_BASE || '';

async function fetchProposals(date, spot, impliedMove, ticker = 'SPX') {
  const params = new URLSearchParams({ date, spot, implied_move: impliedMove, ticker });
  const r = await fetch(`${API_BASE}/api/setup/proposals?${params}`);
  if (r.status === 404) {
    const body = await r.json().catch(() => ({}));
    throw Object.assign(new Error(body.error || `No data for ${date}`), { is404: true });
  }
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

// Spot and implied_move defaults pulled from URL params for quick linking.
function parseQS() {
  const q = new URLSearchParams(window.location.search);
  return {
    date: q.get('date') || mostRecentTradingDay(),
    spot: q.get('spot') ? Number(q.get('spot')) : null,
    impliedMove: q.get('implied_move') ? Number(q.get('implied_move')) : null,
    ticker: q.get('ticker') || 'SPX',
  };
}

// Reasonable default spot from the context once loaded.
const DEFAULT_SPOT = 7400;
const DEFAULT_IM = 50;

export default function App() {
  const qs = parseQS();
  const [date, setDate] = useState(qs.date);
  const [ticker] = useState(qs.ticker);
  const [spot, setSpot] = useState(qs.spot || DEFAULT_SPOT);
  const [impliedMove, setImpliedMove] = useState(qs.impliedMove || DEFAULT_IM);

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const json = await fetchProposals(date, spot, impliedMove, ticker);
      if (!json.ok) throw new Error(json.error || 'API error');
      setData(json);
      // After first load, update spot/IM from the context (the DB's table_spot).
      if (json.context?.spot) setSpot(json.context.spot);
      if (json.context?.implied_move) setImpliedMove(json.context.implied_move);
    } catch (e) {
      setError(e.message);
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [date, spot, impliedMove, ticker]);

  useEffect(() => { load(); }, [load]);

  const context = data?.context;
  const proposals = data?.proposals || [];

  return (
    <div className="page">
      {/* ── Header ── */}
      <div className="header">
        <h1>Today's Setup</h1>
        <span className="ticker-pill">{ticker}</span>
        <input
          className="date-picker"
          type="date"
          value={date}
          onChange={e => setDate(e.target.value)}
        />
        {context && (
          <>
            <span className="spot-readout">
              Spot: <span className="spot-value">{Number(context.spot).toFixed(1)}</span>
            </span>
            {context.implied_move > 0 && (
              <span className="spot-readout">
                ±1σ: <span className="spot-value">{Number(context.implied_move).toFixed(1)}pt</span>
              </span>
            )}
          </>
        )}
        <a
          href="/"
          style={{ marginLeft: 'auto', color: '#60a5fa', fontSize: 12, textDecoration: 'none' }}
        >
          ← Dashboard
        </a>
      </div>

      {/* ── Context strip ── */}
      {context && <ContextStrip context={context} />}

      {/* ── Status ── */}
      {loading && <div className="loading-msg">Loading proposals…</div>}
      {error && (
        <div className="error-msg">
          {error.startsWith('No data for') || error.includes('landscape')
            ? `No data for ${date} — pick a different trade date.`
            : `Error: ${error}`}
        </div>
      )}

      {/* ── Proposals ── */}
      {!loading && !error && proposals.length > 0 && (
        <>
          <p className="section-heading">Proposals</p>
          <div className="proposals-grid">
            {proposals.map((p, i) => (
              <ProposalCard key={`${p.template_id}-${i}`} proposal={p} />
            ))}
          </div>
        </>
      )}

      {!loading && !error && data && proposals.length === 0 && (
        <div className="loading-msg">No proposals for this date.</div>
      )}
    </div>
  );
}
