import { useState, useEffect, useRef, useCallback } from 'react';
import LegTable from './LegTable';
import { ProposalEdgeChart } from 'web-shared';

const TEMPLATE_LABELS = {
  pin_butterfly_tight:         'Pin Butterfly — Tight Wings',
  pin_butterfly_medium:        'Pin Butterfly — Medium Wings',
  pin_butterfly_wide:          'Pin Butterfly — Wide Wings',
  directional_spread_to_target:'Directional Spread to Target',
  debit_spread_to_target:      'Debit Spread to Target',
  bounded_iron_condor:         'Bounded Iron Condor',
  feature_no_trade:            'No Trade',
};

const RECIPE_LABELS = {
  half_fwhm: 'half-FWHM',
  full_fwhm: 'full-FWHM',
  sigma_1x:  '1σ',
};

const TIMEFRAMES = ['t1', 't5', 't15'];
const DEFAULT_TIMEFRAME = 't5';

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Add `days` calendar days to a "YYYY-MM-DD" string. */
function addCalendarDays(dateStr, days) {
  const d = new Date(dateStr + 'T00:00:00Z');
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

/** Map proposal leg shape → pl-data leg shape. */
function adaptLeg(leg, expiration) {
  return {
    strike:     leg.strike,
    expiration,
    flag:       leg.type === 'call' ? 'c' : 'p',
    side:       leg.side,
    qty:        leg.quantity ?? 1,
  };
}

/** Build the POST /api/proposals/pl-data request body. */
function buildPlDataBody(proposal, date, ticker, timeframe, contextRegime) {
  const expiration = addCalendarDays(date, proposal.expiry_dte_target ?? 0);
  const regime = proposal.source?.regime || contextRegime || 'amplification';
  return {
    trade_date:   date,
    ticker,
    timeframe,
    regime_block: { regime },
    legs:         (proposal.legs || []).map(leg => adaptLeg(leg, expiration)),
  };
}

// ── Sub-components ────────────────────────────────────────────────────────────

function SourceLine({ source, wingRecipe }) {
  if (!source) return null;
  const parts = [];
  if (source.type === 'cluster') {
    parts.push(`Cluster @ ${Number(source.cluster_center).toFixed(1)}`);
    if (source.cluster_quality) parts.push(`quality: ${source.cluster_quality}`);
    if (source.cluster_max_gex) {
      const gexB = (Number(source.cluster_max_gex) / 1e9).toFixed(0);
      parts.push(`${gexB}B GEX`);
    }
    if (source.cluster_avg_fwhm) parts.push(`FWHM ${Number(source.cluster_avg_fwhm).toFixed(0)}pt`);
    if (wingRecipe && RECIPE_LABELS[wingRecipe]) {
      parts.push(`wings: ${RECIPE_LABELS[wingRecipe]}`);
    }
  } else if (source.type === 'regime_target') {
    parts.push(`Drift target @ ${Number(source.drift_target).toFixed(1)}`);
    if (source.dominant_wall_gex_b) parts.push(`${source.dominant_wall_gex_b}B GEX`);
    parts.push(`regime: ${source.regime}`);
  } else if (source.type === 'containment_zone') {
    parts.push(`Containment zone: ${Number(source.lower_price).toFixed(1)}–${Number(source.upper_price).toFixed(1)}`);
    if (source.lower_gex_b) parts.push(`L ${source.lower_gex_b}B / U ${source.upper_gex_b}B`);
  } else if (source.regime) {
    parts.push(`regime: ${source.regime}`);
  }
  return <div className="source-line">{parts.join(' · ')}</div>;
}

// ── Expanded chart panel ───────────────────────────────────────────────────────

function ExpandedPanel({ proposal, date, ticker, apiBase, contextRegime }) {
  const [timeframe, setTimeframe]   = useState(DEFAULT_TIMEFRAME);
  const [chartData, setChartData]   = useState(undefined); // undefined = not yet fetched
  const cacheRef = useRef({});   // timeframe → resolved data (or error shape)
  const containerRef = useRef(null);
  const [chartWidth, setChartWidth] = useState(640);

  // ResizeObserver — tracks panel width for chart sizing.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => {
      const w = entries[0]?.contentRect?.width;
      if (w > 0) setChartWidth(Math.floor(w));
    });
    ro.observe(el);
    setChartWidth(Math.floor(el.clientWidth) || 640);
    return () => ro.disconnect();
  }, []);

  // Fetch pl-data when timeframe changes (or on first mount).
  const fetchPlData = useCallback(async (tf) => {
    if (cacheRef.current[tf] !== undefined) {
      setChartData(cacheRef.current[tf]);
      return;
    }
    // Loading state while request is in-flight.
    setChartData(undefined);
    const body = buildPlDataBody(proposal, date, ticker, tf, contextRegime);
    try {
      const resp = await fetch(`${apiBase}/api/proposals/pl-data`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(body),
      });
      const json = await resp.json();
      cacheRef.current[tf] = json;
      setChartData(json);
    } catch (err) {
      const errShape = { ok: false, error: String(err) };
      cacheRef.current[tf] = errShape;
      setChartData(errShape);
    }
  }, [proposal, date, ticker, apiBase, contextRegime]);

  useEffect(() => {
    fetchPlData(timeframe);
  }, [timeframe, fetchPlData]);

  function handleTimeframeChange(tf) {
    if (tf === timeframe) return;
    // Invalidate cache for the new timeframe so a fresh fetch is made.
    delete cacheRef.current[tf];
    setTimeframe(tf);
  }

  return (
    <div
      data-testid="proposal-expanded-panel"
      style={{ borderTop: '1px solid #1f2937', paddingTop: 12, display: 'flex', flexDirection: 'column', gap: 8 }}
    >
      {/* Timeframe selector + legend header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 700 }}>
          Edge Chart
        </span>
        <div
          style={{ display: 'inline-flex', borderRadius: 5, overflow: 'hidden', border: '1px solid #334155' }}
          aria-label="Timeframe"
          data-testid="timeframe-selector"
        >
          {TIMEFRAMES.map(tf => (
            <button
              key={tf}
              type="button"
              aria-pressed={tf === timeframe}
              onClick={() => handleTimeframeChange(tf)}
              style={{
                padding:     '2px 9px',
                fontSize:    10,
                fontWeight:  700,
                border:      'none',
                cursor:      'pointer',
                background:  tf === timeframe ? '#1d4ed8' : '#0f172a',
                color:       tf === timeframe ? '#dbeafe' : '#64748b',
                fontFamily:  'inherit',
              }}
            >
              {tf}
            </button>
          ))}
        </div>
        <span style={{ fontSize: 9, color: '#475569', marginLeft: 'auto' }}>
          Green = struct prob &gt; implied · Red = struct prob &lt; implied
        </span>
      </div>

      {/* Chart area — full card width */}
      <div ref={containerRef} style={{ width: '100%' }}>
        {/* null data → ProposalEdgeChart renders its own skeleton */}
        <ProposalEdgeChart
          data={chartData === undefined ? null : chartData}
          width={chartWidth}
          height={340}
        />
      </div>
    </div>
  );
}

// ── ProposalCard ──────────────────────────────────────────────────────────────

export default function ProposalCard({
  proposal,
  // New props for pl-data fetch (optional — gracefully absent for no-trade cards)
  date,
  ticker,
  apiBase,
  context,
}) {
  const {
    template_id,
    template_kind,
    rationale,
    legs,
    expiry_dte_target,
    expiry_dte_bucket,
    source,
    wing_distance_recipe,
  } = proposal;

  const isNoTrade    = template_kind === 'no_trade';
  const canExpand    = !isNoTrade && Array.isArray(legs) && legs.length > 0 && !!date;
  const label        = TEMPLATE_LABELS[template_id] || template_id;
  const contextRegime = context?.regime;

  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={`proposal-card${isNoTrade ? ' no-trade' : ''}`}
      data-testid="proposal-card"
    >
      <div className="proposal-label">
        <span className={`kind-badge kind-${template_kind}`}>
          {template_kind.replace('_', ' ')}
        </span>
        {label}
      </div>

      {isNoTrade ? (
        <div className="no-trade-headline">NO TRADE</div>
      ) : (
        <LegTable legs={legs} />
      )}

      {expiry_dte_bucket && (
        <div className="expiry-line">
          Target DTE: <strong>{expiry_dte_target}d</strong>
          {' '}({expiry_dte_bucket} bucket)
        </div>
      )}

      <div className="rationale">{rationale}</div>

      <SourceLine source={source} wingRecipe={wing_distance_recipe} />

      {/* Expand / collapse affordance */}
      {canExpand && (
        <button
          type="button"
          data-testid="expand-toggle"
          onClick={() => setExpanded(v => !v)}
          style={{
            alignSelf:     'flex-start',
            padding:       '3px 10px',
            fontSize:      10,
            fontWeight:    700,
            border:        '1px solid #1d4ed844',
            borderRadius:  5,
            background:    expanded ? '#1e3a5f' : '#0f172a',
            color:         expanded ? '#93c5fd' : '#64748b',
            cursor:        'pointer',
            fontFamily:    'inherit',
          }}
          aria-expanded={expanded}
        >
          {expanded ? 'Hide edge chart ▲' : 'Show edge chart ▾'}
        </button>
      )}

      {/* Expanded chart panel */}
      {expanded && canExpand && (
        <ExpandedPanel
          proposal={proposal}
          date={date}
          ticker={ticker}
          apiBase={apiBase}
          contextRegime={contextRegime}
        />
      )}
    </div>
  );
}
