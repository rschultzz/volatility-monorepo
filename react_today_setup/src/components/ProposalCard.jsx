import { useState, useEffect, useRef, useCallback } from 'react';
import LegTable from './LegTable';
import { ProposalEdgeChart, GreeksDisplay } from 'web-shared';

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
function buildPlDataBody(proposal, date, ticker, timeframe, contextRegime, contextDriftTarget) {
  const expiration = addCalendarDays(date, proposal.expiry_dte_target ?? 0);
  const regime = proposal.source?.regime || contextRegime || 'amplification';
  const regime_block = { regime };
  const src = proposal.source || {};
  if (src.drift_target != null)    regime_block.drift_target    = src.drift_target;
  if (src.tolerance != null)       regime_block.tolerance       = src.tolerance;
  if (src.lower_price != null)     regime_block.lower_price     = src.lower_price;
  if (src.upper_price != null)     regime_block.upper_price     = src.upper_price;
  if (regime_block.drift_target == null && contextDriftTarget != null) {
    regime_block.drift_target = contextDriftTarget;
  }
  return {
    trade_date:   date,
    ticker,
    timeframe,
    regime_block,
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

/** Small inline badge for data-quality warnings. */
function WarningsBadge({ warnings }) {
  if (!warnings || warnings.length === 0) return null;
  return (
    <div
      title={warnings.join('\n')}
      style={{
        display:      'inline-flex',
        alignItems:   'center',
        gap:          4,
        fontSize:     9,
        fontWeight:   700,
        color:        '#f59e0b',
        background:   '#292524',
        border:       '1px solid #78350f',
        borderRadius: 4,
        padding:      '1px 5px',
        cursor:       'help',
      }}
    >
      ⚠ {warnings.length === 1 ? warnings[0].slice(0, 40) : `${warnings.length} data warnings`}
    </div>
  );
}

/** Show expiry date(s) from priced legs.
 *
 * Single-expiry (common): one "Expiry: YYYY-MM-DD" line.
 * Mixed-expiry: one line per unique date grouped with leg roles.
 */
function ExpiryLine({ pricedLegs }) {
  if (!pricedLegs || pricedLegs.length === 0) return null;

  const dates = [...new Set(pricedLegs.map(l => l.expiration).filter(Boolean))];
  if (dates.length === 0) return null;

  if (dates.length === 1) {
    return (
      <div style={{ fontSize: 10, color: '#94a3b8' }}>
        Expiry: <strong style={{ color: '#cbd5e1' }}>{dates[0]}</strong>
      </div>
    );
  }

  // Mixed-expiry: show per-leg
  return (
    <div style={{ fontSize: 10, color: '#94a3b8', display: 'flex', flexDirection: 'column', gap: 1 }}>
      {pricedLegs.filter(l => l.expiration).map((l, i) => (
        <span key={i}>
          {l.side === 'long' ? '+' : '−'}{l.flag?.toUpperCase()} {l.strike_spx}
          {' '}→ <strong style={{ color: '#cbd5e1' }}>{l.expiration}</strong>
        </span>
      ))}
    </div>
  );
}

/** Render net debit or net credit from the pl-data response. */
function NetCostLine({ netCost }) {
  if (netCost === undefined) return null;   // not yet loaded

  const label = netCost === null
    ? '—'
    : netCost >= 0
      ? `Net debit  $${netCost.toFixed(2)}`
      : `Net credit $${Math.abs(netCost).toFixed(2)}`;

  return (
    <div
      style={{
        fontSize:   10,
        fontWeight: 700,
        color:      netCost === null ? '#475569' : netCost >= 0 ? '#f87171' : '#4ade80',
        fontVariantNumeric: 'tabular-nums',
      }}
    >
      {label}
    </div>
  );
}

/** Render struct/implied/edge-ratio from trade_thesis.
 *
 * Color convention (consistent with edge-zone chart overlay):
 *   edge_ratio > 1  → struct > implied → green
 *   edge_ratio < 1  → struct < implied → red
 *   edge_ratio null → implied unavailable → neutral
 */
function EdgeBlock({ tradeTthesis }) {
  if (!tradeTthesis) return null;
  const { structural_prob, implied_prob, edge_ratio } = tradeTthesis;
  if (structural_prob == null) return null;

  const structPct  = (structural_prob * 100).toFixed(1) + '%';
  const impliedTxt = implied_prob != null
    ? (implied_prob * 100).toFixed(1) + '%'
    : 'unavailable';
  const edgeTxt    = edge_ratio != null
    ? edge_ratio.toFixed(2) + '×'
    : '—';

  // Edge color: green when struct > implied (edge_ratio > 1), red otherwise
  const hasEdge = edge_ratio != null && edge_ratio > 1;
  const noEdge  = edge_ratio != null && edge_ratio <= 1;
  const edgeColor = hasEdge ? '#4ade80' : noEdge ? '#f87171' : '#94a3b8';

  return (
    <div style={{
      fontSize: 10,
      color: '#64748b',
      display: 'flex',
      gap: 6,
      flexWrap: 'wrap',
      alignItems: 'center',
    }}>
      <span>Struct <strong style={{ color: '#94a3b8' }}>{structPct}</strong></span>
      <span style={{ color: '#334155' }}>·</span>
      <span>Implied <strong style={{ color: implied_prob != null ? '#94a3b8' : '#475569' }}>{impliedTxt}</strong></span>
      <span style={{ color: '#334155' }}>·</span>
      <span>Edge <strong style={{ color: edgeColor }}>{edgeTxt}</strong></span>
    </div>
  );
}

// ── Expanded chart panel ───────────────────────────────────────────────────────

function ExpandedPanel({
  timeframe, setTimeframe, chartData, containerRef, chartWidth,
}) {
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
              onClick={() => setTimeframe(tf)}
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
        <ProposalEdgeChart
          data={chartData === undefined ? null : chartData}
          width={chartWidth}
          height={340}
        />
      </div>

      {/* Greeks row */}
      <GreeksDisplay
        greeks={chartData?.greeks ?? null}
        evaluationTime={chartData?.evaluation_time ?? null}
      />
    </div>
  );
}

// ── ProposalCard ──────────────────────────────────────────────────────────────

export default function ProposalCard({
  proposal,
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

  const isNoTrade         = template_kind === 'no_trade';
  const canExpand         = !isNoTrade && Array.isArray(legs) && legs.length > 0 && !!date;
  const label             = TEMPLATE_LABELS[template_id] || template_id;
  const contextRegime     = context?.regime;
  const contextDriftTarget = source?.drift_target ?? context?.top_cluster?.center_price ?? null;

  const [expanded, setExpanded]     = useState(false);
  const [timeframe, setTimeframe]   = useState(DEFAULT_TIMEFRAME);
  const [chartWidth, setChartWidth] = useState(640);
  const containerRef = useRef(null);

  // Per-timeframe result cache (timeframe → data|error).
  // Lifted to card level so the prefetch and the expanded chart share state.
  const cacheRef = useRef({});
  const [chartData, setChartData] = useState(undefined);

  // ResizeObserver
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

  // ── Fetch helper (shared by prefetch and expand) ────────────────────────
  const fetchPlData = useCallback(async (tf) => {
    if (cacheRef.current[tf] !== undefined) {
      setChartData(cacheRef.current[tf]);
      return;
    }
    setChartData(undefined);
    const body = buildPlDataBody(proposal, date, ticker, tf, contextRegime, contextDriftTarget);
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
  }, [proposal, date, ticker, apiBase, contextRegime, contextDriftTarget]);

  // ── Step 5-A + timeframe handling ─────────────────────────────────────
  // On mount (timeframe = DEFAULT_TIMEFRAME), this acts as a non-blocking
  // day-load prefetch: fires before expand, populates cacheRef, and fills
  // the LegTable mid column progressively.  Re-loading an already-seen day
  // (cacheRef hit) fires zero ORATS calls.  On timeframe change it re-fetches
  // the new timeframe.  One effect covers both cases to avoid double-fetching.
  useEffect(() => {
    if (!canExpand) return;
    fetchPlData(timeframe);
  }, [timeframe, canExpand, fetchPlData]);

  // ── Pricing data for the LegTable (default timeframe) ──────────────────
  // Use the default-timeframe cached data so the header LegTable shows mids
  // even before the card is expanded.
  const defaultData  = cacheRef.current[DEFAULT_TIMEFRAME] || chartData;
  const pricedLegs   = defaultData?.ok ? (defaultData.legs ?? null) : null;
  const pricingWarns = defaultData?.ok ? (defaultData.warnings ?? []) : [];
  // net_cost: undefined = not yet fetched, null = unavailable (leg missing mid)
  const netCost    = defaultData?.ok ? defaultData.net_cost    : undefined;
  const tradeTthesis = defaultData?.ok ? defaultData.trade_thesis : null;

  function handleTimeframeChange(tf) {
    if (tf === timeframe) return;
    setTimeframe(tf);
  }

  return (
    <div
      className={[
        'proposal-card',
        isNoTrade ? 'no-trade' : '',
        expanded ? 'proposal-card--expanded' : '',
      ].filter(Boolean).join(' ')}
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
        <LegTable legs={legs} pricedLegs={pricedLegs} />
      )}

      {/* Expiry: real calendar date from payload (Step 9) + DTE/bucket context */}
      {pricedLegs
        ? <ExpiryLine pricedLegs={pricedLegs} />
        : expiry_dte_bucket && (
          <div className="expiry-line">
            Target DTE: <strong>{expiry_dte_target}d</strong>
            {' '}({expiry_dte_bucket} bucket)
          </div>
        )
      }

      <div className="rationale">{rationale}</div>

      <SourceLine source={source} wingRecipe={wing_distance_recipe} />

      {/* Net debit/credit (Step 8) */}
      {canExpand && <NetCostLine netCost={netCost} />}

      {/* Edge block: struct / implied / edge-ratio (Step 10) */}
      {canExpand && tradeTthesis && <EdgeBlock tradeTthesis={tradeTthesis} />}

      {/* Data-quality warnings badge */}
      {pricingWarns.length > 0 && <WarningsBadge warnings={pricingWarns} />}

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
          timeframe={timeframe}
          setTimeframe={handleTimeframeChange}
          chartData={chartData}
          containerRef={containerRef}
          chartWidth={chartWidth}
        />
      )}
    </div>
  );
}
