import LegTable from './LegTable';

const TEMPLATE_LABELS = {
  pin_butterfly_tight:         'Pin Butterfly — Tight Wings',
  pin_butterfly_medium:        'Pin Butterfly — Medium Wings',
  pin_butterfly_wide:          'Pin Butterfly — Wide Wings',
  directional_spread_to_target:'Directional Spread to Target',
  bounded_iron_condor:         'Bounded Iron Condor',
  feature_no_trade:            'No Trade',
};

const RECIPE_LABELS = {
  half_fwhm: 'half-FWHM',
  full_fwhm: 'full-FWHM',
  sigma_1x:  '1σ',
};

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

export default function ProposalCard({ proposal }) {
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

  const isNoTrade = template_kind === 'no_trade';
  const label = TEMPLATE_LABELS[template_id] || template_id;

  return (
    <div className={`proposal-card${isNoTrade ? ' no-trade' : ''}`}>
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
    </div>
  );
}
