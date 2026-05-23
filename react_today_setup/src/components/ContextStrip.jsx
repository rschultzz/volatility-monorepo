function regimeCls(regime) {
  if (!regime) return 'regime-unknown';
  return 'regime-' + regime.replace(/-/g, '-');
}

function regimeLabel(regime) {
  if (!regime) return '—';
  return regime.replace(/-/g, '‑'); // non-breaking hyphen for wrapping
}

export default function ContextStrip({ context }) {
  if (!context) return null;
  const { regime, dominant_bucket, top_cluster, spot, implied_move } = context;

  return (
    <div className="context-strip">
      <span className={`regime-badge ${regimeCls(regime)}`}>
        {regimeLabel(regime)}
      </span>

      {dominant_bucket && (
        <>
          <span className="context-sep">|</span>
          <span className="context-item">Bucket: <strong>{dominant_bucket}</strong></span>
        </>
      )}

      {spot != null && (
        <>
          <span className="context-sep">|</span>
          <span className="context-item">Spot: <strong>{Number(spot).toFixed(1)}</strong></span>
        </>
      )}

      {implied_move != null && implied_move > 0 && (
        <>
          <span className="context-sep">|</span>
          <span className="context-item">±1σ: <strong>{Number(implied_move).toFixed(1)}pt</strong></span>
        </>
      )}

      {top_cluster && (
        <>
          <span className="context-sep">|</span>
          <span className="context-item">
            Top cluster: <strong>{Number(top_cluster.center_price).toFixed(1)}</strong>
            {' '}({top_cluster.quality})
            {top_cluster.max_gex && (
              <> · {(Number(top_cluster.max_gex) / 1e9).toFixed(0)}B</>
            )}
          </span>
        </>
      )}
    </div>
  );
}
