function pct(v) {
  return v != null ? `${(v * 100).toFixed(1)}%` : '—';
}

export default function StructuralProbabilityBlock({ sp }) {
  if (!sp) return null;

  const { outcome_status, regime_kind, k, k_with_outcomes, note } = sp;

  // ── no_data branch ────────────────────────────────────────────────────────
  if (outcome_status === 'no_data') {
    return (
      <>
        <p className="section-heading" style={{ marginTop: 24 }}>Base Rate</p>
        <div className="sp-block">
          <div className="sp-header">
            {regime_kind && (
              <span className={`regime-badge regime-${regime_kind}`}>{regime_kind}</span>
            )}
            <span className="sp-kline">K={k} · {k_with_outcomes} with outcomes</span>
          </div>
          {note && <p className="sp-note">{note}</p>}
        </div>
      </>
    );
  }

  // ── ok branch ─────────────────────────────────────────────────────────────
  const {
    touch_rate, close_rate,
    touch_ci_lower, touch_ci_upper,
    mean_days_to_reach, mean_excursion_pct,
  } = sp;

  const showBadge = close_rate != null && touch_rate != null
    && close_rate < touch_rate * 0.25;

  const ciText = touch_ci_lower != null && touch_ci_upper != null
    ? `[${(touch_ci_lower * 100).toFixed(0)}%, ${(touch_ci_upper * 100).toFixed(0)}%]`
    : null;

  const hasSecondary = mean_days_to_reach != null || mean_excursion_pct != null;

  return (
    <>
      <p className="section-heading" style={{ marginTop: 24 }}>Base Rate</p>
      <div className="sp-block">

        <div className="sp-header">
          {regime_kind && (
            <span className={`regime-badge regime-${regime_kind}`}>{regime_kind}</span>
          )}
          <span className="sp-kline">K={k} · {k_with_outcomes} with outcomes</span>
        </div>

        <div className="sp-stats">
          <div className="sp-stat-row">
            <span className="sp-stat-label">Touch rate</span>
            <span className="sp-stat-value">{pct(touch_rate)}</span>
            {ciText && <span className="sp-ci">{ciText}</span>}
          </div>
          <div className="sp-stat-row">
            <span className="sp-stat-label">Close rate</span>
            <span className="sp-stat-value">{pct(close_rate)}</span>
            {showBadge && (
              <span className="sp-divergence-badge">touches but doesn't hold</span>
            )}
          </div>
        </div>

        {hasSecondary && (
          <div className="sp-secondary">
            {mean_days_to_reach != null && (
              <span className="sp-meta">
                Avg days to reach: <strong>{mean_days_to_reach.toFixed(1)}d</strong>
              </span>
            )}
            {mean_days_to_reach != null && mean_excursion_pct != null && (
              <span className="context-sep">·</span>
            )}
            {mean_excursion_pct != null && (
              <span className="sp-meta">
                Avg excursion: <strong>{mean_excursion_pct.toFixed(2)}× IM</strong>
              </span>
            )}
          </div>
        )}

        {note && <p className="sp-note">{note}</p>}
      </div>
    </>
  );
}
