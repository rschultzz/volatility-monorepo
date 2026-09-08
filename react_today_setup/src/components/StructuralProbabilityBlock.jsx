// ── Formatters ──────────────────────────────────────────────────────────────

function pct(v) {
  return v != null ? `${(v * 100).toFixed(1)}%` : '—';
}

function pctInt(v) {
  return v != null ? `${Math.round(v * 100)}%` : '—';
}

/** Format a Wilson CI [lo, hi] pair as "[lo-hi%]". */
function ciStr(pair) {
  if (!pair || pair[0] == null || pair[1] == null) return '';
  return `[${Math.round(pair[0] * 100)}-${Math.round(pair[1] * 100)}%]`;
}

// ── DTE → timeframe row mapping ─────────────────────────────────────────────

/**
 * Returns which timeframe row (t1 / t5 / t15) a proposed trade's DTE maps to.
 * Rule: DTE ≤ 3 → T+1, 4–9 → T+5, ≥ 10 → T+15. Returns null when dte is null.
 */
function dteToRow(dte) {
  if (dte == null) return null;
  if (dte <= 3) return 't1';
  if (dte <= 9) return 't5';
  return 't15';
}

// ── Post-touch advisory line (CR-AV decision 2) ─────────────────────────────
//
// The post-touch pattern is evidence, not a direction gate (CR-AL: the label
// does not predict P&L; CR-AM: no reversal). The block states the label, the
// sample size and the continuation fraction at the trade's timeframe — no
// badge implying support or caution, and nothing here reads filter_mode.

/** "post-touch pattern: stepping-stone · n=23 · t5 above 61%" from the
 *  server-side advisory block; falls back to the raw post_touch fields. */
function advisoryText(pt, dte) {
  const adv = pt?.advisory || {};
  const label = adv.pattern_label ?? pt?.pattern_label ?? null;
  const n = adv.n ?? pt?.same_bucket_n ?? pt?.total_touchers ?? null;
  const tf = adv.timeframe ?? dteToRow(dte);
  const dir = adv.direction ?? null;
  let fraction = adv.fraction;
  if (fraction == null && tf && dir) fraction = pt?.fractions?.[tf]?.[dir];
  const parts = [`post-touch pattern: ${label ?? 'unlabeled'}`];
  if (n != null) parts.push(`n=${n}`);
  if (tf && dir && fraction != null) parts.push(`${tf} ${dir} ${Math.round(fraction * 100)}%`);
  return parts.join(' · ');
}

// ── Stacked bar row ──────────────────────────────────────────────────────────

const SEGMENT_COLORS = {
  below: 'var(--pt-below)',
  at:    'var(--pt-at)',
  above: 'var(--pt-above)',
};

function StackedBarRow({ label, fracs, cis, dteMarker }) {
  const segments = ['below', 'at', 'above'].map(key => ({
    key,
    fraction: fracs?.[key] ?? 0,
    ci: cis?.[key],
    color: SEGMENT_COLORS[key],
  })).filter(s => s.fraction > 0);

  return (
    <div className="pt-row">
      <span className="pt-tf-label">{label}</span>
      <div className="pt-bar" aria-label={`${label} distribution`}>
        {segments.map(({ key, fraction, ci, color }) => {
          const full = `${pctInt(fraction)} ${ciStr(ci)}`.trim();
          // Narrow-segment label rules (fraction of total bar width):
          //   ≥ 26% → show "14% [4-32%]"
          //   10-25% → show "14%" only
          //   < 10% → no inline text (bar width carries the signal; title attr for a11y)
          const labelText = fraction >= 0.26
            ? `${pctInt(fraction)} ${ciStr(ci)}`.trim()
            : fraction >= 0.10
              ? pctInt(fraction)
              : '';
          return (
            <div
              key={key}
              className="pt-segment"
              style={{ flexGrow: fraction, background: color }}
              title={full}
              role="img"
              aria-label={`${key}: ${full}`}
            >
              {labelText && <span className="pt-seg-label">{labelText}</span>}
            </div>
          );
        })}
      </div>
      {dteMarker != null && (
        <span className="pt-dte-marker">← {dteMarker}d trade</span>
      )}
    </div>
  );
}

// ── Post-touch distribution section (sub-section of sp-block) ────────────────

function PostTouchSection({ pt, dte }) {
  if (!pt) return null;

  const { filter_mode, same_bucket_n, total_touchers, fractions, wilson_cis } = pt;
  const dteRow = dteToRow(dte);

  // ── Graceful fallbacks for thin / 0DTE corpora ──────────────────────────
  if (filter_mode === 'insufficient') {
    return (
      <div className="pt-section">
        <span className="pt-title">Post-touch close distribution (conditional on touch)</span>
        <p className="sp-note">
          Insufficient post-touch sample ({total_touchers ?? 0} touchers — need ≥ 4)
        </p>
      </div>
    );
  }

  if (filter_mode === 'zero_dte_corpus_insufficient') {
    return (
      <div className="pt-section">
        <span className="pt-title">Post-touch close distribution (conditional on touch)</span>
        <p className="sp-note">0DTE corpus insufficient (3 days in corpus)</p>
      </div>
    );
  }

  // ── strict / pooled-fallback ─────────────────────────────────────────────
  const isPooled = filter_mode === 'pooled-fallback';
  const denomLabel = isPooled
    ? `bucket-pooled fallback, N=${total_touchers}`
    : `same-bucket touchers only, N=${same_bucket_n}`;

  const TF_ROWS = [
    { key: 't1',  label: 'T+1'  },
    { key: 't5',  label: 'T+5'  },
    { key: 't15', label: 'T+15' },
  ];

  // CR-AV: advisory block (decision 2) — rendered when the payload marks the
  // post-touch data advisory_only; the label is evidence, never a direction call.
  const advisory = pt.advisory_only ? advisoryText(pt, dte) : null;

  return (
    <div className="pt-section">
      <div className="pt-header">
        <span className="pt-title">Post-touch close distribution (conditional on touch)</span>
        {isPooled && <span className="sp-bucket-fallback-badge">pooled fallback</span>}
      </div>
      <p className="pt-subline">{denomLabel}</p>
      <p className="pt-caption">Where analogues closed after touching — not a pre-entry probability. For the unconditional read, see the edge table.</p>

      <div className="pt-bars">
        {TF_ROWS.map(({ key, label }) => (
          <StackedBarRow
            key={key}
            label={label}
            fracs={fractions?.[key]}
            cis={wilson_cis?.[key]}
            dteMarker={dteRow === key ? dte : null}
          />
        ))}
        <div className="pt-bar-legend">
          <span className="pt-legend-item">
            <span className="pt-legend-dot" style={{ background: 'var(--pt-below)' }} />
            below
          </span>
          <span className="pt-legend-item">
            <span className="pt-legend-dot" style={{ background: 'var(--pt-at)' }} />
            at
          </span>
          <span className="pt-legend-item">
            <span className="pt-legend-dot" style={{ background: 'var(--pt-above)' }} />
            above
          </span>
        </div>
      </div>

      {advisory && (
        <p className="pt-advisory" data-testid="pt-advisory">{advisory}</p>
      )}
    </div>
  );
}

// ── Main export ──────────────────────────────────────────────────────────────

/**
 * Renders the Base Rate block: touch/close rates + post-touch distribution.
 *
 * Props:
 *   sp  — structural_probability object from /api/setup/proposals
 *   dte — integer DTE from the primary proposal (expiry_dte_target); drives
 *          the "← Nd trade" row marker on the appropriate timeframe.
 */
export default function StructuralProbabilityBlock({ sp, dte }) {
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
    post_touch,
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

        <PostTouchSection pt={post_touch} dte={dte} />

      </div>
    </>
  );
}
