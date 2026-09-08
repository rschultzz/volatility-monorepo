import { clampPct, factorValue, longDate, ordinal } from '../format';

function readText(r) {
  if (!r.populated) return <span className="np">Not populated — not in the match.</span>;
  if (r.in_band === false) return <span className="out">Outside the analogue band.</span>;
  if (r.band_lo != null && r.band_hi != null && r.band_lo === r.band_hi) return 'Same value as every analogue.';
  return 'Inside the analogue band.';
}

function PercentileBar({ r }) {
  const p = clampPct(r.percentile);
  const lo = clampPct(r.band_lo_pct), hi = clampPct(r.band_hi_pct);
  const cls = `pb${r.in_band === false ? ' out' : ''}${!r.populated ? ' np' : ''}`;
  return (
    <div className={cls}>
      {r.populated && lo != null && hi != null
        ? <div className="band" style={{ left: `${lo}%`, width: `${Math.max(1, hi - lo)}%` }} />
        : <div className="band" style={{ left: 0, width: '100%' }} />}
      {p != null && <div className="m" style={{ left: `${p}%` }} />}
      {p != null && <span className="t" style={{ left: `${p}%` }}>{ordinal(p)}</span>}
      <span className="e">{r.lo_label}</span><span className="e r">{r.hi_label}</span>
    </div>
  );
}

export default function KnnFactorsTab({ card }) {
  const knn = card.knn || {};
  const factors = knn.factors || [];
  const mq = knn.match_quality || { in_band: 0, total: 0, outliers: [] };
  const groups = [];
  factors.forEach(f => {
    const g = groups.find(x => x.name === f.group);
    if (g) g.rows.push(f); else groups.push({ name: f.group, rows: [f] });
  });
  const analogueN = knn.analogue_n ?? card.analogues?.k ?? 0;
  const outlierLabels = (mq.outliers || []).map(k => factors.find(f => f.key === k)?.label || k);

  return (
    <div className="pane on kf" data-testid="knn-tab">
      <p className="lead">
        Every feature the KNN uses to pick today's analogues ({analogueN} on {longDate(card.date)} — every corpus day inside the
        similarity ceiling of {knn.distance_ceiling}, not a cap), with today's value placed against the whole magnet-above corpus
        ({knn.corpus_n} days, before {longDate(card.date)}). The shaded band is where the middle half of the analogues sit; the white
        line is today. A marker inside the band means today is typical of its own analogues; outside means the match is stretching.
        None of these rows is a rule — they explain <em>which</em> days the base rates come from.
      </p>
      <table className="kt">
        <thead>
          <tr><th>Factor</th><th>Today</th><th>Weight</th><th>Percentile vs magnet-above corpus &nbsp;·&nbsp; shaded = analogue middle half</th><th>Read</th></tr>
        </thead>
        <tbody>
          {groups.map(g => (
            <GroupRows key={g.name} group={g} />
          ))}
        </tbody>
      </table>
      <p className="foot" data-testid="match-quality">
        Match quality: {mq.in_band} of {mq.total} populated factors inside the analogue band
        {outlierLabels.length ? `; ${outlierLabels.join(', ')} ${outlierLabels.length === 1 ? 'is the outlier' : 'are the outliers'}` : ''}.
        {' '}Recency decay half-life {knn.half_life_months} months. Weights are the current KNN config ({knn.config_version}) — chosen by hand, not fitted.
      </p>
    </div>
  );
}

function GroupRows({ group }) {
  return (
    <>
      <tr><td className="group" colSpan={5}>{group.name}</td></tr>
      {group.rows.map(r => (
        <tr key={r.key} data-testid={`factor-${r.key}`}>
          <td className="f">{r.label}<small>{r.key}</small></td>
          <td className="v">{factorValue(r.key, r.today)}</td>
          <td className="w">×{Number(r.weight).toFixed(r.weight % 1 ? 2 : 1)}</td>
          <td className="p"><PercentileBar r={r} /></td>
          <td className="r">{readText(r)}</td>
        </tr>
      ))}
    </>
  );
}
