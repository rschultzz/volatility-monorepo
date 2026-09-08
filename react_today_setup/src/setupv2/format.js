// Formatters for the Setup v2 card (CR-AW). Pure, no React.

export function pts(v, digits = 2) {
  if (v == null || Number.isNaN(Number(v))) return '—';
  return Number(v).toFixed(digits);
}

export function signedPts(v, digits = 2) {
  if (v == null || Number.isNaN(Number(v))) return '—';
  const n = Number(v);
  return (n >= 0 ? '+' : '−') + Math.abs(n).toFixed(digits);
}

export function pct(v, digits = 0) {
  if (v == null || Number.isNaN(Number(v))) return '—';
  return (Number(v) * 100).toFixed(digits) + '%';
}

export function pctInterval(lo, hi) {
  if (lo == null || hi == null) return '';
  return `[${Math.round(lo * 100)}, ${Math.round(hi * 100)}]`;
}

export function ordinal(p) {
  if (p == null || Number.isNaN(Number(p))) return '—';
  const n = Math.round(Number(p));
  const mod100 = n % 100;
  if (mod100 >= 11 && mod100 <= 13) return `${n}th`;
  switch (n % 10) {
    case 1: return `${n}st`;
    case 2: return `${n}nd`;
    case 3: return `${n}rd`;
    default: return `${n}th`;
  }
}

/** "3 Sep 2026" from "2026-09-03". */
export function longDate(iso) {
  if (!iso) return '—';
  const [y, m, d] = iso.split('-').map(Number);
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${d} ${months[m - 1]} ${y}`;
}

export function shortDate(iso) {
  if (!iso) return '—';
  const [, m, d] = iso.split('-').map(Number);
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${d} ${months[m - 1]}`;
}

export function strike(v) {
  if (v == null) return '—';
  const n = Number(v);
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

export function gexB(v) {
  if (v == null) return '—';
  return `${Math.round(Number(v))}B`;
}

/** "34 × 5, 28 × 20, 6 × 60 sessions" from {"5": 34, "20": 28, "60": 6}. */
export function horizonMixText(mix) {
  if (!mix) return '';
  const parts = Object.keys(mix)
    .map(k => [Number(k), mix[k]])
    .sort((a, b) => a[0] - b[0])
    .map(([h, n]) => `${n} × ${h}`);
  return parts.length ? `${parts.join(', ')} sessions` : '';
}

/** Today's value for a KNN factor, formatted by key. */
export function factorValue(key, v) {
  if (v == null) return 'not populated';
  const n = Number(v);
  if (key.endsWith('_signed_distance_sigma')) return `${n >= 0 ? '+' : '−'}${Math.abs(n).toFixed(2)} IM`;
  if (key.endsWith('_max_gex') || key === 'total_neg_max_gex') return `${Math.round(n)}B`;
  if (key.startsWith('dominance_')) return `${n.toFixed(0)}%`;
  if (key.startsWith('is_')) return n ? 'yes' : 'no';
  if (key === 'magnet_direction_signed') return n > 0 ? 'above' : n < 0 ? 'below' : 'none';
  if (key.endsWith('_quality_ordinal')) return `${n.toFixed(0)} of 4`;
  if (key.startsWith('n_')) return `${n.toFixed(0)}`;
  if (key === 'implied_move_1d') return `${n.toFixed(1)} pt`;
  if (key === 'top_cluster_fraction_of_total_max_gex') return n.toFixed(2);
  return Number.isInteger(n) ? String(n) : n.toFixed(2);
}

/** Clamp a percentile to the bar. */
export function clampPct(p) {
  if (p == null) return null;
  return Math.max(0, Math.min(100, Number(p)));
}
