export default function LegTable({ legs, pricedLegs, netCost }) {
  if (!legs || legs.length === 0) return null;

  // Build a lookup from SPX strike + flag to priced leg data (from pl-data response).
  const priceMap = {};
  if (pricedLegs) {
    for (const pl of pricedLegs) {
      const key = `${pl.strike_spx}-${pl.flag}`;
      priceMap[key] = pl;
    }
  }

  const hasPricing = pricedLegs && pricedLegs.some(l => l.mid != null);
  // netCost: undefined = not yet fetched (hide footer); null = unavailable; number = real value
  const showNetCost = hasPricing && netCost !== undefined;

  return (
    <table className="leg-table">
      <thead>
        <tr>
          <th>Side</th>
          <th>Type</th>
          <th>ES Level</th>
          <th>SPX Strike</th>
          {hasPricing && <th>Mid</th>}
          <th>Qty</th>
        </tr>
      </thead>
      <tbody>
        {legs.map((leg, i) => {
          // Try to match by index (same order) before falling back to strike+flag
          const pl = pricedLegs?.[i] || priceMap[`${leg.strike_spx}-${leg.type === 'call' ? 'c' : 'p'}`];
          const mid = pl?.mid;
          return (
            <tr key={i}>
              <td className={leg.side === 'long' ? 'leg-long' : 'leg-short'}>
                {leg.side === 'long' ? '+' : '−'}
              </td>
              <td>{leg.type}</td>
              <td>{Number(leg.strike).toFixed(1)}</td>
              <td>{leg.strike_spx != null ? leg.strike_spx : '—'}</td>
              {hasPricing && (
                <td style={{ color: mid != null ? '#94a3b8' : '#475569', fontVariantNumeric: 'tabular-nums' }}>
                  {mid != null ? mid.toFixed(2) : '—'}
                </td>
              )}
              <td>{leg.quantity}</td>
            </tr>
          );
        })}
      </tbody>
      {showNetCost && (
        <tfoot>
          <tr>
            <td
              colSpan={hasPricing ? 4 : 3}
              style={{
                borderTop: '1px solid #1f2937',
                textAlign: 'right',
                color: '#475569',
                fontSize: 9,
                paddingRight: 6,
                paddingTop: 3,
              }}
            >
              Net
            </td>
            <td style={{
              borderTop: '1px solid #1f2937',
              fontWeight: 700,
              fontVariantNumeric: 'tabular-nums',
              paddingTop: 3,
              color: netCost === null
                ? '#475569'
                : netCost >= 0 ? '#f87171' : '#4ade80',
            }}>
              {netCost === null
                ? '—'
                : netCost >= 0
                  ? `Debit  $${netCost.toFixed(2)}`
                  : `Credit $${Math.abs(netCost).toFixed(2)}`}
            </td>
            <td style={{ borderTop: '1px solid #1f2937', paddingTop: 3 }} />
          </tr>
        </tfoot>
      )}
    </table>
  );
}
