export default function LegTable({ legs }) {
  if (!legs || legs.length === 0) return null;
  return (
    <table className="leg-table">
      <thead>
        <tr>
          <th>Side</th>
          <th>Type</th>
          <th>Strike</th>
          <th>Qty</th>
        </tr>
      </thead>
      <tbody>
        {legs.map((leg, i) => (
          <tr key={i}>
            <td className={leg.side === 'long' ? 'leg-long' : 'leg-short'}>
              {leg.side === 'long' ? '+' : '−'}
            </td>
            <td>{leg.type}</td>
            <td>{Number(leg.strike).toFixed(1)}</td>
            <td>{leg.quantity}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
