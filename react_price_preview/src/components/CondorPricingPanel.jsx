// Floating overlay rendered inside PriceChart when the σ-band anchor is
// active. Surfaces entry credit, eval net, and gross P&L from the
// /api/condor-pricing payload. Polling is owned by App.jsx; this is a
// pure presentation component.

function fmt(x, digits = 2) {
  if (x === null || x === undefined) return '—'
  const n = Number(x)
  if (!Number.isFinite(n)) return '—'
  return n.toFixed(digits)
}

function pnlColor(gross) {
  if (gross === null || gross === undefined || !Number.isFinite(Number(gross))) {
    return '#bbb'
  }
  const g = Number(gross)
  if (g > 0) return '#4caf50'
  if (g < 0) return '#ef5350'
  return '#bbb'
}

export default function CondorPricingPanel({ condorPricing, positionStyle, onHandleMouseDown }) {
  if (!condorPricing) return null

  const { sigma_pts, entry, eval: evalBlock, pnl, warnings } = condorPricing
  const isLive = Boolean(evalBlock?.is_live)
  const gross = pnl?.gross
  const warningsLen = Array.isArray(warnings) ? warnings.length : 0

  return (
    <div
      data-condor-panel
      style={{
        position: 'absolute',
        ...positionStyle,
        zIndex: 5,
        background: 'rgba(20, 24, 33, 0.92)',
        border: '1px solid #2a3140',
        borderRadius: 4,
        padding: '6px 8px',
        fontFamily: 'monospace',
        fontSize: 10,
        color: '#ddd',
        lineHeight: 1.4,
        minWidth: 140,
        pointerEvents: 'none',
      }}
    >
      <div
        onMouseDown={onHandleMouseDown}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontWeight: 'bold',
          marginBottom: 2,
          color: '#9fb',
          cursor: 'grab',
          userSelect: 'none',
          pointerEvents: 'auto',
        }}
      >
        <span>condor (1σ)</span>
      </div>
      <div>σ: {fmt(sigma_pts)} pts</div>
      <div>
        entry {entry?.snapshot_pt}: {fmt(entry?.net_credit)}
      </div>
      <div>
        eval {evalBlock?.snapshot_pt}
        {isLive && (
          <span
            style={{
              display: 'inline-block',
              width: 6,
              height: 6,
              borderRadius: 3,
              background: '#4caf50',
              marginLeft: 4,
              verticalAlign: 'middle',
            }}
            title="live"
          />
        )}
        : {fmt(evalBlock?.net_cost_to_close)}
      </div>
      <div style={{ color: pnlColor(gross), fontWeight: 'bold' }}>
        P&L: {fmt(gross)}
      </div>
      {warningsLen > 0 && (
        <div style={{ color: '#ffb74d', marginTop: 2 }} title={warnings.join('\n')}>
          ⚠ {warningsLen} leg{warningsLen === 1 ? '' : 's'} missing
        </div>
      )}
    </div>
  )
}
