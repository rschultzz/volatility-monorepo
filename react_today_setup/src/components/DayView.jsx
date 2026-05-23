// DayView — renders the GEX landscape + mini price chart for one day,
// with flag controls (regime_wrong, not_a_true_analogue).
import { useState } from 'react'
import { GexLandscape } from 'web-shared'
import { MiniPriceChart } from 'web-shared'

const REGIME_COLORS = {
  'magnetic-pin': '#10b981',
  'magnet-above': '#fbbf24',
  'magnet-below': '#fbbf24',
  bounded: '#10b981',
  amplification: '#06b6d4',
  'broken-magnet': '#a78bfa',
  untethered: '#94a3b8',
  pinned: '#10b981',
}

const VALID_REGIMES = [
  'magnetic-pin', 'magnet-above', 'magnet-below',
  'bounded', 'amplification', 'untethered', 'broken-magnet',
]

export default function DayView({
  label,            // "Anchor" | "Selected"
  date,             // "YYYY-MM-DD"
  ticker,
  apiBase,
  landscapeData,    // data for GexLandscape (or null)
  regime,           // effective regime string
  autoRegime,       // stored auto regime
  flag,             // existing flag object or null
  allowPairFlag,    // bool — show "not a true analogue" control
  onRegimeFlag,     // (correctedRegime) => void
  onPromote,        // () => void
  onDemote,         // () => void
  onDeleteFlag,     // () => void
  onPairFlag,       // () => void — flag as not a true analogue
}) {
  const [showRegimePicker, setShowRegimePicker] = useState(false)
  const [selectedCorrection, setSelectedCorrection] = useState('')

  const regimeColor = REGIME_COLORS[regime] || '#94a3b8'
  const isFlagged = flag != null
  const isPromoted = flag?.promoted === true

  const clusters = Array.isArray(landscapeData?.confluences)
    ? landscapeData.confluences.map(c => ({
        center_price: c.center_price,
        max_gex: c.max_gex,
      }))
    : []

  function handleSubmitRegimeFlag() {
    if (!selectedCorrection) return
    onRegimeFlag?.(selectedCorrection)
    setShowRegimePicker(false)
    setSelectedCorrection('')
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, position: 'relative' }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          {label}
        </span>
        {date && (
          <span style={{ fontSize: 12, color: '#e2e8f0', fontFamily: 'monospace' }}>{date}</span>
        )}
        {regime && (
          <span style={{
            fontSize: 10, fontWeight: 700, letterSpacing: '0.05em',
            padding: '2px 7px', borderRadius: 5,
            color: regimeColor, background: `${regimeColor}22`, border: `1px solid ${regimeColor}55`,
          }}>
            {regime.replace(/-/g, ' ')}
          </span>
        )}
        {isFlagged && (
          <span style={{ fontSize: 9, color: '#ef4444', fontWeight: 700, letterSpacing: '0.04em' }}>
            ⚑ FLAGGED{isPromoted ? ' (promoted)' : ''}
          </span>
        )}
        {isFlagged && flag.corrected_regime && flag.corrected_regime !== autoRegime && (
          <span style={{ fontSize: 10, color: '#94a3b8' }}>
            auto: {autoRegime} → yours: {flag.corrected_regime}
          </span>
        )}
      </div>

      {/* Landscape */}
      <div style={{ height: 320, position: 'relative', flexShrink: 0 }}>
        {landscapeData ? (
          <GexLandscape
            data={landscapeData}
            spotMode="OPEN"
            onSpotModeChange={() => {}}
          />
        ) : (
          <div style={emptyBox}>
            {date ? 'No landscape data' : 'Select a day'}
          </div>
        )}
      </div>

      {/* Mini price chart */}
      {date && (
        <MiniPriceChart
          date={date}
          ticker={ticker}
          apiBase={apiBase}
          clusters={clusters}
          height={160}
        />
      )}

      {/* Flag controls */}
      {date && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {/* Regime flag */}
          {!isFlagged ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <button
                type="button"
                onClick={() => setShowRegimePicker(v => !v)}
                style={smallBtn('#1e3a5f', '#60a5fa')}
              >
                ⚑ Regime is wrong
              </button>
              {showRegimePicker && (
                <>
                  <select
                    value={selectedCorrection}
                    onChange={e => setSelectedCorrection(e.target.value)}
                    style={selectStyle}
                  >
                    <option value="">Pick regime…</option>
                    {VALID_REGIMES.map(r => (
                      <option key={r} value={r}>{r}</option>
                    ))}
                  </select>
                  <button
                    type="button"
                    onClick={handleSubmitRegimeFlag}
                    disabled={!selectedCorrection}
                    style={smallBtn('#1a3a1a', '#22c55e')}
                  >
                    Submit
                  </button>
                  <button
                    type="button"
                    onClick={() => { setShowRegimePicker(false); setSelectedCorrection('') }}
                    style={smallBtn('#2d1a1a', '#ef4444')}
                  >
                    Cancel
                  </button>
                </>
              )}
            </div>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <button
                type="button"
                onClick={onDeleteFlag}
                style={smallBtn('#2d1a1a', '#ef4444')}
              >
                ✕ Remove flag
              </button>
              {!isPromoted ? (
                <button type="button" onClick={onPromote} style={smallBtn('#1e3a5f', '#60a5fa')}>
                  ↑ Promote
                </button>
              ) : (
                <button type="button" onClick={onDemote} style={smallBtn('#2d1f0a', '#fbbf24')}>
                  ↓ Demote
                </button>
              )}
            </div>
          )}

          {/* Pair flag — only in analogue mode for the selected-day panel */}
          {allowPairFlag && (
            <button
              type="button"
              onClick={onPairFlag}
              style={smallBtn('#2d1f0a', '#fb923c')}
            >
              ✗ Not a true analogue
            </button>
          )}
        </div>
      )}
    </div>
  )
}

const emptyBox = {
  height: '100%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: '#475569',
  fontSize: 13,
  border: '1px dashed #334155',
  borderRadius: 8,
}

function smallBtn(bg, color) {
  return {
    padding: '3px 10px',
    fontSize: 10,
    fontWeight: 700,
    border: `1px solid ${color}44`,
    borderRadius: 5,
    background: bg,
    color,
    cursor: 'pointer',
    fontFamily: 'inherit',
  }
}

const selectStyle = {
  background: '#0f172a',
  color: '#e2e8f0',
  border: '1px solid #334155',
  borderRadius: 5,
  padding: '2px 6px',
  fontSize: 11,
  fontFamily: 'inherit',
}
