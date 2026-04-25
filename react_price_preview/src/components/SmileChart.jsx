import React, { useMemo } from 'react'
import Plotly from 'plotly.js-dist-min'
import createPlotlyComponent from 'react-plotly.js/factory'

const Plot = createPlotlyComponent(Plotly)

const COLORWAY = [
  '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
  '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
]

export default function SmileChart({
  data,
  width,
  height,
  // Anchor controls — optional, only render anchor buttons when provided
  snapshots = null,
  activeAnchorTime = null,   // string e.g. "08:52" or null when off
  onAnchorChange = null,     // (snapshot | null) => void
}) {
  const traces = useMemo(() => {
    if (!data || !Array.isArray(data.traces)) return []
    return data.traces.map((tr) => ({
      ...tr,
      type: 'scatter',
      mode: tr.mode || (tr.x?.length === 1 ? 'markers' : 'lines+markers'),
      marker: {
        ...tr.marker,
        size: tr.marker?.size || 4
      },
      line: {
        ...tr.line,
        width: tr.line?.width || 1.5
      },
    }))
  }, [data])

  // Anchor buttons render only when caller provides snapshots + handler.
  // We filter to snapshots that have both stock_price and atm_iv_pct
  // (without those, sigma bands cannot be computed).
  const anchorEntries = useMemo(() => {
    if (!Array.isArray(snapshots) || !onAnchorChange) return []
    return snapshots.filter(
      (s) => s && s.atm_iv_pct != null && s.stock_price != null
    )
  }, [snapshots, onAnchorChange])

  const showAnchorBar = anchorEntries.length > 0

  // Color a snapshot button to match its trace color in the smile plot
  const colorForSnapshot = (snapshot, idx) => {
    if (snapshot?.is_live) return '#fbbf24'
    return COLORWAY[idx % COLORWAY.length]
  }

  const layout = useMemo(() => ({
    autosize: true,
    width: width,
    height: height,
    template: 'plotly_dark',
    margin: { l: 35, r: 10, t: 10, b: 30 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    xaxis: {
      gridcolor: 'rgba(148, 163, 184, 0.05)',
      zeroline: false,
      tickfont: { size: 9 },
      fixedrange: true,
    },
    yaxis: {
      gridcolor: 'rgba(148, 163, 184, 0.05)',
      zeroline: false,
      tickfont: { size: 9 },
      fixedrange: true,
    },
    showlegend: true,
    legend: {
      orientation: 'h',
      x: 0.5,
      y: -0.2,
      xanchor: 'center',
      font: { size: 9 },
      bgcolor: 'rgba(0,0,0,0)',
    },
    colorway: COLORWAY,
    hovermode: 'closest',
  }), [width, height])

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      {showAnchorBar && (
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '4px',
            padding: '4px 0',
            borderBottom: '1px solid rgba(148,163,184,0.08)',
            marginBottom: '4px',
            alignItems: 'center',
          }}
          onMouseDown={(e) => e.stopPropagation()}
        >
          <span
            style={{
              fontSize: '9px',
              color: '#64748b',
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
              marginRight: '4px',
            }}
          >
            Bands @
          </span>
          {anchorEntries.map((snap, idx) => {
            const isActive = activeAnchorTime === snap.time
            const traceColor = colorForSnapshot(snap, idx)
            return (
              <button
                key={`${snap.time}-${idx}`}
                onClick={(e) => {
                  e.stopPropagation()
                  onAnchorChange(isActive ? null : snap)
                }}
                title={
                  `Anchor sigma bands at ${snap.label}\n` +
                  `SPX: ${snap.stock_price?.toFixed(2)}\n` +
                  `IV: ${snap.atm_iv_pct?.toFixed(2)}%`
                }
                style={{
                  padding: '2px 8px',
                  fontSize: '10px',
                  fontWeight: 600,
                  borderRadius: '6px',
                  border: isActive ? `1.5px solid ${traceColor}` : '1px solid #334155',
                  background: isActive ? `${traceColor}33` : '#0f172a',
                  color: isActive ? traceColor : '#cbd5e1',
                  cursor: 'pointer',
                  transition: 'background 120ms, border-color 120ms',
                  fontFamily: 'inherit',
                }}
              >
                {isActive ? '📍 ' : ''}{snap.label}
              </button>
            )
          })}
          {activeAnchorTime != null && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                onAnchorChange(null)
              }}
              title="Turn off sigma bands"
              style={{
                padding: '2px 8px',
                fontSize: '10px',
                fontWeight: 600,
                borderRadius: '6px',
                border: '1px solid #475569',
                background: 'rgba(239,68,68,0.10)',
                color: '#fca5a5',
                cursor: 'pointer',
                marginLeft: '4px',
                fontFamily: 'inherit',
              }}
            >
              ✕ Off
            </button>
          )}
        </div>
      )}
      <div style={{ flex: 1, minHeight: 0 }}>
        <Plot
          data={traces}
          layout={layout}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    </div>
  )
}
