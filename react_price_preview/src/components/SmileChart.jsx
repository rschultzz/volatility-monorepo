import React, { useMemo } from 'react'
import Plotly from 'plotly.js-dist-min'
import createPlotlyComponent from 'react-plotly.js/factory'

const Plot = createPlotlyComponent(Plotly)

const COLORWAY = [
  '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
  '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
]

export default function SmileChart({ data, width, height }) {
  const traces = useMemo(() => {
    if (!data || !Array.isArray(data.traces)) return []
    return data.traces.map((tr) => ({
      ...tr,
      type: 'scatter',
      mode: tr.mode || 'lines+markers',
      marker: { ...tr.marker, size: 4 },
      line: { ...tr.line, width: tr.line?.width || 1.5 },
    }))
  }, [data])

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
    <Plot
      data={traces}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%', height: '100%' }}
    />
  )
}
