// CR-DQ items 2 / 3 — the box panel on the price chart (replaces CR-DP's popup). One section per drawn box, oldest at the
// top, the current box last and highlighted; frozen boxes are one line, each section expands to the stats (current
// collapsed by default too). Text comes from flowV3.boxPanelSections (CR-DP's boxPopupLines), so the numbers are the
// popup's and the old box hover's. Dragged by its header; the caller owns the place (remembered per browser), the
// double-click reset and the ×. Palette only (C1): the chart's tag background, magenta border, slate / light text.
import { useState } from 'react'
import { TAG_BG } from './flowV3Primitives.js'

const MAGENTA = '232,121,249'
const TEXT = '#e2e8f0'
const MUTED = '#94a3b8'

function Stats({ s, testid }) {
  const l = s.lines
  return (
    <div data-testid={testid} style={{ paddingLeft: 10 }}>
      <div data-testid={s.current ? 'flow-v3-box-popup-head' : undefined}>{l.head}</div>
      <div data-testid={s.current ? 'flow-v3-box-popup-top' : undefined}>{l.top}</div>
      <div data-testid={s.current ? 'flow-v3-box-popup-bottom' : undefined}>{l.bottom}</div>
      <div style={{ color: MUTED }}>{s.probes}</div>
      {l.mixed && <div data-testid={s.current ? 'flow-v3-box-popup-mixed' : undefined} style={{ color: '#fbbf24' }}>{l.mixed}</div>}
    </div>
  )
}

export default function BoxPanel({
  panel, place, panelRef, maxHeight, dragged, onClose, onHeaderPointerDown, onHeaderPointerMove, onHeaderPointerUp,
  onHeaderDoubleClick, renderCurrentTop, renderBadges,
}) {
  const [open, setOpen] = useState({}) // section key -> expanded
  const toggle = (k) => setOpen((o) => ({ ...o, [k]: !o[k] }))
  return (
    <div
      ref={panelRef}
      data-testid="flow-v3-box-popup"
      data-dragged={dragged ? '1' : '0'}
      style={{
        position: 'absolute', left: place?.left ?? 0, top: place?.top ?? 0, visibility: place ? 'visible' : 'hidden',
        zIndex: 5, background: TAG_BG(0.92), border: `1px solid rgba(${MAGENTA},0.35)`, borderRadius: 4,
        fontSize: '11px', fontFamily: 'sans-serif', color: TEXT, lineHeight: 1.6, maxWidth: 380, minWidth: 220,
        display: 'flex', flexDirection: 'column', maxHeight: maxHeight || undefined,
      }}
    >
      <div
        data-testid="flow-v3-box-panel-header"
        title="Drag to move · double-click to return bottom-left"
        onPointerDown={onHeaderPointerDown}
        onPointerMove={onHeaderPointerMove}
        onPointerUp={onHeaderPointerUp}
        onPointerCancel={onHeaderPointerUp}
        onDoubleClick={onHeaderDoubleClick}
        style={{ cursor: 'move', userSelect: 'none', touchAction: 'none', padding: '2px 20px 2px 8px', color: MUTED,
          borderBottom: `1px solid rgba(${MAGENTA},0.2)`, flex: '0 0 auto' }}
      >
        Boxes · visual aid — untested
        <button
          type="button"
          data-popup-close="1"
          data-testid="flow-v3-box-popup-close"
          aria-label="Hide box stats until the next anchor"
          title="Hide until the next anchor"
          onPointerDown={(e) => e.stopPropagation()}
          onClick={onClose}
          style={{ position: 'absolute', top: 2, right: 3, width: 14, height: 14, padding: 0, border: 'none', background: 'transparent', color: MUTED, cursor: 'pointer', fontSize: '12px', lineHeight: '14px' }}
        >
          ×
        </button>
      </div>
      <div data-testid="flow-v3-box-panel-body" style={{ overflowY: 'auto', padding: '4px 8px 6px', flex: '1 1 auto' }}>
        {panel.droppedText && <div data-testid="flow-v3-box-panel-dropped" style={{ color: MUTED }}>{panel.droppedText}</div>}
        {panel.sections.map((s) => {
          const isOpen = Boolean(open[s.key])
          return (
            <div
              key={s.key}
              data-testid={s.current ? 'flow-v3-box-panel-current' : 'flow-v3-box-panel-frozen'}
              style={s.current
                ? { marginTop: 4, padding: '3px 6px', border: `1px solid rgba(${MAGENTA},0.8)`, borderRadius: 3 }
                : { marginTop: 2, padding: '0 6px', color: '#cbd5e1' }}
            >
              {s.current && renderCurrentTop?.(s)}
              <div
                role="button"
                tabIndex={0}
                data-testid="flow-v3-box-panel-summary"
                aria-expanded={isOpen}
                onClick={() => toggle(s.key)}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(s.key) } }}
                style={{ cursor: 'pointer', color: s.current ? '#f0abfc' : undefined, fontWeight: s.current ? 600 : 400 }}
              >
                <span style={{ color: MUTED, display: 'inline-block', width: 10 }}>{isOpen ? '▾' : '▸'}</span>
                {s.summary}
                {renderBadges?.(s, 'summary')}
              </div>
              <div style={{ display: isOpen ? 'block' : 'none' }}>
                {renderBadges?.(s, 'stats')}
                <Stats s={s} testid={s.current ? 'flow-v3-box-panel-current-stats' : undefined} />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
