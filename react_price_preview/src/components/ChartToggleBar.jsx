// Horizontal row of toggle pills that sit above the Price chart and
// collapse/expand the SMILE, SIGNALS, and GEX sub-panels. Adding a new
// toggle is a matter of appending an entry to the `pills` prop — the
// container owns layout, the parent only owns state.
//
// Each pill is hidden via visibility:hidden (rather than removed) while
// its sub-panel is open, so the remaining pills don't shift sideways.

const PILL_BASE_STYLE = {
  cursor: 'pointer',
  background: 'rgba(15, 23, 42, 0.92)',
  border: '1px solid #1f2937',
  borderRadius: '10px',
  padding: '6px 14px',
  boxShadow: '0 10px 25px rgba(0,0,0,0.4)',
  color: '#60a5fa',
  fontSize: '13px',
  fontWeight: 800,
  textTransform: 'uppercase',
  letterSpacing: '0.05em',
  userSelect: 'none',
  height: '32px',
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  lineHeight: 1,
  fontFamily: 'inherit',
}

export default function ChartToggleBar({ pills }) {
  if (!pills || pills.length === 0) return null
  return (
    <div
      style={{
        position: 'absolute',
        top: 8,
        left: 64,
        zIndex: 10,
        display: 'flex',
        gap: '8px',
        alignItems: 'center',
        pointerEvents: 'none',
      }}
    >
      {pills.map(pill => (
        <button
          key={pill.key}
          type="button"
          onClick={pill.onToggle}
          title={pill.title || `Show ${pill.label}`}
          aria-label={pill.title || `Show ${pill.label}`}
          style={{
            ...PILL_BASE_STYLE,
            visibility: pill.isOpen ? 'hidden' : 'visible',
            pointerEvents: pill.isOpen ? 'none' : 'auto',
          }}
        >
          {pill.label}
        </button>
      ))}
    </div>
  )
}
