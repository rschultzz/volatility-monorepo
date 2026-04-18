import { useState } from 'react';

const FIELD_HELP = {
  displayName: 'The human-readable name shown in the strategy selector.',
  strategyKey: 'Unique internal key used to store and retrieve this strategy in the database.',
  notes: 'Free-text notes saved with the strategy. Useful for recording what you changed and why.',
  startDate: 'First trade date to include in the backtest scan.',
  endDate: 'Last trade date to include in the backtest scan.',
  minLevelGexBn: 'Minimum absolute GEX (in billions) a wall must carry to qualify. Uses absolute value — so a wall qualifies if its GEX is ≥ +50BN or ≤ −50BN. Filters out weak walls unlikely to act as real resistance or support.',
  zoneMergeDistancePts: 'Adjacent GEX levels within this many points are merged into a single zone. Prevents treating clustered levels as separate targets.',
  minCleanMovePoints: 'Minimum distance (pts) of clear space between the source zone and the target zone. Ensures there is room for a meaningful directional move.',
  targetProximityPts: 'How close price must come to a GEX wall (pts) to register the wall as hit and begin the consolidation observation window.',
  maxZoneBreachPts: 'During consolidation, if price closes this many points above the wall high, the range is discarded and the system promotes to the next GEX wall above.',
  pivotStrengthBars: 'Number of bars on each side required to confirm a swing high or low as a valid pivot. Higher values find stronger, less frequent pivots.',
  levelFamily: 'Which GEX wall types to include. Primary walls carry the most gamma. Strong walls are secondary. Both combines them.',
  minMinutesAfterOpen: 'The pivot that starts the up move must occur at least this many minutes after the 6:30 PT RTH open. Filters out chaotic early-session moves. Set to 0 to disable.',
  consolidationWindowMinutes: 'After price hits the target wall, the system observes price action for up to this many minutes to build the consolidation range. The skew signal must fire within this window.',
  maxMoveLossPct: 'If price gives back more than this fraction of the up move during consolidation, the setup is invalidated. Example: 0.75 means a 40pt move that retraces more than 30pts is dead.',
  shortPutSkewIncreasePct: 'Minimum percentage increase in put skew (relative to model expectation) required to trigger the short signal. Captures defensive options positioning near the wall.',
  shortCallSkewMaxPct: 'Maximum allowed change in call skew at signal time. Keeps the signal from firing when the market is still bidding calls aggressively, suggesting continued upside.',
  longPutSkewMinDecreasePct: 'Put skew must decrease by at least this percentage (fear unwinding after a selloff). A reading of 80 means delta_put_skew must be ≤ −80%.',
  longCallSkewMinIncreasePct: 'Call skew must increase by at least this percentage (upside being bid). A reading of 30 means delta_call_skew must be ≥ +30%.',
  entryWithinTopPts: 'After consolidation closes, entry triggers when price trades within this many points of the confirmed range high.',
  entrySearchWindowMinutes: 'How many minutes after consolidation closes to look for an entry. If price does not return to the top of range in this window, no trade is taken.',
  initialStopPts: 'Initial stop loss distance above the entry price.',
  trailActivateProfitPts: 'The trade must reach this many points of profit before the trailing stop activates.',
  trailingStopPts: 'Once activated, the trailing stop sits this many points above the lowest low reached since entry.',
  takeProfitPts: 'Fixed take-profit target below entry. Trade exits immediately when price reaches this level.',
  maxPriorDownUpRatio: 'If the largest prior down move of the session exceeds this multiple of the current up move, the setup is treated as a bounce off the lows and is invalidated.',
  maxStartPctOfRange: "The up move pivot must start above this percentile of the session range so far. 0.20 means the pivot cannot be in the bottom 20% of the day's range.",
  maxResults: 'Maximum number of result rows returned per scan. Increase for long date ranges.',
};

const SECTION_CARD = {
  background: '#0b1220',
  border: '1px solid #1f2937',
  borderRadius: '14px',
  padding: '16px 18px 18px',
  display: 'flex',
  flexDirection: 'column',
  gap: '14px',
  marginBottom: '4px',
};

const SECTION_GRID = {
  display: 'grid',
  gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
  gap: '14px',
};

function InfoIcon({ fieldKey }) {
  const [open, setOpen] = useState(false);
  const text = FIELD_HELP[fieldKey];
  if (!text) return null;
  return (
    <span style={{ position: 'relative', display: 'inline-flex', alignItems: 'center', marginLeft: '6px' }}>
      <button
        type="button"
        onClick={(e) => { e.preventDefault(); setOpen(v => !v); }}
        style={{
          width: '15px',
          height: '15px',
          borderRadius: '50%',
          border: '1px solid #334155',
          background: open ? '#1e3a5f' : '#0f172a',
          color: open ? '#93c5fd' : '#475569',
          fontSize: '9px',
          fontWeight: '700',
          cursor: 'pointer',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          lineHeight: 1,
          padding: 0,
          flexShrink: 0,
        }}
        aria-label={`Info for ${fieldKey}`}
      >
        i
      </button>
      {open && (
        <span style={{
          position: 'absolute',
          bottom: '20px',
          left: '0',
          zIndex: 100,
          background: '#0f172a',
          border: '1px solid #334155',
          borderRadius: '10px',
          padding: '10px 12px',
          fontSize: '11px',
          lineHeight: '1.55',
          color: '#cbd5e1',
          width: '230px',
          boxShadow: '0 12px 32px rgba(0,0,0,0.5)',
          pointerEvents: 'none',
        }}>
          {text}
        </span>
      )}
    </span>
  );
}

function SectionCard({ title, children }) {
  return (
    <div style={SECTION_CARD}>
      <div style={{
        borderBottom: '1px solid #1f2937',
        paddingBottom: '10px',
        marginBottom: '2px',
      }}>
        <span style={{
          fontSize: '11px',
          fontWeight: '700',
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: '#93c5fd',
        }}>
          {title}
        </span>
      </div>
      <div style={SECTION_GRID}>
        {children}
      </div>
    </div>
  );
}

function Field({ label, fieldKey, wide, children }) {
  return (
    <label className={`field${wide ? ' field-wide' : ''}`} style={wide ? { gridColumn: '1 / -1' } : {}}>
      <span style={{ display: 'flex', alignItems: 'center' }}>
        {label}
        <InfoIcon fieldKey={fieldKey} />
      </span>
      {children}
    </label>
  );
}

export default function SettingsModal({
  isOpen,
  settingsDraft,
  strategyMetaDraft,
  creatingStrategy,
  onStrategyMetaChange,
  onCreateStrategy,
  onChange,
  onClose,
  onRun,
}) {
  if (!isOpen) return null;

  const isDownMove = (strategyMetaDraft.strategyKey || '').includes('down_move') ||
                     (strategyMetaDraft.baseStrategyKey || '').includes('down_move');

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" style={{ width: 'min(980px, 100%)', padding: '20px' }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <div className="eyebrow">Backtests</div>
            <h2>Strategy &amp; Scan Settings</h2>
          </div>
          <button className="ghost-button" onClick={onClose}>Close</button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>

          {/* ── Strategy ── */}
          <SectionCard title="Strategy">
            <Field label="Strategy name" fieldKey="displayName" wide>
              <input type="text" value={strategyMetaDraft.displayName || ''} onChange={(e) => onStrategyMetaChange('displayName', e.target.value)} />
            </Field>
            <Field label="Strategy key" fieldKey="strategyKey" wide>
              <input type="text" value={strategyMetaDraft.strategyKey || ''} onChange={(e) => onStrategyMetaChange('strategyKey', e.target.value)} />
            </Field>
            <Field label="Notes" fieldKey="notes" wide>
              <textarea
                value={strategyMetaDraft.notes || ''}
                onChange={(e) => onStrategyMetaChange('notes', e.target.value)}
                rows={2}
                style={{ width: '100%', backgroundColor: '#020617', border: '1px solid #334155', color: '#e5e7eb', borderRadius: '10px', padding: '10px 12px', resize: 'vertical' }}
              />
            </Field>
            <Field label="Start date" fieldKey="startDate">
              <input type="date" value={settingsDraft.startDate || ''} onChange={(e) => onChange('startDate', e.target.value)} />
            </Field>
            <Field label="End date" fieldKey="endDate">
              <input type="date" value={settingsDraft.endDate || ''} onChange={(e) => onChange('endDate', e.target.value)} />
            </Field>
          </SectionCard>

          {/* ── Move detection ── */}
          <SectionCard title="Move detection">
            <Field label="Minimum level GEX (BN)" fieldKey="minLevelGexBn">
              <input type="number" step="1" value={settingsDraft.minLevelGexBn} onChange={(e) => onChange('minLevelGexBn', e.target.value)} />
            </Field>
            <Field label="Zone merge distance (pts)" fieldKey="zoneMergeDistancePts">
              <input type="number" step="1" value={settingsDraft.zoneMergeDistancePts} onChange={(e) => onChange('zoneMergeDistancePts', e.target.value)} />
            </Field>
            <Field label="Minimum clean move (pts)" fieldKey="minCleanMovePoints">
              <input type="number" step="1" value={settingsDraft.minCleanMovePoints} onChange={(e) => onChange('minCleanMovePoints', e.target.value)} />
            </Field>
            <Field label="Target proximity (pts)" fieldKey="targetProximityPts">
              <input type="number" step="0.5" value={settingsDraft.targetProximityPts} onChange={(e) => onChange('targetProximityPts', e.target.value)} />
            </Field>
            <Field label="Pivot strength (bars)" fieldKey="pivotStrengthBars">
              <input type="number" step="1" min="1" value={settingsDraft.pivotStrengthBars} onChange={(e) => onChange('pivotStrengthBars', e.target.value)} />
            </Field>
            <Field label="Level family" fieldKey="levelFamily">
              <select value={settingsDraft.levelFamily} onChange={(e) => onChange('levelFamily', e.target.value)}>
                <option value="primary">Primary walls only</option>
                <option value="strong">Strong walls only</option>
                <option value="both">Primary + strong</option>
              </select>
            </Field>
            <Field label="Min minutes after open" fieldKey="minMinutesAfterOpen">
              <input type="number" step="1" min="0" value={settingsDraft.minMinutesAfterOpen} onChange={(e) => onChange('minMinutesAfterOpen', e.target.value)} />
            </Field>
            <Field label="Prior context: max down/up ratio" fieldKey="maxPriorDownUpRatio">
              <input type="number" step="0.1" min="0" value={settingsDraft.maxPriorDownUpRatio} onChange={(e) => onChange('maxPriorDownUpRatio', e.target.value)} />
            </Field>
            <Field label="Prior context: max start % of range" fieldKey="maxStartPctOfRange">
              <input type="number" step="0.01" min="0" max="1" value={settingsDraft.maxStartPctOfRange} onChange={(e) => onChange('maxStartPctOfRange', e.target.value)} />
            </Field>
          </SectionCard>

          {/* ── Range definition ── */}
          <SectionCard title="Range definition">
            <Field label="Target consolidation window (min)" fieldKey="consolidationWindowMinutes">
              <input type="number" step="1" min="1" value={settingsDraft.consolidationWindowMinutes} onChange={(e) => onChange('consolidationWindowMinutes', e.target.value)} />
            </Field>
            <Field label="Max zone breach (pts)" fieldKey="maxZoneBreachPts">
              <input type="number" step="0.5" value={settingsDraft.maxZoneBreachPts} onChange={(e) => onChange('maxZoneBreachPts', e.target.value)} />
            </Field>
            <Field label="Max move loss before invalidation" fieldKey="maxMoveLossPct">
              <input type="number" step="0.05" min="0" max="1" value={settingsDraft.maxMoveLossPct} onChange={(e) => onChange('maxMoveLossPct', e.target.value)} />
            </Field>
          </SectionCard>

          {/* ── Vol signal ── */}
          <SectionCard title="Vol signal">
            {isDownMove ? (
              <>
                <Field label="Long trigger: min Δ Put Skew decrease %" fieldKey="longPutSkewMinDecreasePct">
                  <input type="number" step="1" min="0" value={settingsDraft.longPutSkewMinDecreasePct} onChange={(e) => onChange('longPutSkewMinDecreasePct', e.target.value)} />
                </Field>
                <Field label="Long trigger: min Δ Call Skew increase %" fieldKey="longCallSkewMinIncreasePct">
                  <input type="number" step="1" min="0" value={settingsDraft.longCallSkewMinIncreasePct} onChange={(e) => onChange('longCallSkewMinIncreasePct', e.target.value)} />
                </Field>
              </>
            ) : (
              <>
                <Field label="Short trigger: min Δ Put Skew %" fieldKey="shortPutSkewIncreasePct">
                  <input type="number" step="1" value={settingsDraft.shortPutSkewIncreasePct} onChange={(e) => onChange('shortPutSkewIncreasePct', e.target.value)} />
                </Field>
                <Field label="Short trigger: max Δ Call Skew %" fieldKey="shortCallSkewMaxPct">
                  <input type="number" step="1" value={settingsDraft.shortCallSkewMaxPct} onChange={(e) => onChange('shortCallSkewMaxPct', e.target.value)} />
                </Field>
              </>
            )}
          </SectionCard>

          {/* ── Trade entry ── */}
          <SectionCard title="Trade entry">
            <Field label="Entry within top of range (pts)" fieldKey="entryWithinTopPts">
              <input type="number" step="0.5" min="0" value={settingsDraft.entryWithinTopPts} onChange={(e) => onChange('entryWithinTopPts', e.target.value)} />
            </Field>
            <Field label="Entry search window (min)" fieldKey="entrySearchWindowMinutes">
              <input type="number" step="1" min="1" value={settingsDraft.entrySearchWindowMinutes} onChange={(e) => onChange('entrySearchWindowMinutes', e.target.value)} />
            </Field>
          </SectionCard>

          {/* ── Trade management ── */}
          <SectionCard title="Trade management">
            <Field label="Initial stop (pts)" fieldKey="initialStopPts">
              <input type="number" step="0.5" min="0.5" value={settingsDraft.initialStopPts} onChange={(e) => onChange('initialStopPts', e.target.value)} />
            </Field>
            <Field label="Trail activate profit (pts)" fieldKey="trailActivateProfitPts">
              <input type="number" step="0.5" min="0.5" value={settingsDraft.trailActivateProfitPts} onChange={(e) => onChange('trailActivateProfitPts', e.target.value)} />
            </Field>
            <Field label="Trailing stop (pts)" fieldKey="trailingStopPts">
              <input type="number" step="0.5" min="0.5" value={settingsDraft.trailingStopPts} onChange={(e) => onChange('trailingStopPts', e.target.value)} />
            </Field>
            <Field label="Take profit (pts)" fieldKey="takeProfitPts">
              <input type="number" step="0.5" min="0.5" value={settingsDraft.takeProfitPts} onChange={(e) => onChange('takeProfitPts', e.target.value)} />
            </Field>
          </SectionCard>

          {/* ── Scan limits ── */}
          <SectionCard title="Scan limits">
            <Field label="Maximum results" fieldKey="maxResults" wide>
              <input type="number" step="100" value={settingsDraft.maxResults} onChange={(e) => onChange('maxResults', e.target.value)} />
            </Field>
          </SectionCard>

        </div>

        <div className="modal-actions" style={{ marginTop: '18px' }}>
          <button className="ghost-button" onClick={onCreateStrategy}>
            {creatingStrategy ? 'Creating…' : 'Save as new strategy'}
          </button>
          <button className="ghost-button" onClick={onClose}>Cancel</button>
          <button className="primary-button" onClick={onRun}>Save &amp; Run Scan</button>
        </div>
      </div>
    </div>
  );
}
