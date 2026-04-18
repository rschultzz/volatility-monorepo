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

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <div className="eyebrow">Backtests</div>
            <h2>Strategy & Scan Settings</h2>
          </div>
          <button className="ghost-button" onClick={onClose}>Close</button>
        </div>

        <div className="form-grid">
          <label className="field field-wide">
            <span>Strategy name</span>
            <input
              type="text"
              value={strategyMetaDraft.displayName || ''}
              onChange={(e) => onStrategyMetaChange('displayName', e.target.value)}
            />
          </label>

          <label className="field field-wide">
            <span>Strategy key</span>
            <input
              type="text"
              value={strategyMetaDraft.strategyKey || ''}
              onChange={(e) => onStrategyMetaChange('strategyKey', e.target.value)}
            />
          </label>

          <label className="field field-wide">
            <span>Notes</span>
            <textarea
              value={strategyMetaDraft.notes || ''}
              onChange={(e) => onStrategyMetaChange('notes', e.target.value)}
              rows={3}
              style={{
                width: '100%',
                backgroundColor: '#111827',
                border: '1px solid #374151',
                color: 'white',
                borderRadius: '8px',
                padding: '10px 12px',
                resize: 'vertical',
              }}
            />
          </label>

          <label className="field">
            <span>Start date</span>
            <input
              type="date"
              value={settingsDraft.startDate || ''}
              onChange={(e) => onChange('startDate', e.target.value)}
            />
          </label>

          <label className="field">
            <span>End date</span>
            <input
              type="date"
              value={settingsDraft.endDate || ''}
              onChange={(e) => onChange('endDate', e.target.value)}
            />
          </label>

          <label className="field">
            <span>Minimum level GEX (BN)</span>
            <input
              type="number"
              step="1"
              value={settingsDraft.minLevelGexBn}
              onChange={(e) => onChange('minLevelGexBn', e.target.value)}
            />
          </label>

          <label className="field">
            <span>Zone merge distance (pts)</span>
            <input
              type="number"
              step="1"
              value={settingsDraft.zoneMergeDistancePts}
              onChange={(e) => onChange('zoneMergeDistancePts', e.target.value)}
            />
          </label>

          <label className="field">
            <span>Minimum clean move (pts)</span>
            <input
              type="number"
              step="1"
              value={settingsDraft.minCleanMovePoints}
              onChange={(e) => onChange('minCleanMovePoints', e.target.value)}
            />
          </label>

          <label className="field">
            <span>Target proximity (pts)</span>
            <input
              type="number"
              step="0.5"
              value={settingsDraft.targetProximityPts}
              onChange={(e) => onChange('targetProximityPts', e.target.value)}
            />
          </label>

          <label className="field">
            <span>Max zone breach (pts)</span>
            <input
              type="number"
              step="0.5"
              value={settingsDraft.maxZoneBreachPts}
              onChange={(e) => onChange('maxZoneBreachPts', e.target.value)}
            />
          </label>

          <label className="field">
            <span>Pivot strength (bars)</span>
            <input
              type="number"
              step="1"
              min="1"
              value={settingsDraft.pivotStrengthBars}
              onChange={(e) => onChange('pivotStrengthBars', e.target.value)}
            />
          </label>

          <label className="field">
            <span>Level family</span>
            <select
              value={settingsDraft.levelFamily}
              onChange={(e) => onChange('levelFamily', e.target.value)}
            >
              <option value="primary">Primary walls only</option>
              <option value="strong">Strong walls only</option>
              <option value="both">Primary + strong</option>
            </select>
          </label>

          <label className="field">
            <span>Target consolidation window (min)</span>
            <input
              type="number"
              step="1"
              min="1"
              value={settingsDraft.consolidationWindowMinutes}
              onChange={(e) => onChange('consolidationWindowMinutes', e.target.value)}
            />
          </label>

          <label className="field">
            <span>Short trigger: min Δ Put Skew %</span>
            <input
              type="number"
              step="1"
              value={settingsDraft.shortPutSkewIncreasePct}
              onChange={(e) => onChange('shortPutSkewIncreasePct', e.target.value)}
            />
          </label>

          <label className="field">
            <span>Short trigger: max Δ Call Skew %</span>
            <input
              type="number"
              step="1"
              value={settingsDraft.shortCallSkewMaxPct}
              onChange={(e) => onChange('shortCallSkewMaxPct', e.target.value)}
            />
          </label>

          <label className="field">
            <span>Entry within top of range (pts)</span>
            <input
              type="number"
              step="0.5"
              min="0"
              value={settingsDraft.entryWithinTopPts}
              onChange={(e) => onChange('entryWithinTopPts', e.target.value)}
            />
          </label>

          <label className="field">
            <span>Entry search window (min)</span>
            <input
              type="number"
              step="1"
              min="1"
              value={settingsDraft.entrySearchWindowMinutes}
              onChange={(e) => onChange('entrySearchWindowMinutes', e.target.value)}
            />
          </label>

          <label className="field">
            <span>Initial stop (pts)</span>
            <input
              type="number"
              step="0.5"
              min="0.5"
              value={settingsDraft.initialStopPts}
              onChange={(e) => onChange('initialStopPts', e.target.value)}
            />
          </label>

          <label className="field">
            <span>Trail activate profit (pts)</span>
            <input
              type="number"
              step="0.5"
              min="0.5"
              value={settingsDraft.trailActivateProfitPts}
              onChange={(e) => onChange('trailActivateProfitPts', e.target.value)}
            />
          </label>

          <label className="field">
            <span>Trailing stop (pts)</span>
            <input
              type="number"
              step="0.5"
              min="0.5"
              value={settingsDraft.trailingStopPts}
              onChange={(e) => onChange('trailingStopPts', e.target.value)}
            />
          </label>

          <label className="field">
            <span>Take profit (pts)</span>
            <input
              type="number"
              step="0.5"
              min="0.5"
              value={settingsDraft.takeProfitPts}
              onChange={(e) => onChange('takeProfitPts', e.target.value)}
            />
          </label>

          <label className="field">
            <span>Prior context: max down/up ratio</span>
            <input
              type="number"
              step="0.1"
              min="0"
              value={settingsDraft.maxPriorDownUpRatio}
              onChange={(e) => onChange('maxPriorDownUpRatio', e.target.value)}
            />
          </label>

          <label className="field">
            <span>Prior context: max start % of range</span>
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={settingsDraft.maxStartPctOfRange}
              onChange={(e) => onChange('maxStartPctOfRange', e.target.value)}
            />
          </label>

          <label className="field">
            <span>Max move loss before invalidation</span>
            <input
              type="number"
              step="0.05"
              min="0"
              max="1"
              value={settingsDraft.maxMoveLossPct}
              onChange={(e) => onChange('maxMoveLossPct', e.target.value)}
            />
            <div style={{ fontSize: '10px', color: '#94a3b8', marginTop: '4px' }}>
              If price gives back this fraction of the up move during consolidation, the setup is invalidated. Default 0.75 = 75%.
            </div>
          </label>

          <label className="field field-wide">
            <span>Maximum results</span>
            <input
              type="number"
              step="100"
              value={settingsDraft.maxResults}
              onChange={(e) => onChange('maxResults', e.target.value)}
            />
          </label>
        </div>

        <div className="helper-note">
          This keeps one shared settings window, but adds strategy metadata so you can save the current strategy or create a new named variant. That leaves room for dynamic per-strategy fields later.
        </div>

        <div className="modal-actions">
          <button className="ghost-button" onClick={onCreateStrategy}>
            {creatingStrategy ? 'Creating…' : 'Save as new strategy'}
          </button>
          <button className="ghost-button" onClick={onClose}>Cancel</button>
          <button className="primary-button" onClick={onRun}>Save & Run Scan</button>
        </div>
      </div>
    </div>
  );
}
