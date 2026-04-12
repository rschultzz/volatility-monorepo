export default function SettingsModal({ isOpen, settingsDraft, onChange, onClose, onRun }) {
  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <div className="eyebrow">Backtests</div>
            <h2>Scan Settings</h2>
          </div>
          <button className="ghost-button" onClick={onClose}>Close</button>
        </div>

        <div className="form-grid">
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
          Strategy selection now lives in the page header. This modal stays focused on scan and trade parameters.
        </div>

        <div className="modal-actions">
          <button className="ghost-button" onClick={onClose}>Cancel</button>
          <button className="primary-button" onClick={onRun}>Save & Run Scan</button>
        </div>
      </div>
    </div>
  );
}
