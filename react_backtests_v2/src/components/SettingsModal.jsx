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
              value={settingsDraft.startDate}
              onChange={(e) => onChange('startDate', e.target.value)}
            />
          </label>

          <label className="field">
            <span>End date</span>
            <input
              type="date"
              value={settingsDraft.endDate}
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
            <span>Minimum move (points)</span>
            <input
              type="number"
              step="1"
              value={settingsDraft.minMovePoints}
              onChange={(e) => onChange('minMovePoints', e.target.value)}
            />
          </label>

          <label className="field">
            <span>Touch buffer (points)</span>
            <input
              type="number"
              step="0.25"
              value={settingsDraft.touchBufferPoints}
              onChange={(e) => onChange('touchBufferPoints', e.target.value)}
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
          RTH-only and same-day constraints are enforced in this first build.
        </div>

        <div className="modal-actions">
          <button className="ghost-button" onClick={onClose}>Cancel</button>
          <button className="primary-button" onClick={onRun}>Save & Run Scan</button>
        </div>
      </div>
    </div>
  );
}
