import React from 'react';

export default function ColumnSettingsModal({
  isOpen,
  onClose,
  columns,
  onUpdateColumns,
}) {
  if (!isOpen) return null;

  const handleToggle = (id) => {
    const next = columns.map((col) => {
      if (col.id === id) return { ...col, visible: !col.visible };
      return col;
    });
    onUpdateColumns(next);
  };

  const move = (idx, direction) => {
    const next = [...columns];
    const targetIdx = idx + direction;
    if (targetIdx < 0 || targetIdx >= next.length) return;
    const [moved] = next.splice(idx, 1);
    next.splice(targetIdx, 0, moved);
    onUpdateColumns(next);
  };

  const reset = () => {
    const next = columns.map(c => ({ ...c, visible: true }));
    // We don't easily have the original order here unless we pass it in, 
    // but we can at least turn them all back on.
    onUpdateColumns(next);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" style={{ width: 480 }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <div className="eyebrow">Table Settings</div>
            <h2>Configure Columns</h2>
          </div>
          <button className="ghost-button" onClick={onClose}>Close</button>
        </div>

        <div className="column-list" style={{ 
          display: 'flex', 
          flexDirection: 'column', 
          gap: '8px', 
          maxHeight: '500px', 
          overflowY: 'auto',
          padding: '4px',
          marginBottom: '16px'
        }}>
          {columns.map((col, idx) => (
            <div 
              key={col.id} 
              className={`column-item ${!col.visible ? 'hidden' : ''}`}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '8px 12px',
                background: '#0f172a',
                border: '1px solid #1f2937',
                borderRadius: '8px',
                opacity: col.visible ? 1 : 0.5
              }}
            >
              <input 
                type="checkbox" 
                checked={!!col.visible} 
                onChange={() => handleToggle(col.id)}
                disabled={col.alwaysVisible}
              />
              <span style={{ flex: 1, fontSize: '13px' }}>{col.label}</span>
              
              <div style={{ display: 'flex', gap: '4px' }}>
                <button 
                  className="icon-button" 
                  onClick={() => move(idx, -1)}
                  disabled={idx === 0}
                  style={{ background: 'none', border: '1px solid #334155', color: '#fff', cursor: 'pointer', borderRadius: '4px', width: '28px' }}
                >
                  ↑
                </button>
                <button 
                  className="icon-button" 
                  onClick={() => move(idx, 1)}
                  disabled={idx === columns.length - 1}
                  style={{ background: 'none', border: '1px solid #334155', color: '#fff', cursor: 'pointer', borderRadius: '4px', width: '28px' }}
                >
                  ↓
                </button>
              </div>
            </div>
          ))}
        </div>

        <div className="modal-actions">
          <button className="ghost-button" onClick={onClose}>Done</button>
        </div>
      </div>
    </div>
  );
}
