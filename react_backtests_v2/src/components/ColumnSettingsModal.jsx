import React, { useState } from 'react';

export default function ColumnSettingsModal({
  isOpen,
  onClose,
  columns,
  onUpdateColumns,
}) {
  // Drag state — the id of the row being dragged, and the id of the row
  // currently under the cursor as a drop target.
  const [draggingId, setDraggingId] = useState(null);
  const [dragOverId, setDragOverId] = useState(null);

  if (!isOpen) return null;

  const handleToggle = (id) => {
    const next = columns.map((col) => (
      col.id === id ? { ...col, visible: !col.visible } : col
    ));
    onUpdateColumns(next);
  };

  const handleDragStart = (e, id) => {
    setDraggingId(id);
    e.dataTransfer.effectAllowed = 'move';
    // Firefox needs a setData call to actually start dragging.
    try { e.dataTransfer.setData('text/plain', id); } catch (_) {}
  };

  const handleDragOver = (e, id) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    if (id !== dragOverId) setDragOverId(id);
  };

  const handleDragLeave = (id) => {
    if (dragOverId === id) setDragOverId(null);
  };

  const handleDrop = (e, dropTargetId) => {
    e.preventDefault();
    const sourceId = draggingId;
    setDraggingId(null);
    setDragOverId(null);
    if (!sourceId || sourceId === dropTargetId) return;

    const fromIdx = columns.findIndex(c => c.id === sourceId);
    const toIdx = columns.findIndex(c => c.id === dropTargetId);
    if (fromIdx === -1 || toIdx === -1) return;

    const next = [...columns];
    const [moved] = next.splice(fromIdx, 1);
    next.splice(toIdx, 0, moved);
    onUpdateColumns(next);
  };

  const handleDragEnd = () => {
    setDraggingId(null);
    setDragOverId(null);
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

        <div style={{
          fontSize: 11,
          color: '#64748b',
          marginBottom: 8,
          letterSpacing: '0.02em',
        }}>
          Drag rows to reorder · Click checkbox to show or hide
        </div>

        <div className="column-list" style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '4px',
          maxHeight: '500px',
          overflowY: 'auto',
          padding: '4px',
          marginBottom: '16px',
        }}>
          {columns.map((col) => {
            const isDragging = draggingId === col.id;
            const isDropTarget = dragOverId === col.id && draggingId && draggingId !== col.id;
            const canDrag = !col.alwaysVisible;
            return (
              <div
                key={col.id}
                draggable={canDrag}
                onDragStart={(e) => canDrag && handleDragStart(e, col.id)}
                onDragOver={(e) => handleDragOver(e, col.id)}
                onDragLeave={() => handleDragLeave(col.id)}
                onDrop={(e) => handleDrop(e, col.id)}
                onDragEnd={handleDragEnd}
                className={`column-item ${!col.visible ? 'hidden' : ''}`}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  padding: '8px 12px',
                  background: isDropTarget ? 'rgba(37, 99, 235, 0.18)' : '#0f172a',
                  border: `1px solid ${isDropTarget ? '#2563eb' : '#1f2937'}`,
                  borderRadius: '8px',
                  opacity: isDragging ? 0.4 : (col.visible ? 1 : 0.55),
                  cursor: canDrag ? 'grab' : 'default',
                  userSelect: 'none',
                  transition: 'background 0.1s ease, border-color 0.1s ease',
                }}
              >
                <span
                  aria-hidden="true"
                  title={canDrag ? 'Drag to reorder' : 'Required column'}
                  style={{
                    color: canDrag ? '#64748b' : '#334155',
                    fontSize: 16,
                    fontWeight: 700,
                    letterSpacing: '-2px',
                    width: 12,
                    flexShrink: 0,
                  }}
                >
                  ⋮⋮
                </span>
                <input
                  type="checkbox"
                  checked={!!col.visible}
                  onChange={() => handleToggle(col.id)}
                  disabled={col.alwaysVisible}
                  // Stop the click from initiating a drag on the row itself
                  onClick={(e) => e.stopPropagation()}
                  onMouseDown={(e) => e.stopPropagation()}
                />
                <span style={{ flex: 1, fontSize: '13px' }}>{col.label}</span>
              </div>
            );
          })}
        </div>

        <div className="modal-actions">
          <button className="ghost-button" onClick={onClose}>Done</button>
        </div>
      </div>
    </div>
  );
}
