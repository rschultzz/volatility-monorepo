import React, { useState } from 'react';

/**
 * Two-pane column selector modeled after Quickbase's report column picker.
 *
 * Props:
 *   isOpen            – boolean
 *   onClose           – () => void
 *   columns           – Array<{ id, label, visible, alwaysVisible? }>
 *   onUpdateColumns   – (nextColumns) => void
 *   defaultColumns?   – Array<string> of column ids representing the default
 *                       column set, in default order. If supplied, a
 *                       "Set to default columns" button is rendered.
 *
 * Behavior:
 *   - Available pane shows columns where `visible: false` (filtered by search).
 *   - Report Columns pane shows columns where `visible: true`, in array order.
 *   - Center arrows move the selected column between panes.
 *   - Right-side stacked arrows reorder the selected Report column.
 *   - Double-click on any row moves it to the other pane.
 *   - alwaysVisible columns are pinned to the Report pane and tagged "req".
 *     They can still be reordered, just not removed.
 */
export default function ColumnSettingsModal({
  isOpen,
  onClose,
  columns,
  onUpdateColumns,
  defaultColumns,
}) {
  const [search, setSearch] = useState('');
  const [availableSelectedId, setAvailableSelectedId] = useState(null);
  const [reportSelectedId, setReportSelectedId] = useState(null);

  if (!isOpen) return null;

  // -------------------- derived state --------------------
  const visibleCols = columns.filter((c) => c.visible);
  const hiddenCols = columns.filter((c) => !c.visible);
  const filteredHidden = search
    ? hiddenCols.filter((c) =>
        c.label.toLowerCase().includes(search.toLowerCase())
      )
    : hiddenCols;

  // -------------------- helpers --------------------
  const lastVisibleIndex = (arr) => {
    for (let i = arr.length - 1; i >= 0; i--) {
      if (arr[i].visible) return i;
    }
    return -1;
  };

  const showColumn = (id) => {
    const target = columns.find((c) => c.id === id);
    if (!target || target.visible) return;
    const without = columns.filter((c) => c.id !== id);
    const insertIdx = lastVisibleIndex(without) + 1;
    const next = [
      ...without.slice(0, insertIdx),
      { ...target, visible: true },
      ...without.slice(insertIdx),
    ];
    onUpdateColumns(next);
    setReportSelectedId(id);
    setAvailableSelectedId(null);
  };

  const hideColumn = (id) => {
    const target = columns.find((c) => c.id === id);
    if (!target || target.alwaysVisible) return;
    const next = columns.map((c) =>
      c.id === id ? { ...c, visible: false } : c
    );
    onUpdateColumns(next);
    setAvailableSelectedId(id);
    setReportSelectedId(null);
  };

  const swapInArray = (arr, idxA, idxB) => {
    const next = [...arr];
    [next[idxA], next[idxB]] = [next[idxB], next[idxA]];
    return next;
  };

  const moveSelectedUp = () => {
    if (!reportSelectedId) return;
    const visIdx = visibleCols.findIndex((c) => c.id === reportSelectedId);
    if (visIdx <= 0) return;
    const prevId = visibleCols[visIdx - 1].id;
    const fromIdx = columns.findIndex((c) => c.id === reportSelectedId);
    const toIdx = columns.findIndex((c) => c.id === prevId);
    onUpdateColumns(swapInArray(columns, fromIdx, toIdx));
  };

  const moveSelectedDown = () => {
    if (!reportSelectedId) return;
    const visIdx = visibleCols.findIndex((c) => c.id === reportSelectedId);
    if (visIdx === -1 || visIdx >= visibleCols.length - 1) return;
    const nextId = visibleCols[visIdx + 1].id;
    const fromIdx = columns.findIndex((c) => c.id === reportSelectedId);
    const toIdx = columns.findIndex((c) => c.id === nextId);
    onUpdateColumns(swapInArray(columns, fromIdx, toIdx));
  };

  const moveSelectedToTop = () => {
    if (!reportSelectedId) return;
    const target = columns.find((c) => c.id === reportSelectedId);
    if (!target) return;
    const without = columns.filter((c) => c.id !== reportSelectedId);
    onUpdateColumns([target, ...without]);
  };

  const moveSelectedToBottom = () => {
    if (!reportSelectedId) return;
    const target = columns.find((c) => c.id === reportSelectedId);
    if (!target) return;
    const without = columns.filter((c) => c.id !== reportSelectedId);
    const insertIdx = lastVisibleIndex(without) + 1;
    onUpdateColumns([
      ...without.slice(0, insertIdx),
      target,
      ...without.slice(insertIdx),
    ]);
  };

  const handleSetDefaults = () => {
    if (!defaultColumns?.length) return;
    const idSet = new Set(defaultColumns);
    const byId = new Map(columns.map((c) => [c.id, c]));
    const ordered = [];
    defaultColumns.forEach((id) => {
      const c = byId.get(id);
      if (c) ordered.push({ ...c, visible: true });
    });
    columns.forEach((c) => {
      if (!idSet.has(c.id)) {
        ordered.push({ ...c, visible: !!c.alwaysVisible });
      }
    });
    onUpdateColumns(ordered);
    setReportSelectedId(null);
    setAvailableSelectedId(null);
  };

  // -------------------- styles --------------------
  const paneStyle = {
    flex: 1,
    minWidth: 0,
    display: 'flex',
    flexDirection: 'column',
    background: '#0b1220',
    border: '1px solid #1f2937',
    borderRadius: 8,
    overflow: 'hidden',
  };

  const paneHeaderStyle = {
    padding: '8px 12px',
    borderBottom: '1px solid #1f2937',
    fontSize: 11,
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
    color: '#94a3b8',
    background: '#0f172a',
  };

  const listStyle = {
    flex: 1,
    overflowY: 'auto',
    minHeight: 320,
    maxHeight: 420,
    padding: 4,
  };

  const itemStyle = (selected) => ({
    padding: '6px 10px',
    fontSize: 13,
    color: '#e2e8f0',
    background: selected ? 'rgba(37, 99, 235, 0.22)' : 'transparent',
    border: `1px solid ${selected ? '#2563eb' : 'transparent'}`,
    borderRadius: 6,
    cursor: 'pointer',
    userSelect: 'none',
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginBottom: 2,
  });

  const arrowBtnStyle = (disabled) => ({
    width: 36,
    height: 32,
    background: disabled ? '#0f172a' : '#1e293b',
    color: disabled ? '#475569' : '#e2e8f0',
    border: '1px solid #1f2937',
    borderRadius: 6,
    cursor: disabled ? 'not-allowed' : 'pointer',
    fontSize: 14,
    lineHeight: 1,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  });

  // -------------------- enablement --------------------
  const canMoveRight = !!availableSelectedId;
  const canMoveLeft = (() => {
    if (!reportSelectedId) return false;
    const c = columns.find((x) => x.id === reportSelectedId);
    return !!c && !c.alwaysVisible;
  })();
  const visIdxOfSelected = reportSelectedId
    ? visibleCols.findIndex((c) => c.id === reportSelectedId)
    : -1;
  const canMoveUp = visIdxOfSelected > 0;
  const canMoveDown =
    visIdxOfSelected !== -1 && visIdxOfSelected < visibleCols.length - 1;

  // -------------------- render --------------------
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-card"
        style={{ width: 760 }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <div className="eyebrow">Table Settings</div>
            <h2>Configure Columns</h2>
          </div>
          <button className="ghost-button" onClick={onClose}>
            Close
          </button>
        </div>

        <div
          style={{
            fontSize: 11,
            color: '#64748b',
            marginBottom: 10,
            letterSpacing: '0.02em',
          }}
        >
          Select a column, then use the arrows to add, remove, or reorder ·
          Double-click to move
        </div>

        <div
          style={{
            display: 'flex',
            alignItems: 'stretch',
            gap: 10,
            marginBottom: 16,
          }}
        >
          {/* AVAILABLE PANE */}
          <div style={paneStyle}>
            <div style={paneHeaderStyle}>Available</div>
            <div style={{ padding: 8, borderBottom: '1px solid #1f2937' }}>
              <input
                type="text"
                value={search}
                placeholder="Search columns…"
                onChange={(e) => setSearch(e.target.value)}
                style={{
                  width: '100%',
                  padding: '6px 10px',
                  fontSize: 12,
                  background: '#0f172a',
                  color: '#e2e8f0',
                  border: '1px solid #1f2937',
                  borderRadius: 6,
                  outline: 'none',
                  boxSizing: 'border-box',
                }}
              />
            </div>
            <div style={listStyle}>
              {filteredHidden.length === 0 && (
                <div
                  style={{
                    padding: '12px',
                    color: '#475569',
                    fontSize: 12,
                    fontStyle: 'italic',
                  }}
                >
                  {search ? 'No matches' : 'All columns are in the report'}
                </div>
              )}
              {filteredHidden.map((col) => {
                const selected = availableSelectedId === col.id;
                return (
                  <div
                    key={col.id}
                    style={itemStyle(selected)}
                    onClick={() => setAvailableSelectedId(col.id)}
                    onDoubleClick={() => showColumn(col.id)}
                  >
                    <span style={{ flex: 1 }}>{col.label}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* CENTER MOVE ARROWS */}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              gap: 8,
            }}
          >
            <button
              style={arrowBtnStyle(!canMoveRight)}
              disabled={!canMoveRight}
              onClick={() =>
                availableSelectedId && showColumn(availableSelectedId)
              }
              title="Add to report"
            >
              →
            </button>
            <button
              style={arrowBtnStyle(!canMoveLeft)}
              disabled={!canMoveLeft}
              onClick={() => reportSelectedId && hideColumn(reportSelectedId)}
              title="Remove from report"
            >
              ←
            </button>
          </div>

          {/* REPORT COLUMNS PANE */}
          <div style={paneStyle}>
            <div style={paneHeaderStyle}>
              Report Columns ({visibleCols.length})
            </div>
            {defaultColumns?.length > 0 && (
              <div style={{ padding: 8, borderBottom: '1px solid #1f2937' }}>
                <button
                  className="ghost-button"
                  style={{ width: '100%', fontSize: 12 }}
                  onClick={handleSetDefaults}
                >
                  Set to default columns
                </button>
              </div>
            )}
            <div style={listStyle}>
              {visibleCols.map((col) => {
                const selected = reportSelectedId === col.id;
                return (
                  <div
                    key={col.id}
                    style={itemStyle(selected)}
                    onClick={() => setReportSelectedId(col.id)}
                    onDoubleClick={() =>
                      !col.alwaysVisible && hideColumn(col.id)
                    }
                    title={
                      col.alwaysVisible
                        ? 'Required column · cannot be removed'
                        : ''
                    }
                  >
                    <span style={{ flex: 1 }}>{col.label}</span>
                    {col.alwaysVisible && (
                      <span
                        style={{
                          fontSize: 10,
                          color: '#64748b',
                          textTransform: 'uppercase',
                          letterSpacing: '0.05em',
                        }}
                      >
                        req
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* RIGHT REORDER ARROWS */}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              gap: 8,
            }}
          >
            <button
              style={arrowBtnStyle(!canMoveUp)}
              disabled={!canMoveUp}
              onClick={moveSelectedToTop}
              title="Move to top"
            >
              ⇈
            </button>
            <button
              style={arrowBtnStyle(!canMoveUp)}
              disabled={!canMoveUp}
              onClick={moveSelectedUp}
              title="Move up"
            >
              ↑
            </button>
            <button
              style={arrowBtnStyle(!canMoveDown)}
              disabled={!canMoveDown}
              onClick={moveSelectedDown}
              title="Move down"
            >
              ↓
            </button>
            <button
              style={arrowBtnStyle(!canMoveDown)}
              disabled={!canMoveDown}
              onClick={moveSelectedToBottom}
              title="Move to bottom"
            >
              ⇊
            </button>
          </div>
        </div>

        <div className="modal-actions">
          <button className="ghost-button" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
