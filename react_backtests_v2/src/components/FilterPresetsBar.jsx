import React, { useState, useEffect } from 'react';
import { isFilterActive } from './columnFilters';

// ─────────────────────────────────────────────────────────────────────
// FilterPresetsBar
//   Horizontal strip of saved filter presets for the current scan.
//   Each preset captures: column filters + (optionally) the long/short
//   forward-outcomes view. Click a preset to load it; that preset
//   becomes "active" and the bar shows an Update button alongside
//   Save-as-new — mirroring Save vs. Save As semantics.
//
// Props:
//   presets:               [{ id, name, notes, filters, view_direction, created_at, updated_at }]
//   activeFilters:         current { [columnId]: filterObject } map
//   activeView:            current 'long' | 'short' (forward-outcomes view)
//   originalTrade:         'long' | 'short' (default for the scan)
//   activePresetId:        id of the preset most recently applied/created (or null)
//   onApplyPreset:         (preset) => void   — apply preset's filters + view
//   onCreatePreset:        ({ name, notes, filters, view_direction }) => void
//   onUpdatePreset:        (id, { name, notes }) => void   — name/notes only
//   onUpdatePresetFilters: (id) => void   — replace preset's filters with current
//   onDeletePreset:        (id) => void
// ─────────────────────────────────────────────────────────────────────

export default function FilterPresetsBar({
  presets = [],
  activeFilters = {},
  activeView,
  originalTrade,
  activePresetId,
  onApplyPreset,
  onCreatePreset,
  onUpdatePreset,
  onUpdatePresetFilters,
  onDeletePreset,
}) {
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [editingPreset, setEditingPreset] = useState(null);

  const activeFilterCount = Object.values(activeFilters).filter(isFilterActive).length;
  const canSaveCurrent = activeFilterCount > 0;

  const activePreset = activePresetId
    ? presets.find(p => p.id === activePresetId)
    : null;

  return (
    <>
      <div style={{
        background: '#0b1220',
        border: '1px solid #1f2937',
        borderRadius: 14,
        padding: '8px 12px',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        flexWrap: 'wrap',
      }}>
        <span style={{
          fontSize: 11,
          color: '#94a3b8',
          fontWeight: 700,
          textTransform: 'uppercase',
          letterSpacing: '0.04em',
          marginRight: 4,
        }}>
          Saved Presets
        </span>

        {presets.length === 0 ? (
          <span style={{ color: '#64748b', fontSize: 12, flex: 1 }}>
            No presets yet — apply some filters and click &quot;Save current&quot; to remember them.
          </span>
        ) : (
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', flex: 1 }}>
            {presets.map((p) => (
              <PresetChip
                key={p.id}
                preset={p}
                isActive={p.id === activePresetId}
                originalTrade={originalTrade}
                onApply={() => onApplyPreset(p)}
                onEdit={() => setEditingPreset(p)}
                onDelete={() => {
                  if (window.confirm(`Delete preset "${p.name}"?`)) {
                    onDeletePreset(p.id);
                  }
                }}
              />
            ))}
          </div>
        )}

        {/* Save / Save-As cluster on the right. When a preset is
            active and current filters can be saved, both buttons
            appear; otherwise just the create button. */}
        {activePreset && canSaveCurrent && (
          <button
            type="button"
            onClick={() => {
              if (window.confirm(`Update "${activePreset.name}" with the current filters and view?`)) {
                onUpdatePresetFilters(activePreset.id);
              }
            }}
            title={`Replace "${activePreset.name}" with the current filter set and view direction`}
            style={{
              background: 'rgba(37, 99, 235, 0.18)',
              border: '1px solid #2563eb',
              color: '#bfdbfe',
              padding: '4px 10px',
              borderRadius: 6,
              cursor: 'pointer',
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: '0.04em',
              textTransform: 'uppercase',
            }}
          >
            ↻ Update preset
          </button>
        )}

        <button
          type="button"
          onClick={() => setSaveDialogOpen(true)}
          disabled={!canSaveCurrent}
          title={canSaveCurrent
            ? (activePreset
                ? 'Save the current filter set as a NEW preset (instead of overwriting the active one)'
                : 'Save the current filter set (and direction view) as a named preset')
            : 'Apply at least one filter before saving as a preset'}
          style={{
            background: canSaveCurrent ? 'rgba(34, 197, 94, 0.15)' : 'transparent',
            border: `1px solid ${canSaveCurrent ? '#22c55e' : '#334155'}`,
            color: canSaveCurrent ? '#86efac' : '#64748b',
            padding: '4px 10px',
            borderRadius: 6,
            cursor: canSaveCurrent ? 'pointer' : 'not-allowed',
            fontSize: 11,
            fontWeight: 700,
            letterSpacing: '0.04em',
            textTransform: 'uppercase',
          }}
        >
          {activePreset ? '+ Save as new' : '+ Save current'}
        </button>
      </div>

      {saveDialogOpen && (
        <PresetDialog
          mode="create"
          initialName=""
          initialNotes=""
          onCancel={() => setSaveDialogOpen(false)}
          onSubmit={({ name, notes }) => {
            onCreatePreset({
              name,
              notes,
              filters: activeFilters,
              view_direction: activeView,
            });
            setSaveDialogOpen(false);
          }}
        />
      )}

      {editingPreset && (
        <PresetDialog
          mode="edit"
          initialName={editingPreset.name}
          initialNotes={editingPreset.notes || ''}
          onCancel={() => setEditingPreset(null)}
          onSubmit={({ name, notes }) => {
            onUpdatePreset(editingPreset.id, { name, notes });
            setEditingPreset(null);
          }}
        />
      )}
    </>
  );
}

function PresetChip({ preset, isActive, originalTrade, onApply, onEdit, onDelete }) {
  const [hovered, setHovered] = useState(false);
  const filterCount = Object.values(preset.filters || {}).filter(isFilterActive).length;
  const dir = preset.view_direction;
  // Subtle annotation if this preset will flip the direction from the scan default.
  const flipsDirection = dir && originalTrade && dir !== originalTrade;

  // Active preset gets a brighter border + background so the user can
  // see which one is "loaded" — and so the Update button has obvious context.
  const borderColor = isActive ? '#60a5fa' : (hovered ? '#2563eb' : 'rgba(37, 99, 235, 0.40)');
  const bgColor = isActive
    ? 'rgba(37, 99, 235, 0.32)'
    : (hovered ? 'rgba(37, 99, 235, 0.20)' : 'rgba(37, 99, 235, 0.10)');

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        background: bgColor,
        border: `1px solid ${borderColor}`,
        borderRadius: 999,
        padding: '2px 4px 2px 10px',
        fontSize: 12,
        color: '#bfdbfe',
        transition: 'background 0.1s, border-color 0.1s',
        boxShadow: isActive ? '0 0 0 1px rgba(96, 165, 250, 0.25)' : 'none',
      }}
      title={[
        preset.name,
        isActive ? ' (active)' : '',
        preset.notes ? `\n${preset.notes}` : '',
        `\n${filterCount} filter${filterCount === 1 ? '' : 's'}`,
        flipsDirection ? `\nFlips view to ${dir.toUpperCase()}` : '',
      ].join('')}
    >
      <button
        type="button"
        onClick={onApply}
        style={{
          background: 'none',
          border: 'none',
          color: 'inherit',
          padding: 0,
          cursor: 'pointer',
          fontSize: 12,
          fontWeight: isActive ? 700 : 600,
          fontFamily: 'inherit',
        }}
      >
        {isActive && <span style={{ marginRight: 4, opacity: 0.9 }}>●</span>}
        {preset.name}
        <span style={{
          marginLeft: 6,
          fontSize: 10,
          color: '#7dd3fc',
          fontWeight: 500,
          opacity: 0.7,
        }}>
          {filterCount}
        </span>
        {flipsDirection && (
          <span style={{
            marginLeft: 4,
            fontSize: 9,
            color: '#fcd34d',
            fontWeight: 700,
            letterSpacing: '0.05em',
          }}>
            ↺{dir.toUpperCase()}
          </span>
        )}
      </button>

      <button
        type="button"
        onClick={onEdit}
        title="Edit name and notes"
        style={iconBtnStyle(hovered)}
      >
        ✎
      </button>
      <button
        type="button"
        onClick={onDelete}
        title="Delete preset"
        style={{ ...iconBtnStyle(hovered), color: '#fca5a5' }}
      >
        ×
      </button>
    </div>
  );
}

function iconBtnStyle(hovered) {
  return {
    background: 'none',
    border: 'none',
    color: '#bfdbfe',
    padding: '2px 4px',
    cursor: 'pointer',
    fontSize: 11,
    lineHeight: 1,
    opacity: hovered ? 0.9 : 0.5,
    fontFamily: 'inherit',
  };
}

function PresetDialog({ mode, initialName, initialNotes, onCancel, onSubmit }) {
  const [name, setName] = useState(initialName);
  const [notes, setNotes] = useState(initialNotes);

  // Keep state in sync if the props change (e.g. switching which preset is edited)
  useEffect(() => {
    setName(initialName);
    setNotes(initialNotes);
  }, [initialName, initialNotes]);

  const trimmed = name.trim();
  const canSubmit = trimmed.length > 0;
  const title = mode === 'edit' ? 'Edit Preset' : 'Save Filter Preset';
  const submitLabel = mode === 'edit' ? 'Save Changes' : 'Save Preset';

  const handleSubmit = (e) => {
    e?.preventDefault?.();
    if (!canSubmit) return;
    onSubmit({ name: trimmed, notes: notes.trim() });
  };

  return (
    <div className="modal-backdrop" onClick={onCancel}>
      <form
        className="modal-card"
        style={{ width: 480 }}
        onClick={(e) => e.stopPropagation()}
        onSubmit={handleSubmit}
      >
        <div className="modal-header">
          <div>
            <div className="eyebrow">Filter Presets</div>
            <h2>{title}</h2>
          </div>
          <button type="button" className="ghost-button" onClick={onCancel}>Close</button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 16 }}>
          <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <span style={{
              fontSize: 11,
              color: '#94a3b8',
              fontWeight: 700,
              letterSpacing: '0.04em',
              textTransform: 'uppercase',
            }}>
              Name
            </span>
            <input
              type="text"
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. High GEX targets only"
              style={{
                background: '#0f172a',
                border: '1px solid #334155',
                borderRadius: 6,
                color: '#e2e8f0',
                padding: '8px 10px',
                fontSize: 13,
                fontFamily: 'inherit',
              }}
            />
          </label>

          <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <span style={{
              fontSize: 11,
              color: '#94a3b8',
              fontWeight: 700,
              letterSpacing: '0.04em',
              textTransform: 'uppercase',
            }}>
              Notes (optional)
            </span>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="What makes this preset useful? When would you reach for it?"
              rows={4}
              style={{
                background: '#0f172a',
                border: '1px solid #334155',
                borderRadius: 6,
                color: '#e2e8f0',
                padding: '8px 10px',
                fontSize: 13,
                fontFamily: 'inherit',
                resize: 'vertical',
              }}
            />
          </label>
        </div>

        <div className="modal-actions">
          <button type="button" className="ghost-button" onClick={onCancel}>Cancel</button>
          <button
            type="submit"
            className="primary-button"
            disabled={!canSubmit}
          >
            {submitLabel}
          </button>
        </div>
      </form>
    </div>
  );
}
