const FUNNEL_GROUPS = [
  {
    id: "levels",
    title: "Levels",
    unitLabel: "levels",
    stageKeys: ["gex_level_qualifies"],
  },
  {
    id: "zones",
    title: "Zones",
    unitLabel: "zones",
    stageKeys: ["zone_built"],
  },
  {
    id: "episodes",
    title: "Episodes",
    unitLabel: "pivots",
    stageKeys: ["zone_episode_valid", "pivot_after_open"],
  },
  {
    id: "target_attempts",
    title: "Target attempts",
    unitLabel: "target attempts",
    stageKeys: ["clean_space_sufficient", "target_hit"],
  },
  {
    id: "confirmed_ranges",
    title: "Confirmed ranges",
    unitLabel: "confirmed ranges",
    stageKeys: ["consolidation_not_invalidated", "consolidation_window_complete"],
  },
  {
    id: "setups",
    title: "Setups",
    unitLabel: "setups",
    stageKeys: ["prior_context_valid"],
  },
  {
    id: "signals_and_entries",
    title: "Signals + entries",
    unitLabel: "entries",
    stageKeys: ["skew_signal_fired", "trade_window_open", "entry_band_hit"],
  },
];

function fmt(value, digits = 2) {
  if (value === null || value === undefined || value === '') return '—';
  const num = Number(value);
  if (Number.isNaN(num)) return String(value);
  return num.toFixed(digits);
}

function StatCell({ label, value }) {
  return (
    <div className="diag-stat-card">
      <div className="diag-stat-label">{label}</div>
      <div className="diag-stat-value">{value}</div>
    </div>
  );
}

function sortRowsForEquity(rows) {
  return [...(rows || [])].sort((a, b) => {
    const aKey = `${a.trade_date || ''} ${a.trade_entry_ts_pt || a.short_signal_ts_pt || a.target_ts_pt || a.start_ts_pt || ''}`;
    const bKey = `${b.trade_date || ''} ${b.trade_entry_ts_pt || b.short_signal_ts_pt || b.target_ts_pt || b.start_ts_pt || ''}`;
    return aKey.localeCompare(bKey);
  });
}

function computePerformance(rows) {
  const executedTradeRows = sortRowsForEquity(rows).filter(
    (row) => row.trade_entry_found && row.trade_realized_points !== null && row.trade_realized_points !== undefined
  );

  const ordered = executedTradeRows
    .map((row) => Number(row.trade_realized_points))
    .filter((v) => Number.isFinite(v));

  if (!ordered.length) {
    return {
      totalPnLPts: null,
      evPtsPerTrade: null,
      maxDrawdownPts: null,
    };
  }

  const totalPnLPts = ordered.reduce((sum, v) => sum + v, 0);
  const evPtsPerTrade = totalPnLPts / ordered.length;

  let equity = 0;
  let peak = 0;
  let maxDrawdown = 0;

  for (const pnl of ordered) {
    equity += pnl;
    peak = Math.max(peak, equity);
    maxDrawdown = Math.max(maxDrawdown, peak - equity);
  }

  return {
    totalPnLPts,
    evPtsPerTrade,
    maxDrawdownPts: maxDrawdown,
  };
}

function FunnelStage({ stage, index, expectedCandidatesIn }) {
  const { label, kind, bypassed, candidates_in, kept, dropped, drop_reasons } = stage;
  const pct = candidates_in > 0 ? (kept / candidates_in) * 100 : 0;
  
  const kindColor = kind === 'construction' ? '#3b82f6' : kind === 'filter' ? '#10b981' : '#64748b';
  const barColor = bypassed ? '#475569' : kindColor;
  
  const hasMismatch = expectedCandidatesIn !== null && expectedCandidatesIn !== undefined && expectedCandidatesIn !== candidates_in;
  
  return (
    <div style={{
      background: '#0f172a',
      border: '1px solid #1e293b',
      borderRadius: '10px',
      padding: '12px 14px',
      marginBottom: '8px',
      marginLeft: '12px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ 
            fontSize: '11px', 
            fontWeight: '700', 
            color: '#64748b',
            minWidth: '20px',
          }}>
            {index + 1}.
          </span>
          <span style={{ 
            fontSize: '13px', 
            fontWeight: '600', 
            color: bypassed ? '#64748b' : '#e2e8f0',
            opacity: bypassed ? 0.6 : 1,
          }}>
            {label}
          </span>
          {bypassed && (
            <span style={{
              fontSize: '9px',
              fontWeight: '700',
              color: '#64748b',
              background: '#1e293b',
              padding: '2px 6px',
              borderRadius: '4px',
              letterSpacing: '0.05em',
            }}>
              BYPASSED
            </span>
          )}
          {hasMismatch && (
            <span 
              style={{
                fontSize: '11px',
                color: '#fb923c',
                marginLeft: '4px',
              }}
              title={`Expected ${expectedCandidatesIn} from previous stage`}
            >
              ⚠
            </span>
          )}
        </div>
        <div style={{ fontSize: '12px', color: '#94a3b8', fontWeight: '500' }}>
          {kept} / {candidates_in}
          {dropped > 0 && !bypassed && (
            <span style={{ color: '#f87171', marginLeft: '8px' }}>
              (−{dropped})
            </span>
          )}
        </div>
      </div>
      
      <div style={{
        height: '6px',
        background: '#1e293b',
        borderRadius: '3px',
        overflow: 'hidden',
      }}>
        <div style={{
          height: '100%',
          width: `${pct}%`,
          background: barColor,
          transition: 'width 0.3s ease',
        }} />
      </div>
      
      {!bypassed && dropped > 0 && drop_reasons && Object.keys(drop_reasons).length > 0 && (
        <div style={{ marginTop: '8px', fontSize: '11px', color: '#64748b' }}>
          {Object.entries(drop_reasons).map(([reason, count]) => (
            <div key={reason} style={{ marginTop: '2px' }}>
              • {reason.replace(/_/g, ' ')}: {count}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function groupFunnel(funnel) {
  const grouped = [];
  const ungrouped = [];
  const stageKeyToIndex = new Map();
  
  funnel.forEach((stage, idx) => {
    stageKeyToIndex.set(stage.key, idx);
  });
  
  FUNNEL_GROUPS.forEach((groupDef) => {
    const stages = [];
    groupDef.stageKeys.forEach((key) => {
      const idx = stageKeyToIndex.get(key);
      if (idx !== undefined) {
        stages.push({ stage: funnel[idx], globalIndex: idx });
      }
    });
    
    if (stages.length > 0) {
      grouped.push({ groupDef, stages });
    }
  });
  
  const allGroupedKeys = new Set(FUNNEL_GROUPS.flatMap(g => g.stageKeys));
  funnel.forEach((stage, idx) => {
    if (!allGroupedKeys.has(stage.key)) {
      ungrouped.push({ stage, globalIndex: idx });
    }
  });
  
  return { grouped, ungrouped };
}

function GroupTransition({ fromStage, toStage, fromUnit, toUnit }) {
  // Defensive checks for undefined/null values
  if (!fromStage || !toStage) {
    return null;
  }
  
  const fromCount = fromStage.kept ?? 0;
  const toCount = toStage.candidates_in ?? 0;
  
  let text;
  if (fromCount === toCount) {
    text = `↓  ${fromCount.toLocaleString()} ${fromUnit}`;
  } else {
    text = `↓  ${fromCount.toLocaleString()} ${fromUnit} produced ${toCount.toLocaleString()} ${toUnit}`;
  }
  
  return (
    <div style={{
      fontSize: '11px',
      color: '#64748b',
      padding: '12px 0',
      textAlign: 'left',
    }}>
      {text}
    </div>
  );
}

export default function DiagnosticsPanel({ diagnostics, rows = [], funnel = [] }) {
  if (!diagnostics) return null;

  const perf = computePerformance(rows);

  return (
    <div className="diag-card">
      <div className="results-header">
        <div>
          <h2>Diagnostics</h2>
          <p>
            This section emphasizes strategy performance first, with a smaller set of useful trade counts.
          </p>
        </div>
      </div>

      {funnel && funnel.length > 0 && (
        <div style={{ marginBottom: '20px' }}>
          <h3 style={{ 
            fontSize: '13px', 
            fontWeight: '700', 
            color: '#93c5fd', 
            marginBottom: '12px',
            letterSpacing: '0.05em',
            textTransform: 'uppercase',
          }}>
            Pipeline Funnel
          </h3>
          {(() => {
            try {
              const { grouped, ungrouped } = groupFunnel(funnel);
            
            return (
              <>
                {grouped.map((group, groupIdx) => {
                  const { groupDef, stages } = group;
                  const prevGroup = groupIdx > 0 ? grouped[groupIdx - 1] : null;
                  
                  return (
                    <div key={groupDef.id}>
                      {prevGroup && (
                        <GroupTransition
                          fromStage={prevGroup.stages[prevGroup.stages.length - 1].stage}
                          toStage={stages[0].stage}
                          fromUnit={prevGroup.groupDef.unitLabel}
                          toUnit={groupDef.unitLabel}
                        />
                      )}
                      
                      <div style={{
                        fontSize: '14px',
                        fontWeight: '700',
                        color: '#93c5fd',
                        marginBottom: '8px',
                        letterSpacing: '0.03em',
                      }}>
                        {groupDef.title}
                      </div>
                      
                      {stages.map((stageInfo, stageIdx) => {
                        const prevStageInfo = stageIdx > 0 ? stages[stageIdx - 1] : null;
                        const expectedCandidatesIn = prevStageInfo ? prevStageInfo.stage.kept : null;
                        
                        return (
                          <FunnelStage
                            key={stageInfo.stage.key}
                            stage={stageInfo.stage}
                            index={stageInfo.globalIndex}
                            expectedCandidatesIn={expectedCandidatesIn}
                          />
                        );
                      })}
                    </div>
                  );
                })}
                
                {ungrouped.length > 0 && (
                  <div>
                    <div style={{
                      fontSize: '14px',
                      fontWeight: '700',
                      color: '#93c5fd',
                      marginTop: '16px',
                      marginBottom: '8px',
                      letterSpacing: '0.03em',
                    }}>
                      Ungrouped
                    </div>
                    {ungrouped.map((stageInfo) => (
                      <FunnelStage
                        key={stageInfo.stage.key}
                        stage={stageInfo.stage}
                        index={stageInfo.globalIndex}
                        expectedCandidatesIn={null}
                      />
                    ))}
                  </div>
                )}
              </>
            );
            } catch (err) {
              console.error('Error rendering funnel:', err);
              return (
                <div style={{ color: '#f87171', fontSize: '12px', padding: '10px' }}>
                  Error rendering funnel. Check console for details.
                </div>
              );
            }
          })()}
        </div>
      )}

      <div className="diag-stat-grid diag-stat-grid-compact">
        <StatCell label="Days total" value={diagnostics.days_total ?? '—'} />
        <StatCell label="Valid instances" value={diagnostics.valid_instances ?? '—'} />
        <StatCell label="Up short setups" value={diagnostics.up_short_setups_found ?? '—'} />
        <StatCell label="Actual trades" value={diagnostics.actual_trades_found ?? '—'} />
        <StatCell label="Winning trades" value={diagnostics.winning_trades ?? '—'} />
        <StatCell label="Total P/L (pts)" value={fmt(perf.totalPnLPts)} />
        <StatCell label="EV (pts/trade)" value={fmt(perf.evPtsPerTrade)} />
        <StatCell label="Max drawdown (pts)" value={fmt(perf.maxDrawdownPts)} />
      </div>
    </div>
  );
}
