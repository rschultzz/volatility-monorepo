import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import PriceChart from './components/PriceChart'
import AggressorFlowPanel from './components/AggressorFlowPanel'

const FLOW_HEIGHT_STORAGE_KEY = 'ib-react-flow-panel-height'
const FLOW_MIN_HEIGHT = 140
const FLOW_MAX_HEIGHT = 520

function cleanBaseUrl(value) {
  const s = String(value || '').trim()
  if (!s) return ''
  return s.replace(/\/+$/, '')
}

function inferApiBase() {
  const params = new URLSearchParams(window.location.search)

  const explicit = cleanBaseUrl(params.get('api_base'))
  if (explicit) return explicit

  const envBase = cleanBaseUrl(
    import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_BASE
  )
  if (envBase) return envBase

  try {
    if (document.referrer) {
      return cleanBaseUrl(new URL(document.referrer).origin)
    }
  } catch (err) {
    // ignore
  }

  try {
    if (window.location?.origin) {
      return cleanBaseUrl(window.location.origin)
    }
  } catch (err) {
    // ignore
  }

  return 'http://127.0.0.1:8060'
}

function parseSelectedTimes(params) {
  const raw = params.get('selected_times') || ''
  return raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
}

function parseBool(value, fallback = true) {
  if (value == null || value === '') return fallback
  return !['0', 'false', 'no', 'off'].includes(String(value).trim().toLowerCase())
}

function parseFloatOrNull(value) {
  if (value == null || value === '') return null
  const num = Number(value)
  return Number.isFinite(num) ? num : null
}

function parseIntOrDefault(value, fallback) {
  const num = Number.parseInt(String(value ?? ''), 10)
  return Number.isFinite(num) ? num : fallback
}

function dedupeAndSortBars(values) {
  const byTime = new Map()
  for (const bar of Array.isArray(values) ? values : []) {
    const t = Number(bar?.time)
    if (!Number.isFinite(t)) continue
    byTime.set(t, bar)
  }
  return Array.from(byTime.values()).sort((a, b) => a.time - b.time)
}

function dedupeSegments(values) {
  const byKey = new Map()
  for (const seg of Array.isArray(values) ? values : []) {
    const key = [
      seg?.session_date,
      seg?.start_time,
      seg?.end_time,
      seg?.level,
    ].join('|')
    byKey.set(key, seg)
  }
  return Array.from(byKey.values())
}

function dedupeAndSortFlowPoints(values) {
  const byTime = new Map()
  for (const point of Array.isArray(values) ? values : []) {
    const t = Number(point?.time)
    const value = Number(point?.value)
    if (!Number.isFinite(t) || !Number.isFinite(value)) continue
    byTime.set(t, { ...point, time: t, value })
  }
  return Array.from(byTime.values()).sort((a, b) => a.time - b.time)
}

function normalizeDateList(values) {
  const seen = new Set()
  const out = []
  for (const item of Array.isArray(values) ? values : []) {
    const s = String(item || '').trim()
    if (!/^\d{4}-\d{2}-\d{2}$/.test(s)) continue
    if (seen.has(s)) continue
    seen.add(s)
    out.push(s)
  }
  return out.sort()
}

function clampFlowHeight(value) {
  const n = Number(value)
  if (!Number.isFinite(n)) return 220
  return Math.max(FLOW_MIN_HEIGHT, Math.min(FLOW_MAX_HEIGHT, Math.round(n)))
}

function normalizeLogicalRange(range) {
  if (!range) return null
  const from = Number(range.from)
  const to = Number(range.to)
  if (!Number.isFinite(from) || !Number.isFinite(to)) return null
  return { from, to }
}

function rangesClose(a, b, eps = 1) {
  if (!a || !b) return false
  return Math.abs(Number(a.from) - Number(b.from)) <= eps && Math.abs(Number(a.to) - Number(b.to)) <= eps
}

export default function App() {
  const params = useMemo(() => new URLSearchParams(window.location.search), [])
  const tradeDate = params.get('trade_date') || new Date().toISOString().slice(0, 10)
  const interval = params.get('interval') || '1min'
  const gexEnabled = parseBool(params.get('gex_enabled'), true)
  const gexMinAbsB = parseFloatOrNull(params.get('gex_min_abs_b'))
  const daysEitherSide = Math.max(0, parseIntOrDefault(params.get('days_either_side'), 5))

  const flowEnabled = parseBool(params.get('flow_enabled'), true)
  const flowSession = (params.get('flow_session') || 'FULL').toUpperCase()
  const defaultFlowResample = '1m'
  const flowResample = (params.get('flow_resample') || defaultFlowResample).toLowerCase()
  const defaultFlowEmaLen = flowResample === '1m' || flowResample === '60s' ? 14 : 840
  const flowEmaLen = Math.max(1, parseIntOrDefault(params.get('flow_ema_len'), defaultFlowEmaLen))
  const flowHistAlpha = parseFloatOrNull(params.get('flow_hist_alpha')) ?? 0.30

  const effectiveDaysEitherSide = daysEitherSide

  const apiBase = useMemo(() => inferApiBase(), [])
  const initialSelectedTimes = useMemo(() => parseSelectedTimes(params), [params])

  const dragStateRef = useRef(null)

  const [centerBars, setCenterBars] = useState([])
  const [extraBars, setExtraBars] = useState([])
  const [centerGexSegments, setCenterGexSegments] = useState([])
  const [extraGexSegments, setExtraGexSegments] = useState([])
  const [centerLoadedDates, setCenterLoadedDates] = useState([])
  const [extraLoadedDates, setExtraLoadedDates] = useState([])
  const [meta, setMeta] = useState(null)

  const [flowPoints, setFlowPoints] = useState([])
  const [flowLoading, setFlowLoading] = useState(false)
  const [flowError, setFlowError] = useState('')

  const [loadingCenter, setLoadingCenter] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState('')

  const [sharedLogicalRange, setSharedLogicalRange] = useState(null)

  const [flowPanelHeight, setFlowPanelHeight] = useState(() => {
    try {
      return clampFlowHeight(window.localStorage.getItem(FLOW_HEIGHT_STORAGE_KEY) || 220)
    } catch (err) {
      return 220
    }
  })

  const mergedBars = useMemo(
    () => dedupeAndSortBars([...centerBars, ...extraBars]),
    [centerBars, extraBars]
  )

  const mergedGexSegments = useMemo(
    () => dedupeSegments([...centerGexSegments, ...extraGexSegments]),
    [centerGexSegments, extraGexSegments]
  )

  const loadedFlowDates = useMemo(
    () => normalizeDateList([...centerLoadedDates, ...extraLoadedDates]),
    [centerLoadedDates, extraLoadedDates]
  )

  useEffect(() => {
    try {
      window.localStorage.setItem(FLOW_HEIGHT_STORAGE_KEY, String(flowPanelHeight))
    } catch (err) {
      // ignore
    }
  }, [flowPanelHeight])

  useEffect(() => {
    const controller = new AbortController()
    const url = new URL(`${apiBase}/api/ironbeam/bars`)
    url.searchParams.set('trade_date', tradeDate)
    url.searchParams.set('interval', interval)
    url.searchParams.set('gex_enabled', gexEnabled ? '1' : '0')
    url.searchParams.set('phase', 'center')
    if (gexMinAbsB != null) {
      url.searchParams.set('gex_min_abs_b', String(gexMinAbsB))
    }

    async function loadCenter() {
      try {
        setLoadingCenter(true)
        setLoadingMore(false)
        setError('')
        setCenterBars([])
        setExtraBars([])
        setCenterGexSegments([])
        setExtraGexSegments([])
        setCenterLoadedDates([])
        setExtraLoadedDates([])
        setMeta(null)
        setSharedLogicalRange(null)

        const response = await fetch(url.toString(), {
          method: 'GET',
          credentials: 'include',
          signal: controller.signal,
          cache: 'no-store',
        })

        if (response.status === 401) {
          throw new Error(
            `Unauthorized from Dash backend at ${apiBase}. Open the Dash app first and make sure you are logged in there.`
          )
        }

        if (!response.ok) {
          throw new Error(
            `Backend returned ${response.status} ${response.statusText || ''}`.trim()
          )
        }

        const payload = await response.json()
        setCenterBars(Array.isArray(payload.bars) ? payload.bars : [])
        setCenterGexSegments(Array.isArray(payload.gex_segments) ? payload.gex_segments : [])
        setCenterLoadedDates(
          normalizeDateList(
            Array.isArray(payload.loaded_dates) && payload.loaded_dates.length
              ? payload.loaded_dates
              : [payload?.effective_trade_date || payload?.trade_date || tradeDate]
          )
        )
        setMeta(payload)
      } catch (err) {
        if (err?.name === 'AbortError') return
        setCenterBars([])
        setExtraBars([])
        setCenterGexSegments([])
        setExtraGexSegments([])
        setCenterLoadedDates([])
        setExtraLoadedDates([])
        setMeta(null)
        setError(err?.message || `Could not load bars from ${url.toString()}`)
      } finally {
        setLoadingCenter(false)
      }
    }

    loadCenter()
    return () => controller.abort()
  }, [apiBase, tradeDate, interval, gexEnabled, gexMinAbsB])

  useEffect(() => {
    if (loadingCenter || error || effectiveDaysEitherSide <= 0) return

    const controller = new AbortController()
    const url = new URL(`${apiBase}/api/ironbeam/bars`)
    url.searchParams.set('trade_date', tradeDate)
    url.searchParams.set('interval', interval)
    url.searchParams.set('gex_enabled', gexEnabled ? '1' : '0')
    url.searchParams.set('phase', 'multi')
    url.searchParams.set('days_either_side', String(effectiveDaysEitherSide))
    if (gexMinAbsB != null) {
      url.searchParams.set('gex_min_abs_b', String(gexMinAbsB))
    }

    async function loadMore() {
      try {
        setLoadingMore(true)

        const response = await fetch(url.toString(), {
          method: 'GET',
          credentials: 'include',
          signal: controller.signal,
          cache: 'no-store',
        })

        if (response.status === 401) {
          throw new Error(
            `Unauthorized from Dash backend at ${apiBase}. Open the Dash app first and make sure you are logged in there.`
          )
        }

        if (!response.ok) {
          throw new Error(
            `Backend returned ${response.status} ${response.statusText || ''}`.trim()
          )
        }

        const payload = await response.json()
        setExtraBars(Array.isArray(payload.bars) ? payload.bars : [])
        setExtraGexSegments(Array.isArray(payload.gex_segments) ? payload.gex_segments : [])
        setExtraLoadedDates(normalizeDateList(payload.loaded_dates))
      } catch (err) {
        if (err?.name === 'AbortError') return
        console.error('React preview multi-day load failed', err)
      } finally {
        setLoadingMore(false)
      }
    }

    loadMore()
    return () => controller.abort()
  }, [apiBase, tradeDate, interval, effectiveDaysEitherSide, loadingCenter, error, gexEnabled, gexMinAbsB])

  useEffect(() => {
    if (!flowEnabled || loadingCenter || error) {
      setFlowPoints([])
      setFlowError('')
      setFlowLoading(false)
      return
    }

    const controller = new AbortController()
    const datesToLoad = loadedFlowDates.length ? loadedFlowDates : [tradeDate]

    async function loadFlow() {
      try {
        setFlowLoading(true)
        setFlowError('')
        setFlowPoints([])

        const settled = await Promise.allSettled(
          datesToLoad.map(async (dateStr) => {
            const url = new URL(`${apiBase}/api/ironbeam/flow`)
            url.searchParams.set('trade_date', dateStr)
            url.searchParams.set('session', flowSession)
            url.searchParams.set('resample', flowResample)
            url.searchParams.set('ema_len', String(flowEmaLen))

            const response = await fetch(url.toString(), {
              method: 'GET',
              credentials: 'include',
              signal: controller.signal,
              cache: 'no-store',
            })

            if (response.status === 401) {
              throw new Error(
                `Unauthorized from Dash backend at ${apiBase}. Open the Dash app first and make sure you are logged in there.`
              )
            }

            if (!response.ok) {
              throw new Error(
                `Flow request failed for ${dateStr}: ${response.status} ${response.statusText || ''}`.trim()
              )
            }

            const payload = await response.json()
            return Array.isArray(payload.flow_points) ? payload.flow_points : []
          })
        )

        const merged = []
        const failures = []

        for (const result of settled) {
          if (result.status === 'fulfilled') {
            merged.push(...result.value)
          } else if (result.reason?.name !== 'AbortError') {
            failures.push(result.reason?.message || 'Unknown flow load error')
          }
        }

        if (!merged.length && failures.length) {
          throw new Error(failures[0])
        }

        setFlowPoints(dedupeAndSortFlowPoints(merged))
        setFlowError('')
      } catch (err) {
        if (err?.name === 'AbortError') return
        setFlowPoints([])
        setFlowError(err?.message || 'Could not load multi-day flow')
      } finally {
        setFlowLoading(false)
      }
    }

    loadFlow()
    return () => controller.abort()
  }, [apiBase, tradeDate, flowEnabled, flowSession, flowResample, flowEmaLen, loadingCenter, error, loadedFlowDates])

  useEffect(() => {
    const handleMove = (event) => {
      const state = dragStateRef.current
      if (!state) return
      const delta = state.startY - event.clientY
      setFlowPanelHeight(clampFlowHeight(state.startHeight + delta))
      document.body.classList.add('resizing-panels')
    }

    const handleUp = () => {
      dragStateRef.current = null
      document.body.classList.remove('resizing-panels')
    }

    window.addEventListener('mousemove', handleMove)
    window.addEventListener('mouseup', handleUp)

    return () => {
      window.removeEventListener('mousemove', handleMove)
      window.removeEventListener('mouseup', handleUp)
      document.body.classList.remove('resizing-panels')
    }
  }, [])

  function beginResize(event) {
    dragStateRef.current = {
      startY: event.clientY,
      startHeight: flowPanelHeight,
    }
    document.body.classList.add('resizing-panels')
    event.preventDefault()
  }

  const handleSharedLogicalRangeChange = useCallback((nextRange) => {
    const next = normalizeLogicalRange(nextRange)
    if (!next) return
    setSharedLogicalRange((prev) => (rangesClose(prev, next) ? prev : next))
  }, [])

  return (
    <div className="app-shell compact-shell">
      <div className="card compact-card">
        {error ? (
          <div className="status-card status-card-error">
            <div className="status-title">React preview could not load data</div>
            <div className="status-text">{error}</div>
            <div className="status-text" style={{ marginTop: '6px', opacity: 0.8 }}>
              Requested from {apiBase}/api/ironbeam/bars
            </div>
          </div>
        ) : loadingCenter ? (
          <div className="status-card">
            <div className="status-title">Loading center session…</div>
            <div className="status-text">Fetching from {apiBase}/api/ironbeam/bars</div>
          </div>
        ) : mergedBars.length === 0 ? (
          <div className="status-card">
            <div className="status-title">No bars returned</div>
            <div className="status-text">
              The backend responded, but this session did not return any ES bars.
            </div>
            <div className="status-text" style={{ marginTop: '6px', opacity: 0.8 }}>
              Source: {apiBase}/api/ironbeam/bars
            </div>
          </div>
        ) : (
          <div className="react-preview-layout">
            <div className="react-top-pane">
              {loadingMore && (
                <div className="top-pane-status">Loading surrounding sessions…</div>
              )}
              <PriceChart
                candles={mergedBars}
                interval={interval}
                initialSelectedTimes={initialSelectedTimes}
                gexSegments={mergedGexSegments}
                gexEnabled={Boolean(meta?.gex_enabled ?? gexEnabled)}
                onVisibleLogicalRangeChange={handleSharedLogicalRangeChange}
              />
            </div>

            {flowEnabled && (
              <>
                <div
                  className="panel-splitter"
                  onMouseDown={beginResize}
                  title="Drag to resize panels"
                />
                <div className="react-bottom-pane" style={{ height: `${flowPanelHeight}px` }}>
                  <AggressorFlowPanel
                    dataPoints={flowPoints}
                    candles={mergedBars}
                    visibleLogicalRange={sharedLogicalRange}
                    height={flowPanelHeight}
                    loading={flowLoading}
                    error={flowError}
                    histAlpha={flowHistAlpha}
                  />
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
