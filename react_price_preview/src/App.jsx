import { useEffect, useMemo, useState } from 'react'
import PriceChart from './components/PriceChart'

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
    // ignore and continue
  }

  try {
    if (window.location?.origin) {
      return cleanBaseUrl(window.location.origin)
    }
  } catch (err) {
    // ignore and continue
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

export default function App() {
  const params = useMemo(() => new URLSearchParams(window.location.search), [])
  const tradeDate = params.get('trade_date') || new Date().toISOString().slice(0, 10)
  const interval = params.get('interval') || '1min'
  const gexEnabled = parseBool(params.get('gex_enabled'), true)
  const gexMinAbsB = parseFloatOrNull(params.get('gex_min_abs_b'))
  const daysEitherSide = Math.max(0, parseIntOrDefault(params.get('days_either_side'), 5))
  const apiBase = useMemo(() => inferApiBase(), [])
  const initialSelectedTimes = useMemo(() => parseSelectedTimes(params), [params])

  const [centerBars, setCenterBars] = useState([])
  const [extraBars, setExtraBars] = useState([])
  const [centerGexSegments, setCenterGexSegments] = useState([])
  const [extraGexSegments, setExtraGexSegments] = useState([])
  const [meta, setMeta] = useState(null)

  const [loadingCenter, setLoadingCenter] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState('')

  const mergedBars = useMemo(
    () => dedupeAndSortBars([...centerBars, ...extraBars]),
    [centerBars, extraBars]
  )

  const mergedGexSegments = useMemo(
    () => dedupeSegments([...centerGexSegments, ...extraGexSegments]),
    [centerGexSegments, extraGexSegments]
  )

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
        setMeta(null)

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
        setMeta(payload)
      } catch (err) {
        if (err?.name === 'AbortError') return
        setCenterBars([])
        setExtraBars([])
        setCenterGexSegments([])
        setExtraGexSegments([])
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
    if (loadingCenter || error || daysEitherSide <= 0) return

    const controller = new AbortController()
    const url = new URL(`${apiBase}/api/ironbeam/bars`)
    url.searchParams.set('trade_date', tradeDate)
    url.searchParams.set('interval', interval)
    url.searchParams.set('gex_enabled', gexEnabled ? '1' : '0')
    url.searchParams.set('phase', 'multi')
    url.searchParams.set('days_either_side', String(daysEitherSide))
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
      } catch (err) {
        if (err?.name === 'AbortError') return
        console.error('React preview multi-day load failed', err)
      } finally {
        setLoadingMore(false)
      }
    }

    loadMore()
    return () => controller.abort()
  }, [apiBase, tradeDate, interval, daysEitherSide, loadingCenter, error, gexEnabled, gexMinAbsB])

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
          <>
            {loadingMore && (
              <div className="status-text" style={{ marginBottom: '8px', opacity: 0.8 }}>
                Loading surrounding sessions…
              </div>
            )}
            <PriceChart
              candles={mergedBars}
              interval={interval}
              initialSelectedTimes={initialSelectedTimes}
              gexSegments={mergedGexSegments}
              gexEnabled={Boolean(meta?.gex_enabled ?? gexEnabled)}
            />
          </>
        )}
      </div>
    </div>
  )
}