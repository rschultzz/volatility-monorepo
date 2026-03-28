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
    // ignore and fall through
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

export default function App() {
  const params = useMemo(() => new URLSearchParams(window.location.search), [])
  const tradeDate = params.get('trade_date') || new Date().toISOString().slice(0, 10)
  const interval = params.get('interval') || '1min'
  const gexEnabled = parseBool(params.get('gex_enabled'), true)
  const gexMinAbsB = parseFloatOrNull(params.get('gex_min_abs_b'))
  const apiBase = useMemo(() => inferApiBase(), [])
  const initialSelectedTimes = useMemo(() => parseSelectedTimes(params), [params])

  const [bars, setBars] = useState([])
  const [meta, setMeta] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    const url = new URL(`${apiBase}/api/ironbeam/bars`)
    url.searchParams.set('trade_date', tradeDate)
    url.searchParams.set('interval', interval)
    url.searchParams.set('gex_enabled', gexEnabled ? '1' : '0')
    if (gexMinAbsB != null) {
      url.searchParams.set('gex_min_abs_b', String(gexMinAbsB))
    }

    async function load() {
      try {
        setLoading(true)
        setError('')

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
        setBars(Array.isArray(payload.bars) ? payload.bars : [])
        setMeta(payload)
      } catch (err) {
        if (err?.name === 'AbortError') return
        setBars([])
        setMeta(null)
        setError(err?.message || `Could not load bars from ${url.toString()}`)
      } finally {
        setLoading(false)
      }
    }

    load()
    return () => controller.abort()
  }, [apiBase, tradeDate, interval, gexEnabled, gexMinAbsB])

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
        ) : loading ? (
          <div className="status-card">
            <div className="status-title">Loading bars…</div>
            <div className="status-text">Fetching from {apiBase}/api/ironbeam/bars</div>
          </div>
        ) : bars.length === 0 ? (
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
          <PriceChart
            candles={bars}
            interval={interval}
            initialSelectedTimes={initialSelectedTimes}
            gexLevels={Array.isArray(meta?.gex_levels) ? meta.gex_levels : []}
            gexEnabled={Boolean(meta?.gex_enabled ?? gexEnabled)}
          />
        )}
      </div>
    </div>
  )
}