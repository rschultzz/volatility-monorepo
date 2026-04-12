import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import PriceChart from './components/PriceChart'
import AggressorFlowPanel from './components/AggressorFlowPanel'

const FLOW_HEIGHT_STORAGE_KEY = 'ib-react-flow-panel-height'
const FLOW_EMA_MINUTES_STORAGE_KEY = 'ib-react-flow-ema-minutes'
const GEX_MIN_ABS_B_STORAGE_KEY = 'ib-react-gex-min-abs-b'
const FLOW_MIN_HEIGHT = 140
const FLOW_MAX_HEIGHT = 520
const FULL_BARS_LIVE_POLL_MS = 30000
const LIVE_OVERLAY_POLL_MS = 1000
const FLOW_LIVE_POLL_MS = 3000
const SKEW_DATA_POLL_MS = 10000
const SMILE_DATA_POLL_MS = 10000

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

function normalizeTimes(value) {
  if (!Array.isArray(value)) return []
  const seen = new Set()
  const out = []
  for (const item of value) {
    const s = String(item || '').trim()
    if (!s) continue
    if (!/^\d{2}:\d{2}$/.test(s)) continue
    if (seen.has(s)) continue
    seen.add(s)
    out.push(s)
  }
  return out.sort()
}

function parseSelectedTimes(params) {
  const raw = params.get('selected_times') || ''
  return normalizeTimes(raw.split(',').filter(Boolean))
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

function coerceGexMinAbsB(value, fallback = 10) {
  const num = Number(value)
  if (!Number.isFinite(num)) return fallback
  return Math.max(0, Math.min(200, Math.round(num)))
}

function coercePositiveInt(value, fallback) {
  const num = Number.parseInt(String(value ?? ''), 10)
  if (!Number.isFinite(num) || num < 1) return fallback
  return num
}

function flowResampleToSeconds(value) {
  const s = String(value || '').trim().toLowerCase()
  if (!s) return 60

  if (s === '1m' || s === '1min' || s === '60s') return 60

  const secMatch = s.match(/^(\d+)\s*s$/)
  if (secMatch) return Math.max(1, Number(secMatch[1]))

  const minMatch = s.match(/^(\d+)\s*(m|min)$/)
  if (minMatch) return Math.max(60, Number(minMatch[1]) * 60)

  return 60
}

function deriveDefaultFlowEmaMinutes(params, flowResample) {
  const explicitMinutes = coercePositiveInt(
    params.get('flow_ema_minutes') || params.get('flow_ema_min'),
    NaN
  )
  if (Number.isFinite(explicitMinutes)) return explicitMinutes

  const legacyBars = coercePositiveInt(params.get('flow_ema_len'), NaN)
  if (Number.isFinite(legacyBars)) {
    const stepSeconds = flowResampleToSeconds(flowResample)
    return Math.max(1, Math.round((legacyBars * stepSeconds) / 60))
  }

  return 14
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

function normalizeTradeDateKey(value) {
  const s = String(value || '').trim()
  return /^\d{4}-\d{2}-\d{2}$/.test(s) ? s : ''
}

function zonedDateParts(date, timeZone = 'America/Los_Angeles') {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone,
    hour12: false,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).formatToParts(date)

  const out = {}
  for (const part of parts) {
    if (part.type !== 'literal') out[part.type] = part.value
  }

  return {
    year: Number(out.year),
    month: Number(out.month),
    day: Number(out.day),
    hour: Number(out.hour),
    minute: Number(out.minute),
    second: Number(out.second),
  }
}

function dateKeyFromParts(parts) {
  const yyyy = String(parts?.year || '').padStart(4, '0')
  const mm = String(parts?.month || '').padStart(2, '0')
  const dd = String(parts?.day || '').padStart(2, '0')
  if (!/^\d{4}$/.test(yyyy) || !/^\d{2}$/.test(mm) || !/^\d{2}$/.test(dd)) return ''
  return `${yyyy}-${mm}-${dd}`
}

function parseDateKey(value) {
  const key = normalizeTradeDateKey(value)
  if (!key) return null
  const [year, month, day] = key.split('-').map((part) => Number(part))
  if (!Number.isFinite(year) || !Number.isFinite(month) || !Number.isFinite(day)) return null
  return { year, month, day }
}

function shiftDateKey(value, days) {
  const parsed = parseDateKey(value)
  if (!parsed) return ''
  const base = new Date(Date.UTC(parsed.year, parsed.month - 1, parsed.day + Number(days || 0), 12, 0, 0))
  return dateKeyFromParts({
    year: base.getUTCFullYear(),
    month: base.getUTCMonth() + 1,
    day: base.getUTCDate(),
  })
}

function rollForwardToWeekdayKey(value) {
  let key = normalizeTradeDateKey(value)
  if (!key) return ''

  for (let i = 0; i < 7; i += 1) {
    const parsed = parseDateKey(key)
    if (!parsed) return ''
    const weekday = new Date(Date.UTC(parsed.year, parsed.month - 1, parsed.day, 12, 0, 0)).getUTCDay()
    if (weekday !== 0 && weekday !== 6) return key
    key = shiftDateKey(key, 1)
  }

  return key
}

function currentSessionTradeDateKey(timeZone = 'America/Los_Angeles') {
  const parts = zonedDateParts(new Date(), timeZone)
  let key = dateKeyFromParts(parts)
  if (!key) return ''

  if (parts.hour > 15 || (parts.hour === 15 && parts.minute >= 0)) {
    key = shiftDateKey(key, 1)
  }

  return rollForwardToWeekdayKey(key)
}

function zonedLocalToUtcEpochSec(parts, timeZone = 'America/Los_Angeles') {
  const year = Number(parts?.year)
  const month = Number(parts?.month)
  const day = Number(parts?.day)
  const hour = Number(parts?.hour || 0)
  const minute = Number(parts?.minute || 0)
  const second = Number(parts?.second || 0)

  if (
    !Number.isFinite(year) ||
    !Number.isFinite(month) ||
    !Number.isFinite(day) ||
    !Number.isFinite(hour) ||
    !Number.isFinite(minute) ||
    !Number.isFinite(second)
  ) {
    return null
  }

  const utcGuess = new Date(Date.UTC(year, month - 1, day, hour, minute, second))
  const zoned = zonedDateParts(utcGuess, timeZone)
  const zonedAsUtc = Date.UTC(
    Number(zoned.year),
    Number(zoned.month) - 1,
    Number(zoned.day),
    Number(zoned.hour),
    Number(zoned.minute),
    Number(zoned.second)
  )
  const targetAsUtc = Date.UTC(year, month - 1, day, hour, minute, second)
  return Math.round((utcGuess.getTime() + (targetAsUtc - zonedAsUtc)) / 1000)
}

function sessionWindowUtcRangeForTradeDate(tradeDate, timeZone = 'America/Los_Angeles') {
  const current = parseDateKey(tradeDate)
  const priorKey = shiftDateKey(tradeDate, -1)
  const prior = parseDateKey(priorKey)
  if (!current || !prior) return null

  const from = zonedLocalToUtcEpochSec({ ...prior, hour: 15, minute: 0, second: 0 }, timeZone)
  const to = zonedLocalToUtcEpochSec({ ...current, hour: 13, minute: 0, second: 0 }, timeZone)
  if (!Number.isFinite(from) || !Number.isFinite(to)) return null
  return { from, to }
}

function replaceFlowPointsForTradeDate(existingPoints, replacementPoints, tradeDate) {
  const sessionRange = sessionWindowUtcRangeForTradeDate(tradeDate)
  if (!sessionRange) {
    return dedupeAndSortFlowPoints(replacementPoints)
  }

  const preserved = (Array.isArray(existingPoints) ? existingPoints : []).filter((point) => {
    const t = Number(point?.time)
    if (!Number.isFinite(t)) return false
    return t < sessionRange.from || t >= sessionRange.to
  })

  return dedupeAndSortFlowPoints([
    ...preserved,
    ...(Array.isArray(replacementPoints) ? replacementPoints : []),
  ])
}

function barsPayloadSignature(payload) {
  const bars = Array.isArray(payload?.bars) ? payload.bars : []
  const segs = Array.isArray(payload?.gex_segments) ? payload.gex_segments : []
  const lastBar = bars.length ? bars[bars.length - 1] : null
  const lastSeg = segs.length ? segs[segs.length - 1] : null
  return [
    bars.length,
    lastBar?.time ?? '',
    lastBar?.open ?? '',
    lastBar?.high ?? '',
    lastBar?.low ?? '',
    lastBar?.close ?? '',
    lastBar?.volume ?? lastBar?.vol ?? lastBar?.size ?? '',
    segs.length,
    lastSeg?.session_date ?? '',
    lastSeg?.level ?? '',
    lastSeg?.start_time ?? '',
    lastSeg?.end_time ?? '',
    lastSeg?.net_gamma ?? '',
  ].join('|')
}

function overlayBarsSignature(bars) {
  const list = Array.isArray(bars) ? bars : []
  const last = list.length ? list[list.length - 1] : null
  return [
    list.length,
    last?.time ?? '',
    last?.open ?? '',
    last?.high ?? '',
    last?.low ?? '',
    last?.close ?? '',
  ].join('|')
}

function flowPointsSignature(points) {
  const list = Array.isArray(points) ? points : []
  const last = list.length ? list[list.length - 1] : null
  return [list.length, last?.time ?? '', last?.value ?? ''].join('|')
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
  const expirationDate = params.get('expiration_date') || tradeDate
  const expectedOn = params.get('expected') !== 'off'

  const initialInterval = useMemo(() => {
    const raw = params.get('interval') || '1min'
    const s = String(raw).trim().toLowerCase()
    return s.includes('5') ? '5min' : '1min'
  }, [params])

  const gexEnabled = parseBool(params.get('gex_enabled'), true)
  const initialGexMinAbsB = parseFloatOrNull(params.get('gex_min_abs_b'))
  const explicitDaysEitherSideRaw = params.get('days_either_side')
  const explicitDaysEitherSide =
    explicitDaysEitherSideRaw == null || explicitDaysEitherSideRaw === ''
      ? null
      : Math.max(0, parseIntOrDefault(explicitDaysEitherSideRaw, 0))

  const flowEnabled = parseBool(params.get('flow_enabled'), true)
  const flowSession = (params.get('flow_session') || 'FULL').toUpperCase()
  const defaultFlowResample = '1m'
  const flowResample = (params.get('flow_resample') || defaultFlowResample).toLowerCase()
  const defaultFlowEmaMinutes = deriveDefaultFlowEmaMinutes(params, flowResample)
  const flowHistAlpha = parseFloatOrNull(params.get('flow_hist_alpha')) ?? 0.30

  const apiBase = useMemo(() => inferApiBase(), [])
  
  const initialTimes = useMemo(() => parseSelectedTimes(params), [params])
  const [selectedTimes, setSelectedTimes] = useState(initialTimes)

  // Sync state if URL params change without component re-mount
  useEffect(() => {
    setSelectedTimes((prev) => {
      const next = initialTimes
      return prev.join(',') === next.join(',') ? prev : next
    })
  }, [initialTimes])

  const [interval, setInterval] = useState(initialInterval)

  const effectiveDaysEitherSide = useMemo(() => {
    if (interval === initialInterval && explicitDaysEitherSide != null) {
      return explicitDaysEitherSide
    }
    return interval === '5min' ? 10 : 1
  }, [explicitDaysEitherSide, interval, initialInterval])

  const dragStateRef = useRef(null)

  const [centerBars, setCenterBars] = useState([])
  const [extraBars, setExtraBars] = useState([])
  const [centerGexSegments, setCenterGexSegments] = useState([])
  const [extraGexSegments, setExtraGexSegments] = useState([])
  const [centerExpectedMoveLevels, setCenterExpectedMoveLevels] = useState([])
  const [extraExpectedMoveLevels, setExtraExpectedMoveLevels] = useState([])
  const [centerLoadedDates, setCenterLoadedDates] = useState([])
  const [extraLoadedDates, setExtraLoadedDates] = useState([])
  const [meta, setMeta] = useState(null)
  const [liveTradeBars, setLiveTradeBars] = useState([])

  const [flowPoints, setFlowPoints] = useState([])
  const [flowLoading, setFlowLoading] = useState(false)
  const [flowError, setFlowError] = useState('')

  const [loadingCenter, setLoadingCenter] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState('')

  const [sharedLogicalRange, setSharedLogicalRange] = useState(null)
  const [linkedCrosshair, setLinkedCrosshair] = useState(null)
  const [isChartInteracting, setIsChartInteracting] = useState(false)

  const [skewData, setSkewData] = useState([])
  const [smileData, setSmileData] = useState(null)

  const [flowPanelHeight, setFlowPanelHeight] = useState(() => {
    try {
      return clampFlowHeight(window.localStorage.getItem(FLOW_HEIGHT_STORAGE_KEY) || 220)
    } catch (err) {
      return 220
    }
  })

  const [flowEmaMinutes, setFlowEmaMinutes] = useState(() => {
    try {
      return coercePositiveInt(
        window.localStorage.getItem(FLOW_EMA_MINUTES_STORAGE_KEY),
        defaultFlowEmaMinutes
      )
    } catch (err) {
      return defaultFlowEmaMinutes
    }
  })

  const [gexMinAbsB, setGexMinAbsB] = useState(() => {
    const fallback = coerceGexMinAbsB(initialGexMinAbsB, 10)
    try {
      return coerceGexMinAbsB(
        window.localStorage.getItem(GEX_MIN_ABS_B_STORAGE_KEY),
        fallback
      )
    } catch (err) {
      return fallback
    }
  })

  const [liveSessionKey, setLiveSessionKey] = useState(() => currentSessionTradeDateKey())

  const selectedSessionTradeDateKey = useMemo(() => {
    const effective = normalizeTradeDateKey(meta?.effective_trade_date)
    if (effective) return effective
    return normalizeTradeDateKey(tradeDate)
  }, [meta, tradeDate])

  const lastLiveBarsSignatureRef = useRef('')
  const lastLiveOverlaySignatureRef = useRef('')
  const lastLiveFlowSignatureRef = useRef('')

  const flowEmaLen = useMemo(() => {
    const stepSeconds = flowResampleToSeconds(flowResample)
    return Math.max(1, Math.round((flowEmaMinutes * 60) / stepSeconds))
  }, [flowEmaMinutes, flowResample])

  useEffect(() => {
    const handleParentTimeslices = (event) => {
      const data = event?.data
      if (!data || data.type !== 'ib-parent-timeslices') return
      const next = normalizeTimes(Array.isArray(data.times) ? data.times : [])
      setSelectedTimes((prev) => {
        if (prev.join(',') === next.join(',')) return prev
        return next
      })
    }
    window.addEventListener('message', handleParentTimeslices)
    return () => window.removeEventListener('message', handleParentTimeslices)
  }, [])

  useEffect(() => {
    const tick = () => {
      setLiveSessionKey((prev) => {
        const next = currentSessionTradeDateKey()
        return next && next !== prev ? next : prev
      })
    }

    const id = window.setInterval(tick, 30000)
    return () => window.clearInterval(id)
  }, [])

  const isLiveTradeDate = useMemo(() => {
    const selected = normalizeTradeDateKey(selectedSessionTradeDateKey)
    return Boolean(selected) && selected === normalizeTradeDateKey(liveSessionKey)
  }, [selectedSessionTradeDateKey, liveSessionKey])

  const mergedBars = useMemo(
    () => dedupeAndSortBars([...centerBars, ...extraBars]),
    [centerBars, extraBars]
  )

  const liveTradeTimelineSignature = useMemo(
    () => (Array.isArray(liveTradeBars) ? liveTradeBars : []).map((bar) => Number(bar?.time) || '').join('|'),
    [liveTradeBars]
  )

  const mergedBarsWithLiveTimeline = useMemo(
    () => dedupeAndSortBars([
      ...centerBars,
      ...extraBars,
      ...(Array.isArray(liveTradeBars) ? liveTradeBars : []).map((bar) => ({
        time: Number(bar?.time),
        is_center: bar?.is_center,
      })),
    ]),
    [centerBars, extraBars, liveTradeTimelineSignature]
  )

  const mergedGexSegments = useMemo(
    () => dedupeSegments([...centerGexSegments, ...extraGexSegments]),
    [centerGexSegments, extraGexSegments]
  )

  const mergedExpectedMoveLevels = useMemo(
    () => [...centerExpectedMoveLevels, ...extraExpectedMoveLevels],
    [centerExpectedMoveLevels, extraExpectedMoveLevels]
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
    try {
      window.localStorage.setItem(FLOW_EMA_MINUTES_STORAGE_KEY, String(flowEmaMinutes))
    } catch (err) {
      // ignore
    }
  }, [flowEmaMinutes])

  useEffect(() => {
    try {
      window.localStorage.setItem(GEX_MIN_ABS_B_STORAGE_KEY, String(gexMinAbsB))
    } catch (err) {
      // ignore
    }
  }, [gexMinAbsB])

  useEffect(() => {
    try {
      const nextUrl = new URL(window.location.href)
      nextUrl.searchParams.set('interval', interval)
      window.history.replaceState(null, '', nextUrl.toString())
    } catch (err) {
      // ignore
    }
  }, [interval])

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
        setCenterExpectedMoveLevels([])
        setExtraExpectedMoveLevels([])
        setCenterLoadedDates([])
        setExtraLoadedDates([])
        setMeta(null)
        setLiveTradeBars([])
        setSharedLogicalRange(null)
        lastLiveBarsSignatureRef.current = ''
        lastLiveOverlaySignatureRef.current = ''
        lastLiveFlowSignatureRef.current = ''

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
        setCenterExpectedMoveLevels(Array.isArray(payload.expected_move_levels) ? payload.expected_move_levels : [])
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
        setCenterExpectedMoveLevels([])
        setExtraExpectedMoveLevels([])
        setCenterLoadedDates([])
        setExtraLoadedDates([])
        setMeta(null)
        setLiveTradeBars([])
        lastLiveBarsSignatureRef.current = ''
        lastLiveOverlaySignatureRef.current = ''
        lastLiveFlowSignatureRef.current = ''
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
        setExtraExpectedMoveLevels(Array.isArray(payload.expected_move_levels) ? payload.expected_move_levels : [])
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
    if (isLiveTradeDate) return
    setLiveTradeBars([])
    lastLiveOverlaySignatureRef.current = ''
  }, [isLiveTradeDate, selectedSessionTradeDateKey])

  useEffect(() => {
    if (!isLiveTradeDate || loadingCenter || error || isChartInteracting) return undefined

    let disposed = false
    let inFlight = false
    let activeController = null

    const tick = async () => {
      if (disposed || inFlight) return
      inFlight = true
      const controller = new AbortController()
      activeController = controller

      try {
        const url = new URL(`${apiBase}/api/ironbeam/bars`)
        url.searchParams.set('trade_date', selectedSessionTradeDateKey || tradeDate)
        url.searchParams.set('interval', interval)
        url.searchParams.set('gex_enabled', gexEnabled ? '1' : '0')
        url.searchParams.set('phase', 'center')
        if (gexMinAbsB != null) {
          url.searchParams.set('gex_min_abs_b', String(gexMinAbsB))
        }

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
        if (disposed) return

        const nextSignature = barsPayloadSignature(payload)
        if (nextSignature !== lastLiveBarsSignatureRef.current) {
          lastLiveBarsSignatureRef.current = nextSignature
          setCenterBars(Array.isArray(payload.bars) ? payload.bars : [])
          setCenterGexSegments(Array.isArray(payload.gex_segments) ? payload.gex_segments : [])
          setCenterExpectedMoveLevels(Array.isArray(payload.expected_move_levels) ? payload.expected_move_levels : [])
          setCenterLoadedDates(
            normalizeDateList(
              Array.isArray(payload.loaded_dates) && payload.loaded_dates.length
                ? payload.loaded_dates
                : [payload?.effective_trade_date || payload?.trade_date || tradeDate]
            )
          )
          setMeta((prev) => {
            const prevSig = prev ? barsPayloadSignature(prev) : ''
            return prevSig === nextSignature ? prev : payload
          })
        }
      } catch (err) {
        if (err?.name !== 'AbortError') {
          console.error('React preview live bars refresh failed', err)
        }
      } finally {
        if (activeController === controller) {
          activeController = null
        }
        inFlight = false
      }
    }

    tick()
    const timer = window.setInterval(tick, FULL_BARS_LIVE_POLL_MS)

    return () => {
      disposed = true
      window.clearInterval(timer)
      if (activeController) {
        activeController.abort()
      }
    }
  }, [apiBase, tradeDate, selectedSessionTradeDateKey, interval, gexEnabled, gexMinAbsB, loadingCenter, error, isLiveTradeDate, isChartInteracting])

  useEffect(() => {
    if (!isLiveTradeDate || loadingCenter || error) return undefined

    let disposed = false
    let inFlight = false
    let activeController = null

    const tick = async () => {
      if (disposed || inFlight) return
      inFlight = true
      const controller = new AbortController()
      activeController = controller

      try {
        const url = new URL(`${apiBase}/api/ironbeam/live-trades-overlay`)
        url.searchParams.set('trade_date', selectedSessionTradeDateKey || tradeDate)
        url.searchParams.set('interval', interval)

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
            `Live overlay request failed: ${response.status} ${response.statusText || ''}`.trim()
          )
        }

        const payload = await response.json()
        if (disposed) return

        const nextBars = Array.isArray(payload.bars) ? payload.bars : []
        const nextSignature = overlayBarsSignature(nextBars)
        if (nextSignature !== lastLiveOverlaySignatureRef.current) {
          lastLiveOverlaySignatureRef.current = nextSignature
          setLiveTradeBars(nextBars)
        }
      } catch (err) {
        if (err?.name !== 'AbortError') {
          console.error('React preview live trades overlay refresh failed', err)
        }
      } finally {
        if (activeController === controller) {
          activeController = null
        }
        inFlight = false
      }
    }

    tick()
    const timer = window.setInterval(tick, LIVE_OVERLAY_POLL_MS)

    return () => {
      disposed = true
      window.clearInterval(timer)
      if (activeController) {
        activeController.abort()
      }
    }
  }, [apiBase, tradeDate, selectedSessionTradeDateKey, interval, loadingCenter, error, isLiveTradeDate])

  useEffect(() => {
    if (!isLiveTradeDate || !flowEnabled || loadingCenter || error || isChartInteracting) return undefined

    let disposed = false
    let inFlight = false
    let activeController = null

    const tick = async () => {
      if (disposed || inFlight) return
      inFlight = true
      const controller = new AbortController()
      activeController = controller

      try {
        const url = new URL(`${apiBase}/api/ironbeam/flow`)
        url.searchParams.set('trade_date', selectedSessionTradeDateKey || tradeDate)
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
            `Flow request failed for ${tradeDate}: ${response.status} ${response.statusText || ''}`.trim()
          )
        }

        const payload = await response.json()
        if (disposed) return

        const nextPoints = Array.isArray(payload.flow_points) ? payload.flow_points : []
        const nextSignature = flowPointsSignature(nextPoints)
        if (nextSignature !== lastLiveFlowSignatureRef.current) {
          lastLiveFlowSignatureRef.current = nextSignature
          setFlowPoints((prev) => replaceFlowPointsForTradeDate(prev, nextPoints, selectedSessionTradeDateKey || tradeDate))
        }
      } catch (err) {
        if (err?.name !== 'AbortError') {
          console.error('React preview live flow refresh failed', err)
        }
      } finally {
        if (activeController === controller) {
          activeController = null
        }
        inFlight = false
      }
    }

    tick()
    const timer = window.setInterval(tick, FLOW_LIVE_POLL_MS)

    return () => {
      disposed = true
      window.clearInterval(timer)
      if (activeController) {
        activeController.abort()
      }
    }
  }, [apiBase, tradeDate, selectedSessionTradeDateKey, flowEnabled, flowSession, flowResample, flowEmaLen, loadingCenter, error, isLiveTradeDate])

  // --- Skew Data Fetching ---
  useEffect(() => {
    if (loadingCenter || error) return undefined

    let disposed = false
    let inFlight = false
    let activeController = null

    const tick = async () => {
      if (disposed || inFlight) return
      inFlight = true
      const controller = new AbortController()
      activeController = controller

      try {
        const url = new URL(`${apiBase}/api/ironbeam/skew-data`)
        url.searchParams.set('trade_date', tradeDate)
        url.searchParams.set('expiration_date', expirationDate)
        url.searchParams.set('expected', expectedOn ? 'on' : 'off')
        if (selectedTimes && selectedTimes.length > 0) {
          url.searchParams.set('times', selectedTimes.join(','))
        }

        // Live data from parent store if available
        try {
          const parentData = window.parent?.document?.getElementById('live-data-store')?.textContent
          if (parentData) {
            url.searchParams.set('live_data', parentData)
          }
        } catch (e) {
          // ignore
        }

        const response = await fetch(url.toString(), {
          method: 'GET',
          credentials: 'include',
          signal: controller.signal,
          cache: 'no-store',
        })

        if (!response.ok) throw new Error('Skew data fetch failed')

        const payload = await response.json()
        if (disposed) return
        setSkewData(payload.skew_data || [])
      } catch (err) {
        if (err?.name !== 'AbortError') {
          console.error('Skew data refresh failed', err)
        }
      } finally {
        if (activeController === controller) activeController = null
        inFlight = false
      }
    }

    tick()
    const timer = window.setInterval(tick, SKEW_DATA_POLL_MS)

    return () => {
      disposed = true
      window.clearInterval(timer)
      if (activeController) activeController.abort()
    }
  }, [apiBase, tradeDate, expirationDate, selectedTimes, expectedOn, loadingCenter, error])

  // --- Smile Data Fetching ---
  useEffect(() => {
    if (loadingCenter || error) return undefined

    let disposed = false
    let inFlight = false
    let activeController = null

    const tick = async () => {
      if (disposed || inFlight) return
      inFlight = true
      const controller = new AbortController()
      activeController = controller

      try {
        const url = new URL(`${apiBase}/api/ironbeam/smile-data`)
        url.searchParams.set('trade_date', tradeDate)
        url.searchParams.set('expiration_date', expirationDate)
        url.searchParams.set('expected', expectedOn ? 'on' : 'off')
        if (selectedTimes && selectedTimes.length > 0) {
          url.searchParams.set('times', selectedTimes.join(','))
        }

        try {
          const parentData = window.parent?.document?.getElementById('live-data-store')?.textContent
          if (parentData) {
            url.searchParams.set('live_data', parentData)
          }
        } catch (e) {}

        const response = await fetch(url.toString(), {
          method: 'GET',
          credentials: 'include',
          signal: controller.signal,
          cache: 'no-store',
        })

        if (!response.ok) throw new Error('Smile data fetch failed')

        const payload = await response.json()
        if (disposed) return
        setSmileData(payload)
      } catch (err) {
        if (err?.name !== 'AbortError') {
          console.error('Smile data refresh failed', err)
        }
      } finally {
        if (activeController === controller) activeController = null
        inFlight = false
      }
    }

    tick()
    const timer = window.setInterval(tick, SKEW_DATA_POLL_MS)

    return () => {
      disposed = true
      window.clearInterval(timer)
      if (activeController) activeController.abort()
    }
  }, [apiBase, tradeDate, expirationDate, selectedTimes, expectedOn, loadingCenter, error])

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

  const handleFlowEmaMinutesChange = useCallback((nextValue) => {
    setFlowEmaMinutes((prev) => {
      const next = coercePositiveInt(nextValue, prev)
      return next === prev ? prev : next
    })
  }, [])

  const handleGexMinAbsBChange = useCallback((nextValue) => {
    setGexMinAbsB((prev) => {
      const next = coerceGexMinAbsB(nextValue, prev)
      return next === prev ? prev : next
    })
  }, [])

  const handleIntervalChange = useCallback((nextValue) => {
    const next = String(nextValue || '').trim() === '5min' ? '5min' : '1min'
    setInterval((prev) => (prev === next ? prev : next))
  }, [])

  const handleInteractionActiveChange = useCallback((nextValue) => {
    const next = Boolean(nextValue)
    setIsChartInteracting((prev) => (prev === next ? prev : next))
  }, [])

  const handleLinkedCrosshairChange = useCallback((nextValue) => {
    if (!nextValue) {
      setLinkedCrosshair(null)
      return
    }

    const logical = Number(nextValue.logical)
    const shiftedTime = Number(nextValue.shiftedTime)
    if (!Number.isFinite(logical) && !Number.isFinite(shiftedTime)) {
      setLinkedCrosshair(null)
      return
    }

    const next = {
      logical: Number.isFinite(logical) ? logical : null,
      shiftedTime: Number.isFinite(shiftedTime) ? shiftedTime : null,
    }

    setLinkedCrosshair((prev) => {
      if (
        prev &&
        prev.logical === next.logical &&
        prev.shiftedTime === next.shiftedTime
      ) {
        return prev
      }
      return next
    })
  }, [])

  return (
    <div className="app-shell compact-shell" style={{ width: '100%', minWidth: 0, height: '100%', minHeight: '100%' }}>
      <div className="card compact-card" style={{ width: '100%', minWidth: 0, height: '100%', minHeight: 0, display: 'flex', flexDirection: 'column' }}>
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
          <div className="react-preview-layout" style={{ width: '100%', minWidth: 0, height: '100%', minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <div className="react-top-pane" style={{ width: '100%', minWidth: 0, flex: '1 1 auto', minHeight: 0 }}>
              {loadingMore && (
                <div className="top-pane-status">Loading surrounding sessions…</div>
              )}
              <PriceChart
                candles={mergedBars}
                tradeDate={tradeDate}
                interval={interval}
                liveTradeCandles={liveTradeBars}
                selectedTimes={selectedTimes}
                gexSegments={mergedGexSegments}
                gexEnabled={Boolean(meta?.gex_enabled ?? gexEnabled)}
                gexMinAbsB={gexMinAbsB}
                expectedMoveLevels={mergedExpectedMoveLevels}
                onApplyGexMinAbsB={handleGexMinAbsBChange}
                onApplyIntervalChange={handleIntervalChange}
                onVisibleLogicalRangeChange={handleSharedLogicalRangeChange}
                onLinkedCrosshairChange={handleLinkedCrosshairChange}
                onInteractionActiveChange={handleInteractionActiveChange}
                skewData={skewData}
                smileData={smileData}
              />
            </div>

            {flowEnabled && (
              <>
                <div
                  className="panel-splitter"
                  onMouseDown={beginResize}
                  title="Drag to resize panels"
                />
                <div className="react-bottom-pane" style={{ width: '100%', minWidth: 0, height: `${flowPanelHeight}px` }}>
                  <AggressorFlowPanel
                    dataPoints={flowPoints}
                    candles={mergedBarsWithLiveTimeline}
                    visibleLogicalRange={sharedLogicalRange}
                    height={flowPanelHeight}
                    loading={flowLoading}
                    error={flowError}
                    histAlpha={flowHistAlpha}
                    linkedCrosshair={linkedCrosshair}
                    emaMinutes={flowEmaMinutes}
                    onApplyEmaMinutes={handleFlowEmaMinutesChange}
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
