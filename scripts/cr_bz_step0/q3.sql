WITH f AS (
  SELECT date_trunc('minute', ts_utc) AS m,
         count(DISTINCT symbol) nsym, min(symbol) sym,
         (array_agg(open ORDER BY ts_utc))[1] o, max(high) h, min(low) l,
         (array_agg(close ORDER BY ts_utc DESC))[1] c, sum(volume) v, count(*) n
  FROM ironbeam_es_flow_1s WHERE ts_utc >= '2026-01-01' GROUP BY 1),
j AS (
  SELECT b.datetime, f.*, b.open bo, b.high bh, b.low bl, b.close bc, b.volume bv, b.source,
    ((b.datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::time >= '06:30' AND
     (b.datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::time < '13:00') rth
  FROM ironbeam_es_1m_bars b JOIN f ON f.m = b.datetime AT TIME ZONE 'UTC'
  WHERE b.datetime >= '2026-01-01'
    AND NOT (b.datetime >= '2026-09-13 22:00' AND b.datetime < '2026-09-18 21:00'))
SELECT to_char(datetime,'YYYY-MM') mon, rth, count(*) n,
  sum((nsym>1)::int) two_sym,
  sum((greatest(abs(bh-h),abs(bl-l),abs(bc-c))>0.25)::int) px_diff,
  round(100.0*avg((greatest(abs(bh-h),abs(bl-l),abs(bc-c))>0.25)::int),2) px_pct,
  sum((greatest(abs(bh-h),abs(bl-l),abs(bc-c))>0.25 AND bv < v)::int) px_bar_vol_lt,
  sum((greatest(abs(bh-h),abs(bl-l),abs(bc-c))>0.25 AND bv > v)::int) px_bar_vol_gt,
  percentile_cont(0.5) within group (order by greatest(abs(bh-h),abs(bl-l),abs(bc-c))) filter (where greatest(abs(bh-h),abs(bl-l),abs(bc-c))>0.25) med_pts,
  max(greatest(abs(bh-h),abs(bl-l),abs(bc-c))) max_pts,
  sum((abs(bv-v) > 0.10*greatest(bv,v))::int) vol10,
  round(percentile_cont(0.5) within group (order by bv/nullif(v,0))::numeric,3) vol_ratio_med,
  sum((abs(bv-v) > 0.10*greatest(bv,v) OR greatest(abs(bh-h),abs(bl-l),abs(bc-c))>0.25)::int) l5_hits,
  sum((n < 60 AND rth)::int) short_flow_rth
FROM j GROUP BY 1,2 ORDER BY 1,2
