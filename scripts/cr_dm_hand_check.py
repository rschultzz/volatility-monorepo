"""CR-DM smoke: the live leg at several 2026-10-29 replay times, by hand from the minute data, vs the v2 payload.

    apps/web/.venv/bin/python -u scripts/cr_dm_hand_check.py

For each `until` the smoke backend (port 8065, read-only) serves the v2 payload (effort=1, swing 8). Independently,
from minutes of the payload's symbol in [06:30 PT, until), aggregated from `ironbeam_es_flow_1s` (the 1-minute table
lags a live session):
  start = the last confirmed leg's end (time, price) from that payload; the leg runs the other way;
  extreme = max high (up) / min low (down) of the minutes after start (first minute reaching it);
  pts, volume, delta = sums over (start, extreme]; cpp = vol / |pts|; ratio = cpp / the previous confirmed leg
  the same way.
Also picks one leg confirmed in the window and checks the hand-off: at until = its confirm minute the live leg is that
leg (same numbers); one minute later it is in `legs` and the live leg is the next one.
Writes scripts/logs/cr-dm/smoke/hand_check.json (read by tests/e2e/cr-dm.spec.js). Read-only; DB errors print the type.
"""
import datetime as dt
import json
import os
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for line in open(os.path.join(REPO, ".env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

import psycopg  # noqa: E402

API = os.environ.get("CRDM_API", "http://127.0.0.1:8065")
DAY = "2026-10-29"
PT = dt.timezone(dt.timedelta(hours=-7))   # PDT on 2026-10-29
QS = "session=FULL&resample=1m&ema_len=14&mode=v2&effort=1&m_swing=8&m_push=500&m_div=300&m_fade=0.1&m_gap=30"
OUT = os.path.join(REPO, "scripts", "logs", "cr-dm", "smoke", "hand_check.json")


def payload(until=None):
    url = f"{API}/api/ironbeam/flow?trade_date={DAY}&{QS}" + (f"&until={DAY}T{until}:00" if until else "")
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def epoch(hhmm):
    h, m = map(int, hhmm.split(":"))
    return int(dt.datetime(2026, 10, 29, h, m, tzinfo=PT).timestamp())


def hhmm(t):
    return dt.datetime.fromtimestamp(t, PT).strftime("%H:%M")


def hand(conn, symbol, until_epoch, legs):
    a = epoch("06:30")
    # minutes built from the 1-s table (the 1-minute table lags the live session; equal where both exist, CR-CP)
    rows = conn.execute(
        "SELECT (floor(extract(epoch FROM ts_utc) / 60) * 60)::bigint AS t, max(high), min(low), sum(volume), "
        "sum(buy_vol) - sum(sell_vol) FROM ironbeam_es_flow_1s WHERE symbol = %s AND ts_utc >= to_timestamp(%s) "
        "AND ts_utc < to_timestamp(%s) GROUP BY 1 ORDER BY 1", (symbol, a, until_epoch)).fetchall()
    last = legs[-1]
    up = last["dir"] == "down"
    after = [r for r in rows if r[0] > last["end_time"]]
    ext = max(float(r[1]) for r in after) if up else min(float(r[2]) for r in after)
    ext_t = next(r[0] for r in after if float(r[1 if up else 2]) == ext)
    span = [r for r in after if r[0] <= ext_t]
    vol = sum(float(r[3]) for r in span)
    delta = sum(float(r[4]) for r in span)
    pts = ext - last["end_px"]
    cpp = vol / abs(pts) if pts else None
    prev = next((lg for lg in reversed(legs) if lg["dir"] == ("up" if up else "down")), None)
    ratio = cpp / prev["cpp"] if prev and cpp is not None and prev["cpp"] else None
    return {"dir": "up" if up else "down", "start_time": last["end_time"], "start_px": last["end_px"], "end_time": ext_t,
            "end_px": ext, "pts": pts, "vol": vol, "delta": delta, "cpp": cpp, "ratio": ratio,
            "prev_cpp": None if prev is None else prev["cpp"], "minutes": len(span),
            "last_time": max(r[0] for r in rows if float(r[3]) > 0)}


def label(lv, min_move=4.0):
    if abs(lv["pts"]) < min_move or lv["prev_cpp"] is None:
        return None
    arrow = "↑" if lv["dir"] == "up" else "↓"
    ratio = "–" if lv["ratio"] is None else f"{lv['ratio']:.1f}×"
    sign = "+" if lv["pts"] >= 0 else "-"
    return f"{arrow} {sign}{abs(lv['pts']):.1f} pts so far · {ratio} vs last {arrow}"


def close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) < 1e-6 if isinstance(a, (int, float)) else a == b


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    keys = ("dir", "start_time", "start_px", "end_time", "end_px", "pts", "vol", "delta", "cpp", "ratio", "prev_cpp",
            "minutes", "last_time")
    full = payload("08:00")
    window = [lg for lg in full["legs"] if epoch("07:05") <= lg["confirm_time"] <= epoch("07:55")]
    pick = max(window, key=lambda lg: abs(lg["pts"]))     # a clear leg confirmed in the window
    c = hhmm(pick["confirm_time"])
    c1 = hhmm(pick["confirm_time"] + 60)
    times = ["06:50", "07:05", "07:20", "07:35", "07:50", c, c1]
    out = {"day": DAY, "symbol": full["symbol"], "handoff": {"confirm": c, "after": c1, "leg": pick}, "cases": []}
    ok = True
    try:
        with psycopg.connect(os.environ["BACKFILL_DATABASE_URL"], options="-c default_transaction_read_only=on") as conn:
            for t in times:
                p = payload(t)
                live = p["live_leg"]
                h = hand(conn, p["symbol"], epoch(t), p["legs"])
                diff = [k for k in keys if not close(live.get(k), h[k])]
                after_until = live["last_time"] >= epoch(t) or any(b["time"] >= epoch(t) for b in p["bars"])
                ok &= not diff and not after_until
                out["cases"].append({"until": t, "live": live, "hand": h, "diff": diff, "label": label(h),
                                     "bars_after_until": after_until, "n_legs": len(p["legs"]),
                                     "last_leg": p["legs"][-1]})
                rt = "–" if live["ratio"] is None else f"{live['ratio']:.2f}"
                print(f"until {t}: live {live['dir']} {live['start_px']} -> {live['end_px']} ({hhmm(live['end_time'])}), "
                      f"pts {live['pts']:+.2f}, vol {live['vol']:,.0f}, cpp {live['cpp']:,.0f}, ratio "
                      f"{rt} | hand diff {diff or 'none'} | "
                      f"bars >= until: {after_until} | label {label(h)!r}")
    except psycopg.Error as e:
        print("DB error:", type(e).__name__)
        sys.exit(1)
    pc, pa = out["cases"][-2], out["cases"][-1]
    same = {k: pc["live"][k] for k in ("dir", "start_time", "end_time", "start_px", "end_px", "pts", "vol", "cpp", "ratio")}
    conf = {k: pick[k] for k in same}
    handoff_ok = same == conf and pa["last_leg"]["end_time"] == pick["end_time"] and pa["live"]["dir"] != pick["dir"]
    out["handoff"]["ok"] = handoff_ok
    print(f"hand-off: leg {pick['dir']} {pick['start_px']} -> {pick['end_px']} confirmed {c}: provisional at until {c} "
          f"== confirmed leg: {same == conf}; at until {c1} it is the last confirmed leg and the live leg is "
          f"{pa['live']['dir']}: {handoff_ok}")
    out["ok"] = bool(ok and handoff_ok)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("ALL OK" if out["ok"] else "MISMATCH")


if __name__ == "__main__":
    main()
