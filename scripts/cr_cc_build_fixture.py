"""CR-CC step 6 — build the 5-event smoke fixture `research/gallery/cc-fixture.json` (set "cc:fixture").

    apps/web/.venv/bin/python -m scripts.cr_cc_build_fixture

UI smoke data, not a research result. Read-only DB (backfill role, read-only transaction).

Source events: CR-BW's real level touches (`scripts/logs/cr_bw/events_real.pkl`, machine-local,
git-ignored): session `d`, level `L` (CR-BT corrected placement, the chart's Corrected ES frame),
`net` and `mag` in $ (net = call + put, mag = |call| + |put|), touch time `t` (UTC epoch seconds).
One event per month Jan–May 2026 (train, <= 2026-06-05; the March roll week excluded): the first
touch of the month at or after 07:00 PT (past the opening rotation) for which the rule below finds a
confirmation.

Fixture rule (ES 1-minute bars, `ironbeam_es_1m_bars`, bar start UTC):
  - touch bar = the bar containing `t`; approach side from the close 5 bars earlier:
    above L -> long (fade back up), below L -> short;
  - confirm bar = the first bar within 10 bars after the touch bar that closes back on the
    approach side of L (long: close > L);
  - entry_ts_pt = confirm bar start + 1 min (the close of the signal bar, A1.2); entry = its close;
  - extreme = lowest low (long) / highest high (short) from the touch bar to the confirm bar;
    stop = extreme -/+ 1.0 pt;
  - outcome: race of target entry +/- 8 pts vs the stop over the bars starting at entry_ts_pt up to
    12:59 PT; a bar that reaches both counts as the stop; unresolved -> exit at the 12:59 close.
    exit_ts_pt = the resolving bar's start; pnl_pts in ES points.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EVENTS = REPO / "scripts" / "logs" / "cr_bw" / "events_real.pkl"
OUT = REPO / "research" / "gallery" / "cc-fixture.json"
PT = ZoneInfo("America/Los_Angeles")
UTC = ZoneInfo("UTC")
TARGET_PTS = 8.0
STOP_PAD = 1.0
ROLL_WEEK = (dt.date(2026, 3, 16), dt.date(2026, 3, 20))
TRAIN_END = dt.date(2026, 6, 5)
EARLIEST_TOUCH_PT = dt.time(7, 0)


def pt_naive(ts_utc: pd.Timestamp) -> dt.datetime:
    return ts_utc.tz_localize(UTC).tz_convert(PT).tz_localize(None).to_pydatetime()


def session_bars(conn, d: dt.date) -> pd.DataFrame:
    start = dt.datetime.combine(d, dt.time(6, 30), tzinfo=PT).astimezone(UTC).replace(tzinfo=None)
    end = dt.datetime.combine(d, dt.time(13, 0), tzinfo=PT).astimezone(UTC).replace(tzinfo=None)
    rows = conn.execute(
        "SELECT datetime, open, high, low, close FROM ironbeam_es_1m_bars "
        "WHERE datetime >= %s AND datetime < %s ORDER BY datetime",
        (start, end),
    ).fetchall()
    return pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close"])


def build_event(bars: pd.DataFrame, ev) -> dict | None:
    L = float(ev.L)
    touch_utc = pd.Timestamp(int(ev.t), unit="s")
    idx = bars.index[bars["datetime"] <= touch_utc]
    if len(idx) == 0 or idx[-1] < 5:
        return None
    k = int(idx[-1])
    long = float(bars.at[k - 5, "close"]) > L
    confirm = None
    for j in range(k + 1, min(k + 11, len(bars))):
        c = float(bars.at[j, "close"])
        if (long and c > L) or (not long and c < L):
            confirm = j
            break
    if confirm is None or confirm + 1 >= len(bars):
        return None
    window = bars.iloc[k:confirm + 1]
    extreme = float(window["low"].min()) if long else float(window["high"].max())
    stop = extreme - STOP_PAD if long else extreme + STOP_PAD
    entry = float(bars.at[confirm, "close"])
    target = entry + TARGET_PTS if long else entry - TARGET_PTS

    exit_k, exit_px, result = None, None, None
    for j in range(confirm + 1, len(bars)):
        hi, lo = float(bars.at[j, "high"]), float(bars.at[j, "low"])
        hit_stop = lo <= stop if long else hi >= stop
        hit_target = hi >= target if long else lo <= target
        if hit_stop or hit_target:
            exit_k = j
            exit_px, result = (stop, "stop") if hit_stop else (target, "target")
            break
    if exit_k is None:
        exit_k, exit_px, result = len(bars) - 1, float(bars.at[len(bars) - 1, "close"]), "eod"
    pnl = (exit_px - entry) if long else (entry - exit_px)

    entry_ts = pt_naive(bars.at[confirm, "datetime"]) + dt.timedelta(minutes=1)
    return {
        "direction": "long" if long else "short",
        "level_price": L,
        "level_net_bn": round(float(ev.net) / 1e9, 2),
        "level_gross_bn": round(float(ev.mag) / 1e9, 2),
        "test_ts_pt": pt_naive(bars.at[k, "datetime"]).isoformat(),
        "confirm_ts_pt": pt_naive(bars.at[confirm, "datetime"]).isoformat(),
        "retest_ts_pt": None,
        "entry_ts_pt": entry_ts.isoformat(),
        "entry_price": entry,
        "stop_price": stop,
        "extreme_price": extreme,
        "meta": {
            "overshoot_pts": round(abs(L - extreme), 2),
            "touch_utc": touch_utc.isoformat(),
            "source": "CR-BW events_real touch",
        },
        "outcome": {
            "exit_ts_pt": pt_naive(bars.at[exit_k, "datetime"]).isoformat(),
            "exit_price": exit_px,
            "pnl_pts": round(pnl, 2),
            "result": result,
            "by_exit": {"target_pts": TARGET_PTS, "stop_pad_pts": STOP_PAD},
        },
    }


def main() -> None:
    from dotenv import load_dotenv
    from packages.shared.backfill_safety import assert_role_or_die, get_backfill_db_conn

    load_dotenv(REPO / ".env")

    ev = pd.read_pickle(EVENTS)
    ev["d"] = pd.to_datetime(ev["d"]).dt.date
    ev = ev[(ev["d"] <= TRAIN_END) & ~ev["d"].between(*ROLL_WEEK)].sort_values(["d", "t"])
    touch_pt = pd.to_datetime(ev["t"], unit="s", utc=True).dt.tz_convert(PT).dt.time
    ev = ev[touch_pt >= EARLIEST_TOUCH_PT]

    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    conn.execute("SET default_transaction_read_only = on")
    events = []
    for month in range(1, 6):
        for _, row in ev[[d.month == month for d in ev["d"]]].iterrows():
            out = build_event(session_bars(conn, row["d"]), row)
            if out is not None:
                events.append({"id": len(events) + 1, "session_date": row["d"].isoformat(),
                               "cell": "net+" if row["net"] > 0 else "net-", **out})
                break
    conn.close()

    doc = {
        "cr_id": "CR-CC",
        "set_name": "cc:fixture",
        "level_frame": "es_corrected",
        "generated_at": dt.datetime.now(PT).replace(tzinfo=None, microsecond=0).isoformat(),
        "events": events,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    for e in events:
        print(e["id"], e["session_date"], e["direction"], e["level_price"], e["entry_ts_pt"],
              e["entry_price"], e["stop_price"], e["outcome"]["result"], e["outcome"]["pnl_pts"])
    print(f"wrote {OUT} ({len(events)} events)")


if __name__ == "__main__":
    main()
