"""CR-DF Step 0 (read-only, mechanical display measurements; no direction, outcome or feature statistic).

Through the real cells endpoint builder (flow_v3.build_cells_payload, K 3 / M 3, RTH view), per session:
  - tier (exact / approximate) of the RTH segment and the number of absorption marks (= levels, DF1);
  - contracts per level (aggressor + merged_vol) — sets DF2's log thickness scale;
  - held minutes, share broken;
  - at every RTH minute T ("now"), how many levels are confirmed by T and how many are < 60 min old — how often
    DF3's cap of 8 binds;
  - the 1-minute flow bars the edge readout will use: minutes per segment vs minutes with cells.
Train sessions (Databento via es_flow_price_1m, every ~10th of 2026-01-06..06-05) and the retained Ironbeam
sessions (ironbeam_es_trades). B4: train + forward only, counts only."""
import datetime as dt
import json
import os
import sys

REPO = sys.argv[1]
sys.path[:0] = [REPO, os.path.join(REPO, "apps", "web")]
from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(REPO, ".env"))
import numpy as np  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from modules.Ironbeam import flow_v2, flow_v3  # noqa: E402
from packages.shared import flow_heat as fh  # noqa: E402

engine = create_engine(os.environ["BACKFILL_DATABASE_URL"], pool_size=1, max_overflow=0,
                       connect_args={"options": "-c default_transaction_read_only=on"})
with engine.connect() as c:
    assert c.execute(text("SHOW default_transaction_read_only")).scalar() == "on"
    assert c.execute(text("SELECT current_user")).scalar() == "dash_backfill_writer"
    train = [r[0] for r in c.execute(text(
        "SELECT DISTINCT (minute_utc AT TIME ZONE 'America/Los_Angeles')::date FROM es_flow_price_1m "
        "WHERE source = 'databento' ORDER BY 1"))]
    live = [r[0] for r in c.execute(text(
        "SELECT DISTINCT (ts_utc AT TIME ZONE 'America/Los_Angeles')::date FROM ironbeam_es_trades "
        "WHERE symbol = (SELECT symbol FROM ironbeam_es_trades ORDER BY ts_utc DESC LIMIT 1) "
        "AND ts_utc >= now() - interval '3 days' ORDER BY 1"))]
HOLDOUT = (dt.date(2026, 6, 6), dt.date(2026, 10, 26))  # B4: never read
sessions = train[::10] + [d for d in live if d.weekday() < 5 and not HOLDOUT[0] <= d <= HOLDOUT[1]]
print("sessions", [str(d) for d in sessions], flush=True)

pct = lambda a, q: float(np.percentile(a, q)) if len(a) else None
out, all_c = [], []
for d in sessions:
    p, _ = flow_v3.build_cells_payload(engine, str(d), session="RTH", until_pt=None, params=fh.AbsorbParams(),
                                       session_date=d, session_note=None)
    rth = next((s for s in p["segments"] if s["name"] == "rth"), None)
    if rth is None:
        out.append({"date": str(d), "note": p["notes"]}); continue
    marks = rth["marks"] or []
    cont = [mk["aggressor"] + mk["merged_vol"] for mk in marks]
    all_c += cont
    conc, young = [], []
    for T in range(rth["start"] + 60, min(rth["end"], int(dt.datetime.now(dt.timezone.utc).timestamp())) + 1, 60):
        conf = [mk for mk in marks if mk["confirm_time"] <= T]
        conc.append(len(conf)); young.append(sum(1 for mk in conf if T - mk["time"] < 3600))
    conc, young = np.array(conc), np.array(young)
    ncell_min = len(set(rth["cells"]["t"]))
    out.append({"date": str(d), "tier": rth["tier"], "source": rth["source"], "symbol": rth["symbol"],
                "marks": len(marks), "sellers": sum(mk["kind"] == "sellers_absorbed" for mk in marks),
                "contracts_p50": pct(cont, 50), "contracts_max": max(cont) if cont else None,
                "held_p50": pct([mk["held_min"] for mk in marks], 50),
                "broken_share": (sum(mk["break_time"] is not None for mk in marks) / len(marks)) if marks else None,
                "confirmed_gt8_share": float((conc > 8).mean()) if len(conc) else None,
                "young_max": int(young.max()) if len(young) else 0,
                "young_gt8_share": float((young > 8).mean()) if len(young) else None,
                "cell_minutes": ncell_min})
    print(json.dumps(out[-1]), flush=True)
summary = {"n_levels": len(all_c), "contracts_p10": pct(all_c, 10), "p25": pct(all_c, 25), "p50": pct(all_c, 50),
           "p75": pct(all_c, 75), "p90": pct(all_c, 90), "p99": pct(all_c, 99), "max": max(all_c) if all_c else None}
print(json.dumps(summary))
json.dump({"sessions": out, "contracts": summary}, open(os.path.join(os.path.dirname(__file__), "q_levels.out.json"), "w"), indent=1)
