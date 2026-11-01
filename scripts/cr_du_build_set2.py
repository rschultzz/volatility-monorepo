#!/usr/bin/env python3
"""CR-DU item 0 — build set 2 (`set-2:cp-calls`): the CR-CP A1-primary D2 calls on 2026 train that are not in CR-CZ's
`cr-cz:cp-ranges` sample. READ-ONLY (DB reads via cr_cp_lib), train only.

    /Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python scripts/cr_du_build_set2.py

Cards are built exactly as CR-CZ's (`cr_cz_build_gallery.card`: same meta and outcome fields, entry_ts_pt = the close of
the decision bar, outcomes stored and hidden until reveal). Set 2 = the 235 frozen calls minus CR-CZ's seeded sample,
minus any call whose (session, decision close, side) equals a CR-CZ card (CR-CZ card 31 duplicated card 30 that way);
duplicates inside set 2 are kept once. Numbered 1..n by session and decision bar.

Writes scripts/logs/cr_du/set-2-cp-calls.json, git-ignored because it holds the outcomes; prints counts only (no
outcome). Load with scripts/load_setup_candidates.py --run-cr CR-DU.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import cr_cp_lib as C
import cr_cz_build_gallery as Z

CR_ID, SET_NAME = "CR-DU", "set-2:cp-calls"
GALLERY = C.REPO_ROOT / "scripts" / "logs" / "cr_du" / "set-2-cp-calls.json"   # git-ignored: holds the outcomes


def main():
    D, a, FZ = Z.frozen_schedule()
    picked = set(Z.sample_index(a))
    sig = lambda r: (str(r.d), int(r.t), int(r.side))                    # (session, decision bar, side)
    cz = {sig(r) for r in a.loc[sorted(picked)].itertuples()}
    rest = a.drop(index=list(picked)).sort_values(["d", "t"]).reset_index(drop=True)
    rows, seen, dup_cz, dup_self = [], set(), 0, 0
    for r in rest.itertuples():
        s = sig(r)
        if s in cz:
            dup_cz += 1
            continue
        if s in seen:
            dup_self += 1
            continue
        seen.add(s)
        rows.append(r)
    events = [Z.card(n, r, D, FZ) for n, r in enumerate(rows, start=1)]
    assert all(e["session_date"] <= str(Z.TRAIN_END) for e in events)   # B4
    GALLERY.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(cr_id=CR_ID, set_name=SET_NAME, level_frame="es", generated_at=datetime.now(timezone.utc).isoformat(),
               source="CR-CP A1 primary D2 calls not in cr-cz:cp-ranges (scripts/logs/cr_cp/run_primary_trades.parquet), "
                      "Databento 1-min bars; cards as scripts/cr_cz_build_gallery.py",
               sample=dict(population=len(a), cr_cz_sample=len(picked), same_as_cr_cz_card=dup_cz, duplicate_in_set=dup_self),
               events=events)
    GALLERY.write_text(json.dumps(doc, indent=1))
    n_s = sum(e["direction"] == "short" for e in events)
    print(f"population {len(a)}, CR-CZ sample {len(picked)}, rest {len(rest)}: same call as a CR-CZ card {dup_cz}, "
          f"duplicate inside set 2 {dup_self}")
    print(f"wrote {GALLERY}: {len(events)} events (short {n_s}, long {len(events) - n_s}), "
          f"{len({e['session_date'] for e in events})} sessions, {events[0]['session_date']} .. {events[-1]['session_date']}")


if __name__ == "__main__":
    main()
