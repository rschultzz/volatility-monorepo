"""CR-DV — record the 2026-10-30 RTH fixture for the live-marks tests (read-only).

cells: (t, p ticks, buy, sell) from ironbeam_es_trades XCME:ES.Z26 (same sums as flow_v3.fetch_trade_cells);
secs : per-second (ts, h, l, last receipt epoch) from ironbeam_es_trades, so tests can build the 1-minute bars as of
       any moment (the flow-1s worker's rows, built from the same trades).
Writes packages/shared/tests/fixtures/cr_dv_trades_1030_rth.npz
"""
import sys, datetime as dt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
import numpy as np
from db import conn
from packages.shared import flow_heat as fh
U = dt.timezone.utc; SYM = "XCME:ES.Z26"
A, B = dt.datetime(2026,10,30,13,30,tzinfo=U), dt.datetime(2026,10,30,20,0,tzinfo=U)
c = conn()
r = np.array(c.execute("""select (floor(extract(epoch from ts_utc)/60)*60)::bigint, round(price*4)::bigint,
        coalesce(sum(size) filter (where aggressor_side='BUY'),0)::float8, coalesce(sum(size) filter (where aggressor_side='SELL'),0)::float8,
        max(extract(epoch from received_at))::float8
        from ironbeam_es_trades where symbol=%s and ts_utc>=%s and ts_utc<%s group by 1,2 order by 1,2""", (SYM, A, B)).fetchall(), dtype=float)
s = np.array(c.execute("""select floor(extract(epoch from ts_utc))::bigint, max(price), min(price), max(extract(epoch from received_at))::float8
        from ironbeam_es_trades where symbol=%s and ts_utc>=%s and ts_utc<%s group by 1 order by 1""", (SYM, A, B)).fetchall(), dtype=float)
out = ROOT / "packages/shared/tests/fixtures/cr_dv_trades_1030_rth.npz"
np.savez_compressed(out, cell_t=r[:,0].astype(np.int64), cell_p=r[:,1].astype(np.int64), cell_buy=r[:,2], cell_sell=r[:,3],
                    cell_rcv=r[:,4], sec_ts=s[:,0].astype(np.int64), sec_h=s[:,1], sec_l=s[:,2], sec_rcv=s[:,3])
print(out, len(r), "cells", len(s), "seconds", out.stat().st_size, "bytes")
