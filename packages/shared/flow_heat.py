"""CR-DC — Flow v3 price cells and absorption marks (visual aids, untested as signals).

A cell is one 1-minute bar x one 0.25-pt price, with two values: aggressive SELL volume (market sells
hitting the bid at that price) and aggressive BUY volume (market buys lifting the offer). Prices are
carried as integer ticks (price / 0.25) so cell keys never depend on float rounding.

Three ways to build cells (spec A1.3, DC17), each returning the same frame (t, p, buy, sell):
  - `cells_from_price_1m`: rows of the exact `es_flow_price_1m` table (CR-DD);
  - `cells_from_trades`:   individual trades (`ironbeam_es_trades`), each at its own price (exact);
  - `cells_from_seconds`:  1-second flow, each second's sell volume at its LOW and buy volume at its
    HIGH (approximate: Step 0 Q3 measured 77 % same-cell vs exact, sells biased -0.16 pts).

`absorption_marks` implements the spec's DC9 reading (A1.2: callers only pass exact cells). Every
mark at time t uses cells and bars before the mark's confirmation only.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
import pandas as pd

TICK = 0.25
CELL_COLUMNS = ("t", "p", "buy", "sell")


@dataclass(frozen=True)
class AbsorbParams:
    k: float = 3.0          # aggressor volume at P >= k x the running median cell volume of that side
    m: int = 3              # nothing trades beyond P -/+ 1 tick for the cell's minute plus the next m minutes
    min_cells: int = 15     # no mark until the side has this many prior non-zero cells


def to_ticks(price, floor: bool = False) -> np.ndarray:
    x = np.asarray(price, dtype=float) / TICK
    return (np.floor(x + 1e-9) if floor else np.rint(x)).astype(np.int64)


def _empty() -> pd.DataFrame:
    return pd.DataFrame({"t": pd.Series(dtype=np.int64), "p": pd.Series(dtype=np.int64),
                         "buy": pd.Series(dtype=float), "sell": pd.Series(dtype=float)})


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty()
    out = df.groupby(["t", "p"], sort=True, as_index=False)[["buy", "sell"]].sum()
    out = out[(out["buy"] > 0) | (out["sell"] > 0)]
    out["t"] = out["t"].astype(np.int64)
    out["p"] = out["p"].astype(np.int64)
    return out[list(CELL_COLUMNS)].reset_index(drop=True)


def minute_of(epoch) -> np.ndarray:
    return (np.asarray(epoch, dtype=np.int64) // 60) * 60


def cells_from_seconds(ts, high, low, buy, sell) -> pd.DataFrame:
    """1-s rule: each second's sell volume at that second's LOW, buy volume at its HIGH."""
    t = minute_of(ts)
    sells = pd.DataFrame({"t": t, "p": to_ticks(low), "buy": 0.0, "sell": np.asarray(sell, dtype=float)})
    buys = pd.DataFrame({"t": t, "p": to_ticks(high), "buy": np.asarray(buy, dtype=float), "sell": 0.0})
    return _finish(pd.concat([sells, buys], ignore_index=True))


def cells_from_trades(ts, price, size, side) -> pd.DataFrame:
    """Exact: each trade at its own price; `side` 'BUY' / 'SELL' (anything else is not drawn)."""
    side = np.asarray(side).astype(str)
    size = np.asarray(size, dtype=float)
    df = pd.DataFrame({"t": minute_of(ts), "p": to_ticks(price),
                       "buy": np.where(side == "BUY", size, 0.0), "sell": np.where(side == "SELL", size, 0.0)})
    return _finish(df)


def cells_from_price_1m(minute_epoch, price, buy, sell) -> pd.DataFrame:
    """Exact: `es_flow_price_1m` rows; prices off the 0.25 grid are floored to their cell (A1 readings)."""
    df = pd.DataFrame({"t": minute_of(minute_epoch), "p": to_ticks(price, floor=True),
                       "buy": np.asarray(buy, dtype=float), "sell": np.asarray(sell, dtype=float)})
    return _finish(df)


def cut_cells(cells: pd.DataFrame, start_epoch: int, end_epoch: int) -> pd.DataFrame:
    return cells[(cells["t"] >= start_epoch) & (cells["t"] < end_epoch)].reset_index(drop=True)


def absorption_marks(cells: pd.DataFrame, bars: pd.DataFrame, start_epoch: int, end_epoch: int,
                     params: AbsorbParams = AbsorbParams(), provisional: bool = False) -> list[dict]:
    """'Sellers absorbed' / 'Buyers absorbed' marks (spec DC9), for cells with start <= t < end.

    cells: (t, p ticks, buy, sell); bars: 1-minute bars indexed by bar-start epoch with columns h, l
    (the segment's flow highs / lows). A sell cell at minute t, price P qualifies when its sell volume
    >= k x the median of all earlier minutes' non-zero sell cells (>= min_cells of them), and the low
    of bars t .. t + m minutes is >= P - 1 tick; only the lowest qualifying P per minute is kept. The
    mark is confirmed at t + (m + 1) minutes, which must be <= end. A qualifying cell within 1 tick of
    an earlier, still unbroken mark of the same side merges into it. Break = the first later bar with
    low < P - 1 tick (sellers) / high > P + 1 tick (buyers), before `end`. Mirror for buyers.

    CR-DV: live callers pass `end` = the last closed minute, so nothing still forming is read. With
    `provisional`, a candidate whose hold window has not yet elapsed (confirm > end) and that no bar before
    `end` has broken is returned with `provisional: True` (it adds no volume to an earlier mark it would merge
    into, and is not returned when within 1 tick of one); confirmed marks carry `provisional: False`.
    """
    c = cells[(cells["t"] >= start_epoch) & (cells["t"] < end_epoch)]
    if c.empty or bars is None or len(bars) == 0:
        return []
    bt = np.asarray(bars.index, dtype=np.int64)
    bh = bars["h"].to_numpy(dtype=float)
    bl = bars["l"].to_numpy(dtype=float)
    k_ok = bt < end_epoch
    bt, bh, bl = bt[k_ok], bh[k_ok], bl[k_ok]
    hold_s = (params.m + 1) * 60
    out = []
    for side, opp in (("sell", "buy"), ("buy", "sell")):
        prior: list[float] = []
        marks: list[dict] = []
        for t, g in c.groupby("t", sort=True):
            t = int(t)
            vols = g[side].to_numpy(dtype=float)
            nz = vols[vols > 0]
            if len(prior) >= params.min_cells and len(nz):
                n = len(prior)
                med = prior[n // 2] if n % 2 else 0.5 * (prior[n // 2 - 1] + prior[n // 2])
                q = g[vols >= params.k * med]
                q = q[q[side] > 0]
                confirm = t + hold_s
                pending = confirm > end_epoch
                if not q.empty and (not pending or provisional):
                    row = q.loc[q["p"].idxmin()] if side == "sell" else q.loc[q["p"].idxmax()]
                    p = int(row["p"])
                    price = p * TICK
                    w = (bt >= t) & (bt < confirm)
                    held = (bl[w].min() >= price - TICK - 1e-9) if side == "sell" else (bh[w].max() <= price + TICK + 1e-9)
                    if w.any() and held:
                        merged = False
                        for mk in marks:
                            unbroken = mk["break_time"] is None or mk["break_time"] > t
                            if unbroken and abs(mk["p"] - p) <= 1:
                                if not pending:
                                    mk["merged_vol"] += float(row[side])
                                merged = True
                                break
                        if not merged:
                            after = bt >= confirm
                            brk = (bl < price - TICK - 1e-9) if side == "sell" else (bh > price + TICK + 1e-9)
                            hit = np.flatnonzero(after & brk)
                            marks.append({"kind": "sellers_absorbed" if side == "sell" else "buyers_absorbed",
                                          "time": t, "p": p, "price": price, "confirm_time": confirm,
                                          "aggressor": float(row[side]), "opposite": float(row[opp]),
                                          "median": float(med), "merged_vol": 0.0,
                                          "break_time": int(bt[hit[0]]) if len(hit) else None,
                                          "provisional": pending})
            for v in nz:
                bisect.insort(prior, float(v))
        out += marks
    for mk in out:
        stop = mk["break_time"] if mk["break_time"] is not None else end_epoch
        mk["held_min"] = int((stop - mk["time"]) // 60)
    out.sort(key=lambda mk: (mk["time"], mk["kind"]))
    return out
