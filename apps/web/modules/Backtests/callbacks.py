# apps/web/modules/Backtests/callbacks.py
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text as sql_text

from dash import Input, Output, State, html, ctx, ALL, no_update
from dash.exceptions import PreventUpdate

from packages.backtests.registry import get_strategies

from .components import (
    BT_STRATEGY_ID,
    BT_EXPECTED_TOGGLE_ID,
    BT_MODE_TOGGLE_ID,
    BT_DATE_RANGE_ID,
    BT_PARAM_PANEL_ID,
    BT_RUN_BTN_ID,
    BT_SUMMARY_ID,
    BT_TRADES_STORE_ID,
    BT_TRADES_TABLE_ID,
    BT_DOWNLOAD_BTN_ID,
    BT_DOWNLOAD_ID,
    render_params,
)

# Dashboard control IDs
DASH_TRADE_DATE_ID = "trade-date"
DASH_EXPIRATION_ID = "expiration-date-pick"
DASH_TIME_SLICES_ID = "smile-time-input"

# Main tabs ID / values (must match app.py)
MAIN_TABS_ID = "main-tabs"
TAB_DASHBOARD = "tab-dashboard"


@lru_cache(maxsize=1)
def _engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL env var is not set")
    return create_engine(db_url)


def _load_minutes(start_date: str, end_date: str) -> pd.DataFrame:
    view = os.getenv("ES_MINUTES_FEATURES_VIEW", "es_minutes_with_features")
    q = f"""
    SELECT *
    FROM {view}
    WHERE trade_date >= :start_date
      AND trade_date <= :end_date
    ORDER BY trade_date, ts_utc
    """
    with _engine().connect() as conn:
        return pd.read_sql_query(sql_text(q), conn, params={"start_date": start_date, "end_date": end_date})


def _summary_block(summary: Dict[str, Any]) -> html.Div:
    n = summary.get("n_trades", 0)
    win_rate = summary.get("win_rate", 0.0)
    avg_r = summary.get("avg_r", 0.0)
    total_r = summary.get("total_r", 0.0)

    def row(lbl: str, val: str) -> html.Div:
        return html.Div(
            [
                html.Div(lbl, style={"color": "#9ca3af", "fontSize": "12px"}),
                html.Div(val, style={"color": "#e5e7eb", "fontSize": "18px", "fontWeight": "800"}),
            ],
            style={"minWidth": "160px"},
        )

    return html.Div(
        [
            row("Trades", f"{n:,}"),
            row("Win rate", f"{win_rate * 100.0:.1f}%"),
            row("Avg R", f"{avg_r:.2f}"),
            row("Total R", f"{total_r:.2f}"),
        ],
        style={"display": "flex", "gap": "18px", "flexWrap": "wrap"},
    )


def _parse_yyyy_mm_dd(v: Any) -> Optional[str]:
    if v is None or v == "":
        return None
    try:
        return pd.to_datetime(v).date().isoformat()
    except Exception:
        s = str(v)
        return s[:10] if len(s) >= 10 else None


def _parse_hhmm(v: Any) -> Optional[str]:
    if v is None or v == "":
        return None
    s = str(v)
    m = re.search(r"(\d{1,2}):(\d{2})", s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = m.group(2)
    return f"{hh:02d}:{mm}"


def register_callbacks(app, expected_toggle_id: str):
    # --- Keep main Expected(SS) and backtest Expected(SS) in sync ---
    @app.callback(
        Output(expected_toggle_id, "value"),
        Output(BT_EXPECTED_TOGGLE_ID, "value"),
        Input(expected_toggle_id, "value"),
        Input(BT_EXPECTED_TOGGLE_ID, "value"),
        prevent_initial_call=True,
    )
    def _sync_expected_ss(main_val, bt_val):
        trig = ctx.triggered_id
        if trig == expected_toggle_id:
            return main_val, main_val
        if trig == BT_EXPECTED_TOGGLE_ID:
            return bt_val, bt_val
        raise PreventUpdate

    # --- Update param panel + description when strategy changes ---
    @app.callback(
        Output(BT_PARAM_PANEL_ID, "children"),
        Output("bt-strategy-desc", "children"),
        Input(BT_STRATEGY_ID, "value"),
    )
    def _on_strategy_change(strategy_key: str):
        strategies = get_strategies()
        if not strategy_key or strategy_key not in strategies:
            raise PreventUpdate
        spec = strategies[strategy_key]
        return render_params(strategy_key), spec.description

    # --- Backtest Mode toggle: enable/disable single-select radio column ---
    @app.callback(
        Output(BT_TRADES_TABLE_ID, "row_selectable"),
        Output(BT_TRADES_TABLE_ID, "selected_rows", allow_duplicate=True),
        Input(BT_MODE_TOGGLE_ID, "value"),
        prevent_initial_call=True,
    )
    def _toggle_backtest_mode(mode_val: str):
        if mode_val == "on":
            return "single", []
        return None, []

    # --- Run backtest (also clears any selected row) ---
    @app.callback(
        Output(BT_TRADES_STORE_ID, "data"),
        Output(BT_TRADES_TABLE_ID, "data"),
        Output(BT_TRADES_TABLE_ID, "columns"),
        Output(BT_SUMMARY_ID, "children"),
        Output(BT_TRADES_TABLE_ID, "selected_rows", allow_duplicate=True),
        Input(BT_RUN_BTN_ID, "n_clicks"),
        State(BT_STRATEGY_ID, "value"),
        State(BT_DATE_RANGE_ID, "start_date"),
        State(BT_DATE_RANGE_ID, "end_date"),
        State(BT_EXPECTED_TOGGLE_ID, "value"),
        State({"type": "bt-param", "strategy": ALL, "name": ALL}, "value"),
        State({"type": "bt-param", "strategy": ALL, "name": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _run_backtest(_n, strategy_key: str, start_date: str, end_date: str, expected_toggle: str, values, ids):
        if not strategy_key or not start_date or not end_date:
            raise PreventUpdate

        strategies = get_strategies()
        if strategy_key not in strategies:
            raise PreventUpdate

        raw_params: Dict[str, Any] = {}
        if values and ids:
            for v, idd in zip(values, ids):
                if isinstance(idd, dict) and idd.get("strategy") == strategy_key:
                    raw_params[idd.get("name")] = v

        ctx_obj = {"compare_to_expected_ss": (expected_toggle == "on")}

        df = _load_minutes(start_date, end_date)
        trades_df, summary = strategies[strategy_key].run(df, raw_params, ctx_obj)

        if trades_df is None or trades_df.empty:
            return [], [], [], html.Div("No trades found for this run.", style={"color": "#fca5a5"}), []

        out_df = trades_df.copy()
        for c in out_df.columns:
            if pd.api.types.is_datetime64_any_dtype(out_df[c]):
                out_df[c] = out_df[c].astype(str)

        records = out_df.to_dict("records")
        columns = [{"name": c, "id": c} for c in out_df.columns]
        return records, records, columns, _summary_block(summary), []

    # --- Selecting a row pushes values into the Dashboard controls AND switches to Dashboard tab ---
    @app.callback(
        Output(DASH_TRADE_DATE_ID, "date", allow_duplicate=True),
        Output(DASH_EXPIRATION_ID, "date", allow_duplicate=True),
        Output(DASH_TIME_SLICES_ID, "value", allow_duplicate=True),
        Output(MAIN_TABS_ID, "value"),  # switch tabs
        Input(BT_TRADES_TABLE_ID, "selected_rows"),
        State(BT_TRADES_STORE_ID, "data"),
        State(BT_MODE_TOGGLE_ID, "value"),
        prevent_initial_call=True,
    )
    def _load_trade_into_dashboard(selected_rows: List[int], rows: List[Dict[str, Any]], mode_val: str):
        if mode_val != "on":
            raise PreventUpdate
        if not selected_rows or not rows:
            raise PreventUpdate

        i = selected_rows[0]
        if i is None or i < 0 or i >= len(rows):
            raise PreventUpdate

        r = rows[i]

        trade_date = _parse_yyyy_mm_dd(r.get("trade_date") or r.get("entry_trade_date") or r.get("date"))
        expir = _parse_yyyy_mm_dd(r.get("smile_expir_primary") or r.get("expir_date") or r.get("expiration"))

        anchor_hhmm = _parse_hhmm(r.get("anchor_test_ts_pt"))
        confirm_hhmm = _parse_hhmm(r.get("confirm_test_ts_pt"))

        times: List[str] = []
        for t in [anchor_hhmm, confirm_hhmm]:
            if t and t not in times:
                times.append(t)

        # Even if parsing fails, still switch to dashboard
        out_trade = trade_date if trade_date else no_update
        out_expir = expir if expir else no_update
        out_times = times if times else no_update

        return out_trade, out_expir, out_times, TAB_DASHBOARD

    # --- Download CSV ---
    @app.callback(
        Output(BT_DOWNLOAD_ID, "data"),
        Input(BT_DOWNLOAD_BTN_ID, "n_clicks"),
        State(BT_TRADES_STORE_ID, "data"),
        prevent_initial_call=True,
    )
    def _download(_n, rows):
        if not rows:
            raise PreventUpdate
        df = pd.DataFrame(rows)
        return dict(content=df.to_csv(index=False), filename="backtest_trades.csv")
