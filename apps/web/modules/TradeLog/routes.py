"""
apps/web/modules/TradeLog/routes.py

Flask Blueprint for the Trade Log feature.

REGISTRATION — add to your main Flask app factory:

    from apps.web.modules.TradeLog.routes import trade_log_bp
    app.register_blueprint(trade_log_bp)

All routes are prefixed  /api/trade-log/
"""

import logging

from flask import Blueprint, jsonify, request

from . import service

log = logging.getLogger(__name__)

trade_log_bp = Blueprint('trade_log', __name__, url_prefix='/api/trade-log')


# ── Upload ────────────────────────────────────────────────────

@trade_log_bp.post('/upload')
def upload():
    """
    POST /api/trade-log/upload
    Body: multipart/form-data with field 'file' (CSV)

    Parses the TV CSV, FIFO-pairs fills into round-trip trades,
    inserts new records, and auto-computes market context.
    """
    if 'file' not in request.files:
        return jsonify({'ok': False, 'error': 'No file in request'}), 400

    f = request.files['file']
    if not f.filename:
        return jsonify({'ok': False, 'error': 'Empty filename'}), 400

    try:
        result = service.upload_csv(f.read())
        status = 200 if result.get('ok') else 400
        return jsonify(result), status
    except Exception as e:
        log.exception('Trade log upload failed')
        return jsonify({'ok': False, 'error': str(e)}), 500


# ── Trade list ────────────────────────────────────────────────

@trade_log_bp.get('/trades')
def list_trades():
    """
    GET /api/trade-log/trades[?date=YYYY-MM-DD]

    Returns all trades, newest date first.
    Optional ?date= filter narrows to one session.
    """
    date_filter = request.args.get('date') or None
    try:
        trades = service.list_trades(date_filter)
        return jsonify({'ok': True, 'trades': trades})
    except Exception as e:
        log.exception('list_trades failed')
        return jsonify({'ok': False, 'error': str(e)}), 500


# ── Single trade ──────────────────────────────────────────────

@trade_log_bp.get('/trades/<int:trade_id>')
def get_trade(trade_id):
    """GET /api/trade-log/trades/<id>"""
    try:
        trade = service.get_trade(trade_id)
        if trade is None:
            return jsonify({'ok': False, 'error': 'Not found'}), 404
        return jsonify({'ok': True, 'trade': trade})
    except Exception as e:
        log.exception('get_trade failed')
        return jsonify({'ok': False, 'error': str(e)}), 500


# ── Update annotation ─────────────────────────────────────────

@trade_log_bp.patch('/trades/<int:trade_id>')
def update_trade(trade_id):
    """
    PATCH /api/trade-log/trades/<id>
    Body: JSON with any subset of:
        setup_start_ts_pt   — datetime-local string (YYYY-MM-DDTHH:MM), interpreted as PT
        setup_target_ts_pt  — datetime-local string
        setup_direction     — 'long' | 'short' | ''
        notes               — free text
    """
    body = request.get_json(silent=True) or {}
    try:
        trade = service.update_trade(trade_id, body)
        if trade is None:
            return jsonify({'ok': False, 'error': 'Not found'}), 404
        return jsonify({'ok': True, 'trade': trade})
    except Exception as e:
        log.exception('update_trade failed')
        return jsonify({'ok': False, 'error': str(e)}), 500


# ── Recompute context ─────────────────────────────────────────

@trade_log_bp.post('/trades/<int:trade_id>/recompute_context')
def recompute_context(trade_id):
    """
    POST /api/trade-log/trades/<id>/recompute_context

    Refetches market context (IV, skew deltas, minutes-to-close)
    using the trade's current setup_start_ts_pt / setup_target_ts_pt.
    """
    try:
        result = service.recompute_context(trade_id)
        status = 200 if result.get('ok') else 404
        return jsonify(result), status
    except Exception as e:
        log.exception('recompute_context failed')
        return jsonify({'ok': False, 'error': str(e)}), 500


# ── Delete trade ──────────────────────────────────────────────

@trade_log_bp.delete('/trades/<int:trade_id>')
def delete_trade(trade_id):
    """DELETE /api/trade-log/trades/<id>"""
    try:
        deleted = service.delete_trade(trade_id)
        if not deleted:
            return jsonify({'ok': False, 'error': 'Not found'}), 404
        return jsonify({'ok': True})
    except Exception as e:
        log.exception('delete_trade failed')
        return jsonify({'ok': False, 'error': str(e)}), 500


# ── Aggregate stats ───────────────────────────────────────────

@trade_log_bp.get('/aggregate')
def aggregate():
    """
    GET /api/trade-log/aggregate[?date=YYYY-MM-DD]

    Returns win rate, avg winner/loser, R:R, total P&L, total fees.
    """
    date_filter = request.args.get('date') or None
    try:
        result = service.get_aggregate(date_filter)
        return jsonify(result)
    except Exception as e:
        log.exception('aggregate failed')
        return jsonify({'ok': False, 'error': str(e)}), 500
