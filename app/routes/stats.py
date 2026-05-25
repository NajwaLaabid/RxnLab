"""Read-only traffic stats from the page_views table, guarded by a token.

Disabled (404) unless RXNLAB_STATS_TOKEN is set and matches ?token=. Bots are
excluded by default; pass ?include_bots=1 to see everything.
"""
from sqlalchemy import Date, cast, distinct, func, select

from flask import Blueprint, current_app, jsonify, request

from app.db import db_enabled, get_session
from app.models_db import PageView

bp = Blueprint('stats', __name__)


@bp.route('/api/stats')
def stats():
    token = current_app.config.get('STATS_TOKEN')
    if not token or request.args.get('token') != token:
        return jsonify({'error': 'not found'}), 404
    if not db_enabled():
        return jsonify({'error': 'db disabled'}), 503

    include_bots = request.args.get('include_bots') == '1'

    def humans(stmt):
        return stmt if include_bots else stmt.where(PageView.is_bot.is_(False))

    with get_session() as s:
        total = s.scalar(humans(select(func.count()).select_from(PageView)))
        uniques = s.scalar(humans(select(func.count(distinct(PageView.session_id)))))
        bot_total = s.scalar(
            select(func.count()).select_from(PageView).where(PageView.is_bot.is_(True))
        )
        day = cast(PageView.created_at, Date).label('day')
        by_day = s.execute(
            humans(select(
                day,
                func.count().label('views'),
                func.count(distinct(PageView.session_id)).label('uniques'),
            )).group_by(day).order_by(day)
        ).all()
        top_paths = s.execute(
            humans(select(PageView.path, func.count().label('views')))
            .group_by(PageView.path).order_by(func.count().desc()).limit(20)
        ).all()

    return jsonify({
        'total_pageviews': total,
        'unique_visitors': uniques,
        'bot_pageviews': bot_total,
        'bots_included': include_bots,
        'by_day': [
            {'day': str(r.day), 'views': r.views, 'uniques': r.uniques} for r in by_day
        ],
        'top_paths': [{'path': r.path, 'views': r.views} for r in top_paths],
    })
