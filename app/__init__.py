"""RxnLab Flask application factory."""
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import flask
from flask import g, request


def create_app() -> flask.Flask:
    # Import TensorFlow first, before anything pulls in torch/dgl. TF, torch, and dgl
    # each ship their own OpenMP runtime; whichever loads first wins, and loading TF
    # *after* torch/dgl segfaults the process. Blueprint registration below imports
    # torch (via syntheseus), so TF has to come before all of it. See
    # _warm_classifier_sync for the full rationale. Guarded: no TF ⇒ no MEGAN ⇒ no clash.
    try:
        import tensorflow  # noqa: F401
    except Exception:
        pass

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / 'DiffAlign'))

    app = flask.Flask(
        __name__,
        template_folder='templates',
        static_folder='static',
    )

    from app.config import Config
    app.config.from_object(Config)

    if Config.DATABASE_URL:
        from app.db import init_db
        init_db(Config.DATABASE_URL)
        app.logger.info('DB enabled.')
    else:
        app.logger.warning(
            'DATABASE_URL unset — running without DB; feedback will not be persisted.'
        )

    _register_lifecycle(app)

    from app.routes.compare import bp as compare_bp
    from app.routes.feedback import bp as feedback_bp
    from app.routes.health import bp as health_bp
    from app.routes.landing import bp as landing_bp
    from app.routes.predict import bp as predict_bp
    from app.routes.search import bp as search_bp
    from app.routes.stats import bp as stats_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(landing_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(compare_bp)
    app.register_blueprint(feedback_bp)
    app.register_blueprint(stats_bp)

    @app.errorhandler(413)
    def _payload_too_large(_e):
        return flask.jsonify({'error': 'Request body too large.'}), 413

    _warm_classifier_sync()

    return app


def _warm_classifier_sync() -> None:
    """Initialize the heavy native runtimes on the MAIN thread at boot, in a safe order,
    before any request is served. This pre-loads rxn_insight/RXNMapper (a ~7s import) so
    the first description doesn't pay it, but it is also load-bearing for correctness:

    - RXNMapper pulls in TensorFlow, and the older async (daemon-thread) warm-up
      initialized TF/XLA off the main thread, which segfaults. gunicorn also serves
      requests in worker threads, so TF must be initialized on the main thread here, or
      the first classify in a worker thread can crash.
    - dgl (LocalRetro) and TF each ship their own OpenMP runtime; importing dgl *before*
      TF segfaults. Importing TF first here makes every later dgl import safe.

    The model-comparison endpoint co-loads dgl + TF stacks in one process, which makes
    both failure modes systematic — hence the synchronous, main-thread warm-up. (TF
    itself is imported first thing in create_app, before any torch/dgl import.)
    Best-effort: a failure here must not stop the app from booting."""
    try:
        from app.search import warm_classifier
        warm_classifier()
    except Exception:
        pass


# UA substrings that mark a non-human client. Matched case-insensitively;
# "bot" covers Googlebot/GPTBot/etc., "kube-probe" covers the rahti health probe.
_BOT_UA_MARKERS = (
    'bot', 'crawl', 'spider', 'slurp', 'monitor', 'curl', 'wget',
    'python-requests', 'python-urllib', 'go-http-client', 'httpx',
    'headless', 'kube-probe', 'uptime', 'scan', 'facebookexternalhit',
    'embedly', 'preview', 'fetch',
)


def _looks_like_bot(ua: str) -> bool:
    ua = (ua or '').lower()
    return any(marker in ua for marker in _BOT_UA_MARKERS)


def _register_lifecycle(app: flask.Flask) -> None:
    from app.db import db_enabled, get_session
    from app.models_db import SessionRow

    @app.before_request
    def _open_session_and_cookie():
        g.db = get_session() if db_enabled() else None
        g.session_id = None
        g.cookie_to_set = None
        if g.db is None:
            return

        cookie = request.cookies.get(app.config['SESSION_COOKIE_NAME'])
        sid = None
        if cookie:
            try:
                sid = uuid.UUID(cookie)
            except ValueError:
                sid = None

        now = datetime.now(timezone.utc)
        ua = (request.headers.get('User-Agent') or '')[:500] or None

        if sid is not None:
            row = g.db.get(SessionRow, sid)
            if row is not None:
                row.last_seen_at = now
            else:
                # Cookie carries an unknown UUID — recreate row so feedback joins succeed.
                g.db.add(SessionRow(
                    session_id=sid, created_at=now, last_seen_at=now, user_agent=ua
                ))
        else:
            sid = uuid.uuid4()
            g.db.add(SessionRow(
                session_id=sid, created_at=now, last_seen_at=now, user_agent=ua
            ))
            g.cookie_to_set = str(sid)

        g.db.commit()
        g.session_id = sid

    @app.before_request
    def _record_pageview():
        # Runs after _open_session_and_cookie, so g.session_id is set. Records one
        # row per HTML page GET; analytics must never break a request, so failures
        # are swallowed. Bots are tagged (not dropped) so they can be audited.
        db = getattr(g, 'db', None)
        if db is None or request.method != 'GET':
            return
        path = request.path
        if (path.startswith('/static') or path.startswith('/api')
                or path in ('/health', '/favicon.ico')):
            return

        from app.models_db import PageView
        # Read the header directly: in Werkzeug 3.x request.user_agent is falsy even
        # when a UA is present, so `... if request.user_agent` would drop every UA.
        ua = request.headers.get('User-Agent', '')
        try:
            db.add(PageView(
                session_id=getattr(g, 'session_id', None),
                path=path[:300],
                is_bot=_looks_like_bot(ua),
            ))
            db.commit()
        except Exception:
            db.rollback()

    @app.after_request
    def _set_session_cookie(response):
        cookie_to_set = getattr(g, 'cookie_to_set', None)
        if cookie_to_set:
            response.set_cookie(
                app.config['SESSION_COOKIE_NAME'],
                cookie_to_set,
                max_age=app.config['SESSION_COOKIE_MAX_AGE'],
                httponly=True,
                samesite='Lax',
            )
        return response

    @app.teardown_request
    def _close_db(exc):
        db = getattr(g, 'db', None)
        if db is not None:
            try:
                if exc is not None:
                    db.rollback()
            finally:
                db.close()
