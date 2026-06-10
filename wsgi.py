"""WSGI entry point — gunicorn loads `application` from this module."""
from app import create_app

application = create_app()


if __name__ == "__main__":
    # Local dev only (gunicorn imports `application` and never runs this block, so prod
    # is untouched). On macOS the default 'spawn' start method breaks syntheseus'
    # RootAligned: its canonicalization Pool worker reads a module global that's only set
    # in the parent, so spawned workers NameError. Linux/prod default to 'fork', which
    # inherits it — match that here for parity with prod.
    import sys
    if sys.platform == "darwin":
        import multiprocessing
        multiprocessing.set_start_method("fork", force=True)
    application.run(debug=True, host='0.0.0.0', port=8080)
