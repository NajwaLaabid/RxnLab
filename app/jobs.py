"""In-process background jobs for multi-step search (Phase 3a).

A search runs 30s–minutes, which would block gunicorn's single sync worker and
blow past proxy timeouts if done inline. So we run it in a background thread and
expose a job id the frontend polls. The final route tree is also persisted as a
``prediction_runs`` row so feedback/analytics keep working.

Single-worker assumption: the Procfile/Dockerfile run ``gunicorn --workers 1``,
so a module-level dict is shared across all requests. If we ever scale to >1
worker, this must move to a DB/Redis-backed store.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Optional

# Protect the box (2 vCPU): cap concurrent searches; reject beyond it.
MAX_CONCURRENT = 2
# Keep finished jobs around briefly so the frontend can fetch the result, then GC.
JOB_TTL_S = 1800.0

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _gc_locked() -> None:
    now = time.monotonic()
    stale = [
        jid for jid, j in _JOBS.items()
        if j["status"] in ("done", "error") and now - j["updated"] > JOB_TTL_S
    ]
    for jid in stale:
        del _JOBS[jid]


def _n_active_locked() -> int:
    return sum(1 for j in _JOBS.values() if j["status"] in ("queued", "running"))


def start_search_job(
    *,
    model_id: str,
    product_smiles: str,
    catalog_id: str,
    session_id: Optional[uuid.UUID],
    budget: dict,
) -> Optional[str]:
    """Spawn a background search. Returns the job id, or ``None`` if the box is
    already at ``MAX_CONCURRENT`` searches."""
    with _LOCK:
        _gc_locked()
        if _n_active_locked() >= MAX_CONCURRENT:
            return None
        job_id = str(uuid.uuid4())
        _JOBS[job_id] = {
            "status": "queued",
            "model_id": model_id,
            "catalog_id": catalog_id,
            "product_smiles": product_smiles,
            "result": None,
            "error": None,
            "run_id": None,
            "created": time.monotonic(),
            "updated": time.monotonic(),
        }

    t = threading.Thread(
        target=_run,
        args=(job_id, model_id, product_smiles, catalog_id, session_id, budget),
        daemon=True,
    )
    t.start()
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        return {
            "status": job["status"],
            "model_id": job["model_id"],
            "product_smiles": job["product_smiles"],
            "result": job["result"],
            "error": job["error"],
            "run_id": job["run_id"],
        }


def _set(job_id: str, **fields) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job.update(fields)
        job["updated"] = time.monotonic()


def _run(job_id, model_id, product_smiles, catalog_id, session_id, budget) -> None:
    from app.search import run_search

    _set(job_id, status="running")
    try:
        result = run_search(model_id, product_smiles, catalog_id=catalog_id, **budget)
    except Exception as e:  # surface to the poller; never crash the worker thread
        import traceback
        traceback.print_exc()
        _set(job_id, status="error", error=str(e))
        return

    run_id = _persist(model_id, product_smiles, session_id, result)
    _set(job_id, status="done", result=result, run_id=str(run_id) if run_id else None)


def _persist(model_id, product_smiles, session_id, result) -> Optional[uuid.UUID]:
    """Write the route tree as a prediction_runs row. Background thread → own session."""
    from app.db import db_enabled, get_session
    from app.models_db import PredictionRun

    if not db_enabled() or session_id is None:
        return None
    try:
        with get_session() as s:
            run = PredictionRun(
                session_id=session_id,
                model_id=model_id,
                product_smiles=product_smiles,
                product_inchi_key="",
                params={"search": True, **result.get("stats", {})},
                predictions=result,
                latency_ms=result.get("stats", {}).get("elapsed_ms"),
            )
            s.add(run)
            s.commit()
            return run.run_id
    except Exception:
        import traceback
        traceback.print_exc()
        return None
