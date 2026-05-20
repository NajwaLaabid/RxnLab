"""PubChem lookup with a DB-backed L2 cache on top of the evaluation layer's L1.

Flow per compound:
  1. Canonicalize SMILES.
  2. Look up in Postgres (pubchem_cache table) — ~10ms, shared across all
     gunicorn workers and pods, durable across rollouts.
  3. On L2 miss, defer to evaluation.pubchem_lookup which checks its own
     in-process L1 dict and finally hits PubChem.
  4. On a successful fetch, write back to Postgres so other workers/pods
     don't pay the cost again.

If DATABASE_URL is unset (db_enabled() == False), this transparently falls
through to the evaluation layer's L1-only behavior.
"""
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from evaluation.pubchem_lookup import (
    _canonical,
    get_compound_profile as _l1_get_compound_profile,
)

__all__ = ['get_compound_profile', 'lookup_all_compounds']


def _db_get(key: str) -> dict | None:
    from app.db import get_session
    session = get_session()
    if session is None:
        return None
    try:
        from app.models_db import PubChemCache
        with session as s:
            row = s.get(PubChemCache, key)
            if row is None:
                return None
            return row.data
    except Exception:
        return None


def _db_set(key: str, info: dict) -> None:
    from app.db import get_session
    session = get_session()
    if session is None:
        return
    try:
        from app.models_db import PubChemCache
        data = {k: v for k, v in info.items() if k != 'smiles'}
        with session as s:
            stmt = pg_insert(PubChemCache).values(smiles=key, data=data)
            stmt = stmt.on_conflict_do_update(
                index_elements=['smiles'],
                set_={'data': stmt.excluded.data, 'cached_at': func.now()},
            )
            s.execute(stmt)
            s.commit()
    except Exception:
        pass


def get_compound_profile(smiles: str) -> dict:
    key = _canonical(smiles)
    hit = _db_get(key)
    if hit is not None:
        return {**hit, 'smiles': smiles}

    info = _l1_get_compound_profile(smiles)
    if info.get('found'):
        _db_set(key, info)
    return info


def lookup_all_compounds(smiles_list: list) -> list:
    """Batch lookup compounds on PubChem in parallel, with L1+L2 caching."""
    if not smiles_list:
        return []
    with ThreadPoolExecutor(max_workers=min(5, len(smiles_list))) as ex:
        results = list(ex.map(get_compound_profile, smiles_list))
    results.sort(key=lambda x: x.get('fame_score', 0), reverse=True)
    return results
