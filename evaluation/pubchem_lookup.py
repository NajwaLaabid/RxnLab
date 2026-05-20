"""PubChem compound profiling and literature statistics.

Hits PUG-REST directly (skipping pubchempy's lazy property fetches), batched
across compounds with a ThreadPool. A process-wide rate limiter keeps us
under PubChem's 5 req/sec cap.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

_DB_PREFIXES = (
    'SCHEMBL', 'DTXSID', 'ZINC', 'AKOS', 'MFCD', 'SBB', 'BBV',
    'BCP', 'BDBM', 'STK', 'HY-', 'CS-', 'FT-',
)


class _RateLimiter:
    """Token-bucket-ish: enforces an upper bound on requests per second across threads."""

    def __init__(self, max_per_sec: int) -> None:
        self._lock = threading.Lock()
        self._min_interval = 1.0 / max_per_sec
        self._next_ok = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            sleep_for = self._next_ok - now
            self._next_ok = max(now, self._next_ok) + self._min_interval
        if sleep_for > 0:
            time.sleep(sleep_for)


_limiter = _RateLimiter(max_per_sec=5)

_cache_lock = threading.Lock()
_cache: dict[str, dict] = {}
_CACHE_MAX = 2048


def _canonical(smiles: str) -> str:
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return smiles
        return Chem.MolToSmiles(m)
    except Exception:
        return smiles


def _pub_get(path: str, timeout: int = 15):
    _limiter.wait()
    try:
        r = requests.get(f"{PUBCHEM_BASE}/{path}", timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def _xref_count(cid: int, xref: str, timeout: int = 30) -> int:
    """Return len() of a single xref list, or 0 on any failure.

    PubChem itself sometimes 504s on very large xref lists (e.g. PatentID
    for ethanol) — matches the prior implementation's silent-zero behavior.
    """
    data = _pub_get(f"compound/cid/{cid}/xrefs/{xref}/JSON", timeout=timeout)
    if not data:
        return 0
    rows = data.get('InformationList', {}).get('Information', [])
    if not rows:
        return 0
    return len(rows[0].get(xref, []) or [])


def _cid_for_smiles(smiles: str, timeout: int = 10) -> int | None:
    # POST so that SMILES special chars (#, /, \, [, ]) don't break the URL.
    _limiter.wait()
    try:
        r = requests.post(
            f"{PUBCHEM_BASE}/compound/smiles/cids/JSON",
            data={'smiles': smiles},
            timeout=timeout,
        )
        if r.status_code != 200:
            return None
        cids = r.json().get('IdentifierList', {}).get('CID', [])
        if cids and cids[0]:
            return cids[0]
    except Exception:
        return None
    return None


def _fetch_compound_profile(smiles: str) -> dict:
    cid = _cid_for_smiles(smiles)
    if not cid:
        return {'smiles': smiles, 'found': False}

    info: dict = {'smiles': smiles, 'found': True, 'cid': cid}

    props_data = _pub_get(
        f"compound/cid/{cid}/property/MolecularFormula,MolecularWeight,IUPACName/JSON"
    )
    props = {}
    if props_data:
        rows = props_data.get('PropertyTable', {}).get('Properties', [])
        if rows:
            props = rows[0]
    info['iupac'] = props.get('IUPACName')
    info['formula'] = props.get('MolecularFormula')
    info['mw'] = props.get('MolecularWeight')

    syn_data = _pub_get(f"compound/cid/{cid}/synonyms/JSON")
    syns: list[str] = []
    if syn_data:
        rows = syn_data.get('InformationList', {}).get('Information', [])
        if rows:
            syns = rows[0].get('Synonym', []) or []
    info['n_synonyms'] = len(syns)
    info['short_names'] = [
        s for s in syns[:30]
        if len(s) < 30 and not any(s.startswith(p) for p in _DB_PREFIXES)
    ][:10]

    # n_sources: count of unique deposit-source names (cheap, bounded — was
    # len(c.sids) before, which fetched the full SID list).
    info['n_sources'] = _xref_count(cid, 'SourceName')

    # Patent + PubMed kept as two calls — the combined xref endpoint times
    # out server-side for popular molecules (e.g. ethanol PatentID alone 504s).
    info['n_patents'] = _xref_count(cid, 'PatentID')
    info['n_pubmed'] = _xref_count(cid, 'PubMedID')

    info['fame_score'] = (
        info['n_sources']
        + info['n_patents'] * 5
        + info['n_pubmed'] * 10
        + info['n_synonyms']
    )
    return info


def get_compound_profile(smiles: str) -> dict:
    """Get a comprehensive profile of a compound from PubChem.

    Returns a dict with: smiles, found, cid, iupac, formula, mw,
    n_synonyms, short_names, n_sources, n_patents, n_pubmed, fame_score.
    Cached by canonical SMILES; transient failures are not cached.
    """
    key = _canonical(smiles)

    with _cache_lock:
        hit = _cache.get(key)
    if hit is not None:
        return {**hit, 'smiles': smiles}

    info = _fetch_compound_profile(key)
    if info.get('found'):
        with _cache_lock:
            if len(_cache) >= _CACHE_MAX:
                _cache.pop(next(iter(_cache)))
            _cache[key] = info
    return {**info, 'smiles': smiles}


def lookup_all_compounds(smiles_list: list) -> list:
    """Batch lookup compounds on PubChem in parallel.

    Returns a list of profile dicts sorted by fame_score descending.
    """
    if not smiles_list:
        return []
    with ThreadPoolExecutor(max_workers=min(5, len(smiles_list))) as ex:
        results = list(ex.map(get_compound_profile, smiles_list))
    results.sort(key=lambda x: x.get('fame_score', 0), reverse=True)
    return results
