"""PubChem compound profiling and literature statistics."""

import time
import requests


def get_compound_profile(smiles: str, delay: float = 0.3) -> dict:
    """Get a comprehensive profile of a compound from PubChem.

    Returns a dict with: smiles, found, cid, iupac, formula, mw,
    n_synonyms, short_names, n_sources, n_patents, n_pubmed, fame_score.
    """
    try:
        import pubchempy as pcp
    except ImportError:
        return {'smiles': smiles, 'found': False, 'error': 'pubchempy not installed'}

    try:
        results = pcp.get_compounds(smiles, 'smiles')
        if not results:
            return {'smiles': smiles, 'found': False}
        c = results[0]
        cid = c.cid
        if not cid:
            return {'smiles': smiles, 'found': False}

        info = {
            'smiles': smiles,
            'found': True,
            'cid': cid,
            'iupac': c.iupac_name,
            'formula': c.molecular_formula,
            'mw': c.molecular_weight,
        }

        # Synonyms (common names, trade names, etc.)
        syns = c.synonyms or []
        info['n_synonyms'] = len(syns)
        # Filter out database identifiers, keep recognizable short names
        _db_prefixes = (
            'SCHEMBL', 'DTXSID', 'ZINC', 'AKOS', 'MFCD', 'SBB', 'BBV',
            'BCP', 'BDBM', 'STK', 'HY-', 'CS-', 'FT-',
        )
        info['short_names'] = [
            s for s in syns[:30]
            if len(s) < 30 and not any(s.startswith(p) for p in _db_prefixes)
        ][:10]

        # Number of substance depositors (proxy for vendor/source count)
        sids = c.sids or []
        info['n_sources'] = len(sids)
        time.sleep(delay)

        # Patent count via PUG-REST
        info['n_patents'] = _get_patent_count(cid)
        time.sleep(delay)

        # PubMed references
        info['n_pubmed'] = _get_pubmed_count(cid)
        time.sleep(delay)

        # Composite fame score
        info['fame_score'] = (
            info['n_sources']
            + info['n_patents'] * 5
            + info['n_pubmed'] * 10
            + info['n_synonyms']
        )

        return info
    except Exception as e:
        return {'smiles': smiles, 'found': False, 'error': str(e)}


def lookup_all_compounds(smiles_list: list, delay: float = 0.5) -> list:
    """Batch lookup compounds on PubChem.

    Returns a list of profile dicts sorted by fame_score descending.
    """
    results = []
    for smi in smiles_list:
        results.append(get_compound_profile(smi, delay=delay))
        time.sleep(delay)
    results.sort(key=lambda x: x.get('fame_score', 0), reverse=True)
    return results


def _get_patent_count(cid: int, timeout: int = 10) -> int:
    """Get number of patents mentioning this compound."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/xrefs/PatentID/JSON"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            patents = r.json()['InformationList']['Information'][0].get('PatentID', [])
            return len(patents)
    except Exception:
        pass
    return 0


def _get_pubmed_count(cid: int, timeout: int = 10) -> int:
    """Get number of PubMed articles citing this compound."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/xrefs/PubMedID/JSON"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            pmids = r.json()['InformationList']['Information'][0].get('PubMedID', [])
            return len(pmids)
    except Exception:
        pass
    return 0
