"""Multi-step search routes (Phase 3a).

A search runs in a background thread (see ``app.jobs``); the client starts one with
``POST /api/search`` and polls ``GET /api/search/<job_id>`` until ``status`` is
``done`` or ``error``.
"""
from flask import Blueprint, current_app, g, jsonify, request

from rdkit import Chem

from app.jobs import get_job, start_search_job
from app.registry import registry
from app.search import default_catalog_id, search_enabled
from evaluation.pubchem_lookup import resolve_to_smiles

bp = Blueprint('search', __name__)


def _clamp_int(value, default: int, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _parse_budget(data: dict) -> dict:
    """Demo-safe search knobs from the request. Clamped so a bad value can't blow
    up the box; whole-search time stays at the server default (not user-set)."""
    return {
        "max_routes": _clamp_int(data.get("max_routes"), 5, 1, 15),
        "max_expansion_depth": _clamp_int(data.get("max_expansion_depth"), 6, 1, 12),
    }


@bp.route('/api/search', methods=['POST'])
def start_search():
    data = request.get_json() or {}
    raw_input = (data.get('smiles') or '').strip()
    if not raw_input:
        return jsonify({'error': 'No molecule provided'}), 400
    if len(raw_input) > current_app.config['MAX_SMILES_LEN']:
        return jsonify({'error': f"Input too long (max {current_app.config['MAX_SMILES_LEN']} characters)."}), 400

    model_id = (data.get('model_id') or '').strip() or registry.default_model_id()
    if not registry.is_known(model_id):
        return jsonify({'error': f'Unknown model: {model_id}'}), 400

    catalog_id = (data.get('catalog_id') or '').strip() or default_catalog_id()
    if not search_enabled(model_id, catalog_id):
        return jsonify({
            'error': f'Multi-step search is not available for {model_id}.',
        }), 400

    smiles = raw_input
    resolved_from = None
    if Chem.MolFromSmiles(smiles) is None:
        resolved = resolve_to_smiles(raw_input)
        if resolved.get('smiles') and Chem.MolFromSmiles(resolved['smiles']) is not None:
            smiles = resolved['smiles']
            resolved_from = {'query': raw_input, 'kind': resolved.get('kind'),
                             'source': resolved.get('source')}
        else:
            return jsonify({'error': f"Could not interpret '{raw_input}' as a molecule."}), 400

    job_id = start_search_job(
        model_id=model_id,
        product_smiles=smiles,
        catalog_id=catalog_id,
        session_id=getattr(g, 'session_id', None),
        budget=_parse_budget(data),
    )
    if job_id is None:
        return jsonify({'error': 'The server is busy running other searches. Try again shortly.'}), 429

    return jsonify({'job_id': job_id, 'smiles': smiles, 'resolved_from': resolved_from})


@bp.route('/api/search/<job_id>', methods=['GET'])
def poll_search(job_id):
    job = get_job(job_id)
    if job is None:
        return jsonify({'error': 'Unknown job id'}), 404
    return jsonify(job)


@bp.route('/api/search/describe', methods=['POST'])
def describe():
    data = request.get_json() or {}
    route = data.get('route')
    if not route or not route.get('product'):
        return jsonify({'error': 'No route provided'}), 400
    from app.search import describe_route
    try:
        return jsonify(describe_route(route))
    except Exception as e:
        return jsonify({'error': f'Could not describe route: {e}'}), 500
