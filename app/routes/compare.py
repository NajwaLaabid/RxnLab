"""Model-comparison routes.

A consensus-first side-by-side comparison of several single-step models on one
target. The headline is *agreement*: precursor sets ranked by how many of the
selected models propose them. Per-model ranked lists and reaction-class
distributions sit underneath.

Scores are NOT comparable across models (diffusion likelihood vs. template
probability vs. beam score vs. graph-edit probability live on different scales),
so the UI compares *ranks*, not raw scores.
"""
import time
from collections import Counter, OrderedDict

from flask import Blueprint, current_app, jsonify, render_template, request

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from app.registry import registry
from app.rendering.classify import classify_reactions
from app.rendering.svg import mol_to_svg, mols_to_svg
from evaluation.pubchem_lookup import resolve_to_smiles

bp = Blueprint('compare', __name__)

MAX_MODELS = 4
MIN_MODELS = 2


@bp.route('/compare', methods=['GET'])
def compare_page():
    return render_template(
        'compare.html',
        models=registry.ui_models(),
        default_model_ids=[s.model_id for s in registry.list_specs()][:MIN_MODELS],
        ux_feedback_form_url=current_app.config.get('UX_FEEDBACK_FORM_URL', ''),
    )


def _default_params(model_id: str) -> dict:
    """Each model's ParamSpec defaults — compare doesn't expose per-model knobs,
    so it runs every model at its UI default (matching the single-step page)."""
    return {p.name: p.default for p in registry.get_spec(model_id).params}


def _run_model(model_id: str, smiles: str, product_canon: str, n_precursors: int) -> dict:
    """Run one model and return its processed, classified, ranked predictions."""
    spec = registry.get_spec(model_id)
    t0 = time.monotonic()
    raw = registry.predict(
        model_id, smiles, n_precursors=n_precursors, params=_default_params(model_id)
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    seen = set()
    preds = []
    for pred in raw:
        mols = [Chem.MolFromSmiles(s) for s in pred['precursors'].split('.')]
        mols = [m for m in mols if m is not None]
        if not mols:
            continue
        canon = '.'.join(sorted(Chem.MolToSmiles(m) for m in mols))
        if canon == product_canon or canon in seen:
            continue
        seen.add(canon)
        preds.append({
            'precursors': pred['precursors'],
            'precursors_canon': canon,
            'score': pred['score'],
            'svg': mols_to_svg(mols),
            'formula': ', '.join(CalcMolFormula(m) for m in mols),
        })

    try:
        classify_reactions(preds, smiles)
    except Exception:
        pass
    for p in preds:
        info = p.get('reaction_info') or {}
        p['reaction_class'] = info.get('class')
        p['reaction_name'] = info.get('name')

    for i, p in enumerate(preds):
        p['rank'] = i + 1

    classes = [p['reaction_class'] for p in preds if p.get('reaction_class')]
    distribution = [
        {'label': label, 'count': count}
        for label, count in Counter(classes).most_common()
    ]
    names = [p['reaction_name'] for p in preds if p.get('reaction_name')]
    name_distribution = [
        {'label': label, 'count': count}
        for label, count in Counter(names).most_common()
    ]

    return {
        'model_id': model_id,
        'display_name': spec.display_name,
        'latency_ms': latency_ms,
        'predictions': preds,
        'class_distribution': distribution,
        'name_distribution': name_distribution,
    }


def _build_consensus(model_results: list[dict]) -> list[dict]:
    """Group predictions across models by canonical precursor set. Sorted by
    number of agreeing models (desc), then by best (lowest) rank achieved."""
    groups: OrderedDict[str, dict] = OrderedDict()
    for mr in model_results:
        for p in mr['predictions']:
            key = p['precursors_canon']
            g = groups.get(key)
            if g is None:
                g = groups[key] = {
                    'precursors_canon': key,
                    'precursors': p['precursors'],
                    'svg': p['svg'],
                    'formula': p['formula'],
                    'reaction_class': p.get('reaction_class'),
                    'reaction_name': p.get('reaction_name'),
                    'members': [],
                }
            if g['reaction_class'] is None and p.get('reaction_class'):
                g['reaction_class'] = p['reaction_class']
                g['reaction_name'] = p.get('reaction_name')
            g['members'].append({
                'model_id': mr['model_id'],
                'display_name': mr['display_name'],
                'rank': p['rank'],
                'score': p['score'],
            })

    consensus = list(groups.values())
    for g in consensus:
        g['n_models'] = len(g['members'])
    consensus.sort(key=lambda g: (-g['n_models'], min(m['rank'] for m in g['members'])))
    return consensus


@bp.route('/api/compare', methods=['POST'])
def api_compare():
    data = request.get_json() or {}
    raw_input = (data.get('smiles') or '').strip()
    if not raw_input:
        return jsonify({'error': 'Please enter a molecule.'}), 400
    if len(raw_input) > current_app.config['MAX_SMILES_LEN']:
        return jsonify({'error': f"Input too long (max {current_app.config['MAX_SMILES_LEN']} characters)."}), 400

    model_ids = data.get('model_ids') or []
    if not isinstance(model_ids, list):
        return jsonify({'error': 'model_ids must be a list'}), 400
    model_ids = [m for m in model_ids if registry.is_known(m)]
    # de-dupe, preserve order
    model_ids = list(OrderedDict.fromkeys(model_ids))
    if not (MIN_MODELS <= len(model_ids) <= MAX_MODELS):
        return jsonify(
            {'error': f'Select between {MIN_MODELS} and {MAX_MODELS} models to compare.'}
        ), 400

    n_precursors = data.get('n_precursors', 10)
    try:
        n_precursors = int(n_precursors)
    except (TypeError, ValueError):
        n_precursors = 10
    n_precursors = max(1, min(50, n_precursors))

    smiles = raw_input
    target_mol = Chem.MolFromSmiles(smiles)
    resolved_from = None
    if target_mol is None:
        resolved = resolve_to_smiles(raw_input)
        if resolved.get('smiles'):
            smiles = resolved['smiles']
            target_mol = Chem.MolFromSmiles(smiles)
            resolved_from = {'query': raw_input, 'kind': resolved.get('kind')}
        if target_mol is None:
            return jsonify(
                {'error': f"Could not interpret '{raw_input}' as a molecule."}
            ), 400

    product_canon = Chem.MolToSmiles(target_mol)
    model_results = [
        _run_model(mid, smiles, product_canon, n_precursors) for mid in model_ids
    ]
    consensus = _build_consensus(model_results)

    return jsonify({
        'target': {
            'smiles': smiles,
            'svg': mol_to_svg(target_mol, 150, 150),
            'formula': CalcMolFormula(target_mol),
            'mw': round(Descriptors.MolWt(target_mol), 2),
            'resolved_from': resolved_from,
        },
        'models': model_results,
        'consensus': consensus,
        'n_models': len(model_ids),
    })
