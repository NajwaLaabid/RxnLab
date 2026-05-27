"""DiffAlign prediction routes."""
import time
import uuid
from typing import Optional

from flask import Blueprint, current_app, g, jsonify, render_template, request
from markupsafe import escape

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.inchi import MolToInchiKey
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from DiffAlign.api import predict  # eager model load + DiffAlign-specific inpainting

from app.db import db_enabled
from app.registry import registry
from app.models_db import Event, PredictionRun
from app.rendering.classify import classify_reactions
from app.rendering.pubchem import lookup_all_compounds
from app.rendering.svg import mol_to_svg, mols_to_svg, serialize_results_json
from evaluation.pubchem_lookup import resolve_to_smiles

bp = Blueprint('predict', __name__)


def _strip_svg(predictions: list) -> list:
    """Return predictions with the rendered SVG stripped — we don't want to persist HTML."""
    out = []
    for p in predictions:
        copy = {k: v for k, v in p.items() if k != 'svg'}
        out.append(copy)
    return out


# Stored-prediction schema. Route-aware from Phase 0 so multi-step (Phase 3) slots
# in without migration: the run's product is the tree root, each single-step
# prediction is a depth-1 node, and `children` holds further backward steps later.
PREDICTIONS_SCHEMA = 'route-tree-v1'


def _to_route_tree(product_smiles: str, predictions: list) -> dict:
    """Wrap a flat list of single-step predictions as a route tree. Node `i`
    corresponds to feedback `prediction_index == i`; per-prediction fields are
    preserved verbatim, with `product`/`children` added for tree-awareness."""
    return {
        'schema': PREDICTIONS_SCHEMA,
        'root': product_smiles,
        'nodes': [
            {'product': product_smiles, 'children': [], **p}
            for p in predictions
        ],
    }


def _coerce_params(model_id: str, getter):
    """Coerce + validate the selected model's params from a value source.

    ``getter(name, default)`` pulls a raw value (form field or JSON key). Returns
    ``(params, error)`` — ``error`` is a UI-ready string on the first bad value.
    """
    spec = registry.get_spec(model_id)
    params = {}
    for p in spec.params:
        try:
            params[p.name] = p.coerce(getter(p.name, p.default))
        except ValueError as e:
            return None, str(e)
    return params, None


def _log_run(*, model_id: str, product_smiles: str, target_mol, params: dict,
             predictions: list, latency_ms: int) -> Optional[uuid.UUID]:
    """Persist a prediction_runs row + predict_returned event. No-op if DB is off."""
    if not db_enabled() or g.get('db') is None or g.get('session_id') is None:
        return None
    try:
        inchi_key = MolToInchiKey(target_mol) or ''
    except Exception:
        inchi_key = ''
    run = PredictionRun(
        session_id=g.session_id,
        model_id=model_id,
        product_smiles=product_smiles,
        product_inchi_key=inchi_key,
        params=params,
        predictions=_to_route_tree(product_smiles, _strip_svg(predictions)),
        latency_ms=latency_ms,
    )
    g.db.add(run)
    g.db.flush()  # populate run.run_id without committing yet
    g.db.add(Event(
        session_id=g.session_id,
        run_id=run.run_id,
        event_type='predict_returned',
        payload={'n_predictions': len(predictions), 'latency_ms': latency_ms},
    ))
    g.db.commit()
    return run.run_id


@bp.route('/diffalign', methods=['GET', 'POST'])
def diffalign():
    if request.method == 'GET':
        return render_template(
            'predict.html',
            results_html='',
            models=registry.ui_models(),
            selected_model_id=registry.default_model_id(),
            ux_feedback_form_url=current_app.config.get('UX_FEEDBACK_FORM_URL', ''),
        )

    raw_input = request.form.get('smiles', '').strip()
    smiles = raw_input
    resolved_from = None  # set when we had to resolve a non-SMILES identifier
    n_precursors = request.form.get('n_precursors', 1, type=int)

    model_id = request.form.get('model_id', '').strip() or registry.default_model_id()
    if not registry.is_known(model_id):
        model_id = registry.default_model_id()

    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    def _render(error=None, results=None, target_svg=None, target_mw=0,
                target_formula='', results_json='[]', run_id=None):
        results_html = render_template(
            'partials/results.html',
            error=error,
            results=results,
            results_json=results_json,
            smiles=smiles,
            resolved_from=resolved_from,
            target_svg=target_svg,
            target_mw=target_mw,
            target_formula=target_formula,
            supports_inpainting=registry.supports_inpainting(model_id),
            run_id=str(run_id) if run_id else '',
        )
        if is_ajax:
            return results_html
        return render_template(
            'predict.html',
            smiles=smiles,
            n_precursors=n_precursors,
            models=registry.ui_models(),
            selected_model_id=model_id,
            results_html=results_html,
            ux_feedback_form_url=current_app.config.get('UX_FEEDBACK_FORM_URL', ''),
        )

    if not smiles:
        return _render(error="Please enter a molecule (SMILES, name, InChI, InChIKey, or CAS).")

    if not (1 <= n_precursors <= 100):
        return _render(error="Number of precursors must be between 1 and 100.")

    params, param_error = _coerce_params(model_id, request.form.get)
    if param_error:
        return _render(error=param_error)

    target_mol = Chem.MolFromSmiles(smiles)
    if target_mol is None:
        resolved = resolve_to_smiles(raw_input)
        if resolved.get('smiles'):
            smiles = resolved['smiles']
            resolved_from = {
                'query': raw_input,
                'kind': resolved.get('kind'),
                'source': resolved.get('source'),
            }
            target_mol = Chem.MolFromSmiles(smiles)
        if target_mol is None:
            hint = resolved.get('error') or 'Not found.'
            return _render(
                error=(
                    f"Could not interpret '{escape(raw_input)}' as a SMILES, name, "
                    f"InChI, InChIKey, or CAS number. {escape(hint)}"
                )
            )

    t0 = time.monotonic()
    predictions = registry.predict(
        model_id,
        smiles,
        n_precursors=n_precursors,
        params=params,
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    product_canon = Chem.MolToSmiles(target_mol)
    results = []
    for pred in predictions:
        precursor_mols = [Chem.MolFromSmiles(s) for s in pred['precursors'].split('.')]
        precursor_mols = [m for m in precursor_mols if m is not None]

        # Drop trivial predictions where the precursor set is just the product.
        precursor_canon = '.'.join(sorted(Chem.MolToSmiles(m) for m in precursor_mols))
        if precursor_canon == product_canon:
            continue

        if precursor_mols:
            svg = mols_to_svg(precursor_mols)
            formula = ', '.join(CalcMolFormula(m) for m in precursor_mols)
            results.append({
                'precursors': pred['precursors'],
                'score': pred['score'],
                'svg': svg,
                'formula': formula,
                'sample_data': pred.get('sample_data'),
                'atom_mapping': pred.get('atom_mapping'),
                'mapped_rxn': pred.get('mapped_rxn'),
            })

    if not results:
        return _render(
            error="No valid precursors found for this molecule. Try increasing the number of precursors or diffusion steps."
        )

    try:
        classify_reactions(results, smiles)
    except Exception:
        pass

    run_id = _log_run(
        model_id=model_id,
        product_smiles=smiles,
        target_mol=target_mol,
        params={'n_precursors': n_precursors, **params},
        predictions=results,
        latency_ms=latency_ms,
    )

    return _render(
        results=results,
        target_svg=mol_to_svg(target_mol, 150, 150),
        target_mw=Descriptors.MolWt(target_mol),
        target_formula=CalcMolFormula(target_mol),
        results_json=serialize_results_json(results),
        run_id=run_id,
    )


@bp.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json() or {}
    smiles = data.get('smiles', '').strip()

    if not smiles:
        return jsonify({'error': 'No SMILES provided'}), 400

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({'error': f'Invalid SMILES: {smiles}'}), 400

    n_precursors = data.get('n_precursors', 1)

    model_id = (data.get('model_id') or '').strip() or registry.default_model_id()
    if not registry.is_known(model_id):
        return jsonify({'error': f'Unknown model: {model_id}'}), 400

    params, param_error = _coerce_params(model_id, data.get)
    if param_error:
        return jsonify({'error': param_error}), 400

    t0 = time.monotonic()
    predictions = registry.predict(
        model_id, smiles, n_precursors=n_precursors, params=params,
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    if data.get('evaluate', True):
        try:
            classify_reactions(predictions, smiles)
        except Exception:
            pass

    run_id = _log_run(
        model_id=model_id,
        product_smiles=smiles,
        target_mol=mol,
        params={'n_precursors': n_precursors, **params},
        predictions=predictions,
        latency_ms=latency_ms,
    )

    response = {'target': smiles, 'predictions': predictions}
    if run_id is not None:
        response['run_id'] = str(run_id)
    return jsonify(response)


@bp.route('/api/inpaint', methods=['POST'])
def api_inpaint():
    data = request.get_json() or {}

    model_id = (data.get('model_id') or '').strip() or registry.default_model_id()
    if not registry.is_known(model_id):
        return jsonify({'error': f'Unknown model: {model_id}'}), 400
    if not registry.supports_inpainting(model_id):
        return jsonify({'error': f'Model {model_id} does not support inpainting.'}), 400

    product_smiles = data.get('product_smiles', '').strip()
    previous_sample_data = data.get('previous_sample_data')
    selected_node_indices = data.get('selected_node_indices', [])
    n_precursors = data.get('n_precursors', 1)
    diffusion_steps = data.get('diffusion_steps', 1)

    if not product_smiles:
        return jsonify({'error': 'No product SMILES provided'}), 400
    if not previous_sample_data:
        return jsonify({'error': 'No previous sample data provided'}), 400
    if not selected_node_indices:
        return jsonify({'error': 'No atoms selected for inpainting'}), 400

    if not isinstance(selected_node_indices, list):
        return jsonify({'error': 'selected_node_indices must be a list'}), 400
    if not (1 <= n_precursors <= 100):
        return jsonify({'error': 'n_precursors must be between 1 and 100'}), 400

    # Reject if every real atom is marked fixed — nothing to regenerate.
    node_mask = previous_sample_data.get('node_mask') or []
    node_mask_row = node_mask[0] if node_mask and isinstance(node_mask[0], list) else node_mask
    n_real = sum(1 for v in node_mask_row if v)
    n_fixed = sum(1 for i in selected_node_indices if 0 <= int(i) < len(node_mask_row) and node_mask_row[int(i)])
    if n_real > 0 and n_fixed >= n_real:
        return jsonify({
            'error': 'All atoms are marked fixed. Nothing to regenerate.',
            'hint':  'Deselect at least one atom before submitting.',
        }), 400

    try:
        results, failure_info = predict.predict_with_inpainting(
            product_smiles=product_smiles,
            previous_sample_data=previous_sample_data,
            inpaint_node_indices=selected_node_indices,
            n_precursors=n_precursors,
            diffusion_steps=diffusion_steps,
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        current_app.logger.error(f'Inpainting failed:\n{tb}')
        return jsonify({'error': f'Inpainting failed: {str(e)}', 'traceback': tb.splitlines()[-8:]}), 500

    if not results and failure_info:
        stuck = failure_info.get('stuck_atoms', [])
        if stuck:
            elem_str = ', '.join(a['element'] for a in stuck)
            idx_str = ', '.join(str(a['index']) for a in stuck)
            msg = (
                f"No precursor satisfied the inpainting constraint. Across all "
                f"{failure_info['n_samples']} samples, the following atoms were "
                f"marked to change but stayed the same: {elem_str} "
                f"(positions {idx_str}). Try more diffusion steps, a different "
                f"product, or mark different atoms."
            )
        else:
            msg = (
                f"No precursor satisfied the inpainting constraint across all "
                f"{failure_info['n_samples']} samples. Try more diffusion steps."
            )
        return jsonify({
            'target_smiles': product_smiles,
            'results': [],
            'fixed_atoms_info': f'{len(selected_node_indices)} atoms fixed',
            'failure': {
                'message': msg,
                'stuck_atoms': stuck,
                'requested_change_atoms': failure_info.get('requested_change_atoms', []),
                'n_samples': failure_info['n_samples'],
            },
        })

    for r in results:
        precursor_mols = [Chem.MolFromSmiles(s) for s in r['precursors'].split('.')]
        precursor_mols = [m for m in precursor_mols if m is not None]
        r['svg'] = mols_to_svg(precursor_mols) if precursor_mols else ''
        r['formula'] = ', '.join(CalcMolFormula(m) for m in precursor_mols) if precursor_mols else ''

    return jsonify({
        'target_smiles': product_smiles,
        'results': results,
        'fixed_atoms_info': f'{len(selected_node_indices)} atoms fixed',
    })


@bp.route('/api/evaluate/compound-lookup', methods=['POST'])
def api_compound_lookup():
    data = request.get_json() or {}
    smiles_list = data.get('smiles_list', [])

    if not smiles_list:
        return jsonify({'error': 'No SMILES provided'}), 400

    if len(smiles_list) > 10:
        return jsonify({'error': 'Maximum 10 compounds per request'}), 400

    compounds = lookup_all_compounds(smiles_list)
    return jsonify({'compounds': compounds})
