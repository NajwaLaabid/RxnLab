"""DiffAlign prediction routes."""
import time
import uuid
from typing import Optional

from flask import Blueprint, current_app, g, jsonify, render_template, request
from markupsafe import escape

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.inchi import MolToInchiKey

from DiffAlign.api import predict

from app.config import DIFFALIGN_MODEL_ID
from app.db import db_enabled
from app.models_db import Event, PredictionRun
from app.rendering.classify import classify_reactions
from app.rendering.pubchem import lookup_all_compounds
from app.rendering.svg import mol_to_svg, mols_to_svg, serialize_results_json

bp = Blueprint('predict', __name__)


def _strip_svg(predictions: list) -> list:
    """Return predictions with the rendered SVG stripped — we don't want to persist HTML."""
    out = []
    for p in predictions:
        copy = {k: v for k, v in p.items() if k != 'svg'}
        out.append(copy)
    return out


def _log_run(*, product_smiles: str, target_mol, params: dict,
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
        model_id=DIFFALIGN_MODEL_ID,
        product_smiles=product_smiles,
        product_inchi_key=inchi_key,
        params=params,
        predictions=_strip_svg(predictions),
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
            ux_feedback_form_url=current_app.config.get('UX_FEEDBACK_FORM_URL', ''),
        )

    smiles = request.form.get('smiles', '').strip()
    n_precursors = request.form.get('n_precursors', 1, type=int)
    diffusion_steps = request.form.get('diffusion_steps', 1, type=int)

    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    def _render(error=None, results=None, target_svg=None, target_mw=0,
                results_json='[]', run_id=None):
        results_html = render_template(
            'partials/results.html',
            error=error,
            results=results,
            results_json=results_json,
            smiles=smiles,
            target_svg=target_svg,
            target_mw=target_mw,
            run_id=str(run_id) if run_id else '',
        )
        if is_ajax:
            return results_html
        return render_template(
            'predict.html',
            smiles=smiles,
            n_precursors=n_precursors,
            diffusion_steps=diffusion_steps,
            results_html=results_html,
            ux_feedback_form_url=current_app.config.get('UX_FEEDBACK_FORM_URL', ''),
        )

    if not smiles:
        return _render(error="Please enter a SMILES string.")

    if not (1 <= n_precursors <= 100):
        return _render(error="Number of precursors must be between 1 and 100.")

    if not (1 <= diffusion_steps <= 50):
        return _render(error="Diffusion steps must be between 1 and 50.")

    T = 50
    if T % diffusion_steps != 0:
        valid = [d for d in range(1, T + 1) if T % d == 0]
        return _render(
            error=f"Diffusion steps must evenly divide {T}. Valid values: {valid}"
        )

    target_mol = Chem.MolFromSmiles(smiles)
    if target_mol is None:
        return _render(
            error=(
                f"Invalid SMILES string: '{escape(smiles)}'. "
                "Please check for mismatched parentheses, invalid atom symbols, or incorrect bond notation."
            )
        )

    t0 = time.monotonic()
    predictions = predict.predict_precursors(
        smiles,
        n_precursors=n_precursors,
        diffusion_steps=diffusion_steps,
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    results = []
    for pred in predictions:
        precursor_mols = [Chem.MolFromSmiles(s) for s in pred['precursors'].split('.')]
        precursor_mols = [m for m in precursor_mols if m is not None]

        if precursor_mols:
            svg = mols_to_svg(precursor_mols)
            results.append({
                'precursors': pred['precursors'],
                'score': pred['score'],
                'svg': svg,
                'sample_data': pred.get('sample_data'),
                'atom_mapping': pred.get('atom_mapping'),
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
        product_smiles=smiles,
        target_mol=target_mol,
        params={'n_precursors': n_precursors, 'diffusion_steps': diffusion_steps},
        predictions=results,
        latency_ms=latency_ms,
    )

    return _render(
        results=results,
        target_svg=mol_to_svg(target_mol, 150, 150),
        target_mw=Descriptors.MolWt(target_mol),
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
    diffusion_steps = data.get('diffusion_steps', 1)

    t0 = time.monotonic()
    predictions = predict.predict_precursors(
        smiles, n_precursors=n_precursors, diffusion_steps=diffusion_steps,
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    if data.get('evaluate', True):
        try:
            classify_reactions(predictions, smiles)
        except Exception:
            pass

    run_id = _log_run(
        product_smiles=smiles,
        target_mol=mol,
        params={'n_precursors': n_precursors, 'diffusion_steps': diffusion_steps},
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
            'error': 'All atoms are marked fixed — nothing to regenerate.',
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
