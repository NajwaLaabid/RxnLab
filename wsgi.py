#!/usr/bin/env python3
"""
Retrosynthesis prediction web app demo.
Designed for deployment on CSC Rahti.
Uses SVG rendering to avoid X11 dependencies.
"""
import os
import sys

import flask
from flask import request, render_template_string, jsonify
from markupsafe import escape

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D


# Add project root and DiffAlign submodule to the path
from pathlib import Path
app_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(app_dir / 'DiffAlign'))

from DiffAlign.api import predict
from evaluation import classify_reactions
from evaluation.pubchem_lookup import lookup_all_compounds

# testing
application = flask.Flask(__name__)

# Results fragment template (used by both AJAX and full-page renders)
RESULTS_TEMPLATE = """
{% if error %}
<div class="results-section">
    <div class="error">{{ error }}</div>
</div>
{% elif results %}
<div class="results-section">
    <div class="results-header">
        <h2>Results</h2>
    </div>

    <div class="target-display">
        {{ target_svg | safe }}
        <div class="target-info">
            <h3>Target</h3>
            <div class="smiles">{{ smiles }}</div>
            <div style="margin-top: 8px; font-size: 13px; color: #666;">
                MW: {{ "%.1f"|format(target_mw) }} ·
                Generated {{ results|length }} precursor set(s)
            </div>
        </div>
    </div>

    <h3 style="margin-bottom: 15px;">Predicted Precursor Sets</h3>
    <div class="precursors-grid">
        {% for result in results %}
        <div class="precursor-card">
            <div class="precursor-header">
                <span class="precursor-rank">#{{ loop.index }}</span>
                <span class="precursor-score">Score: {{ "%.3f"|format(result.score) }}</span>
            </div>
            {% if result.reaction_info and result.reaction_info.success %}
            <div class="reaction-class-badge" title="Reaction class: {{ result.reaction_info.class or 'Unknown' }}">
                {{ result.reaction_info.name or result.reaction_info.class or 'Unclassified' }}
            </div>
            {% endif %}
            <div class="mol-svg">{{ result.svg | safe }}</div>
            <div class="precursor-smiles">{{ result.precursors }}</div>
            <button class="btn-lookup" onclick="lookupCompounds(this, '{{ result.precursors }}')">Search PubChem</button>
            <div class="compound-info" style="display:none;"></div>
        </div>
        {% endfor %}
    </div>
</div>
{% endif %}
"""

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>DiffAlign: Retrosynthesis through Diffusion</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1100px;
            margin: 0 auto;
            padding: 20px;
            background: #ffffff;
            min-height: 100vh;
        }
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #4a6fa5, #5a8fd5);
            z-index: 1000;
        }
        .container {
            padding: 30px;
        }
        @media (max-width: 768px) {
            body { padding: 10px; }
            .container { padding: 15px; }
        }
        h1 { color: #1a1a2e; margin-bottom: 8px; }
        .subtitle { color: #666; margin-bottom: 25px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; font-weight: 600; margin-bottom: 6px; color: #333; }
        .label-hint { font-weight: normal; color: #888; font-size: 13px; }
        input[type="text"] {
            width: 100%;
            padding: 12px;
            font-size: 15px;
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
        }
        input[type="text"]:focus { border-color: #4a6fa5; outline: none; }
        .params-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .param-box { background: #f8f9fa; padding: 15px; border-radius: 8px; }
        .param-box label { font-size: 13px; margin-bottom: 8px; }
        .param-hint { font-size: 11px; color: #888; margin-top: 4px; }
        input[type="number"], select {
            width: 100%;
            padding: 10px;
            font-size: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
        }
        .btn-row { display: flex; gap: 10px; margin-top: 20px; }
        button {
            padding: 12px 28px;
            font-size: 15px;
            font-weight: 600;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn-primary { background: #4a6fa5; color: white; }
        .btn-primary:hover { background: #3d5d8a; }
        .btn-primary:disabled { background: #a0b4cc; cursor: not-allowed; }
        .btn-secondary { background: #e9ecef; color: #495057; }
        .btn-secondary:hover { background: #dee2e6; }
        .examples { margin-top: 12px; font-size: 13px; color: #666; }
        .examples code {
            background: #e9ecef;
            padding: 3px 8px;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .examples code:hover { background: #dee2e6; }
        .results-section {
            margin-top: 30px;
            padding-top: 25px;
            border-top: 2px solid #e9ecef;
        }
        .results-header { display: flex; align-items: center; gap: 15px; margin-bottom: 20px; }
        .results-header h2 { margin: 0; color: #1a1a2e; }
        .target-display {
            display: flex;
            align-items: center;
            gap: 20px;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .target-display svg {
            border: 1px solid #ddd;
            border-radius: 6px;
            background: white;
            max-width: 150px;
            height: auto;
            flex-shrink: 0;
        }
        .target-info h3 { margin: 0 0 5px 0; color: #333; }
        .target-info .smiles {
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 13px;
            color: #666;
            word-break: break-all;
        }
        .precursors-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 15px;
        }
        .precursor-card {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            transition: box-shadow 0.2s;
        }
        .precursor-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        .precursor-card .mol-svg { width: 100%; border-radius: 4px; background: #fafafa; overflow: hidden; }
        .mol-svg svg { display: block; width: 100%; height: auto; max-width: 100%; }
        .precursor-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .precursor-rank { font-weight: 700; color: #4a6fa5; }
        .precursor-score {
            background: #e8f4e8;
            color: #2d6a2d;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: 600;
        }
        .precursor-smiles {
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 11px;
            color: #888;
            word-break: break-all;
            overflow-wrap: anywhere;
            max-height: 60px;
            overflow-y: auto;
            margin-top: 10px;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 4px;
        }
        .error {
            color: #c0392b;
            background: #fdf0ef;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #c0392b;
        }
        .reaction-class-badge {
            display: inline-block;
            background: #e8eaf6;
            color: #3949ab;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .btn-lookup {
            display: block;
            width: 100%;
            margin-top: 10px;
            padding: 8px;
            font-size: 13px;
            font-weight: 600;
            border: 1px solid #4a6fa5;
            border-radius: 6px;
            background: white;
            color: #4a6fa5;
            cursor: pointer;
            transition: background 0.2s, color 0.2s;
        }
        .btn-lookup:hover { background: #4a6fa5; color: white; }
        .btn-lookup:disabled { opacity: 0.6; cursor: not-allowed; }
        .compound-info {
            margin-top: 10px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 6px;
            font-size: 12px;
            line-height: 1.6;
        }
        .compound-info .compound-entry { margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #e9ecef; }
        .compound-info .compound-entry:last-child { margin-bottom: 0; padding-bottom: 0; border-bottom: none; }
        .compound-info .compound-name { font-weight: 600; color: #333; }
        .compound-info .compound-detail { color: #666; }
        .fame-score {
            display: inline-block;
            background: #fff3e0;
            color: #e65100;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
        }
        .info-banner {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            background: #edf2f7;
            border-left: 4px solid #4a6fa5;
            padding: 14px 18px;
            border-radius: 6px;
            margin-bottom: 25px;
            font-size: 14px;
            color: #333;
            line-height: 1.5;
        }
        .info-banner-content { flex: 1; }
        .info-banner-content a { color: #4a6fa5; text-decoration: underline; }
        .info-banner-dismiss {
            background: none;
            border: none;
            font-size: 20px;
            color: #888;
            cursor: pointer;
            padding: 0 4px;
            line-height: 1;
        }
        .info-banner-dismiss:hover { color: #333; }
    </style>
</head>
<body>
    <div class="container">
        <h1>DiffAlign: Retrosynthesis through Diffusion</h1>
        <p class="subtitle">Enter your target molecule below</p>

        <div class="info-banner" id="info-banner">
            <div class="info-banner-content">
                <strong>Demo notice:</strong> This app runs DiffAlign on CPU only.
                Expect ~1 min per prediction for small molecules (10–20 atoms) with 50 diffusion steps.
                For full-scale inference, see the
                <a href="https://github.com/Aalto-QuML/DiffAlign" target="_blank" rel="noopener">DiffAlign repository</a>.
            </div>
            <button class="info-banner-dismiss" onclick="dismissBanner()" title="Dismiss">&times;</button>
        </div>

        <form method="post" action="/" id="predict-form">
            <div class="form-group">
                <label>Target Product <span class="label-hint">(SMILES)</span></label>
                <input type="text" name="smiles" placeholder="Enter SMILES string of target molecule..."
                       value="{{ smiles or '' }}" id="smiles-input">
                <div class="examples">
                    Examples:
                    <code onclick="setSmiles('CC(=O)Oc1ccccc1C(=O)O')">Aspirin</code>
                    <code onclick="setSmiles('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')">Caffeine</code>
                    <code onclick="setSmiles('CC(C)Cc1ccc(cc1)C(C)C(=O)O')">Ibuprofen</code>
                    <code onclick="setSmiles('CC(=O)Nc1ccc(O)cc1')">Paracetamol</code>
                </div>
            </div>

            <div class="params-grid">
                <div class="param-box">
                    <label>Number of Precursors</label>
                    <input type="number" name="n_precursors" value="{{ n_precursors or 1 }}" min="1" max="20">
                </div>
                <div class="param-box">
                    <label>Diffusion Steps</label>
                    <input type="number" name="diffusion_steps" value="{{ diffusion_steps or 1 }}" min="1" max="50" step="1">
                    <div class="param-hint">Must divide 50 (e.g. 1, 2, 5, 10, 25, 50)</div>
                </div>
            </div>

            <div class="btn-row">
                <button type="submit" class="btn-primary" id="submit-btn">🔬 Predict Precursors</button>
                <button type="button" class="btn-secondary" onclick="clearForm()">Clear</button>
            </div>

            <div id="progress-container" style="display:none; margin-top:20px;">
                <div style="background:#e9ecef; border-radius:8px; overflow:hidden; height:28px; position:relative;">
                    <div id="progress-bar" style="height:100%; width:0%; background:linear-gradient(90deg,#4a6fa5,#5a8fd5); border-radius:8px; transition:width 0.3s ease;"></div>
                    <span id="progress-text" style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); font-size:13px; font-weight:600; color:#333;"></span>
                </div>
            </div>
        </form>

        <div id="results-container">
            {{ results_html|safe }}
        </div>
    </div>

    <script>
        function dismissBanner() {
            var banner = document.getElementById('info-banner');
            if (banner) banner.style.display = 'none';
            try { localStorage.setItem('infoBannerDismissed', '1'); } catch(e) {}
        }
        (function() {
            try {
                if (localStorage.getItem('infoBannerDismissed') === '1') {
                    var b = document.getElementById('info-banner');
                    if (b) b.style.display = 'none';
                }
            } catch(e) {}
        })();
        function setSmiles(smiles) {
            document.getElementById('smiles-input').value = smiles;
        }
        function clearForm() {
            document.getElementById('smiles-input').value = '';
        }
        function lookupCompounds(btn, precursors) {
            var panel = btn.nextElementSibling;
            if (panel.style.display !== 'none') {
                panel.style.display = 'none';
                btn.textContent = 'Search PubChem';
                return;
            }
            btn.disabled = true;
            btn.textContent = 'Searching PubChem...';
            panel.style.display = 'block';
            panel.innerHTML = '<em>Looking up compounds...</em>';
            var smilesList = precursors.split('.');
            fetch('/api/evaluate/compound-lookup', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({smiles_list: smilesList})
            })
            .then(function(r) { return r.json(); })
            .then(function(data) {
                btn.disabled = false;
                btn.textContent = 'Hide PubChem Info';
                if (data.error) {
                    panel.innerHTML = '<span style="color:#c0392b;">' + data.error + '</span>';
                    return;
                }
                var html = '';
                data.compounds.forEach(function(c) {
                    html += '<div class="compound-entry">';
                    if (c.found) {
                        var name = (c.short_names && c.short_names.length > 0) ? c.short_names[0] : (c.iupac || c.smiles);
                        var aka = (c.short_names && c.short_names.length > 1) ? ' <span class="compound-detail">aka ' + c.short_names.slice(1, 4).join(', ') + '</span>' : '';
                        html += '<div class="compound-name">' + name + aka + '</div>';
                        html += '<div class="compound-detail">' + (c.formula || '') + ' &middot; MW ' + (c.mw || '?') + '</div>';
                        html += '<div class="compound-detail">Patents: ' + (c.n_patents || 0) + ' &middot; PubMed: ' + (c.n_pubmed || 0) + ' <span class="fame-score">Fame ' + (c.fame_score || 0) + '</span></div>';
                        html += '<div class="compound-detail"><a href="https://pubchem.ncbi.nlm.nih.gov/compound/' + c.cid + '" target="_blank" style="color:#4a6fa5;">PubChem CID ' + c.cid + '</a></div>';
                    } else {
                        html += '<div class="compound-detail">Not found: ' + c.smiles + '</div>';
                    }
                    html += '</div>';
                });
                panel.innerHTML = html;
            })
            .catch(function(err) {
                btn.disabled = false;
                btn.textContent = 'Search PubChem';
                panel.innerHTML = '<span style="color:#c0392b;">Lookup failed: ' + err.message + '</span>';
            });
        }

        (function() {
            var form = document.getElementById('predict-form');
            var submitBtn = document.getElementById('submit-btn');
            var progressContainer = document.getElementById('progress-container');
            var progressBar = document.getElementById('progress-bar');
            var progressText = document.getElementById('progress-text');
            var resultsContainer = document.getElementById('results-container');
            var timers = [];

            function setProgress(pct, text) {
                progressBar.style.width = pct + '%';
                progressText.textContent = text;
            }

            function clearTimers() {
                timers.forEach(function(t) { clearTimeout(t); });
                timers = [];
            }

            form.addEventListener('submit', function(e) {
                e.preventDefault();

                var stepsInput = form.querySelector('[name="diffusion_steps"]');
                var nInput = form.querySelector('[name="n_precursors"]');
                var steps = stepsInput ? stepsInput.value : '1';
                var n = nInput ? nInput.value : '1';

                // Show progress, hide old results
                resultsContainer.innerHTML = '';
                progressContainer.style.display = 'block';
                submitBtn.disabled = true;
                clearTimers();

                // Stage 0: immediate
                setProgress(5, 'Validating input...');

                // Stage 1
                timers.push(setTimeout(function() {
                    setProgress(15, 'Running DiffAlign with steps=' + steps + ' for ' + n + ' sample(s)...');
                }, 800));

                // Stage 2
                timers.push(setTimeout(function() {
                    setProgress(20, 'Preparing molecular graph...');
                }, 2500));

                // Stage 3: gradual progress from 20 to 85 over ~120s (asymptotic)
                var startTime = Date.now();
                var gradualInterval = setInterval(function() {
                    var elapsed = (Date.now() - startTime) / 1000;
                    // Asymptotic: 20 + 65 * (1 - e^(-elapsed/60))
                    var pct = 20 + 65 * (1 - Math.exp(-elapsed / 60));
                    if (pct > 84) pct = 84;
                    setProgress(Math.round(pct), 'Running diffusion model...');
                }, 1000);
                timers.push(gradualInterval);

                var formData = new FormData(form);
                fetch('/', {
                    method: 'POST',
                    body: formData,
                    headers: { 'X-Requested-With': 'XMLHttpRequest' }
                })
                .then(function(response) { return response.text(); })
                .then(function(html) {
                    clearTimers();
                    clearInterval(gradualInterval);
                    setProgress(100, 'Complete!');
                    setTimeout(function() {
                        progressContainer.style.display = 'none';
                        progressBar.style.width = '0%';
                        resultsContainer.innerHTML = html;
                        submitBtn.disabled = false;
                    }, 500);
                })
                .catch(function(err) {
                    clearTimers();
                    clearInterval(gradualInterval);
                    progressContainer.style.display = 'none';
                    progressBar.style.width = '0%';
                    resultsContainer.innerHTML = '<div class="results-section"><div class="error">Request failed: ' + err.message + '. Please try again.</div></div>';
                    submitBtn.disabled = false;
                });
            });
        })();
    </script>
</body>
</html>
"""


def mol_to_svg(mol, width=250, height=200):
    """Convert RDKit mol to SVG string (no X11 needed)."""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def mols_to_svg(mols, width=300, height=150):
    """Convert multiple RDKit mols to SVG grid."""
    n_mols = len(mols)
    mols_per_row = min(n_mols, 3)
    n_rows = (n_mols + mols_per_row - 1) // mols_per_row

    cell_width = width // mols_per_row
    cell_height = 120

    drawer = rdMolDraw2D.MolDraw2DSVG(width, cell_height * n_rows, cell_width, cell_height)
    drawer.DrawMolecules(mols)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()



# ============================================================


@application.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200


@application.route('/', methods=['GET', 'POST'])
def index():
    """Main page with form and results."""
    if request.method == 'GET':
        return render_template_string(HTML_TEMPLATE, results_html='')

    smiles = request.form.get('smiles', '').strip()
    n_precursors = request.form.get('n_precursors', 1, type=int)
    diffusion_steps = request.form.get('diffusion_steps', 1, type=int)

    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    def _render(error=None, results=None, target_svg=None, target_mw=0):
        results_html = render_template_string(
            RESULTS_TEMPLATE,
            error=error,
            results=results,
            smiles=smiles,
            target_svg=target_svg,
            target_mw=target_mw,
        )
        if is_ajax:
            return results_html
        return render_template_string(
            HTML_TEMPLATE,
            smiles=smiles,
            n_precursors=n_precursors,
            diffusion_steps=diffusion_steps,
            results_html=results_html,
        )

    if not smiles:
        return _render(error="Please enter a SMILES string.")

    # Range validation
    if not (1 <= n_precursors <= 20):
        return _render(error="Number of precursors must be between 1 and 20.")

    if not (1 <= diffusion_steps <= 50):
        return _render(error="Diffusion steps must be between 1 and 50.")

    # Validate diffusion_steps divides 50
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

    predictions = predict.predict_precursors(
        smiles,
        n_precursors=n_precursors,
        diffusion_steps=diffusion_steps,
    )

    results = []
    for pred in predictions:
        precursor_mols = [Chem.MolFromSmiles(s) for s in pred['precursors'].split('.')]
        precursor_mols = [m for m in precursor_mols if m is not None]

        if precursor_mols:
            svg = mols_to_svg(precursor_mols)
            results.append({
                'precursors': pred['precursors'],
                'score': pred['score'],
                'svg': svg
            })

    if not results:
        return _render(
            error="No valid precursors found for this molecule. Try increasing the number of precursors or diffusion steps."
        )

    try:
        classify_reactions(results, smiles)
    except Exception:
        pass  # graceful degradation

    return _render(
        results=results,
        target_svg=mol_to_svg(target_mol, 150, 150),
        target_mw=Descriptors.MolWt(target_mol),
    )


@application.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    data = request.get_json() or {}
    smiles = data.get('smiles', '').strip()

    if not smiles:
        return jsonify({'error': 'No SMILES provided'}), 400

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({'error': f'Invalid SMILES: {smiles}'}), 400

    predictions = predict.predict_precursors(
        smiles,
        n_precursors=data.get('n_precursors', 1),
        diffusion_steps=data.get('diffusion_steps', 1),
    )

    if data.get('evaluate', True):
        try:
            classify_reactions(predictions, smiles)
        except Exception:
            pass

    return jsonify({'target': smiles, 'predictions': predictions})


@application.route('/api/evaluate/compound-lookup', methods=['POST'])
def api_compound_lookup():
    """PubChem compound lookup endpoint."""
    data = request.get_json() or {}
    smiles_list = data.get('smiles_list', [])

    if not smiles_list:
        return jsonify({'error': 'No SMILES provided'}), 400

    if len(smiles_list) > 10:
        return jsonify({'error': 'Maximum 10 compounds per request'}), 400

    compounds = lookup_all_compounds(smiles_list)
    return jsonify({'compounds': compounds})


if __name__ == "__main__":
    application.run(debug=True, host='0.0.0.0', port=8080)
