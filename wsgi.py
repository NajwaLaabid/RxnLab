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

# testing
application = flask.Flask(__name__)

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
            max-width: 1000px;
            margin: 50px auto;
            padding: 20px;
            background: #f0f2f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.1);
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
        .precursor-card .mol-svg { width: 100%; border-radius: 4px; background: #fafafa; }
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
    </style>
</head>
<body>
    <div class="container">
        <h1>DiffAlign: Retrosynthesis through Diffusion</h1>
        <p class="subtitle">Enter your target molecule below</p>
        
        <form method="post" action="/">
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
                    <input type="number" name="n_precursors" value="{{ n_precursors or 5 }}" min="1" max="20">
                </div>
                <div class="param-box">
                    <label>Diffusion Steps</label>
                    <input type="number" name="diffusion_steps" value="{{ diffusion_steps or 10 }}" min="10" max="1000" step="10">
                </div>
                <div class="param-box">
                    <label>Temperature</label>
                    <input type="number" name="temperature" value="{{ temperature or 1.0 }}" min="0.1" max="2.0" step="0.1">
                </div>
                <div class="param-box">
                    <label>Beam Size</label>
                    <input type="number" name="beam_size" value="{{ beam_size or 10 }}" min="1" max="50">
                </div>
            </div>
            
            <div class="btn-row">
                <button type="submit" class="btn-primary">🔬 Predict Precursors</button>
                <button type="button" class="btn-secondary" onclick="clearForm()">Clear</button>
            </div>
        </form>
        
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
                    <div class="mol-svg">{{ result.svg | safe }}</div>
                    <div class="precursor-smiles">{{ result.precursors }}</div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
    </div>
    
    <script>
        function setSmiles(smiles) {
            document.getElementById('smiles-input').value = smiles;
        }
        function clearForm() {
            document.getElementById('smiles-input').value = '';
        }
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
        return render_template_string(HTML_TEMPLATE)
    
    smiles = request.form.get('smiles', '').strip()
    n_precursors = request.form.get('n_precursors', 2, type=int)
    diffusion_steps = request.form.get('diffusion_steps', 5, type=int)
    temperature = request.form.get('temperature', 1.0, type=float)
    beam_size = request.form.get('beam_size', 10, type=int)
    
    if not smiles:
        return render_template_string(HTML_TEMPLATE, error="Please enter a SMILES string.")
    
    target_mol = Chem.MolFromSmiles(smiles)
    if target_mol is None:
        return render_template_string(
            HTML_TEMPLATE,
            smiles=smiles,
            n_precursors=n_precursors,
            diffusion_steps=diffusion_steps,
            temperature=temperature,
            beam_size=beam_size,
            error=f"Invalid SMILES string: '{escape(smiles)}'"
        )
    
    predictions = predict.predict_precursors(
        smiles, 
        n_precursors=n_precursors,
        diffusion_steps=diffusion_steps,
        temperature=temperature,
        beam_size=beam_size
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
    
    return render_template_string(
        HTML_TEMPLATE,
        smiles=smiles,
        n_precursors=n_precursors,
        diffusion_steps=diffusion_steps,
        temperature=temperature,
        beam_size=beam_size,
        target_svg=mol_to_svg(target_mol, 150, 150),
        target_mw=Descriptors.MolWt(target_mol),
        results=results
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
        n_precursors=data.get('n_precursors', 5),
        diffusion_steps=data.get('diffusion_steps', 100),
        temperature=data.get('temperature', 1.0),
        beam_size=data.get('beam_size', 10)
    )
    
    return jsonify({'target': smiles, 'predictions': predictions})


if __name__ == "__main__":
    application.run(debug=True, host='0.0.0.0', port=8080)