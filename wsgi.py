#!/usr/bin/env python3
"""
Retrosynthesis prediction web app demo.
Designed for deployment on CSC Rahti.
Uses SVG rendering to avoid X11 dependencies.
"""
import json
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
    <script type="application/json" class="generation-results-json">{{ results_json | safe }}</script>
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
        <div class="precursor-card" data-result-index="{{ loop.index0 }}">
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
            <div class="precursor-actions">
                <button class="btn-lookup" onclick="lookupCompounds(this, '{{ result.precursors }}')">Search PubChem</button>
                <button class="btn-inpaint" onclick="enterInpaintMode(this)">Inpaint</button>
            </div>
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
    <script src="https://unpkg.com/@rdkit/rdkit/dist/RDKit_minimal.js"></script>
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
        .precursor-actions {
            display: flex;
            gap: 8px;
            margin-top: 10px;
        }
        .btn-lookup, .btn-inpaint {
            flex: 1;
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
        .btn-lookup:hover, .btn-inpaint:hover { background: #4a6fa5; color: white; }
        .btn-lookup:disabled, .btn-inpaint:disabled { opacity: 0.6; cursor: not-allowed; }
        .btn-inpaint { border-color: #e67e22; color: #e67e22; }
        .btn-inpaint:hover { background: #e67e22; color: white; }
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

        /* ── Inpainting styles ── */
        .inpaint-mode .precursor-card { border-color: #e67e22; }
        .inpaint-mode .mol-svg { cursor: crosshair; }
        .inpaint-toolbar {
            display: none;
            background: #fff8f0;
            border: 1px solid #e67e22;
            border-radius: 8px;
            padding: 12px;
            margin-top: 10px;
        }
        .inpaint-mode .inpaint-toolbar { display: block; }
        .inpaint-toolbar .toolbar-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }
        .inpaint-toolbar .toolbar-row:last-child { margin-bottom: 0; }
        .inpaint-counter {
            font-size: 13px;
            color: #666;
            flex: 1;
        }
        .btn-keep-mol {
            padding: 6px 14px;
            font-size: 12px;
            font-weight: 600;
            border: 1px solid #2d6a2d;
            border-radius: 4px;
            background: white;
            color: #2d6a2d;
            cursor: pointer;
        }
        .btn-keep-mol:hover, .btn-keep-mol.active { background: #2d6a2d; color: white; }
        .btn-regenerate {
            padding: 8px 20px;
            font-size: 14px;
            font-weight: 600;
            border: none;
            border-radius: 6px;
            background: #e67e22;
            color: white;
            cursor: pointer;
        }
        .btn-regenerate:hover { background: #d35400; }
        .btn-regenerate:disabled { background: #f0c8a0; cursor: not-allowed; }
        .btn-cancel-inpaint {
            padding: 8px 16px;
            font-size: 13px;
            border: 1px solid #ccc;
            border-radius: 6px;
            background: white;
            color: #666;
            cursor: pointer;
        }
        .btn-cancel-inpaint:hover { background: #f5f5f5; }

        /* Atom selection highlight */
        [data-selected="true"] { cursor: pointer; }

        /* ── Generation timeline ── */
        .generation-section {
            position: relative;
            margin-bottom: 20px;
        }
        .generation-section.previous { opacity: 0.7; }
        .generation-section.previous:hover { opacity: 1; }
        .generation-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }
        .generation-badge {
            background: #4a6fa5;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        .generation-info {
            font-size: 13px;
            color: #666;
        }
        .generation-connector {
            width: 2px;
            height: 20px;
            background: #4a6fa5;
            margin: 0 auto 10px;
        }
        .generation-connector-label {
            text-align: center;
            font-size: 11px;
            color: #888;
            margin-bottom: 10px;
        }
        .btn-reinpaint {
            padding: 6px 14px;
            font-size: 12px;
            font-weight: 600;
            border: 1px dashed #e67e22;
            border-radius: 6px;
            background: white;
            color: #e67e22;
            cursor: pointer;
        }
        .btn-reinpaint:hover { background: #fff8f0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>DiffAlign: Retrosynthesis through Diffusion</h1>
        <p class="subtitle">Enter your target molecule below</p>

        <div class="info-banner" id="info-banner">
            <div class="info-banner-content">
                <strong>Demo notice:</strong> This app runs DiffAlign on CPU only.
                Expect ~1 min per prediction for small molecules (10-20 atoms) with 50 diffusion steps.
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
                    <input type="number" name="n_precursors" value="{{ n_precursors or 1 }}" min="1" max="100">
                </div>
                <div class="param-box">
                    <label>Diffusion Steps</label>
                    <input type="number" name="diffusion_steps" value="{{ diffusion_steps or 1 }}" min="1" max="50" step="1">
                    <div class="param-hint">Must divide 50 (e.g. 1, 2, 5, 10, 25, 50)</div>
                </div>
            </div>

            <div class="btn-row">
                <button type="submit" class="btn-primary" id="submit-btn">Predict Precursors</button>
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
    /* ── Global state ── */
    var RDKitModule = null;
    var generations = [];  // [{results: [...], fixedInfo: null|str, targetSmiles: str}, ...]
    var currentInpaint = null;  // {genIdx, resultIdx, selectedAtoms: Set}
    var currentTargetSmiles = '';
    var MAX_GENERATIONS = 6;  // 1 initial + 5 inpainting rounds

    /* ── RDKit.js init ── */
    window.initRDKitModule().then(function(RDKit) {
        console.log('RDKit.js version: ' + RDKit.version());
        RDKitModule = RDKit;
    }).catch(function(err) {
        console.warn('RDKit.js failed to load:', err);
    });

    /* ── Utility functions ── */
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
        generations = [];
        currentInpaint = null;
        document.getElementById('results-container').innerHTML = '';
    }

    function lookupCompounds(btn, precursors) {
        var panel = btn.closest('.precursor-card').querySelector('.compound-info');
        if (panel.style.display !== 'none') {
            panel.style.display = 'none';
            btn.textContent = 'Search PubChem';
            return;
        }
        btn.disabled = true;
        btn.textContent = 'Searching...';
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
            btn.textContent = 'Hide PubChem';
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

    /* ── Progress bar helpers ── */
    var progressTimers = [];
    function setProgress(pct, text) {
        document.getElementById('progress-bar').style.width = pct + '%';
        document.getElementById('progress-text').textContent = text;
    }
    function clearProgressTimers() {
        progressTimers.forEach(function(t) { clearTimeout(t); clearInterval(t); });
        progressTimers = [];
    }
    function showProgress(label) {
        var pc = document.getElementById('progress-container');
        pc.style.display = 'block';
        setProgress(5, 'Validating input...');
        progressTimers.push(setTimeout(function() {
            setProgress(15, label);
        }, 800));
        progressTimers.push(setTimeout(function() {
            setProgress(20, 'Preparing molecular graph...');
        }, 2500));
        var startTime = Date.now();
        var iv = setInterval(function() {
            var elapsed = (Date.now() - startTime) / 1000;
            var pct = 20 + 65 * (1 - Math.exp(-elapsed / 60));
            if (pct > 84) pct = 84;
            setProgress(Math.round(pct), 'Running diffusion model...');
        }, 1000);
        progressTimers.push(iv);
    }
    function hideProgress() {
        clearProgressTimers();
        setProgress(100, 'Complete!');
        setTimeout(function() {
            document.getElementById('progress-container').style.display = 'none';
            document.getElementById('progress-bar').style.width = '0%';
        }, 500);
    }

    /* ── Parse results from server HTML and store in generations ── */
    function parseAndStoreGeneration(container, fixedInfo) {
        var jsonScript = container.querySelector('.generation-results-json');
        if (!jsonScript) return;
        try {
            var resultsData = JSON.parse(jsonScript.textContent);
            generations.push({
                results: resultsData,
                fixedInfo: fixedInfo,
                targetSmiles: currentTargetSmiles
            });
        } catch(e) {
            console.warn('Could not parse generation results:', e);
        }
    }

    /* ── Render the full timeline ── */
    function renderTimeline() {
        var container = document.getElementById('results-container');
        // We don't re-render from scratch — the server HTML is already in place.
        // This function re-renders RDKit.js molecules for cards in inpaint mode.
        // For now, it's called after inpaint results arrive.
    }

    /* ── Render precursors with RDKit.js (interactive SVGs with atom classes) ── */
    function renderMolWithRDKit(container, smiles, atomMap) {
        if (!RDKitModule) return false;
        try {
            var mol = RDKitModule.get_mol(smiles);
            if (!mol) return false;
            var svg = mol.get_svg(250, 150);
            mol.delete();
            container.innerHTML = svg;
            // Store atom map on the container
            container.setAttribute('data-atom-map', JSON.stringify(atomMap || {}));
            return true;
        } catch(e) {
            return false;
        }
    }

    /* ── Activate RDKit.js rendering on precursor cards for a generation ── */
    function activateRDKitRendering(genIdx) {
        if (!RDKitModule || !generations[genIdx]) return;
        var gen = generations[genIdx];
        // Find the container: .generation-section for inpaint generations, .results-section for initial
        var genSections = document.querySelectorAll('.generation-section');
        var genSection = genSections[genIdx] || document.querySelector('.results-section');
        if (!genSection) return;

        var cards = genSection.querySelectorAll('.precursor-card');
        cards.forEach(function(card) {
            var resultIdx = parseInt(card.getAttribute('data-result-index'));
            var result = gen.results[resultIdx];
            if (!result || !result.atom_mapping) return;

            var svgContainer = card.querySelector('.mol-svg');
            // Render each reactant molecule
            var precursorSmiles = result.precursors.split('.');
            var html = '';
            for (var m = 0; m < result.atom_mapping.length; m++) {
                var mi = result.atom_mapping[m];
                var molSmiles = mi.smiles;
                try {
                    var mol = RDKitModule.get_mol(molSmiles);
                    if (mol) {
                        var molSvg = mol.get_svg(200, 130);
                        mol.delete();
                        html += '<div class="rdkit-mol" data-mol-index="' + m + '" ' +
                                "data-atom-map='" + JSON.stringify(mi.atom_map) + "' " +
                                'data-smiles="' + molSmiles.replace(/"/g, '&quot;') + '">' +
                                molSvg + '</div>';
                    }
                } catch(e) {}
            }
            if (html) {
                svgContainer.innerHTML = html;
                svgContainer.classList.add('rdkit-rendered');
            }
        });
    }

    /* ── Inpaint mode ── */
    function enterInpaintMode(btn) {
        if (!RDKitModule) {
            alert('RDKit.js is still loading. Please wait a moment and try again.');
            return;
        }
        if (generations.length >= MAX_GENERATIONS) {
            alert('Maximum of ' + (MAX_GENERATIONS - 1) + ' inpainting rounds reached. Start a new prediction to continue.');
            return;
        }
        var card = btn.closest('.precursor-card');
        var genSection = card.closest('.generation-section') || card.closest('.results-section');
        var genIdx = 0;
        var genSections = document.querySelectorAll('.generation-section');
        genSections.forEach(function(s, i) { if (s === genSection) genIdx = i; });
        // If it's the initial results-section (not yet wrapped), genIdx = 0
        if (!genSection.classList.contains('generation-section')) {
            genIdx = generations.length - 1;
        }
        var resultIdx = parseInt(card.getAttribute('data-result-index'));

        // Ensure RDKit rendering is active
        if (!card.querySelector('.rdkit-rendered')) {
            activateRDKitRendering(genIdx);
        }

        currentInpaint = {
            genIdx: genIdx,
            resultIdx: resultIdx,
            selectedAtoms: new Set()
        };

        card.classList.add('inpaint-mode');

        // Add toolbar if not present
        if (!card.querySelector('.inpaint-toolbar')) {
            var toolbar = document.createElement('div');
            toolbar.className = 'inpaint-toolbar';
            toolbar.innerHTML =
                '<div class="toolbar-row">' +
                    '<span class="inpaint-counter">Click atoms to keep (yellow = fixed). Unselected atoms will be regenerated.</span>' +
                '</div>' +
                '<div class="toolbar-row">' +
                    '<span class="inpaint-counter" id="atom-count">0 atoms selected</span>' +
                '</div>' +
                '<div class="toolbar-row" id="keep-mol-buttons"></div>' +
                '<div class="toolbar-row">' +
                    '<button class="btn-regenerate" onclick="submitInpaint()" disabled>Regenerate</button>' +
                    '<button class="btn-cancel-inpaint" onclick="cancelInpaint()">Cancel</button>' +
                '</div>';
            card.appendChild(toolbar);
        } else {
            card.querySelector('.inpaint-toolbar').style.display = 'block';
        }

        // Add "Keep entire molecule" buttons for each reactant
        var gen = generations[genIdx];
        var result = gen.results[resultIdx];
        var keepBtnsContainer = card.querySelector('#keep-mol-buttons');
        keepBtnsContainer.innerHTML = '';
        if (result && result.atom_mapping) {
            result.atom_mapping.forEach(function(mi, molIdx) {
                var btn = document.createElement('button');
                btn.className = 'btn-keep-mol';
                btn.textContent = 'Keep molecule ' + (molIdx + 1) + ' (' + mi.smiles.substring(0, 20) + (mi.smiles.length > 20 ? '...' : '') + ')';
                btn.onclick = function() { toggleKeepMolecule(molIdx, btn); };
                keepBtnsContainer.appendChild(btn);
            });
        }

        // Set up click handlers on SVG atoms
        setupAtomClickHandlers(card);
    }

    function setupAtomClickHandlers(card) {
        var molContainers = card.querySelectorAll('.rdkit-mol');
        molContainers.forEach(function(molDiv) {
            var svgEl = molDiv.querySelector('svg');
            if (!svgEl) return;

            svgEl.addEventListener('click', function(event) {
                var target = event.target;
                // SVG elements have className as SVGAnimatedString; always use baseVal
                var className = '';
                if (target.className && typeof target.className === 'string') {
                    className = target.className;
                } else if (target.className && target.className.baseVal != null) {
                    className = target.className.baseVal;
                }
                if (!className) return;

                // Match atom-N class
                var atomMatch = className.match(/atom-(\\d+)/);
                if (!atomMatch) return;
                var rdkitAtomIdx = atomMatch[1];  // string key for the atom_map
                var atomMap = JSON.parse(molDiv.getAttribute('data-atom-map') || '{}');
                var denseIdx = atomMap[rdkitAtomIdx];
                if (denseIdx === undefined) return;

                // Toggle selection
                if (currentInpaint.selectedAtoms.has(denseIdx)) {
                    currentInpaint.selectedAtoms.delete(denseIdx);
                    clearAtomHighlight(svgEl, rdkitAtomIdx);
                } else {
                    currentInpaint.selectedAtoms.add(denseIdx);
                    highlightAtom(svgEl, rdkitAtomIdx);
                }
                updateAtomCount();
            });
        });
    }

    function getSvgClassName(el) {
        if (el.className && typeof el.className === 'string') return el.className;
        if (el.className && el.className.baseVal != null) return el.className.baseVal;
        return '';
    }

    function highlightAtom(svgEl, atomIdx) {
        var re = new RegExp('(^|\\s)atom-' + atomIdx + '(\\s|$)');
        svgEl.querySelectorAll('[class*="atom-' + atomIdx + '"]').forEach(function(el) {
            if (!re.test(getSvgClassName(el))) return;
            el.setAttribute('data-selected', 'true');
            el.setAttribute('data-orig-stroke', el.getAttribute('stroke') || '');
            el.setAttribute('data-orig-stroke-width', el.getAttribute('stroke-width') || '');
            el.setAttribute('data-orig-fill', el.getAttribute('fill') || '');
            var tag = el.tagName.toLowerCase();
            if (tag === 'path' || tag === 'line') {
                el.setAttribute('stroke', '#e67e22');
                el.setAttribute('stroke-width', '4');
            } else if (tag === 'text' || tag === 'tspan') {
                el.setAttribute('fill', '#e67e22');
                el.style.fontWeight = 'bold';
            } else if (tag === 'ellipse' || tag === 'circle' || tag === 'rect') {
                el.setAttribute('fill', 'rgba(230, 126, 34, 0.3)');
                el.setAttribute('stroke', '#e67e22');
                el.setAttribute('stroke-width', '2');
            }
        });
    }

    function clearAtomHighlight(svgEl, atomIdx) {
        var re = new RegExp('(^|\\s)atom-' + atomIdx + '(\\s|$)');
        svgEl.querySelectorAll('[data-selected="true"]').forEach(function(el) {
            if (!re.test(getSvgClassName(el))) return;
            el.removeAttribute('data-selected');
            el.setAttribute('stroke', el.getAttribute('data-orig-stroke') || '');
            el.setAttribute('stroke-width', el.getAttribute('data-orig-stroke-width') || '');
            el.setAttribute('fill', el.getAttribute('data-orig-fill') || '');
            el.style.fontWeight = '';
        });
    }

    function toggleKeepMolecule(molIdx, btn) {
        if (!currentInpaint) return;
        var genIdx = currentInpaint.genIdx;
        var resultIdx = currentInpaint.resultIdx;
        var gen = generations[genIdx];
        var result = gen.results[resultIdx];
        var mi = result.atom_mapping[molIdx];
        var atomMap = mi.atom_map;
        var allSelected = true;

        // Check if all atoms in this molecule are already selected
        for (var key in atomMap) {
            if (!currentInpaint.selectedAtoms.has(atomMap[key])) {
                allSelected = false;
                break;
            }
        }

        var card = btn.closest('.precursor-card');
        var molDiv = card.querySelectorAll('.rdkit-mol')[molIdx];
        var svgEl = molDiv ? molDiv.querySelector('svg') : null;

        if (allSelected) {
            // Deselect all
            for (var key in atomMap) {
                currentInpaint.selectedAtoms.delete(atomMap[key]);
                if (svgEl) clearAtomHighlight(svgEl, key);
            }
            btn.classList.remove('active');
        } else {
            // Select all
            for (var key in atomMap) {
                currentInpaint.selectedAtoms.add(atomMap[key]);
                if (svgEl) highlightAtom(svgEl, key);
            }
            btn.classList.add('active');
        }
        updateAtomCount();
    }

    function getSelectedSubSmiles() {
        // Build a summary of which atoms are selected per molecule
        if (!currentInpaint) return '';
        var gen = generations[currentInpaint.genIdx];
        if (!gen) return '';
        var result = gen.results[currentInpaint.resultIdx];
        if (!result || !result.atom_mapping) return '';
        var parts = [];
        result.atom_mapping.forEach(function(mi) {
            // Find which rdkit atom indices in this molecule are selected
            var selectedRdkit = [];
            for (var rdkitIdx in mi.atom_map) {
                if (currentInpaint.selectedAtoms.has(mi.atom_map[rdkitIdx])) {
                    selectedRdkit.push(parseInt(rdkitIdx));
                }
            }
            var total = Object.keys(mi.atom_map).length;
            if (selectedRdkit.length === total) {
                parts.push(mi.smiles);
            } else if (selectedRdkit.length > 0) {
                parts.push(selectedRdkit.length + '/' + total + ' atoms of ' + mi.smiles);
            }
        });
        return parts.join(' + ');
    }

    function updateAtomCount() {
        var countEl = document.getElementById('atom-count');
        var regenBtn = document.querySelector('.inpaint-mode .btn-regenerate');
        var n = currentInpaint ? currentInpaint.selectedAtoms.size : 0;
        var summary = getSelectedSubSmiles();
        var text = n + ' atom' + (n !== 1 ? 's' : '') + ' selected';
        if (summary) text += ': ' + summary;
        if (countEl) countEl.textContent = text;
        if (regenBtn) regenBtn.disabled = (n === 0);
    }

    function cancelInpaint() {
        if (!currentInpaint) return;
        var cards = document.querySelectorAll('.inpaint-mode');
        cards.forEach(function(card) {
            card.classList.remove('inpaint-mode');
            var toolbar = card.querySelector('.inpaint-toolbar');
            if (toolbar) toolbar.style.display = 'none';
            // Clear highlights
            card.querySelectorAll('[data-selected="true"]').forEach(function(el) {
                el.style.filter = '';
                el.removeAttribute('data-selected');
            });
        });
        currentInpaint = null;
    }

    /* ── Submit inpainting request ── */
    function submitInpaint() {
        if (!currentInpaint || currentInpaint.selectedAtoms.size === 0) return;

        if (generations.length >= MAX_GENERATIONS) {
            alert('Maximum of ' + (MAX_GENERATIONS - 1) + ' inpainting rounds reached. Start a new prediction to continue.');
            return;
        }

        var genIdx = currentInpaint.genIdx;
        var resultIdx = currentInpaint.resultIdx;
        var gen = generations[genIdx];
        var result = gen.results[resultIdx];

        var selectedNodes = Array.from(currentInpaint.selectedAtoms);
        var nPrecursors = parseInt(document.querySelector('[name="n_precursors"]').value) || 1;
        var diffusionSteps = parseInt(document.querySelector('[name="diffusion_steps"]').value) || 1;

        var summary = getSelectedSubSmiles();
        var fixedInfo = 'Fixed ' + summary + ' (' + selectedNodes.length + ' atoms) from #' + (resultIdx + 1) + ', gen ' + (genIdx + 1);

        cancelInpaint();

        // Show progress
        var submitBtn = document.getElementById('submit-btn');
        submitBtn.disabled = true;
        showProgress('Running inpainting with ' + selectedNodes.length + ' fixed atoms...');

        fetch('/api/inpaint', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                product_smiles: gen.targetSmiles,
                previous_sample_data: result.sample_data,
                selected_node_indices: selectedNodes,
                n_precursors: nPrecursors,
                diffusion_steps: diffusionSteps
            })
        })
        .then(function(r) { return r.json(); })
        .then(function(data) {
            hideProgress();
            submitBtn.disabled = false;
            if (data.error) {
                var rc = document.getElementById('results-container');
                rc.innerHTML += '<div class="results-section"><div class="error">' + data.error + '</div></div>';
                return;
            }
            addInpaintGeneration(data, fixedInfo);
        })
        .catch(function(err) {
            hideProgress();
            submitBtn.disabled = false;
            var rc = document.getElementById('results-container');
            rc.innerHTML += '<div class="results-section"><div class="error">Inpaint request failed: ' + err.message + '</div></div>';
        });
    }

    /* ── Add a new inpaint generation to the timeline ── */
    function addInpaintGeneration(data, fixedInfo) {
        var results = data.results;
        var targetSmiles = data.target_smiles;

        // Store generation
        generations.push({
            results: results,
            fixedInfo: fixedInfo,
            targetSmiles: targetSmiles
        });
        var genIdx = generations.length - 1;

        // Mark previous generations as dimmed
        document.querySelectorAll('.generation-section').forEach(function(s) {
            s.classList.add('previous');
        });
        // Also mark initial results section
        var initialSection = document.querySelector('.results-section:not(.generation-section)');
        if (initialSection && !initialSection.classList.contains('wrapped-as-gen')) {
            initialSection.classList.add('wrapped-as-gen');
            // Wrap it in a generation-section for consistency
            var wrapper = document.createElement('div');
            wrapper.className = 'generation-section previous';
            var header = document.createElement('div');
            header.className = 'generation-header';
            header.innerHTML = '<span class="generation-badge">Generation 1</span>' +
                '<span class="generation-info">Initial prediction</span>' +
                '<button class="btn-reinpaint" onclick="reinpaintFrom(0)">Re-inpaint from here</button>';
            initialSection.parentNode.insertBefore(wrapper, initialSection);
            wrapper.appendChild(header);
            wrapper.appendChild(initialSection);
        }

        // Build generation HTML
        var rc = document.getElementById('results-container');

        // Connector
        var connector = document.createElement('div');
        connector.innerHTML = '<div class="generation-connector"></div>' +
            '<div class="generation-connector-label">' + fixedInfo + '</div>';
        rc.appendChild(connector);

        // Generation section
        var section = document.createElement('div');
        section.className = 'generation-section';
        section.setAttribute('data-gen-index', genIdx);

        var headerHtml = '<div class="generation-header">' +
            '<span class="generation-badge">Generation ' + (genIdx + 1) + '</span>' +
            '<span class="generation-info">Inpainted: ' + fixedInfo + '</span>' +
            '</div>';

        var cardsHtml = '<div class="precursors-grid">';
        results.forEach(function(result, idx) {
            cardsHtml += '<div class="precursor-card" data-result-index="' + idx + '">' +
                '<div class="precursor-header">' +
                    '<span class="precursor-rank">#' + (idx + 1) + '</span>' +
                    '<span class="precursor-score">Score: ' + result.score.toFixed(3) + '</span>' +
                '</div>' +
                '<div class="mol-svg"></div>' +
                '<div class="precursor-smiles">' + result.precursors + '</div>' +
                '<div class="precursor-actions">' +
                    '<button class="btn-lookup" onclick="lookupCompounds(this, &quot;' + result.precursors.replace(/"/g, '&quot;') + '&quot;)">Search PubChem</button>' +
                    '<button class="btn-inpaint" onclick="enterInpaintMode(this)">Inpaint</button>' +
                '</div>' +
                '<div class="compound-info" style="display:none;"></div>' +
                '</div>';
        });
        cardsHtml += '</div>';

        section.innerHTML = headerHtml + cardsHtml;
        rc.appendChild(section);

        // Render molecules with RDKit.js (for atom selection)
        renderInpaintGenerationMols(section, results);

        // Scroll to the new generation
        section.scrollIntoView({behavior: 'smooth', block: 'start'});
    }

    function renderInpaintGenerationMols(section, results) {
        if (!RDKitModule) return;
        var cards = section.querySelectorAll('.precursor-card');
        cards.forEach(function(card) {
            var idx = parseInt(card.getAttribute('data-result-index'));
            var result = results[idx];
            if (!result || !result.atom_mapping) return;
            var svgContainer = card.querySelector('.mol-svg');
            var html = '';
            result.atom_mapping.forEach(function(mi, molIdx) {
                try {
                    var mol = RDKitModule.get_mol(mi.smiles);
                    if (mol) {
                        var svg = mol.get_svg(200, 130);
                        mol.delete();
                        html += '<div class="rdkit-mol" data-mol-index="' + molIdx + '" ' +
                                "data-atom-map='" + JSON.stringify(mi.atom_map) + "' " +
                                'data-smiles="' + mi.smiles.replace(/"/g, '&quot;') + '">' +
                                svg + '</div>';
                    }
                } catch(e) {}
            });
            if (html) {
                svgContainer.innerHTML = html;
                svgContainer.classList.add('rdkit-rendered');
            } else {
                // Fallback: show SMILES text
                svgContainer.innerHTML = '<div style="padding:10px;color:#888;">' + result.precursors + '</div>';
            }
        });
    }

    /* ── Re-inpaint from an earlier generation ── */
    function reinpaintFrom(genIdx) {
        // Remove all generations after genIdx
        while (generations.length > genIdx + 1) {
            generations.pop();
        }
        // Remove DOM elements for removed generations
        var allSections = document.querySelectorAll('.generation-section');
        var allConnectors = document.querySelectorAll('.generation-connector, .generation-connector-label');
        // Keep only up to genIdx
        allSections.forEach(function(s, i) {
            if (i > genIdx) s.remove();
        });
        // Remove orphaned connectors (those after the kept sections)
        var remaining = document.querySelectorAll('.generation-section');
        var lastSection = remaining[remaining.length - 1];
        if (lastSection) {
            var sibling = lastSection.nextElementSibling;
            while (sibling) {
                var next = sibling.nextElementSibling;
                if (sibling.classList.contains('generation-connector') ||
                    sibling.classList.contains('generation-connector-label') ||
                    (sibling.classList.contains('generation-section') && sibling !== lastSection)) {
                    sibling.remove();
                }
                sibling = next;
            }
        }
        // Un-dim the last generation
        if (lastSection) lastSection.classList.remove('previous');
    }

    /* ── Initial form submission (AJAX) ── */
    (function() {
        var form = document.getElementById('predict-form');
        var submitBtn = document.getElementById('submit-btn');
        var resultsContainer = document.getElementById('results-container');

        form.addEventListener('submit', function(e) {
            e.preventDefault();
            var stepsInput = form.querySelector('[name="diffusion_steps"]');
            var nInput = form.querySelector('[name="n_precursors"]');
            var steps = stepsInput ? stepsInput.value : '1';
            var n = nInput ? nInput.value : '1';

            currentTargetSmiles = document.getElementById('smiles-input').value.trim();
            generations = [];
            currentInpaint = null;

            resultsContainer.innerHTML = '';
            submitBtn.disabled = true;
            showProgress('Running DiffAlign with steps=' + steps + ' for ' + n + ' sample(s)...');

            var formData = new FormData(form);
            fetch('/', {
                method: 'POST',
                body: formData,
                headers: { 'X-Requested-With': 'XMLHttpRequest' }
            })
            .then(function(response) { return response.text(); })
            .then(function(html) {
                hideProgress();
                submitBtn.disabled = false;
                resultsContainer.innerHTML = html;
                // Parse and store generation 0
                parseAndStoreGeneration(resultsContainer, null);
                // Activate RDKit.js rendering
                if (generations.length > 0) {
                    activateRDKitRendering(0);
                }
            })
            .catch(function(err) {
                hideProgress();
                submitBtn.disabled = false;
                resultsContainer.innerHTML = '<div class="results-section"><div class="error">Request failed: ' + err.message + '. Please try again.</div></div>';
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


def _serialize_results_json(results):
    """Build JSON-safe version of results for embedding in a <script> tag."""
    safe = []
    for r in results:
        safe.append({
            'precursors': r['precursors'],
            'score': r['score'],
            'sample_data': r.get('sample_data'),
            'atom_mapping': r.get('atom_mapping'),
        })
    # Escape </script> sequences that could break out of the <script> tag
    return json.dumps(safe).replace('</', '<\\/')


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

    def _render(error=None, results=None, target_svg=None, target_mw=0, results_json='[]'):
        results_html = render_template_string(
            RESULTS_TEMPLATE,
            error=error,
            results=results,
            results_json=results_json,
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
    if not (1 <= n_precursors <= 100):
        return _render(error="Number of precursors must be between 1 and 100.")

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
        pass  # graceful degradation

    return _render(
        results=results,
        target_svg=mol_to_svg(target_mol, 150, 150),
        target_mw=Descriptors.MolWt(target_mol),
        results_json=_serialize_results_json(results),
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


@application.route('/api/inpaint', methods=['POST'])
def api_inpaint():
    """Inpainting endpoint: regenerate with selected atoms fixed."""
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

    # Validate
    if not isinstance(selected_node_indices, list):
        return jsonify({'error': 'selected_node_indices must be a list'}), 400
    if not (1 <= n_precursors <= 100):
        return jsonify({'error': 'n_precursors must be between 1 and 100'}), 400

    try:
        results = predict.predict_with_inpainting(
            product_smiles=product_smiles,
            previous_sample_data=previous_sample_data,
            inpaint_node_indices=selected_node_indices,
            n_precursors=n_precursors,
            diffusion_steps=diffusion_steps,
        )
    except Exception as e:
        return jsonify({'error': f'Inpainting failed: {str(e)}'}), 500

    # Add SVGs for initial display
    for r in results:
        precursor_mols = [Chem.MolFromSmiles(s) for s in r['precursors'].split('.')]
        precursor_mols = [m for m in precursor_mols if m is not None]
        r['svg'] = mols_to_svg(precursor_mols) if precursor_mols else ''

    return jsonify({
        'target_smiles': product_smiles,
        'results': results,
        'fixed_atoms_info': f'{len(selected_node_indices)} atoms fixed',
    })


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
