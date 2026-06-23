/* RxnLab single-step/multi-step predict page — progress bar + result parsing/rendering.
   Split from the former inline <script> in predict.html. All functions are
   globals (called from inline handlers); load order set in predict.html. */

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
    pc.scrollIntoView({ behavior: 'smooth', block: 'center' });
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

function escapeHtml(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
