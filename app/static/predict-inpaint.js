/* RxnLab single-step/multi-step predict page — inpainting (atom select, lasso, submit, timeline).
   Split from the former inline <script> in predict.html. All functions are
   globals (called from inline handlers); load order set in predict.html. */

/* ── Inpaint mode ── */
function enterInpaintMode(btn) {
    if (!RDKitModule) {
        alert('RDKit.js is still loading. Please wait a moment and try again.');
        return;
    }
    if (generations.length >= MAX_GENERATIONS) {
        alert('Maximum of ' + (MAX_GENERATIONS - 1) + ' edit rounds reached. Start a new prediction to continue.');
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
        selectedAtoms: new Set(),
        mode: 'regenerate',
        lassoActive: false,
        lassoPoints: []
    };

    card.classList.add('inpaint-mode');
    card.classList.add('inpaint-focus-panel');

    // Store original SVG HTML so we can restore on cancel
    var svgContainer = card.querySelector('.mol-svg');
    if (svgContainer && !svgContainer.getAttribute('data-orig-html')) {
        svgContainer.setAttribute('data-orig-html', svgContainer.innerHTML);
    }

    // Re-render molecules at larger size with atom indices
    var gen = generations[genIdx];
    var result = gen.results[resultIdx];
    renderInpaintMolecules(card, result);

    // Add toolbar if not present
    if (!card.querySelector('.inpaint-toolbar')) {
        var toolbar = document.createElement('div');
        toolbar.className = 'inpaint-toolbar';
        toolbar.innerHTML =
            '<div class="toolbar-row">' +
                '<div class="mode-toggle">' +
                    '<button class="mode-btn active-regenerate" data-mode="regenerate" onclick="setInpaintMode(\'regenerate\', this)">Select atoms to CHANGE</button>' +
                    '<button class="mode-btn" data-mode="keep" onclick="setInpaintMode(\'keep\', this)">Select atoms to KEEP</button>' +
                '</div>' +
            '</div>' +
            '<div class="toolbar-row">' +
                '<span class="inpaint-counter inpaint-instruction">Click atoms you want to regenerate (red = will change). Other atoms stay fixed.</span>' +
            '</div>' +
            '<div class="toolbar-row">' +
                '<span class="inpaint-counter inpaint-atom-count">0 atoms selected</span>' +
                '<button class="btn-cancel-inpaint" onclick="selectAllAtoms()" style="font-size:12px;padding:4px 10px;">Select All</button>' +
                '<button class="btn-cancel-inpaint" onclick="deselectAllAtoms()" style="font-size:12px;padding:4px 10px;">Deselect All</button>' +
            '</div>' +
            '<div class="toolbar-row keep-mol-buttons"></div>' +
            '<div class="toolbar-row" style="gap:16px;">' +
                '<label style="font-size:12px;color:#555;display:flex;align-items:center;gap:4px;">' +
                    'Precursors <input type="number" class="inpaint-n-precursors" min="1" max="100" value="10" style="width:50px;padding:3px 6px;border:1px solid #ccc;border-radius:4px;font-size:12px;">' +
                '</label>' +
                '<label style="font-size:12px;color:#555;display:flex;align-items:center;gap:4px;">' +
                    'Diff. steps ' +
                    '<select class="inpaint-diff-steps" style="padding:3px 6px;border:1px solid #ccc;border-radius:4px;font-size:12px;">' +
                        '<option value="1">1</option>' +
                        '<option value="2">2</option>' +
                        '<option value="5">5</option>' +
                        '<option value="10" selected>10</option>' +
                        '<option value="25">25</option>' +
                        '<option value="50">50</option>' +
                    '</select>' +
                '</label>' +
            '</div>' +
            '<div class="toolbar-row">' +
                '<button class="btn-lasso" onclick="toggleLasso(this)">Lasso Select</button>' +
                '<button class="btn-regenerate" onclick="submitInpaint()" disabled>Regenerate</button>' +
                '<button class="btn-cancel-inpaint" onclick="cancelInpaint()">Cancel</button>' +
            '</div>';
        card.appendChild(toolbar);
    } else {
        var existingToolbar = card.querySelector('.inpaint-toolbar');
        existingToolbar.style.display = 'block';
        // Reset mode toggle to default (regenerate)
        existingToolbar.querySelectorAll('.mode-btn').forEach(function(b) {
            b.className = 'mode-btn';
            if (b.getAttribute('data-mode') === 'regenerate') b.classList.add('active-regenerate');
        });
        var instrEl = existingToolbar.querySelector('.inpaint-instruction');
        if (instrEl) instrEl.textContent = 'Click atoms you want to regenerate (red = will change). Other atoms stay fixed.';
        var countEl = existingToolbar.querySelector('.inpaint-atom-count');
        if (countEl) countEl.textContent = '0 atoms selected to change';
        var regenBtn = existingToolbar.querySelector('.btn-regenerate');
        if (regenBtn) regenBtn.disabled = true;
    }

    // Add molecule shortcut buttons for each reactant
    var keepBtnsContainer = card.querySelector('.keep-mol-buttons');
    keepBtnsContainer.innerHTML = '';
    if (result && result.atom_mapping) {
        result.atom_mapping.forEach(function(mi, molIdx) {
            var molBtn = document.createElement('button');
            molBtn.className = 'btn-keep-mol';
            molBtn.setAttribute('data-mol-idx', molIdx);
            molBtn.textContent = getMolButtonLabel(molIdx, mi.smiles);
            molBtn.onclick = function() { toggleKeepMolecule(molIdx, molBtn); };
            keepBtnsContainer.appendChild(molBtn);
        });
    }

    // Set up click handlers on SVG atoms (event delegation)
    setupAtomClickHandlers(card);
}

/* Render molecules at enlarged size with atom indices for inpaint mode */
function renderInpaintMolecules(card, result) {
    if (!RDKitModule || !result || !result.atom_mapping) return;
    var svgContainer = card.querySelector('.mol-svg');
    var html = '';
    result.atom_mapping.forEach(function(mi, molIdx) {
        try {
            var mol = RDKitModule.get_mol(mi.smiles);
            if (mol) {
                var mdetails = {
                    addAtomIndices: true,
                    annotationFontScale: 0.7,
                    bondLineWidth: 2.0
                };
                var svg = mol.get_svg_with_highlights(JSON.stringify(mdetails));
                mol.delete();
                // Resize SVG to 400x260
                svg = svg.replace(/width='(\d+)px'/, "width='400px'").replace(/height='(\d+)px'/, "height='260px'");
                html += '<div class="rdkit-mol" data-mol-index="' + molIdx + '" ' +
                        "data-atom-map='" + JSON.stringify(mi.atom_map) + "' " +
                        'data-smiles="' + mi.smiles.replace(/"/g, '&quot;') + '">' +
                        svg + '</div>';
            }
        } catch(e) {
            console.warn('RDKit render error for', mi.smiles, e);
        }
    });
    if (html) {
        svgContainer.innerHTML = html;
        svgContainer.classList.add('rdkit-rendered');
        // Add invisible hit-area circles for reliable click targeting
        svgContainer.querySelectorAll('.rdkit-mol').forEach(function(molDiv) {
            addAtomHitAreas(molDiv);
        });
    }
}

/* Get label for molecule shortcut button based on current mode */
function getMolButtonLabel(molIdx, smiles) {
    var mode = (currentInpaint && currentInpaint.mode) || 'regenerate';
    var prefix = mode === 'regenerate' ? 'Regenerate' : 'Keep';
    var truncated = smiles.substring(0, 20) + (smiles.length > 20 ? '...' : '');
    return prefix + ' mol ' + (molIdx + 1) + ' (' + truncated + ')';
}

function setupAtomClickHandlers(card) {
    var molContainers = card.querySelectorAll('.rdkit-mol');
    molContainers.forEach(function(molDiv) {
        // Use event delegation on the molDiv (not svgEl) so handlers survive SVG re-renders
        if (molDiv.getAttribute('data-click-bound')) return;
        molDiv.setAttribute('data-click-bound', 'true');

        molDiv.addEventListener('click', function(event) {
            if (currentInpaint && currentInpaint.lassoActive) return; // skip in lasso mode
            var target = event.target;
            // SVG elements have className as SVGAnimatedString; always use baseVal
            var className = '';
            if (target.className && typeof target.className === 'string') {
                className = target.className;
            } else if (target.className && target.className.baseVal != null) {
                className = target.className.baseVal;
            }
            if (!className) return;

            // Match atom-N class (exact match with word boundary)
            var atomMatch = className.match(/\batom-(\d+)\b/);
            if (!atomMatch) return;
            var rdkitAtomIdx = atomMatch[1];  // string key for the atom_map
            var atomMap = JSON.parse(molDiv.getAttribute('data-atom-map') || '{}');
            var denseIdx = atomMap[rdkitAtomIdx];
            if (denseIdx === undefined) return;

            // Toggle selection
            if (currentInpaint.selectedAtoms.has(denseIdx)) {
                currentInpaint.selectedAtoms.delete(denseIdx);
            } else {
                currentInpaint.selectedAtoms.add(denseIdx);
            }

            // Re-render this molecule with updated highlights
            var selectedRdkit = [];
            for (var key in atomMap) {
                if (currentInpaint.selectedAtoms.has(atomMap[key])) {
                    selectedRdkit.push(parseInt(key));
                }
            }
            reRenderMolWithHighlights(molDiv, selectedRdkit);
            updateAtomCount();
        });
    });
}

/* Parse a molblock and return indices of bonds whose endpoints are both in
 * `selectedSet` (a plain object map from atomIdx -> truthy). Used for
 * bond-inclusive highlight rendering. */
function bondsBetweenSelectedAtoms(mol, selectedSet) {
    var out = [];
    try {
        var molblock = mol.get_molblock();
        var lines = molblock.split('\n');
        // V2000 counts line is line index 3: "%3d%3d..." -> atoms, bonds
        var counts = lines[3];
        var nAtoms = parseInt(counts.substring(0, 3));
        var nBonds = parseInt(counts.substring(3, 6));
        for (var i = 0; i < nBonds; i++) {
            var bondLine = lines[4 + nAtoms + i];
            var a1 = parseInt(bondLine.substring(0, 3)) - 1;  // V2000 is 1-indexed
            var a2 = parseInt(bondLine.substring(3, 6)) - 1;
            if (selectedSet[a1] && selectedSet[a2]) out.push(i);
        }
    } catch (e) { /* best effort; fall back to atom-only highlights */ }
    return out;
}

/* Re-render a single molecule SVG with RDKit.js native highlighting */
function reRenderMolWithHighlights(molDiv, selectedRdkitAtoms) {
    var smiles = molDiv.getAttribute('data-smiles');
    if (!smiles || !RDKitModule) return;
    var mol = RDKitModule.get_mol(smiles);
    if (!mol) return;

    var mode = (currentInpaint && currentInpaint.mode) || 'regenerate';
    var color = mode === 'regenerate' ? [0.91, 0.30, 0.24] : [0.18, 0.62, 0.78];

    // Also highlight bonds whose both endpoints are selected — this makes
    // selection visible on implicit-H carbons that RDKit doesn't label in SVG.
    var selectedSet = {};
    selectedRdkitAtoms.forEach(function(a) { selectedSet[a] = true; });
    var bondsToHighlight = bondsBetweenSelectedAtoms(mol, selectedSet);

    var mdetails = {
        atoms: selectedRdkitAtoms,
        bonds: bondsToHighlight,
        highlightColour: color,
        fillHighlights: true,
        addAtomIndices: true,
        annotationFontScale: 0.7,
        highlightBondWidthMultiplier: 8,
        bondLineWidth: 2.0
    };

    var svg = mol.get_svg_with_highlights(JSON.stringify(mdetails));
    mol.delete();

    // Replace the SVG content (event delegation on molDiv survives this)
    var oldSvg = molDiv.querySelector('svg');
    if (oldSvg) {
        oldSvg.outerHTML = svg;
    } else {
        molDiv.innerHTML = svg;
    }

    // Apply inpaint-mode size via viewBox scaling
    var newSvg = molDiv.querySelector('svg');
    if (newSvg) {
        newSvg.setAttribute('width', '400');
        newSvg.setAttribute('height', '260');
    }

    // Add invisible hit-area circles for reliable click targeting
    addAtomHitAreas(molDiv);
}

/**
 * Add transparent circle overlays at every atom position.
 * Solves: hidden carbons have no SVG elements; visible atoms are too small.
 * Strategy: gather positions from existing atom elements + bond path endpoints,
 * then append large transparent circles with the atom-N class.
 */
function addAtomHitAreas(molDiv) {
    var svgEl = molDiv.querySelector('svg');
    if (!svgEl) return;
    var atomMap = JSON.parse(molDiv.getAttribute('data-atom-map') || '{}');
    var atomPositions = {};  // rdkitIdx -> {x, y}

    // Pass 1: find positions from elements that have atom-N classes (visible atoms)
    for (var rdkitIdx in atomMap) {
        var re = new RegExp('(^|\\s)atom-' + rdkitIdx + '(\\s|$)');
        var els = svgEl.querySelectorAll('[class*="atom-' + rdkitIdx + '"]');
        var cx = 0, cy = 0, count = 0;
        els.forEach(function(el) {
            var cls = (el.className && el.className.baseVal != null) ? el.className.baseVal : (el.className || '');
            if (!re.test(cls)) return;
            if (el.classList && el.classList.contains('atom-hit-area')) return; // skip our own overlays
            try {
                var bbox = el.getBBox();
                if (bbox.width > 0 || bbox.height > 0) {
                    cx += bbox.x + bbox.width / 2;
                    cy += bbox.y + bbox.height / 2;
                    count++;
                }
            } catch(e) {}
        });
        if (count > 0) {
            atomPositions[rdkitIdx] = {x: cx / count, y: cy / count};
        }
    }

    // Pass 2: for atoms without positions, extract from bond path endpoints
    // Bond paths have class="bond-N atom-A atom-B" and d="M x1,y1 L x2,y2 ..."
    var bondPaths = svgEl.querySelectorAll('path[class*="bond-"]');
    bondPaths.forEach(function(path) {
        var cls = (path.className && path.className.baseVal != null) ? path.className.baseVal : '';
        var d = path.getAttribute('d') || '';
        // Extract atom indices from class
        var atomMatches = cls.match(/atom-(\d+)/g);
        if (!atomMatches || atomMatches.length < 2) return;
        var idx0 = atomMatches[0].replace('atom-', '');
        var idx1 = atomMatches[1].replace('atom-', '');
        // Parse first and last coordinates from d attribute
        var coords = extractPathEndpoints(d);
        if (!coords) return;
        // First coord → first atom, last coord → second atom
        if (!atomPositions[idx0] && atomMap[idx0] !== undefined) {
            atomPositions[idx0] = coords.start;
        }
        if (!atomPositions[idx1] && atomMap[idx1] !== undefined) {
            atomPositions[idx1] = coords.end;
        }
    });

    // Pass 3: create transparent circles at each atom position
    for (var idx in atomPositions) {
        if (atomMap[idx] === undefined) continue;
        var pos = atomPositions[idx];
        var circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', pos.x.toFixed(1));
        circle.setAttribute('cy', pos.y.toFixed(1));
        circle.setAttribute('r', '14');
        circle.setAttribute('fill', 'transparent');
        circle.setAttribute('stroke', 'none');
        circle.setAttribute('class', 'atom-' + idx + ' atom-hit-area');
        circle.style.cursor = 'pointer';
        circle.style.pointerEvents = 'all';
        svgEl.appendChild(circle);
    }
}

/** Extract start and end points from an SVG path d attribute */
function extractPathEndpoints(d) {
    // Match M/m commands (move to) and L/l/C/c/Q/q etc for endpoints
    var numbers = d.match(/-?[\d.]+/g);
    if (!numbers || numbers.length < 4) return null;
    return {
        start: {x: parseFloat(numbers[0]), y: parseFloat(numbers[1])},
        end: {x: parseFloat(numbers[numbers.length - 2]), y: parseFloat(numbers[numbers.length - 1])}
    };
}

/* Re-render all molecules in the current inpaint card with updated highlights */
function reRenderAllMolHighlights() {
    if (!currentInpaint) return;
    var card = document.querySelector('.inpaint-mode');
    if (!card) return;
    var molDivs = card.querySelectorAll('.rdkit-mol');
    molDivs.forEach(function(molDiv) {
        var atomMap = JSON.parse(molDiv.getAttribute('data-atom-map') || '{}');
        var selectedRdkit = [];
        for (var key in atomMap) {
            if (currentInpaint.selectedAtoms.has(atomMap[key])) {
                selectedRdkit.push(parseInt(key));
            }
        }
        reRenderMolWithHighlights(molDiv, selectedRdkit);
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

    if (allSelected) {
        // Deselect all
        for (var key in atomMap) {
            currentInpaint.selectedAtoms.delete(atomMap[key]);
        }
        btn.classList.remove('active');
    } else {
        // Select all
        for (var key in atomMap) {
            currentInpaint.selectedAtoms.add(atomMap[key]);
        }
        btn.classList.add('active');
    }

    // Re-render this molecule with updated highlights
    var card = btn.closest('.precursor-card');
    var molDiv = card.querySelectorAll('.rdkit-mol')[molIdx];
    if (molDiv) {
        var selectedRdkit = [];
        for (var key in atomMap) {
            if (currentInpaint.selectedAtoms.has(atomMap[key])) {
                selectedRdkit.push(parseInt(key));
            }
        }
        reRenderMolWithHighlights(molDiv, selectedRdkit);
    }
    updateAtomCount();
}

/* Parse molblock atom + bond sections so we can walk the selected sub-graph.
 * Returns {symbols: {idx: 'C'|'N'|...}, bonds: [{a:i, b:j, order:1|2|3}, ...]}
 * or null on parse failure. */
function parseMolGraph(mol) {
    try {
        var lines = mol.get_molblock().split('\n');
        var counts = lines[3];
        var nAtoms = parseInt(counts.substring(0, 3));
        var nBonds = parseInt(counts.substring(3, 6));
        var symbols = {};
        for (var i = 0; i < nAtoms; i++) {
            var atomLine = lines[4 + i];
            // V2000 atom symbol lives at columns 31-34 (after 3x10-char coords)
            var sym = atomLine.substring(31, 34).trim();
            symbols[i] = sym || '*';
        }
        var bonds = [];
        for (var j = 0; j < nBonds; j++) {
            var bondLine = lines[4 + nAtoms + j];
            bonds.push({
                a: parseInt(bondLine.substring(0, 3)) - 1,
                b: parseInt(bondLine.substring(3, 6)) - 1,
                order: parseInt(bondLine.substring(6, 9)) || 1,
            });
        }
        return {symbols: symbols, bonds: bonds};
    } catch (e) { return null; }
}

/* Walk connected components inside the selected atom subset, emitting a
 * short SMILES-like fragment descriptor per component ("CC", "C=O", etc). */
function selectedFragmentsFor(mol, selectedSet) {
    var g = parseMolGraph(mol);
    if (!g) return [];
    // Adjacency inside the selection only
    var adj = {};
    Object.keys(selectedSet).forEach(function(k) { adj[k] = []; });
    g.bonds.forEach(function(bd) {
        if (selectedSet[bd.a] && selectedSet[bd.b]) {
            adj[bd.a].push({n: bd.b, order: bd.order});
            adj[bd.b].push({n: bd.a, order: bd.order});
        }
    });
    var orderChar = {1: '', 2: '=', 3: '#', 4: ''};  // 4 = aromatic, render as single
    var visited = {};
    var fragments = [];
    Object.keys(selectedSet).sort(function(a, b) { return parseInt(a) - parseInt(b); })
        .forEach(function(startKey) {
            if (visited[startKey]) return;
            var start = parseInt(startKey);
            // DFS emitting symbols and bond orders along first-visit edges
            var frag = g.symbols[start] || '?';
            visited[start] = true;
            var stack = [start];
            while (stack.length) {
                var node = stack.pop();
                (adj[node] || []).forEach(function(edge) {
                    if (!visited[edge.n]) {
                        visited[edge.n] = true;
                        frag += (orderChar[edge.order] || '') + (g.symbols[edge.n] || '?');
                        stack.push(edge.n);
                    }
                });
            }
            fragments.push(frag);
        });
    return fragments;
}

function getSelectedSubSmiles() {
    if (!currentInpaint || !RDKitModule) return '';
    var gen = generations[currentInpaint.genIdx];
    if (!gen) return '';
    var result = gen.results[currentInpaint.resultIdx];
    if (!result || !result.atom_mapping) return '';
    var parts = [];
    result.atom_mapping.forEach(function(mi) {
        var selectedSet = {};
        for (var rdkitIdx in mi.atom_map) {
            if (currentInpaint.selectedAtoms.has(mi.atom_map[rdkitIdx])) {
                selectedSet[parseInt(rdkitIdx)] = true;
            }
        }
        if (Object.keys(selectedSet).length === 0) return;
        var mol = RDKitModule.get_mol(mi.smiles);
        if (!mol) return;
        var frags = selectedFragmentsFor(mol, selectedSet);
        mol.delete();
        if (frags.length) parts.push(frags.join(', '));
    });
    return parts.join(' + ');
}

function updateAtomCount() {
    var countEl = document.querySelector('.inpaint-mode .inpaint-atom-count');
    var regenBtn = document.querySelector('.inpaint-mode .btn-regenerate');
    var n = currentInpaint ? currentInpaint.selectedAtoms.size : 0;
    var mode = (currentInpaint && currentInpaint.mode) || 'regenerate';
    var summary = getSelectedSubSmiles();
    var verb = mode === 'regenerate' ? ' to change' : ' to keep';
    var text = n + ' atom' + (n !== 1 ? 's' : '') + ' selected' + verb;
    if (summary) text += ': ' + summary;
    if (countEl) countEl.textContent = text;
    if (regenBtn) regenBtn.disabled = (n === 0);
}

/* ── Mode toggle: regenerate vs keep ── */
function setInpaintMode(mode, btn) {
    if (!currentInpaint) return;
    currentInpaint.mode = mode;
    // Update toggle button styles
    btn.parentElement.querySelectorAll('.mode-btn').forEach(function(b) {
        b.className = 'mode-btn';
        if (b.getAttribute('data-mode') === mode) {
            b.classList.add(mode === 'regenerate' ? 'active-regenerate' : 'active-keep');
        }
    });
    // Update instruction text
    var instrEl = document.querySelector('.inpaint-mode .inpaint-instruction');
    if (instrEl) {
        if (mode === 'regenerate') {
            instrEl.textContent = 'Click atoms you want to regenerate (red = will change). Other atoms stay fixed.';
        } else {
            instrEl.textContent = 'Click atoms you want to keep (blue = stays fixed). Other atoms will be regenerated.';
        }
    }
    // Update molecule shortcut button labels
    var card = document.querySelector('.inpaint-mode');
    if (card) {
        var gen = generations[currentInpaint.genIdx];
        var result = gen.results[currentInpaint.resultIdx];
        card.querySelectorAll('.btn-keep-mol').forEach(function(molBtn) {
            var molIdx = parseInt(molBtn.getAttribute('data-mol-idx'));
            if (result && result.atom_mapping && result.atom_mapping[molIdx]) {
                molBtn.textContent = getMolButtonLabel(molIdx, result.atom_mapping[molIdx].smiles);
            }
        });
    }
    // Re-render all highlights with new color
    reRenderAllMolHighlights();
    updateAtomCount();
}

/* ── Select All / Deselect All ── */
function selectAllAtoms() {
    if (!currentInpaint) return;
    var gen = generations[currentInpaint.genIdx];
    var result = gen.results[currentInpaint.resultIdx];
    if (!result || !result.atom_mapping) return;
    result.atom_mapping.forEach(function(mi) {
        for (var key in mi.atom_map) {
            currentInpaint.selectedAtoms.add(mi.atom_map[key]);
        }
    });
    reRenderAllMolHighlights();
    updateAtomCount();
    // Update molecule shortcut buttons
    document.querySelectorAll('.btn-keep-mol').forEach(function(b) { b.classList.add('active'); });
}

function deselectAllAtoms() {
    if (!currentInpaint) return;
    currentInpaint.selectedAtoms.clear();
    reRenderAllMolHighlights();
    updateAtomCount();
    document.querySelectorAll('.btn-keep-mol').forEach(function(b) { b.classList.remove('active'); });
}

/* ── Lasso selection ── */
function toggleLasso(btn) {
    if (!currentInpaint) return;
    currentInpaint.lassoActive = !currentInpaint.lassoActive;
    currentInpaint.lassoPoints = [];
    btn.classList.toggle('active', currentInpaint.lassoActive);
    // Update cursor on all molecule SVGs
    document.querySelectorAll('.inpaint-mode .rdkit-mol svg').forEach(function(svg) {
        svg.style.cursor = currentInpaint.lassoActive ? 'crosshair' : 'pointer';
    });
    if (currentInpaint.lassoActive) {
        setupLassoHandlers();
    } else {
        removeLassoHandlers();
    }
}

var lassoMouseDown = null, lassoMouseMove = null, lassoMouseUp = null;

function setupLassoHandlers() {
    var molDivs = document.querySelectorAll('.inpaint-mode .rdkit-mol');
    molDivs.forEach(function(molDiv) {
        var svgEl = molDiv.querySelector('svg');
        if (!svgEl) return;

        lassoMouseDown = function(e) {
            if (!currentInpaint || !currentInpaint.lassoActive) return;
            e.preventDefault();
            currentInpaint.lassoPoints = [];
            var pt = getSvgPoint(svgEl, e);
            currentInpaint.lassoPoints.push(pt);
            // Create polyline overlay
            var polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
            polyline.setAttribute('id', 'lasso-line');
            polyline.setAttribute('fill', 'none');
            polyline.setAttribute('stroke', '#e74c3c');
            polyline.setAttribute('stroke-width', '2');
            polyline.setAttribute('stroke-dasharray', '5,5');
            polyline.setAttribute('pointer-events', 'none');
            svgEl.appendChild(polyline);

            lassoMouseMove = function(ev) {
                if (!currentInpaint || !currentInpaint.lassoActive) return;
                var p = getSvgPoint(svgEl, ev);
                currentInpaint.lassoPoints.push(p);
                var points = currentInpaint.lassoPoints.map(function(pt) { return pt.x + ',' + pt.y; }).join(' ');
                var line = svgEl.querySelector('#lasso-line');
                if (line) line.setAttribute('points', points);
            };

            lassoMouseUp = function(ev) {
                if (!currentInpaint || !currentInpaint.lassoActive) return;
                svgEl.removeEventListener('mousemove', lassoMouseMove);
                svgEl.removeEventListener('mouseup', lassoMouseUp);
                // Remove polyline
                var line = svgEl.querySelector('#lasso-line');
                if (line) line.remove();
                // Select atoms inside the lasso polygon
                if (currentInpaint.lassoPoints.length >= 3) {
                    selectAtomsInLasso(molDiv, currentInpaint.lassoPoints);
                }
                currentInpaint.lassoPoints = [];
            };

            svgEl.addEventListener('mousemove', lassoMouseMove);
            svgEl.addEventListener('mouseup', lassoMouseUp);
        };
        svgEl.addEventListener('mousedown', lassoMouseDown);
        svgEl.setAttribute('data-lasso-bound', 'true');
    });
}

function removeLassoHandlers() {
    document.querySelectorAll('.inpaint-mode .rdkit-mol svg[data-lasso-bound]').forEach(function(svgEl) {
        if (lassoMouseDown) svgEl.removeEventListener('mousedown', lassoMouseDown);
        svgEl.removeAttribute('data-lasso-bound');
    });
}

function getSvgPoint(svgEl, event) {
    var rect = svgEl.getBoundingClientRect();
    var viewBox = svgEl.viewBox.baseVal;
    var scaleX = viewBox.width / rect.width;
    var scaleY = viewBox.height / rect.height;
    return {
        x: (event.clientX - rect.left) * scaleX + viewBox.x,
        y: (event.clientY - rect.top) * scaleY + viewBox.y
    };
}

function selectAtomsInLasso(molDiv, polygon) {
    var svgEl = molDiv.querySelector('svg');
    if (!svgEl) return;
    var atomMap = JSON.parse(molDiv.getAttribute('data-atom-map') || '{}');

    for (var rdkitIdx in atomMap) {
        // Find the center of this atom's SVG elements
        var re = new RegExp('(^|\\s)atom-' + rdkitIdx + '(\\s|$)');
        var elements = svgEl.querySelectorAll('[class*="atom-' + rdkitIdx + '"]');
        var cx = 0, cy = 0, count = 0;
        elements.forEach(function(el) {
            var cls = (el.className && el.className.baseVal != null) ? el.className.baseVal : (el.className || '');
            if (!re.test(cls)) return;
            try {
                var bbox = el.getBBox();
                cx += bbox.x + bbox.width / 2;
                cy += bbox.y + bbox.height / 2;
                count++;
            } catch(e) {}
        });
        if (count === 0) continue;
        cx /= count; cy /= count;

        // Point-in-polygon test (ray casting)
        if (pointInPolygon(cx, cy, polygon)) {
            currentInpaint.selectedAtoms.add(atomMap[rdkitIdx]);
        }
    }
    // Re-render with updated highlights
    var selectedRdkit = [];
    for (var key in atomMap) {
        if (currentInpaint.selectedAtoms.has(atomMap[key])) {
            selectedRdkit.push(parseInt(key));
        }
    }
    reRenderMolWithHighlights(molDiv, selectedRdkit);
    // Re-attach lasso handlers since SVG was replaced
    if (currentInpaint.lassoActive) {
        var newSvg = molDiv.querySelector('svg');
        if (newSvg && lassoMouseDown) {
            newSvg.addEventListener('mousedown', lassoMouseDown);
            newSvg.setAttribute('data-lasso-bound', 'true');
            newSvg.style.cursor = 'crosshair';
        }
    }
    updateAtomCount();
}

function pointInPolygon(x, y, polygon) {
    var inside = false;
    for (var i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        var xi = polygon[i].x, yi = polygon[i].y;
        var xj = polygon[j].x, yj = polygon[j].y;
        var intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

function cancelInpaint() {
    if (!currentInpaint) return;
    // Clean up lasso handlers if active
    if (currentInpaint.lassoActive) {
        removeLassoHandlers();
    }
    var cards = document.querySelectorAll('.inpaint-mode');
    cards.forEach(function(card) {
        card.classList.remove('inpaint-mode');
        card.classList.remove('inpaint-focus-panel');
        var toolbar = card.querySelector('.inpaint-toolbar');
        if (toolbar) toolbar.style.display = 'none';
        // Restore original (compact) SVG HTML
        var svgContainer = card.querySelector('.mol-svg');
        var origHtml = svgContainer ? svgContainer.getAttribute('data-orig-html') : null;
        if (origHtml) {
            svgContainer.innerHTML = origHtml;
            svgContainer.removeAttribute('data-orig-html');
        }
        // Remove click-bound markers so handlers can be re-attached next time
        card.querySelectorAll('.rdkit-mol[data-click-bound]').forEach(function(el) {
            el.removeAttribute('data-click-bound');
        });
    });
    currentInpaint = null;
}

/* ── Submit inpainting request ── */
function submitInpaint() {
    if (!currentInpaint || currentInpaint.selectedAtoms.size === 0) return;

    if (generations.length >= MAX_GENERATIONS) {
        alert('Maximum of ' + (MAX_GENERATIONS - 1) + ' edit rounds reached. Start a new prediction to continue.');
        return;
    }

    var genIdx = currentInpaint.genIdx;
    var resultIdx = currentInpaint.resultIdx;
    var gen = generations[genIdx];
    var result = gen.results[resultIdx];

    // Read from inpaint toolbar controls (fall back to main form)
    var inpaintPrecEl = document.querySelector('.inpaint-mode .inpaint-n-precursors');
    var inpaintStepsEl = document.querySelector('.inpaint-mode .inpaint-diff-steps');
    var nPrecursors = inpaintPrecEl ? (parseInt(inpaintPrecEl.value) || 1) : (parseInt(document.querySelector('[name="n_precursors"]').value) || 1);
    var diffusionSteps = inpaintStepsEl ? (parseInt(inpaintStepsEl.value) || 1) : (parseInt(document.querySelector('[name="diffusion_steps"]').value) || 1);

    // Collect every real atom in this result. Start from the atom_mapping
    // (user-addressable reactant atoms), then fold in any remaining real
    // atom from node_mask so that product/supernode atoms — which the user
    // can't see and the structural mask keeps frozen anyway — are always
    // treated as "keep". Without this, the backend's strict change-must-
    // actually-change check would flag those structurally-fixed atoms as
    // stuck on every sample.
    var allNodes = new Set();
    result.atom_mapping.forEach(function(mi) {
        for (var key in mi.atom_map) {
            allNodes.add(mi.atom_map[key]);
        }
    });
    var sampleData = result.sample_data || {};
    var nodeMask = sampleData.node_mask;
    if (nodeMask) {
        var row = Array.isArray(nodeMask[0]) ? nodeMask[0] : nodeMask;
        for (var i = 0; i < row.length; i++) {
            if (row[i]) allNodes.add(i);
        }
    }

    // In "regenerate" mode, the user selected atoms to CHANGE.
    // The backend expects atoms to KEEP fixed, so we invert the selection.
    var nodesToKeep;
    var mode = currentInpaint.mode || 'regenerate';
    if (mode === 'regenerate') {
        nodesToKeep = Array.from(allNodes).filter(function(n) {
            return !currentInpaint.selectedAtoms.has(n);
        });
    } else {
        // Keep mode: user's selection plus every non-addressable real atom
        // (product/supernode) so the structural-mask atoms don't get
        // double-counted as change-atoms.
        var keep = new Set(currentInpaint.selectedAtoms);
        allNodes.forEach(function(n) {
            // Any real atom not in atom_mapping stays fixed too
            var inMapping = false;
            result.atom_mapping.forEach(function(mi) {
                for (var k in mi.atom_map) if (mi.atom_map[k] === n) inMapping = true;
            });
            if (!inMapping) keep.add(n);
        });
        nodesToKeep = Array.from(keep);
    }

    // Guard: if every real atom is marked fixed there is nothing to
    // regenerate — skip the API call and let the user adjust their
    // selection.
    if (allNodes.size > 0 && nodesToKeep.length >= allNodes.size) {
        alert("You've marked every atom to keep fixed. Nothing to regenerate. "
              + "Deselect at least one atom, or switch to 'Select atoms to change' mode.");
        return;
    }

    var summary = getSelectedSubSmiles();
    var nSelected = currentInpaint.selectedAtoms.size;
    var fixedInfo = (mode === 'regenerate'
        ? 'Regenerating ' + summary + ' (' + nSelected + ' atoms changed)'
        : 'Fixed ' + summary + ' (' + nSelected + ' atoms kept)')
        + ' from #' + (resultIdx + 1) + ', gen ' + (genIdx + 1);

    cancelInpaint();

    // Show progress
    var submitBtn = document.getElementById('submit-btn');
    submitBtn.disabled = true;
    showProgress('Running inpainting with ' + nodesToKeep.length + ' fixed atoms...');

    fetch('/api/inpaint', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            product_smiles: gen.targetSmiles,
            previous_sample_data: result.sample_data,
            selected_node_indices: nodesToKeep,
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
        rc.innerHTML += '<div class="results-section"><div class="error">Edit-precursor request failed: ' + err.message + '</div></div>';
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
        '<span class="generation-info">Edited: ' + fixedInfo + '</span>' +
        '</div>';

    var bodyHtml = '';
    if ((!results || results.length === 0) && data.failure) {
        var stuck = data.failure.stuck_atoms || [];
        var stuckDetail = stuck.map(function(a) {
            return a.element + ' @ ' + a.index;
        }).join(', ') || '(none)';
        var requested = data.failure.requested_change_atoms || [];
        var requestedDetail = requested.map(function(a) {
            return a.element + ' @ ' + a.index;
        }).join(', ') || '(none)';
        bodyHtml = '<div class="inpaint-empty">' +
            '<strong>No valid inpainting result.</strong>' +
            '<p>' + escapeHtml(data.failure.message) + '</p>' +
            '<details>' +
                '<summary>Details</summary>' +
                '<p>Requested to change: <code>' + escapeHtml(requestedDetail) + '</code></p>' +
                '<p>Stuck (unchanged in all ' + data.failure.n_samples + ' samples): <code>' + escapeHtml(stuckDetail) + '</code></p>' +
            '</details>' +
            '</div>';
    } else {
        var cardsHtml = '<div class="precursors-grid">';
        results.forEach(function(result, idx) {
            var formulaHtml = result.formula
                ? '<div class="precursor-formula">Formula: <strong>' + escapeHtml(result.formula) + '</strong></div>'
                : '';
            cardsHtml += '<div class="precursor-card" data-result-index="' + idx + '">' +
                '<div class="precursor-header">' +
                    '<span class="precursor-rank">#' + (idx + 1) + '</span>' +
                    '<span class="precursor-score">Score: ' + result.score.toFixed(3) + '</span>' +
                '</div>' +
                '<div class="mol-svg"></div>' +
                '<div class="precursor-smiles">' + result.precursors + '</div>' +
                formulaHtml +
                '<div class="precursor-actions">' +
                    '<button class="btn-lookup"' +
                            ' title="Queries the PubChem REST API for compound metadata (formula, MW, patents, PubMed mentions). May take a minute for several compounds."' +
                            ' onclick="lookupCompounds(this, &quot;' + result.precursors.replace(/"/g, '&quot;') + '&quot;)">Search PubChem</button>' +
                    '<button class="btn-inpaint" onclick="enterInpaintMode(this)">Edit precursor</button>' +
                    '<span class="info-tip" tabindex="0" role="button" aria-label="About edit precursor">' +
                        '<span class="info-tip-icon">?</span>' +
                        '<span class="info-tip-body" role="tooltip">Mark atoms to keep or to change, then ask the model to regenerate this precursor with your selection in mind.</span>' +
                    '</span>' +
                '</div>' +
                '<div class="compound-info" style="display:none;"></div>' +
                '</div>';
        });
        cardsHtml += '</div>';
        bodyHtml = cardsHtml;
    }

    section.innerHTML = headerHtml + bodyHtml;
    rc.appendChild(section);

    // Render molecules with RDKit.js (for atom selection) — skip if empty
    if (results && results.length > 0) {
        renderInpaintGenerationMols(section, results);
    }

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
