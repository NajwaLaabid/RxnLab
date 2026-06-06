/* RxnLab single-step/multi-step predict page — multi-step search + route tree.
   Split from the former inline <script> in predict.html. All functions are
   globals (called from inline handlers); load order set in predict.html. */

/* ── Multi-step search (Phase 3) ── */
var _searchRoutes = [];
function _escAttr(s) {
    return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;');
}

function _molSvg(smiles, w, h) {
    if (!RDKitModule) return '';
    var m = null;
    try {
        m = RDKitModule.get_mol(smiles);
        if (!m) return '';
        return m.get_svg(w, h);
    } catch (e) { return ''; }
    finally { if (m) m.delete(); }
}

function _routeNodeHtml(node) {
    var buyable = node.purchasable;
    var html = '<div class="route-node">';
    html += '<div class="route-mol-card' + (buyable ? ' buy' : '') + '">';
    html += '<div class="route-mol" data-smiles="' + _escAttr(node.product) + '"></div>';
    html += '<div class="route-smiles">' + escapeHtml(node.product) + '</div>';
    if (buyable) html += '<span class="route-badge">buyable</span>';
    html += '<button type="button" class="route-lookup-btn" data-smiles="'
          + _escAttr(node.product) + '" onclick="lookupCardCompound(this)">Search PubChem</button>';
    html += '<div class="route-compound-info" style="display:none;"></div>';
    html += '</div>';
    if (node.children && node.children.length) {
        var sc = (node.score != null) ? ' · score ' + (+node.score).toFixed(2) : '';
        html += '<div class="route-rxn" data-product="' + _escAttr(node.product) + '">'
              + '<span class="route-rxn-label">made from</span>' + sc + '</div>';
        html += '<div class="route-children">';
        node.children.forEach(function(c) { html += _routeNodeHtml(c); });
        html += '</div>';
    }
    html += '</div>';
    return html;
}

function lookupCardCompound(btn) {
    var card = btn.closest('.route-mol-card');
    var panel = card.querySelector('.route-compound-info');
    if (panel.style.display !== 'none') {
        panel.style.display = 'none';
        btn.textContent = 'Search PubChem';
        return;
    }
    var smiles = btn.getAttribute('data-smiles');
    btn.disabled = true;
    btn.textContent = 'Searching…';
    panel.style.display = 'block';
    panel.innerHTML = '<div class="route-compound-detail">Looking up…</div>';
    fetch('/api/evaluate/compound-lookup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smiles_list: [smiles] })
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        btn.disabled = false;
        btn.textContent = 'Hide PubChem';
        if (data.error) {
            panel.innerHTML = '<span style="color:#c0392b;">' + escapeHtml(data.error) + '</span>';
            return;
        }
        var c = (data.compounds && data.compounds[0]) || {};
        if (!c.found) {
            panel.innerHTML = '<div class="route-compound-detail">Not found in PubChem</div>';
            return;
        }
        var name = (c.short_names && c.short_names.length) ? c.short_names[0] : (c.iupac || c.smiles);
        var aka = (c.short_names && c.short_names.length > 1)
            ? '<div class="route-compound-detail">aka ' + escapeHtml(c.short_names.slice(1, 3).join(', ')) + '</div>' : '';
        var h = '<div class="route-compound-name">' + escapeHtml(name) + '</div>' + aka;
        h += '<div class="route-compound-detail">' + escapeHtml(c.formula || '') + ' · MW ' + (c.mw || '?') + '</div>';
        h += '<div class="route-compound-detail">Patents: ' + (c.n_patents || 0)
           + ' · PubMed: ' + (c.n_pubmed || 0) + '</div>';
        h += '<div class="route-compound-detail"><a href="https://pubchem.ncbi.nlm.nih.gov/compound/'
           + c.cid + '" target="_blank">PubChem CID ' + c.cid + '</a></div>';
        panel.innerHTML = h;
    })
    .catch(function(err) {
        btn.disabled = false;
        btn.textContent = 'Search PubChem';
        panel.innerHTML = '<span style="color:#c0392b;">Lookup failed: ' + escapeHtml(err.message) + '</span>';
    });
}

function _drawRouteMolecules(root) {
    if (!RDKitModule) { setTimeout(function() { _drawRouteMolecules(root); }, 200); return; }
    root.querySelectorAll('.route-mol[data-smiles]').forEach(function(el) {
        if (el.getAttribute('data-drawn')) return;
        var svg = _molSvg(el.getAttribute('data-smiles'), 150, 110);
        if (svg) { el.innerHTML = svg; el.setAttribute('data-drawn', '1'); }
    });
}

function renderRouteTree(result) {
    var rc = document.getElementById('results-container');
    var st = result.stats || {};
    var routes = result.routes || [];
    _searchRoutes = routes;
    var secs = ((st.elapsed_ms || 0) / 1000).toFixed(1);
    var h = '<div class="results-section">';
    if (!routes.length) {
        h += '<div class="route-summary">No synthesis route to purchasable building blocks '
           + 'was found within the search budget (' + secs + 's, ' + (st.n_nodes || 0).toLocaleString()
           + ' nodes explored). Try a simpler target or a different model.</div></div>';
        rc.innerHTML = h;
        return;
    }
    h += '<div class="route-summary"><strong>' + routes.length + ' route'
       + (routes.length > 1 ? 's' : '') + '</strong> found · '
       + (st.n_nodes || 0).toLocaleString() + ' nodes explored in ' + secs + 's</div>';
    routes.forEach(function(r, i) {
        var hasDesc = _hasDescription(r);
        h += '<div class="route-block"><div class="route-title">Route ' + (i + 1) + '</div>'
           + '<button type="button" class="route-describe-btn" onclick="describeRoute(this,' + i + ')">'
           + (hasDesc ? 'Hide description' : 'Describe route') + '</button>'
           + '<div class="route-desc"' + (hasDesc ? '' : ' style="display:none;"') + '></div>'
           + '<div class="route-tree-scroll">' + _routeNodeHtml(r) + '</div></div>';
    });
    h += '</div>';
    rc.innerHTML = h;
    _drawRouteMolecules(rc);
    var blocks = rc.querySelectorAll('.route-block');
    routes.forEach(function(r, i) {
        if (_hasDescription(r)) _fillDescPanel(blocks[i], r.description);
    });
}

function _hasDescription(r) {
    return !!(r && r.description
        && (r.description.summary || (r.description.steps && r.description.steps.length)));
}

function _descInnerHtml(d) {
    var h = '<div class="route-desc-summary">' + escapeHtml(d.summary || '') + '</div>';
    if (d.steps && d.steps.length) {
        h += '<ol class="route-desc-steps">';
        d.steps.forEach(function(s) {
            var detail = '';
            if (s.short_label === 'unidentified reaction type') {
                detail = s.fg_transform || '';
            } else if (s.name && s.name.toLowerCase() !== (s['class'] || '').toLowerCase()) {
                detail = s.name;
            }
            var li = '<strong>' + escapeHtml(s.short_label || 'reaction') + '</strong>';
            if (detail) li += ' <span class="route-desc-detail">— ' + escapeHtml(detail) + '</span>';
            h += '<li>' + li + '</li>';
        });
        h += '</ol>';
    }
    return h;
}

function _fillDescPanel(block, d) {
    var panel = block.querySelector('.route-desc');
    panel.innerHTML = _descInnerHtml(d);
    _annotateRouteTree(block, d.steps);  // label the tree connectors
}

function _annotateRouteTree(block, steps) {
    if (!steps) return;
    var byProduct = {};
    steps.forEach(function(s) { byProduct[s.product] = s; });
    block.querySelectorAll('.route-rxn[data-product]').forEach(function(conn) {
        var s = byProduct[conn.getAttribute('data-product')];
        if (!s) return;
        var lblEl = conn.querySelector('.route-rxn-label');
        if (!lblEl) return;
        lblEl.textContent = s.short_label || 'reaction';
        lblEl.classList.add('route-rxn-named');
        if (s.short_label === 'unidentified reaction type') lblEl.classList.add('route-rxn-unidentified');
        if (s.label) conn.title = s.label;  // full descriptor on hover
    });
}

function describeRoute(btn, idx) {
    var block = btn.closest('.route-block');
    var panel = block.querySelector('.route-desc');
    if (panel.style.display !== 'none') {
        panel.style.display = 'none';
        btn.textContent = 'Describe route';
        return;
    }
    var route = _searchRoutes[idx];
    if (!route) return;
    if (_hasDescription(route)) {  // already classified during search — just show it
        if (!panel.innerHTML) _fillDescPanel(block, route.description);
        panel.style.display = 'block';
        btn.textContent = 'Hide description';
        return;
    }
    btn.disabled = true;
    btn.textContent = 'Analyzing…';
    panel.style.display = 'block';
    panel.innerHTML = '<div class="route-desc-detail">Classifying reactions '
                    + '(atom-mapping runs a few seconds per step)…</div>';
    fetch('/api/search/describe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ route: route })
    })
    .then(function(r) { return r.json().then(function(j) { return { ok: r.ok, j: j }; }); })
    .then(function(res) {
        btn.disabled = false;
        btn.textContent = 'Hide description';
        if (!res.ok) {
            panel.innerHTML = '<span style="color:#c0392b;">' + escapeHtml(res.j.error || 'failed') + '</span>';
            return;
        }
        _fillDescPanel(block, res.j);
    })
    .catch(function(err) {
        btn.disabled = false;
        btn.textContent = 'Describe route';
        panel.innerHTML = '<span style="color:#c0392b;">Description failed: ' + escapeHtml(err.message) + '</span>';
    });
}

function _stopSearch() {
    clearProgressTimers();
    hideProgress();
    var btn = document.getElementById('submit-btn');
    if (btn) btn.disabled = false;
}

function _pollSearchJob(jobId) {
    fetch('/api/search/' + jobId)
        .then(function(r) { return r.json(); })
        .then(function(job) {
            if (job.status === 'done') {
                _stopSearch();
                renderRouteTree(job.result);
            } else if (job.status === 'error') {
                _stopSearch();
                document.getElementById('results-container').innerHTML =
                    '<div class="results-section"><div class="error">Search failed: '
                    + escapeHtml(job.error || 'unknown error') + '</div></div>';
            } else {
                setTimeout(function() { _pollSearchJob(jobId); }, 1500);
            }
        })
        .catch(function() { setTimeout(function() { _pollSearchJob(jobId); }, 2500); });
}

function runMultiStepSearch() {
    var smiles = document.getElementById('smiles-input').value.trim();
    var modelSel = document.getElementById('model-select');
    var modelId = modelSel ? modelSel.value : '';
    var catSel = document.getElementById('catalog-select');
    var catalogId = catSel ? catSel.value : '';
    var maxRoutesEl = document.getElementById('search-max-routes');
    var maxDepthEl = document.getElementById('search-max-depth');
    var rc = document.getElementById('results-container');
    if (!smiles) {
        rc.innerHTML = '<div class="results-section"><div class="error">'
                     + 'Enter a target molecule (SMILES, name, InChI, InChIKey, or CAS) to search.</div></div>';
        return;
    }
    var btn = document.getElementById('submit-btn');
    rc.innerHTML = '';
    if (btn) btn.disabled = true;
    showProgress('Searching backward to purchasable building blocks…');
    fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
        body: JSON.stringify({
            smiles: smiles, model_id: modelId, catalog_id: catalogId,
            max_routes: maxRoutesEl ? maxRoutesEl.value : undefined,
            max_expansion_depth: maxDepthEl ? maxDepthEl.value : undefined
        })
    })
    .then(function(r) { return r.json().then(function(j) { return { ok: r.ok, j: j }; }); })
    .then(function(res) {
        if (!res.ok) throw new Error(res.j.error || 'Search request failed');
        _pollSearchJob(res.j.job_id);
    })
    .catch(function(err) {
        _stopSearch();
        rc.innerHTML = '<div class="results-section"><div class="error">'
                     + escapeHtml(err.message) + '</div></div>';
    });
}
