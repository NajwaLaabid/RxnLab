/* RxnLab single-step/multi-step predict page — form, model/catalog controls, examples, lookup.
   Split from the former inline <script> in predict.html. All functions are
   globals (called from inline handlers); load order set in predict.html. */

/* ── Utility functions ── */
function dismissBanner(bannerId) {
    bannerId = bannerId || 'info-banner';
    var banner = document.getElementById(bannerId);
    if (banner) banner.style.display = 'none';
    try { localStorage.setItem('bannerDismissed:' + bannerId, '1'); } catch(e) {}
}
(function() {
    try {
        // Migrate legacy key for backward compat.
        if (localStorage.getItem('infoBannerDismissed') === '1') {
            localStorage.setItem('bannerDismissed:info-banner', '1');
            localStorage.removeItem('infoBannerDismissed');
        }
        if (localStorage.getItem('bannerDismissed:info-banner') === '1') {
            var b = document.getElementById('info-banner');
            if (b) b.style.display = 'none';
        }
    } catch(e) {}
})();

/* Show only the selected model's param controls; disabled inputs don't submit. */
function onModelChange() {
    var sel = document.getElementById('model-select');
    if (!sel) return;
    var mid = sel.value;
    document.querySelectorAll('.model-param').forEach(function(box) {
        var on = box.getAttribute('data-for-model') === mid;
        box.style.display = on ? '' : 'none';
        box.querySelectorAll('input, select').forEach(function(el) { el.disabled = !on; });
    });
    var desc = document.getElementById('model-desc');
    var opt = sel.options[sel.selectedIndex];
    if (desc && opt) desc.textContent = opt.getAttribute('data-desc') || '';
    onSearchModeChange();
}

function searchModeOn() { return MULTISTEP; }

function onSearchModeChange() {
    var btn = document.getElementById('submit-btn');
    if (btn) btn.textContent = MULTISTEP ? 'Find Synthesis Routes' : 'Predict Reactants';
    if (MULTISTEP) onCatalogChange();
}

function onCatalogChange() {
    var sel = document.getElementById('catalog-select');
    var desc = document.getElementById('catalog-desc');
    if (!sel || !desc) return;
    var opt = sel.options[sel.selectedIndex];
    desc.textContent = opt ? (opt.getAttribute('data-blurb') || '') : '';
}

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
    var smilesList = precursors.split('.');
    // Each compound takes a few seconds (3 sequential PubChem calls each); show a
    // determinate progress bar driven by elapsed time as a proxy.
    var nCompounds = smilesList.length;
    var estPerCompound = 3500;  // ms; rough average per compound
    var estTotal = nCompounds * estPerCompound;
    panel.innerHTML =
        '<div class="pubchem-loading">' +
            '<div class="pubchem-loading-msg">Fetching PubChem metadata for ' + nCompounds +
            ' compound' + (nCompounds === 1 ? '' : 's') + '. This may take a minute…</div>' +
            '<div class="pubchem-progress"><div class="pubchem-progress-bar"></div></div>' +
        '</div>';
    var progressBar = panel.querySelector('.pubchem-progress-bar');
    var pubchemStart = Date.now();
    var pubchemTimer = setInterval(function () {
        var elapsed = Date.now() - pubchemStart;
        var pct = Math.min(95, (elapsed / estTotal) * 100);
        if (progressBar) progressBar.style.width = pct + '%';
    }, 200);
    fetch('/api/evaluate/compound-lookup', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({smiles_list: smilesList})
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        clearInterval(pubchemTimer);
        if (progressBar) progressBar.style.width = '100%';
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
                html += '<div class="compound-detail"><a href="https://pubchem.ncbi.nlm.nih.gov/compound/' + c.cid + '" target="_blank" style="color:#0d8a8a;">PubChem CID ' + c.cid + '</a></div>';
            } else {
                html += '<div class="compound-detail">Not found: ' + c.smiles + '</div>';
            }
            html += '</div>';
        });
        panel.innerHTML = html;
    })
    .catch(function(err) {
        clearInterval(pubchemTimer);
        btn.disabled = false;
        btn.textContent = 'Search PubChem';
        panel.innerHTML = '<span style="color:#c0392b;">Lookup failed: ' + err.message + '</span>';
    });
}
