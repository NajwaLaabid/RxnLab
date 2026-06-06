/* RxnLab single-step/multi-step predict page — CSV export.
   Split from the former inline <script> in predict.html. All functions are
   globals (called from inline handlers); load order set in predict.html. */

/* ── CSV export ── */
function downloadPredictionsCSV() {
    _doCsvDownload();
}

function _csvEscape(v) {
    if (v === null || v === undefined) return '';
    var s = String(v);
    if (s.indexOf(',') >= 0 || s.indexOf('"') >= 0 || s.indexOf('\n') >= 0 || s.indexOf('\r') >= 0) {
        return '"' + s.replace(/"/g, '""') + '"';
    }
    return s;
}

function _toCSV(rows) {
    return rows.map(function(row) {
        return row.map(_csvEscape).join(',');
    }).join('\r\n');
}

/* Pull predictions for the most recent generation, falling back to the embedded JSON. */
function _activeResults() {
    if (generations.length > 0) {
        var gen = generations[generations.length - 1];
        return { results: gen.results || [], targetSmiles: gen.targetSmiles || currentTargetSmiles };
    }
    var script = document.querySelector('.generation-results-json');
    if (script) {
        try { return { results: JSON.parse(script.textContent), targetSmiles: currentTargetSmiles }; }
        catch(e) { return { results: [], targetSmiles: currentTargetSmiles }; }
    }
    return { results: [], targetSmiles: currentTargetSmiles };
}

/* Read populated PubChem panels from the DOM, keyed by result index. */
function _scrapePubchemByIndex() {
    var byIdx = {};
    var cards = document.querySelectorAll('.precursor-card');
    cards.forEach(function(card, i) {
        var panel = card.querySelector('.compound-info');
        if (!panel || panel.style.display === 'none') return;
        var entries = panel.querySelectorAll('.compound-entry');
        if (!entries.length) return;
        var cids = [], names = [], mws = [];
        entries.forEach(function(e) {
            var nameEl = e.querySelector('.compound-name');
            var link = e.querySelector('a[href*="/compound/"]');
            if (nameEl) names.push(nameEl.textContent.replace(/\s+aka\s+.*$/, '').trim());
            if (link) {
                var m = link.href.match(/\/compound\/(\d+)/);
                if (m) cids.push(m[1]);
            }
            var mwFound = '';
            e.querySelectorAll('.compound-detail').forEach(function(d) {
                if (mwFound) return;
                var mwMatch = d.textContent.match(/MW\s+([\d.]+)/);
                if (mwMatch) mwFound = mwMatch[1];
            });
            mws.push(mwFound);
        });
        byIdx[i] = { cids: cids.join('|'), names: names.join('|'), mws: mws.join('|') };
    });
    return byIdx;
}

function _slugForFilename(s) {
    if (!s) return 'target';
    return s.replace(/[^A-Za-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'target';
}

function _doCsvDownload() {
    var active = _activeResults();
    var results = active.results;
    if (!results.length) {
        alert('No predictions to export.');
        return;
    }
    var target = active.targetSmiles ||
                 (document.querySelector('.target-info .smiles') || {}).textContent || '';
    target = target.trim();

    var pubchemByIdx = _scrapePubchemByIndex();
    var anyPubchem = Object.keys(pubchemByIdx).length > 0;
    var anyRxnExtra = results.some(function(r) {
        if (!r.reaction_info) return false;
        var ri = r.reaction_info;
        return Object.keys(ri).some(function(k) {
            return k !== 'success' && k !== 'class' && k !== 'name' && ri[k];
        });
    });

    var header = ['rank','target_smiles','precursors_smiles','score','formula',
                  'reaction_class','reaction_name','mapped_reaction_smiles'];
    if (anyPubchem) header.push('pubchem_cids','pubchem_names','pubchem_mw');
    if (anyRxnExtra) header.push('rxn_insight_extra_json');

    var rows = [header];
    results.forEach(function(r, i) {
        var ri = r.reaction_info || {};
        var row = [
            i + 1,
            target,
            r.precursors || '',
            typeof r.score === 'number' ? r.score.toFixed(6) : (r.score || ''),
            r.formula || '',
            ri.class || '',
            ri.name || '',
            r.mapped_rxn || ''
        ];
        if (anyPubchem) {
            var p = pubchemByIdx[i] || {cids:'', names:'', mws:''};
            row.push(p.cids, p.names, p.mws);
        }
        if (anyRxnExtra) {
            var extras = {};
            Object.keys(ri).forEach(function(k) {
                if (k === 'success' || k === 'class' || k === 'name') return;
                extras[k] = ri[k];
            });
            row.push(Object.keys(extras).length ? JSON.stringify(extras) : '');
        }
        rows.push(row);
    });

    var ts = new Date().toISOString();
    // "sep=," hint makes Excel split on commas regardless of the OS locale's
    // list separator (e.g. ";" on Finnish/European locales); BOM forces UTF-8.
    var csv = '﻿' + 'sep=,\r\n' + _toCSV(rows) + '\r\n';

    var stamp = ts.replace(/[-:]/g, '').replace(/\..*$/, '').replace('T', '-').slice(0, 13);
    var fname = 'rxnlab_' + _slugForFilename(target) + '_' + stamp + '.csv';

    var blob = new Blob([csv], {type: 'text/csv;charset=utf-8'});
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = fname;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function() { URL.revokeObjectURL(url); }, 1000);
}
