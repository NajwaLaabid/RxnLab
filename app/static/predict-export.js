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

/* All generations (initial prediction + any inpaint rounds), falling back to the
   embedded JSON when no JS state exists. Each is {results, fixedInfo, targetSmiles}. */
function _allGenerations() {
    if (generations.length > 0) return generations;
    var script = document.querySelector('.generation-results-json');
    if (script) {
        try {
            return [{ results: JSON.parse(script.textContent), fixedInfo: null,
                      targetSmiles: currentTargetSmiles }];
        } catch(e) { return []; }
    }
    return [];
}

/* Scrape one generation's populated PubChem panels, keyed by result index within
   that generation. Each generation with results renders exactly one
   `.precursors-grid`, so the caller pairs the k-th non-empty generation with the
   k-th grid (failure generations render no grid and contribute no rows). */
function _scrapePubchemForGrid(grid) {
    var byIdx = {};
    if (!grid) return byIdx;
    grid.querySelectorAll('.precursor-card').forEach(function(card, i) {
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
    var gens = _allGenerations();
    // Pair each generation that has results with its rendered grid, in DOM order.
    // Failure generations render no `.precursors-grid` and contribute no rows.
    var grids = document.querySelectorAll('.precursors-grid');
    var nonEmpty = [];
    gens.forEach(function(gen, gi) {
        if ((gen.results || []).length) {
            nonEmpty.push({ gen: gen, genNumber: gi + 1, grid: grids[nonEmpty.length] });
        }
    });
    if (!nonEmpty.length) {
        alert('No predictions to export.');
        return;
    }

    var fallbackTarget = (document.querySelector('.target-info .smiles') || {}).textContent || '';
    fallbackTarget = (currentTargetSmiles || fallbackTarget).trim();

    // Export every generation; tag rows with generation columns only when there's
    // more than one (initial-only export stays clean).
    var multiGen = nonEmpty.length > 1;
    var anyPubchem = false, anyRxnExtra = false;
    nonEmpty.forEach(function(ng) {
        ng.pubchem = _scrapePubchemForGrid(ng.grid);
        if (Object.keys(ng.pubchem).length) anyPubchem = true;
        (ng.gen.results || []).forEach(function(r) {
            var ri = r.reaction_info;
            if (ri && Object.keys(ri).some(function(k) {
                return k !== 'success' && k !== 'class' && k !== 'name' && ri[k];
            })) anyRxnExtra = true;
        });
    });

    var header = [];
    if (multiGen) header.push('generation','generation_label');
    header.push('rank','target_smiles','reactants_smiles','score','formula',
                'reaction_class','reaction_name','mapped_reaction_smiles');
    if (anyPubchem) header.push('pubchem_cids','pubchem_names','pubchem_mw');
    if (anyRxnExtra) header.push('rxn_insight_extra_json');

    var rows = [header];
    nonEmpty.forEach(function(ng) {
        var target = (ng.gen.targetSmiles || fallbackTarget).trim();
        var label = ng.gen.fixedInfo || 'Initial prediction';
        (ng.gen.results || []).forEach(function(r, i) {
            var ri = r.reaction_info || {};
            var row = [];
            if (multiGen) row.push(ng.genNumber, label);
            row.push(
                i + 1,
                target,
                r.precursors || '',
                typeof r.score === 'number' ? r.score.toFixed(6) : (r.score || ''),
                r.formula || '',
                ri.class || '',
                ri.name || '',
                r.mapped_rxn || ''
            );
            if (anyPubchem) {
                var p = ng.pubchem[i] || {cids:'', names:'', mws:''};
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
    });

    var target = (nonEmpty[0].gen.targetSmiles || fallbackTarget).trim();
    var ts = new Date().toISOString();
    // Leading BOM forces UTF-8 detection (Excel) without polluting the data; no
    // "sep=," hint — it broke pandas/programmatic reads (read as a data row).
    var csv = '﻿' + _toCSV(rows) + '\r\n';

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
