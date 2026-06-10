'use strict';

function setSmiles(s) {
    document.getElementById('smiles-input').value = s;
}

function clearCompare() {
    document.getElementById('smiles-input').value = '';
    document.getElementById('compare-results').innerHTML = '';
    hideError();
}

function showError(msg) {
    const el = document.getElementById('compare-error');
    el.textContent = msg;
    el.style.display = 'block';
}

function hideError() {
    document.getElementById('compare-error').style.display = 'none';
}

function esc(s) {
    const d = document.createElement('div');
    d.textContent = s == null ? '' : String(s);
    return d.innerHTML;
}

function selectedModelIds() {
    return Array.from(
        document.querySelectorAll('#model-checks input[name="model_id"]:checked')
    ).map((c) => c.value);
}

// Fake progress: we don't get server-side step events, so animate toward 90%
// while the request is in flight, then snap to 100% on response.
let _progTimer = null;
function startProgress() {
    const wrap = document.getElementById('compare-progress');
    const bar = document.getElementById('compare-bar');
    const text = document.getElementById('compare-text');
    wrap.style.display = 'block';
    let pct = 0;
    bar.style.width = '0%';
    text.textContent = 'Running models…';
    _progTimer = setInterval(() => {
        pct = Math.min(90, pct + Math.max(0.5, (90 - pct) * 0.08));
        bar.style.width = pct + '%';
    }, 400);
}
function stopProgress() {
    if (_progTimer) clearInterval(_progTimer);
    _progTimer = null;
    const bar = document.getElementById('compare-bar');
    const wrap = document.getElementById('compare-progress');
    bar.style.width = '100%';
    setTimeout(() => {
        wrap.style.display = 'none';
        bar.style.width = '0%';
    }, 400);
}

// Comparison view state: the last response + the two layered filters on the
// consensus table (a pill — all/agree/only-one-model — and an optional reaction
// class picked by clicking the overview).
let _cmpData = null;
let _activePill = 'all';
let _activeClass = null;

function renderOverview(data) {
    const models = data.models;
    const consensus = data.consensus || [];
    const total = data.n_models;
    const shared = consensus.filter((g) => g.n_models >= 2).length;
    const agreePct = consensus.length ? Math.round((shared / consensus.length) * 100) : 0;

    const stats = `
    <div class="overview-stats">
        <div class="ov-stat"><span class="ov-num">${consensus.length}</span><span class="ov-cap">distinct precursor sets</span></div>
        <div class="ov-stat"><span class="ov-num">${shared}</span><span class="ov-cap">proposed by ≥2 models (${agreePct}%)</span></div>
        ${models
            .map(
                (m) =>
                    `<div class="ov-stat"><span class="ov-num">${(m.class_distribution || []).length}</span><span class="ov-cap">${esc(m.display_name)} reaction classes</span></div>`
            )
            .join('')}
    </div>`;

    const cols = models
        .map((m) => {
            const dist = m.class_distribution || [];
            const names = m.name_distribution || [];
            if (!dist.length) {
                return `<div class="dist-model"><div class="dm-name">${esc(m.display_name)}</div><div class="dist-empty">No reaction classes detected.</div></div>`;
            }
            const max = Math.max(...dist.map((d) => d.count));
            const rows = dist
                .map(
                    (d) => `
                <div class="dist-row class-row" data-class="${esc(d.label)}" title="Filter consensus to ${esc(d.label)}">
                    <span class="dr-label">${esc(d.label)}</span>
                    <span class="dr-bar-wrap"><span class="dr-bar" style="width:${(d.count / max) * 100}%"></span></span>
                    <span class="dr-count">${d.count}</span>
                </div>`
                )
                .join('');
            const nameChips = names
                .slice(0, 4)
                .map((n) => `<span class="name-chip" title="${n.count}× ${esc(n.label)}">${esc(n.label)}</span>`)
                .join('');
            return `<div class="dist-model">
                <div class="dm-name">${esc(m.display_name)}</div>
                ${rows}
                ${nameChips ? `<div class="name-chips">${nameChips}</div>` : ''}
            </div>`;
        })
        .join('');

    return `
    <div class="compare-section">
        <h2>Disconnection profile</h2>
        <div class="section-hint">Reaction-class mix per model (via RXN-Insight) — each model's disconnection bias. Click a class to filter the consensus below.</div>
        ${stats}
        <div class="dist-grid">${cols}</div>
    </div>`;
}

function renderConsensusTable(data) {
    const consensus = data.consensus || [];
    const models = data.models;
    const total = data.n_models;
    if (!consensus.length) return '';

    const pills =
        `<button class="filter-pill" data-pill="all">All</button>` +
        `<button class="filter-pill" data-pill="agree">Agree (≥2) ✓</button>` +
        models
            .map(
                (m) =>
                    `<button class="filter-pill" data-pill="only:${esc(m.model_id)}">Only ${esc(m.display_name)}</button>`
            )
            .join('');

    const head = `
        <tr>
            <th class="ct-precursors">Precursors</th>
            ${models.map((m) => `<th class="ct-rank">${esc(m.display_name)}</th>`).join('')}
            <th class="ct-class">Reaction class</th>
        </tr>`;

    const rows = consensus
        .map((g) => {
            const full = g.n_models === total;
            const byModel = {};
            g.members.forEach((m) => (byModel[m.model_id] = m.rank));
            const presentIds = g.members.map((m) => m.model_id).join(',');
            const mols = g.precursors.split('.');
            const rankCells = models
                .map((m) => {
                    const r = byModel[m.model_id];
                    return r
                        ? `<td class="ct-rank"><span class="rank-pill">#${r}</span></td>`
                        : `<td class="ct-rank ct-absent">—</td>`;
                })
                .join('');
            return `
            <tr class="ct-row${full ? ' full-agreement' : ''}"
                data-class="${esc(g.reaction_class || '')}"
                data-agree="${g.n_models}"
                data-models="${esc(presentIds)}">
                <td class="ct-precursors">
                    <div class="ct-svg">${g.svg}</div>
                    <div class="ct-names" data-smiles="${esc(g.precursors)}">${mols.map((s) => esc(s)).join(' + ')}</div>
                </td>
                ${rankCells}
                <td class="ct-class">${g.reaction_class ? esc(g.reaction_class) : '<span class="ct-absent">—</span>'}</td>
            </tr>`;
        })
        .join('');

    return `
    <div class="compare-section" id="consensus-section">
        <h2>Consensus</h2>
        <div class="section-hint">Precursor sets ranked by agreement, then best rank. Each model's rank is shown side by side; full agreement is highlighted.</div>
        <div class="filter-bar">
            <div class="filter-pills">${pills}</div>
            <span class="active-class" id="active-class" style="display:none;"></span>
        </div>
        <div class="ct-wrap">
            <table class="consensus-table">
                <thead>${head}</thead>
                <tbody>${rows}</tbody>
            </table>
            <div id="ct-empty" class="dist-empty" style="display:none; padding:14px;">No precursor sets match this filter.</div>
        </div>
        <div class="score-note">Ranks are comparable across models; raw scores are not (each model scores on its own scale). Compound names resolved via PubChem.</div>
    </div>`;
}

function applyFilters() {
    document.querySelectorAll('.filter-pill').forEach((p) =>
        p.classList.toggle('active', p.dataset.pill === _activePill)
    );
    const chip = document.getElementById('active-class');
    if (chip) {
        if (_activeClass) {
            chip.style.display = '';
            chip.innerHTML = `class: ${esc(_activeClass)} <span class="ac-clear">×</span>`;
        } else {
            chip.style.display = 'none';
        }
    }
    document.querySelectorAll('.class-row').forEach((r) =>
        r.classList.toggle('active', r.dataset.class === _activeClass)
    );

    let visible = 0;
    document.querySelectorAll('.ct-row').forEach((row) => {
        const agree = parseInt(row.dataset.agree, 10);
        const present = (row.dataset.models || '').split(',');
        let ok = true;
        if (_activePill === 'agree') ok = agree >= 2;
        else if (_activePill.startsWith('only:')) {
            const mid = _activePill.slice(5);
            ok = present.length === 1 && present[0] === mid;
        }
        if (ok && _activeClass) ok = row.dataset.class === _activeClass;
        row.style.display = ok ? '' : 'none';
        if (ok) visible++;
    });

    const empty = document.getElementById('ct-empty');
    if (empty) empty.style.display = visible ? 'none' : '';
}

function wireFilters() {
    document.querySelectorAll('.filter-pill').forEach((p) =>
        p.addEventListener('click', () => {
            _activePill = p.dataset.pill;
            applyFilters();
        })
    );
    document.querySelectorAll('.class-row').forEach((r) =>
        r.addEventListener('click', () => {
            _activeClass = _activeClass === r.dataset.class ? null : r.dataset.class;
            applyFilters();
            document.getElementById('consensus-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        })
    );
    const chip = document.getElementById('active-class');
    chip?.addEventListener('click', () => {
        _activeClass = null;
        applyFilters();
    });
}

// Lazy PubChem names: collect every unique precursor SMILES across the table,
// resolve via the same cached endpoint the single-step page uses (≤10/request),
// then fill each row's name line. Failures are silent — SMILES stays as fallback.
async function fillPubChemNames() {
    const cells = Array.from(document.querySelectorAll('.ct-names'));
    const unique = [...new Set(cells.flatMap((c) => c.dataset.smiles.split('.')))];
    const nameMap = {};
    for (let i = 0; i < unique.length; i += 10) {
        const chunk = unique.slice(i, i + 10);
        try {
            const resp = await fetch('/api/evaluate/compound-lookup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ smiles_list: chunk }),
            });
            if (!resp.ok) continue;
            const { compounds } = await resp.json();
            (compounds || []).forEach((c) => {
                if (c.name) nameMap[c.smiles] = c.name;
            });
        } catch (e) {
            /* keep SMILES fallback */
        }
    }
    if (!Object.keys(nameMap).length) return;
    cells.forEach((c) => {
        const parts = c.dataset.smiles.split('.').map((s) => nameMap[s] || s);
        c.innerHTML = parts.map((p) => esc(p)).join(' + ');
    });
}

function renderResults(data) {
    _cmpData = data;
    _activePill = 'all';
    _activeClass = null;
    const t = data.target;
    const resolved = t.resolved_from
        ? `<div style="font-size:12px;color:#888;">resolved from "${esc(t.resolved_from.query)}"</div>`
        : '';
    const target = `
    <div class="compare-target">
        <div>${t.svg}</div>
        <div class="ct-info">
            <div class="smiles">${esc(t.smiles)}</div>
            <div>${esc(t.formula)} · MW ${t.mw}</div>
            ${resolved}
        </div>
    </div>`;
    document.getElementById('compare-results').innerHTML =
        target + renderOverview(data) + renderConsensusTable(data);
    wireFilters();
    applyFilters();
    fillPubChemNames();
}

async function runCompare(e) {
    e.preventDefault();
    hideError();
    const smiles = document.getElementById('smiles-input').value.trim();
    if (!smiles) {
        showError('Please enter a molecule.');
        return;
    }
    const modelIds = selectedModelIds();
    if (modelIds.length < 2 || modelIds.length > 4) {
        showError('Select between 2 and 4 models to compare.');
        return;
    }
    const nPrecursors = parseInt(document.getElementById('n-precursors').value, 10) || 10;

    const btn = document.getElementById('compare-btn');
    btn.disabled = true;
    document.getElementById('compare-results').innerHTML = '';
    startProgress();
    try {
        const resp = await fetch('/api/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ smiles, model_ids: modelIds, n_precursors: nPrecursors }),
        });
        const data = await resp.json();
        if (!resp.ok) {
            showError(data.error || 'Comparison failed.');
            return;
        }
        renderResults(data);
    } catch (err) {
        showError('Network error: ' + err.message);
    } finally {
        stopProgress();
        btn.disabled = false;
    }
}

document.getElementById('compare-form').addEventListener('submit', runCompare);

// Prefill from a single-step "Compare with other models" link
// (/compare?smiles=...&models=id1,id2). Fills the target and preselects the
// carried-over model(s); does NOT auto-run — the user picks the opponents.
(function prefillFromQuery() {
    const q = new URLSearchParams(window.location.search);
    const smiles = q.get('smiles');
    if (smiles) document.getElementById('smiles-input').value = smiles;
    const models = (q.get('models') || '').split(',').filter(Boolean);
    if (models.length) {
        document.querySelectorAll('#model-checks input[name="model_id"]').forEach((c) => {
            c.checked = models.includes(c.value);
        });
    }
})();
