/* RxnLab single-step/multi-step predict page — RDKit init + page bootstrap; load LAST.
   Split from the former inline <script> in predict.html. All functions are
   globals (called from inline handlers); load order set in predict.html. */

/* ── RDKit.js init ── */
window.initRDKitModule().then(function(RDKit) {
    console.log('RDKit.js version: ' + RDKit.version());
    RDKitModule = RDKit;
}).catch(function(err) {
    console.warn('RDKit.js failed to load:', err);
});

/* ── Initial form submission (AJAX) ── */
(function() {
    var form = document.getElementById('predict-form');
    var submitBtn = document.getElementById('submit-btn');
    var resultsContainer = document.getElementById('results-container');

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        if (searchModeOn()) { runMultiStepSearch(); return; }
        var modelSel = document.getElementById('model-select');
        var modelName = modelSel ? modelSel.options[modelSel.selectedIndex].text : 'the model';
        var stepsInput = form.querySelector('[name="diffusion_steps"]:not([disabled])');
        var nInput = form.querySelector('[name="n_precursors"]');
        var n = nInput ? nInput.value : '1';
        var label = 'Running ' + modelName +
                    (stepsInput ? ' with steps=' + stepsInput.value : '') +
                    ' for ' + n + ' sample(s)...';

        currentTargetSmiles = document.getElementById('smiles-input').value.trim();
        generations = [];
        currentInpaint = null;

        resultsContainer.innerHTML = '';
        submitBtn.disabled = true;
        showProgress(label);

        var formData = new FormData(form);
        fetch('/lab', {
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

    onModelChange();  // sync param controls to the initially-selected model

    // ?smiles= prefills the target on either page.
    (function applyDeepLink() {
        var smiles = new URLSearchParams(window.location.search).get('smiles');
        if (smiles) {
            var si = document.getElementById('smiles-input');
            if (si) si.value = smiles;
        }
    })();
})();
