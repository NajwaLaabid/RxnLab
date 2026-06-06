/* RxnLab single-step/multi-step predict page — shared state + config.
   Split from the former inline <script> in predict.html. All functions are
   globals (called from inline handlers); load order set in predict.html. */

var RDKitModule = null;
var generations = [];  // [{results: [...], fixedInfo: null|str, targetSmiles: str}, ...]
var currentInpaint = null;  // {genIdx, resultIdx, selectedAtoms: Set}
var currentTargetSmiles = '';
var MAX_GENERATIONS = 4;  // 1 initial + 3 inpainting rounds

// Multi-step lives on its own page (/multistep); single-step on /lab.
var MULTISTEP = window.RXNLAB_MULTISTEP === true;
