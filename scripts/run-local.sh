#!/usr/bin/env bash
# Launch RxnLab locally for UX testing (e.g. the /compare page) via `python wsgi.py`.
# All four models run in-process — DiffAlign in-process is slow (~16s/request); to use
# the fast Modal GPU path instead, set the two RXNLAB_MODAL_* vars at the bottom.
set -euo pipefail

# --- conda env -------------------------------------------------------------
# Project pins Python 3.10 in `diffalign-10`. Use that env's `python`, not system python3.
# `conda` may not be on PATH in a non-interactive shell, so locate the base and source
# its activation hook before activating. Falls back to the laptop's /opt/miniconda3.
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
else
    CONDA_BASE="${CONDA_BASE:-/opt/miniconda3}"
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate diffalign-10

cd "$(dirname "$0")/.."  # repo root (wsgi.py lives there); this script is in scripts/

# --- required env ----------------------------------------------------------
# MEGAN imports a package that calls into GitPython at import time; without a git binary
# on PATH it raises unless this is set (see commit 41153e6, baked into the prod image).
export GIT_PYTHON_REFRESH=quiet

# Dev secret so Flask sessions work; fine for local only.
export SECRET_KEY="dev-local-$(whoami)"

# macOS only: wsgi.py forces the 'fork' multiprocessing start method for R-SMILES parity
# with Linux/prod. Forking after Accelerate/Objective-C libs init aborts unless this is
# set ("may have been in progress in another thread when fork() was called"). No-op on Linux.
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# --- optional env (left unset = sensible local defaults) -------------------
# No DATABASE_URL  -> app runs without a DB (feedback/analytics not persisted). Fine for UX.
# No RXNLAB_MODAL_* -> DiffAlign runs in-process (slow but self-contained).
# Syntheseus checkpoints (LocalRetro/MEGAN/R-SMILES) auto-download to:
#   ${SYNTHESEUS_CACHE_DIR:-$HOME/.cache/torch/syntheseus}  on first use.
# export SYNTHESEUS_CACHE_DIR="$HOME/.cache/torch/syntheseus"

# To test against the deployed Modal DiffAlign GPU instead of in-process, uncomment:
# export RXNLAB_MODAL_DIFFALIGN_URL="https://<your-modal-app>.modal.run"
# export RXNLAB_PROXY_TOKEN="<token>"

# --- launch ----------------------------------------------------------------
# wsgi.py's __main__ block runs Flask's dev server on 0.0.0.0:8080 (debug=True).
echo "Booting RxnLab — first request warms the classifier + models (~15-20s)."
echo "Open http://localhost:8080/compare once it's up."
python wsgi.py
