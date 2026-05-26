"""Phase-0 make-or-break parity check (seeded, end-to-end).

Asserts that routing inference through the syntheseus wrapper (via the model
registry) produces byte-identical predictions to DiffAlign's native
``predict.predict_precursors`` path the app used before Phase 0.

This is now a true seeded end-to-end equivalence test. It became possible after
fixing the dominant nondeterminism source (`transformer_model_with_y.py`: the
Laplacian-PE `eigsh` used a random ARPACK start vector untied to any seed; now
pinned with `v0=np.ones(n)`). A benign ~1e-7 float-level residual remains in the
transformer forward but does not change discrete predictions.

Method: for each (product, n, steps) we pre-build the transition model (so it is
not rebuilt mid-run, which would consume RNG asymmetrically), seed torch+numpy+
random, run the native path, re-seed identically, run the registry path, and
require identical precursors / score / atom_mapping / mapped_rxn / sample_data.

Run:  python scripts/phase0_parity_check.py
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "DiffAlign"))

torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_num_threads(1)

from DiffAlign.api import predict  # noqa: E402  (eager-loads the model once)
from diffalign.inference import _ensure_transition_model  # noqa: E402
from app.registry import registry  # noqa: E402

SEED = 1234
PRODUCTS = [
    "CC(=O)Oc1ccccc1C(=O)O",        # aspirin
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",     # caffeine
    "O=C(O)c1ccccc1",                 # benzoic acid
    "CC(=O)Nc1ccc(O)cc1",             # paracetamol
]
CASES = [(3, 1), (5, 1), (10, 5)]   # (n_precursors, diffusion_steps)

_FIELDS = ("precursors", "score", "atom_mapping", "mapped_rxn", "sample_data")


def _seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def _compare(native, reg, ctx):
    problems = []
    if len(native) != len(reg):
        return [f"{ctx}: count differs native={len(native)} registry={len(reg)}"]
    for i, (a, b) in enumerate(zip(native, reg)):
        for k in _FIELDS:
            if a.get(k) != b.get(k):
                problems.append(f"{ctx}[{i}].{k} differs")
    return problems


def main() -> int:
    model_id = registry.default_model_id()
    model = registry._instance(model_id)

    all_problems = []
    for smiles in PRODUCTS:
        # Feed canonical SMILES to both paths. syntheseus `Molecule` re-serializes
        # to RDKit-canonical, which reorders atoms for non-canonical inputs; that
        # would change sample_data/mapped_rxn (atom ordering) without changing the
        # canonical precursors. Canonicalizing up front isolates the routing change
        # from that input-normalization difference.
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        for n, steps in CASES:
            ctx = f"{smiles} n={n} steps={steps}"

            # Pre-build the transition model for these steps so neither path
            # rebuilds it (and consumes RNG) mid-run.
            model.diffusion_steps = steps
            _ensure_transition_model(model._model, model._cfg, steps)

            _seed()
            native = predict.predict_precursors(smiles, n_precursors=n, diffusion_steps=steps)
            _seed()
            reg = registry.predict(model_id, smiles, n_precursors=n, diffusion_steps=steps)

            probs = _compare(native, reg, ctx)
            print(f"[{'OK' if not probs else 'FAIL'}] {ctx}: native={len(native)} registry={len(reg)}")
            all_problems.extend(probs)

    if all_problems:
        print("\n=== PARITY FAILURES ===")
        for p in all_problems:
            print(" -", p)
        return 1
    print("\nAll cases byte-identical. Seeded end-to-end parity holds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
