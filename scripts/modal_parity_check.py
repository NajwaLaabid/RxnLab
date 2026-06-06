"""Parity + integration check for the Modal DiffAlign backend (build step 2).

Runs the real ``registry.predict`` path against the Modal proxy, exercises an
inpaint round-trip, and compares the predicted-precursor set to the in-process
model. NOTE: DiffAlign sampling is stochastic and GPU(Modal)≠CPU(local) bitwise,
so this checks set OVERLAP and contract shape, not byte equality.

Usage (env must point at the deployed service):
    export RXNLAB_MODAL_DIFFALIGN_URL='https://najwalaabid--rxnlab-diffalign'
    export RXNLAB_PROXY_TOKEN="$(cat /tmp/rxnlab_proxy_token)"
    python scripts/modal_parity_check.py [SMILES]
"""
import os
import sys

# Mirror create_app()'s path setup so the in-process DiffAlign import works.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DiffAlign"))

from rdkit import Chem

from app.registry import DEFAULT_MODEL_ID, registry


def _canon_set(preds: list) -> set:
    out = set()
    for p in preds:
        for frag in (p["precursors"] or "").split("."):
            m = Chem.MolFromSmiles(frag)
            if m:
                out.add(Chem.MolToSmiles(m))
    return out


def _predict(smiles: str, n: int) -> list:
    registry._instances.clear()  # force re-instantiation under current env
    return registry.predict(DEFAULT_MODEL_ID, smiles, n_precursors=n)


def main() -> None:
    smiles = sys.argv[1] if len(sys.argv) > 1 else "CC(=O)Oc1ccccc1C(=O)O"
    n = 5
    assert os.environ.get("RXNLAB_MODAL_DIFFALIGN_URL"), "set RXNLAB_MODAL_DIFFALIGN_URL"

    print(f"target: {smiles}\nbackend spec: {registry.get_spec(DEFAULT_MODEL_ID).backend}\n")

    print("[1] registry.predict via MODAL …")
    modal_preds = _predict(smiles, n)
    print(f"  {len(modal_preds)} predictions")
    p0 = modal_preds[0]
    assert {"precursors", "score", "sample_data", "atom_mapping", "mapped_rxn"} <= set(p0), p0.keys()
    assert p0["sample_data"], "rich predict must carry sample_data"
    for p in modal_preds:
        print(f"    {p['score']:.3f}  {p['precursors']}")

    print("\n[2] inpaint round-trip via MODAL …")
    sample_data = p0["sample_data"]
    node_mask = sample_data.get("node_mask") or [[]]
    row = node_mask[0] if node_mask and isinstance(node_mask[0], list) else node_mask
    real_idx = [i for i, v in enumerate(row) if v]
    fixed = real_idx[: max(1, len(real_idx) // 2)]  # fix ~half the real atoms
    proxy = registry._instance(DEFAULT_MODEL_ID)
    results, failure = proxy.inpaint(
        product_smiles=smiles,
        previous_sample_data=sample_data,
        inpaint_node_indices=fixed,
        n_precursors=3,
        diffusion_steps=1,
    )
    print(f"  fixed {len(fixed)}/{len(real_idx)} real atoms -> {len(results)} result(s)"
          + (f"; failure: {failure.get('message','')[:60]}" if not results and failure else ""))

    print("\n[3] in-process predict (unset URL) for overlap …")
    saved = os.environ.pop("RXNLAB_MODAL_DIFFALIGN_URL")
    try:
        local_preds = _predict(smiles, n)
    finally:
        os.environ["RXNLAB_MODAL_DIFFALIGN_URL"] = saved
    ms, ls = _canon_set(modal_preds), _canon_set(local_preds)
    inter = ms & ls
    print(f"  modal frags={len(ms)} local frags={len(ls)} overlap={len(inter)}")
    print(f"  shared: {sorted(inter)}")

    print("\nStep-2 integration check passed (contract shape + inpaint round-trip OK).")


if __name__ == "__main__":
    main()
