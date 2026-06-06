"""Smoke-test the deployed Modal DiffAlign service (app/modal_app.py).

Posts a single product to each endpoint and checks the response shape — confirms
the GPU container loads the model, the checkpoint is present, and the JSON
contract matches what app/backends/modal_proxy.py will expect.

Usage:
    export RXNLAB_MODAL_BASE_URL='https://<workspace>--rxnlab-diffalign'  # prefix printed by `modal deploy`
    export RXNLAB_PROXY_TOKEN='<same token as the rxnlab-proxy Modal secret>'
    python scripts/modal_smoke_test.py [SMILES]

The base URL prefix + per-endpoint suffix: Modal names each web endpoint
"<prefix>-<classname-lowercased>-<method>.modal.run". With class DiffAlignService
the three URLs are:
    {BASE}-diffalignservice-predict.modal.run
    {BASE}-diffalignservice-get-reactions.modal.run
    {BASE}-diffalignservice-inpaint.modal.run
(Confirm the exact URLs from the `modal deploy` output and override via the
RXNLAB_MODAL_*_URL env vars below if they differ.)
"""
import json
import os
import sys
import urllib.request


def _post(url: str, body: dict, token: str) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-RxnLab-Token": token},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:  # 600s: first call eats cold start
        return json.loads(resp.read())


def _url(suffix: str) -> str:
    base = os.environ["RXNLAB_MODAL_BASE_URL"].rstrip("-")
    explicit = os.environ.get(f"RXNLAB_MODAL_{suffix.upper().replace('-', '_')}_URL")
    return explicit or f"{base}-diffalignservice-{suffix}.modal.run"


def main() -> None:
    token = os.environ["RXNLAB_PROXY_TOKEN"]
    product = sys.argv[1] if len(sys.argv) > 1 else "CC(=O)Oc1ccccc1C(=O)O"  # aspirin

    print(f"product: {product}")

    print("\n[1/2] POST /predict (rich) …")
    pred = _post(_url("predict"), {"smiles_list": [product], "num_results": 3, "diffusion_steps": 1}, token)
    per_input = pred[0]
    assert "output_list" in per_input and "metadata_list" in per_input, per_input
    print(f"  ok: {len(per_input['output_list'])} precursor sets")
    md0 = per_input["metadata_list"][0]
    print(f"  first precursors: {per_input['output_list'][0]}")
    print(f"  metadata keys: {sorted(md0)}  (expect score/sample_data/atom_mapping/mapped_rxn)")
    assert "sample_data" in md0, "rich decode must carry sample_data for inpainting"

    print("\n[2/2] POST /get_reactions (lean) …")
    lean = _post(_url("get-reactions"), {"smiles_list": [product], "num_results": 3, "diffusion_steps": 1}, token)
    lmd0 = lean[0]["metadata_list"][0]
    print(f"  ok: {len(lean[0]['output_list'])} precursor sets; metadata keys: {sorted(lmd0)}")
    assert "probability" in lmd0, "lean decode must carry probability for Retro* cost"

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
