"""Modal-hosted DiffAlign inference service — RxnLab Phase 2 "heavy-model backend".

The 4 GB CPU box can't run DiffAlign fast enough for multi-step search (and loading
it locally eats ~1-2 GB it can't spare). This module hosts DiffAlign on a Modal GPU
and exposes three JSON endpoints that mirror the in-process DiffAlign call paths:

  POST /predict        rich single-step  -> per-input {output_list, metadata_list}
                       (metadata carries score, sample_data, atom_mapping, mapped_rxn)
  POST /inpaint        substructure-locked regen -> {results, failure}
  POST /get_reactions  lean batch for Retro* search -> per-input {output_list, metadata_list}
                       (metadata carries probability only — cheap decode)

The RxnLab Flask app calls these from app/backends/modal_proxy.py (build step 2),
which rebuilds syntheseus reaction objects locally from output_list/metadata_list.
sample_data is the serialized DiffAlign PlaceHolder dict that already round-trips to
the browser today, so the whole contract is plain JSON.

Deploy:
    modal deploy app/modal_app.py
    # prints the public base URL, e.g. https://<workspace>--rxnlab-diffalign-*.modal.run

Auth: every request must send  X-RxnLab-Token: <token>  matching key
RXNLAB_PROXY_TOKEN in the Modal secret named 'rxnlab-proxy':
    modal secret create rxnlab-proxy RXNLAB_PROXY_TOKEN=$(openssl rand -hex 32)
"""
import os

import modal
from fastapi import Header, HTTPException

# Immutable checkpoint asset — same GitHub Release the public Docker image bakes in
# (see Dockerfile). SHA-pinned so the Modal image build is reproducible too.
CKPT_URL = (
    "https://github.com/Aalto-QuML/DiffAlign/releases/download/"
    "checkpoints%2Falign-absorbing-v1/diffalign-align-absorbing-v1.tar.gz"
)
CKPT_SHA256 = "ec8620eb6d18b481f591d6023c6ddc8c39d9dddcaa7632d05a81535705b5dea0"

# ── GPU vs CPU ────────────────────────────────────────────────────────────────
# DiffAlign is small; a T4 (16 GB, cheapest GPU) is plenty. To fall back to CPU
# (zero CUDA-wheel risk — reuses the box's exact torch/PyG install), set GPU = None
# and swap the two torch/PyG `.pip_install(...)` blocks below for the box's
# requirements.txt (torch==2.6.0+cpu + the +cpu PyG find-links).
GPU = "T4"

# ⚠️ DEPLOY-TIME CHECK: these CUDA wheels must exist for torch 2.6.0. cu124 is the
# standard build for 2.6.0 and PyG publishes a matching wheel index, but verify on
# the first `modal deploy` — a wheel mismatch here is the single most likely failure.
CUDA_TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
PYG_FIND_LINKS = "https://data.pyg.org/whl/torch-2.6.0+cu124.html"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libxrender1", "libxext6", "libexpat1", "libcairo2",
        "libfreetype6", "libfontconfig1", "curl", "ca-certificates",
    )
    # torch BEFORE PyG (torch-scatter etc. import torch at build time) — same
    # ordering rationale as the Dockerfile.
    .pip_install("torch==2.6.0", extra_index_url=CUDA_TORCH_INDEX)
    .pip_install(
        "torch-geometric==2.6.1", "torch-scatter==2.1.2", "torch-sparse==0.6.18",
        "torch-cluster==1.6.3", "torch-spline-conv==1.2.2",
        find_links=PYG_FIND_LINKS,
    )
    .pip_install_from_requirements("requirements-modal.txt")
    # copy=True bakes the source into the image so the next run_commands can
    # `pip install -e` it and the checkpoint can land relative to PROJECT_ROOT.
    .add_local_dir(
        "DiffAlign", "/root/DiffAlign", copy=True,
        ignore=["checkpoints", ".git", "**/__pycache__", "*.pyc"],
    )
    .run_commands(
        "pip install --no-cache-dir -e /root/DiffAlign",
        # PROJECT_ROOT = /root/DiffAlign → checkpoint must be at checkpoints/epoch760.pt
        "mkdir -p /root/DiffAlign/checkpoints",
        f"curl -fL -o /tmp/ckpt.tar.gz '{CKPT_URL}'",
        f"echo '{CKPT_SHA256}  /tmp/ckpt.tar.gz' | sha256sum -c -",
        "tar xzf /tmp/ckpt.tar.gz -C /root/DiffAlign/checkpoints",
        "rm /tmp/ckpt.tar.gz",
    )
    .env({"HOME": "/root", "SYNTHESEUS_CACHE_DIR": "/root/.cache/syntheseus"})
)

app = modal.App("rxnlab-diffalign", image=image)


def _serialize(reactions_per_input: list) -> list:
    """syntheseus reactions -> per-input {output_list, metadata_list}, the shape
    app/backends/modal_proxy.py feeds back to process_raw_smiles_outputs_backwards."""
    out = []
    for reactions in reactions_per_input:
        output_list, metadata_list = [], []
        for r in reactions:
            md = dict(r.metadata)
            output_list.append(
                md.get("precursors") or ".".join(sorted(m.smiles for m in r.reactants))
            )
            metadata_list.append(md)
        out.append({"output_list": output_list, "metadata_list": metadata_list})
    return out


@app.cls(
    gpu=GPU,
    scaledown_window=120,  # stance (a): no min_containers → scale to zero, accept cold start
    secrets=[modal.Secret.from_name("rxnlab-proxy")],
)
class DiffAlignService:
    @modal.enter()
    def load(self):
        # Two decode modes share one set of weights: rich (web single-step + inpaint)
        # vs lean (search — skips atom-mapping / mapped-rxn / stereo).
        from diffalign.model import DiffAlignModel

        self.rich = DiffAlignModel(diffusion_steps=1, samples_per_product=100, rich_metadata=True)
        self.lean = DiffAlignModel(diffusion_steps=1, samples_per_product=100, rich_metadata=False)

    def _run(self, model, smiles_list: list, num_results: int, diffusion_steps: int) -> list:
        from syntheseus.interface.molecule import Molecule

        model.diffusion_steps = diffusion_steps  # mirrors the ParamSpec.apply on the box
        mols = [Molecule(smiles=s) for s in smiles_list]
        return _serialize(model(mols, num_results=num_results))

    @modal.fastapi_endpoint(method="POST")
    def predict(self, body: dict, x_rxnlab_token: str = Header(default="")):
        _auth(x_rxnlab_token)
        return self._run(
            self.rich,
            body["smiles_list"],
            int(body.get("num_results", 10)),
            int(body.get("diffusion_steps", 1)),
        )

    @modal.fastapi_endpoint(method="POST")
    def get_reactions(self, body: dict, x_rxnlab_token: str = Header(default="")):
        _auth(x_rxnlab_token)
        return self._run(
            self.lean,
            body["smiles_list"],
            int(body.get("num_results", 10)),
            int(body.get("diffusion_steps", 1)),
        )

    @modal.fastapi_endpoint(method="POST")
    def inpaint(self, body: dict, x_rxnlab_token: str = Header(default="")):
        _auth(x_rxnlab_token)
        from diffalign.inference import predict_with_inpainting

        results, failure = predict_with_inpainting(
            product_smiles=body["product_smiles"],
            previous_sample_data=body["previous_sample_data"],
            inpaint_node_indices=body["selected_node_indices"],
            n_precursors=int(body.get("n_precursors", 1)),
            diffusion_steps=int(body.get("diffusion_steps", 1)),
        )
        return {"results": results, "failure": failure}


def _auth(token: str) -> None:
    if not token or token != os.environ.get("RXNLAB_PROXY_TOKEN"):
        raise HTTPException(status_code=401, detail="missing or bad X-RxnLab-Token")
