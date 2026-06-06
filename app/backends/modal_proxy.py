"""Local client for the Modal-hosted DiffAlign service (app/modal_app.py).

``ModalDiffAlign`` is a ``BackwardReactionModel`` that holds no weights — it POSTs
to the deployed Modal endpoints and rebuilds syntheseus reactions locally via
``process_raw_smiles_outputs_backwards``, so ``registry.predict`` and the Retro*
search both work against it unchanged. The whole contract is plain JSON (DiffAlign's
``sample_data`` already round-trips to the browser, so it serializes fine here too).

Config (env):
- ``RXNLAB_MODAL_DIFFALIGN_URL`` — the base prefix printed by ``modal deploy``,
  e.g. ``https://<workspace>--rxnlab-diffalign``. Per-endpoint URL =
  ``{prefix}-diffalignservice-{predict|get-reactions|inpaint}.modal.run``.
- ``RXNLAB_PROXY_TOKEN`` — shared secret matching the ``rxnlab-proxy`` Modal secret.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence

import requests

from syntheseus.interface.models import BackwardReactionModel
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.utils.inference import (
    process_raw_smiles_outputs_backwards,
)

# Endpoint name segment is the Modal class name lowercased (see app/modal_app.py).
_SERVICE = "diffalignservice"
_TIMEOUT_S = 300  # generous: a cold start can take tens of seconds


class ModalDiffAlign(BackwardReactionModel):
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        rich: bool = True,
        diffusion_steps: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._base_url = (base_url or os.environ["RXNLAB_MODAL_DIFFALIGN_URL"]).rstrip("-")
        self._token = token or os.environ["RXNLAB_PROXY_TOKEN"]
        self.rich = rich
        self.diffusion_steps = diffusion_steps  # mutated by registry ParamSpec.apply

    @property
    def name(self) -> str:
        return "DiffAlign"

    def _url(self, endpoint: str) -> str:
        return f"{self._base_url}-{_SERVICE}-{endpoint}.modal.run"

    def _post(self, endpoint: str, body: dict) -> Any:
        resp = requests.post(
            self._url(endpoint),
            json=body,
            headers={"X-RxnLab-Token": self._token},
            timeout=_TIMEOUT_S,
        )
        resp.raise_for_status()
        return resp.json()

    def _get_reactions(
        self, inputs: List[Molecule], num_results: int
    ) -> List[Sequence[SingleProductReaction]]:
        endpoint = "predict" if self.rich else "get-reactions"
        per_input = self._post(
            endpoint,
            {
                "smiles_list": [m.smiles for m in inputs],
                "num_results": num_results,
                "diffusion_steps": self.diffusion_steps,
            },
        )
        return [
            process_raw_smiles_outputs_backwards(
                input=mol,
                output_list=per["output_list"],
                metadata_list=per["metadata_list"],
            )
            for mol, per in zip(inputs, per_input)
        ]

    def inpaint(
        self,
        *,
        product_smiles: str,
        previous_sample_data: dict,
        inpaint_node_indices: list,
        n_precursors: int = 1,
        diffusion_steps: int = 1,
    ) -> tuple:
        """Mirror of ``DiffAlign.api.predict.predict_with_inpainting`` over Modal:
        returns ``(results, failure_info)``."""
        out = self._post(
            "inpaint",
            {
                "product_smiles": product_smiles,
                "previous_sample_data": previous_sample_data,
                "selected_node_indices": inpaint_node_indices,
                "n_precursors": n_precursors,
                "diffusion_steps": diffusion_steps,
            },
        )
        return out["results"], out["failure"]
