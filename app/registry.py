"""Model registry — single source of truth for the models RxnLab can run.

Phase 0 of the syntheseus upgrade: the Flask app no longer calls DiffAlign's
native ``predict.predict_precursors`` directly. Instead it routes inference
through the syntheseus ``BackwardReactionModel`` interface, driven by this
registry. Each entry carries display metadata, capability flags, a ``backend``,
and a factory that lazily builds the (heavy) wrapper instance on first use.

The registry reconstructs the rich per-prediction dict the rest of the app
expects (``precursors``/``score``/``sample_data``/``atom_mapping``/``mapped_rxn``)
from each reaction's ``metadata``, which the wrapper populates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    version: str
    description: str
    wrapper_factory: Callable[[], Any]
    supports_inpainting: bool = False
    supports_steering: bool = False
    backend: str = "in-process"  # in-process | remote | modal
    metadata: dict = field(default_factory=dict)


def _make_diffalign():
    """Instantiate the DiffAlign syntheseus wrapper.

    ``samples_per_product=1`` so a request for ``n`` precursors draws exactly ``n``
    stochastic samples — matching the native ``predict_precursors`` path
    (``n_samples = n_precursors``) for the Phase-0 parity guarantee. Heavier
    sampling for search/coverage is a per-entry knob, not a parity break.
    """
    from diffalign.model import DiffAlignModel

    return DiffAlignModel(diffusion_steps=1, samples_per_product=1)


DEFAULT_MODEL_ID = "diffalign-align-absorbing-v1"

_SPECS = [
    ModelSpec(
        model_id=DEFAULT_MODEL_ID,
        display_name="DiffAlign (align-absorbing)",
        version="epoch760",
        description="Graph diffusion model for single-step retrosynthesis.",
        wrapper_factory=_make_diffalign,
        supports_inpainting=True,
        supports_steering=False,
        backend="in-process",
        metadata={"arch": "graph-diffusion", "training": "USPTO-50k"},
    ),
]


class ModelRegistry:
    def __init__(self, specs: list[ModelSpec]):
        self._specs = {s.model_id: s for s in specs}
        self._instances: dict[str, Any] = {}

    def list_specs(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def get_spec(self, model_id: str) -> ModelSpec:
        return self._specs[model_id]

    def default_model_id(self) -> str:
        return DEFAULT_MODEL_ID

    def supports_inpainting(self, model_id: str) -> bool:
        return self._specs[model_id].supports_inpainting

    def _instance(self, model_id: str):
        if model_id not in self._instances:
            self._instances[model_id] = self._specs[model_id].wrapper_factory()
        return self._instances[model_id]

    def predict(
        self,
        model_id: str,
        product_smiles: str,
        *,
        n_precursors: int,
        diffusion_steps: int,
    ) -> list[dict]:
        """Run single-step prediction through the syntheseus wrapper and return
        the rich dict shape the app/templates expect."""
        from syntheseus.interface.molecule import Molecule

        model = self._instance(model_id)
        model.diffusion_steps = diffusion_steps  # picked up by _ensure_transition_model
        reactions = model([Molecule(smiles=product_smiles)], num_results=n_precursors)[0]
        return [_reaction_to_dict(r) for r in reactions]


def _reaction_to_dict(reaction) -> dict:
    md = reaction.metadata
    return {
        "precursors": md["precursors"],
        "score": md["score"],
        "sample_data": md.get("sample_data"),
        "atom_mapping": md.get("atom_mapping"),
        "mapped_rxn": md.get("mapped_rxn"),
    }


registry = ModelRegistry(_SPECS)
