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
class ParamSpec:
    """A model-specific inference knob the UI renders beyond the universal
    ``n_precursors``. ``apply`` pushes a validated value onto the (heavy) wrapper
    instance right before prediction.

    ``kind`` is ``"select"`` (value must be one of ``choices``) or ``"int"``
    (``min``/``max`` bounds). DiffAlign's ``diffusion_steps`` is a ``select`` over
    divisors of the training step count, so the old "must evenly divide" check
    collapses to a membership test.
    """
    name: str
    label: str
    kind: str  # "select" | "int"
    default: Any
    apply: Callable[[Any, Any], None]
    help: str = ""
    choices: tuple = ()
    min: Optional[int] = None
    max: Optional[int] = None

    def coerce(self, raw: Any) -> Any:
        """Parse + validate a raw form value; raise ValueError with a UI message."""
        try:
            val = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{self.label} must be a number.")
        if self.kind == "select":
            if val not in self.choices:
                raise ValueError(
                    f"{self.label} must be one of: {', '.join(map(str, self.choices))}."
                )
        elif self.kind == "int":
            if (self.min is not None and val < self.min) or (
                self.max is not None and val > self.max
            ):
                raise ValueError(f"{self.label} must be between {self.min} and {self.max}.")
        return val


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
    params: tuple = ()  # model-specific ParamSpecs (n_precursors is universal)
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


LOCALRETRO_FIGSHARE_ID = 23960745


def _ensure_localretro_checkpoint():
    """Return a populated LocalRetro ``model_dir``, downloading on first use.

    syntheseus' own ``get_figshare_download_link`` builds a *stale* per-file URL for
    this article (Figshare re-versions file IDs), so its auto-download 404/403s. We
    instead hit the Figshare article API, take the live ``download_url``, verify the
    published md5, and extract the bundle (``*.pth`` + ``default_config.json`` +
    ``data/*.csv``) into the syntheseus cache layout.
    """
    import hashlib
    import json
    import os
    import urllib.request
    import zipfile
    from pathlib import Path

    cache_root = Path(
        os.environ.get("SYNTHESEUS_CACHE_DIR", Path.home() / ".cache" / "torch" / "syntheseus")
    )
    model_dir = cache_root / "LocalRetro_backward"
    if list(model_dir.glob("*.pth")):
        return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    api = f"https://api.figshare.com/v2/articles/{LOCALRETRO_FIGSHARE_ID}/files"
    headers = {"User-Agent": "Mozilla/5.0"}
    files = json.load(urllib.request.urlopen(urllib.request.Request(api, headers=headers), timeout=120))
    f = files[0]
    data = urllib.request.urlopen(
        urllib.request.Request(f["download_url"], headers=headers), timeout=600
    ).read()
    if hashlib.md5(data).hexdigest() != f["computed_md5"]:
        raise RuntimeError("LocalRetro checkpoint md5 mismatch")
    zip_path = model_dir / "model.zip"
    zip_path.write_bytes(data)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(model_dir)
    zip_path.unlink()
    return model_dir


def _make_localretro():
    """Instantiate the syntheseus LocalRetro wrapper on the USPTO-50k checkpoint.

    ``weights_only=False`` for construction only — the checkpoint pickles custom
    globals and predates torch 2.6's ``weights_only=True`` default. dgl ships no
    torch-2.6 build, so the env pins ``dgl==1.1.3`` (pre-graphbolt; fine for the
    dgllife featurization LocalRetro uses).
    """
    import functools

    import torch
    from syntheseus.reaction_prediction.inference.local_retro import LocalRetroModel

    model_dir = _ensure_localretro_checkpoint()
    orig_load = torch.load
    torch.load = functools.partial(orig_load, weights_only=False)
    try:
        return LocalRetroModel(model_dir=str(model_dir))
    finally:
        torch.load = orig_load


DEFAULT_MODEL_ID = "diffalign-align-absorbing-v1"
LOCALRETRO_MODEL_ID = "localretro-uspto50k-v1"

_DIFFUSION_STEPS = ParamSpec(
    name="diffusion_steps",
    label="Diffusion steps",
    kind="select",
    default=10,
    choices=(1, 2, 5, 10, 25, 50),
    apply=lambda model, v: setattr(model, "diffusion_steps", v),
    help=(
        "How many denoising steps the model takes to produce each sample. More "
        "steps usually mean cleaner, more confident predictions but slower inference."
    ),
)

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
        params=(_DIFFUSION_STEPS,),
        metadata={"arch": "graph-diffusion", "training": "USPTO-50k"},
    ),
    ModelSpec(
        model_id=LOCALRETRO_MODEL_ID,
        display_name="LocalRetro",
        version="uspto50k",
        description="Template-based graph model for single-step retrosynthesis.",
        wrapper_factory=_make_localretro,
        supports_inpainting=False,
        supports_steering=False,
        backend="in-process",
        params=(),
        metadata={"arch": "template-based", "training": "USPTO-50k"},
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

    def is_known(self, model_id: str) -> bool:
        return model_id in self._specs

    def ui_models(self) -> list[dict]:
        """Serializable model + param-schema info for rendering the form."""
        return [
            {
                "model_id": s.model_id,
                "display_name": s.display_name,
                "description": s.description,
                "supports_inpainting": s.supports_inpainting,
                "params": [
                    {
                        "name": p.name,
                        "label": p.label,
                        "kind": p.kind,
                        "default": p.default,
                        "choices": list(p.choices),
                        "min": p.min,
                        "max": p.max,
                        "help": p.help,
                    }
                    for p in s.params
                ],
            }
            for s in self._specs.values()
        ]

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
        params: Optional[dict] = None,
    ) -> list[dict]:
        """Run single-step prediction through the syntheseus wrapper and return
        the rich dict shape the app/templates expect.

        ``params`` carries model-specific knobs (already coerced/validated against
        the spec's ``ParamSpec``s); each is pushed onto the wrapper via its ``apply``.
        """
        from syntheseus.interface.molecule import Molecule

        spec = self._specs[model_id]
        model = self._instance(model_id)
        params = params or {}
        for p in spec.params:
            if p.name in params:
                p.apply(model, params[p.name])
        reactions = model([Molecule(smiles=product_smiles)], num_results=n_precursors)[0]
        return [_reaction_to_dict(r) for r in reactions]


def _reaction_to_dict(reaction) -> dict:
    """Normalize a syntheseus reaction into the app's per-prediction dict.

    DiffAlign's wrapper carries the rich payload (``precursors``/``score``/
    ``sample_data``/``atom_mapping``/``mapped_rxn``) on ``metadata``. Stock
    syntheseus wrappers (e.g. LocalRetro) instead expose reactants as a
    ``Bag[Molecule]`` and a ``probability`` in metadata — there's no sample
    data / atom-mapping to surface, so those fields are ``None``.
    """
    md = reaction.metadata
    if "precursors" in md:
        return {
            "precursors": md["precursors"],
            "score": md["score"],
            "sample_data": md.get("sample_data"),
            "atom_mapping": md.get("atom_mapping"),
            "mapped_rxn": md.get("mapped_rxn"),
        }
    precursors = ".".join(sorted(m.smiles for m in reaction.reactants))
    return {
        "precursors": precursors,
        "score": md.get("probability", 0.0),
        "sample_data": None,
        "atom_mapping": None,
        "mapped_rxn": None,
    }


registry = ModelRegistry(_SPECS)
