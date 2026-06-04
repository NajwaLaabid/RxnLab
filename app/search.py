"""Multi-step retrosynthesis search (Phase 3).

Wraps a registered single-step model in a syntheseus search algorithm (Retro*),
bottoming out at a buyable building-block inventory, and returns route trees on
the Phase-0 ``route-tree-v1`` schema.

Single-step (``registry.predict``) stays the core instrument; this is the
multi-step layer. LocalRetro is the only search-enabled model in Phase 3a
(template-based, fast per-expansion on CPU); DiffAlign search waits on a faster
backend.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from app.registry import registry

# Models allowed to drive multi-step search. DiffAlign is too slow per-expansion
# on CPU for a real search; gated until a faster backend (Modal / bigger box).
SEARCH_MODEL_IDS = ("localretro-uspto50k-v1",)

# Default search budget — tuned for the 2-vCPU / 4GB box. Overridable per request.
# limit_graph_nodes bounds peak memory (the search graph is the dominant cost);
# ~15k nodes kept peak RSS well under the box ceiling in testing.
DEFAULT_TIME_LIMIT_S = 45.0
DEFAULT_MAX_ROUTES = 5
DEFAULT_LIMIT_ITERATIONS = 100
DEFAULT_LIMIT_GRAPH_NODES = 10000
DEFAULT_MAX_EXPANSION_DEPTH = 6


# Buyable-molecule catalogs offered for multi-step search. Each loads from a file
# on the box's disk, pointed to by an env var. The data (eMolecules / Enamine) is
# licensed for use, not redistribution — never commit it or bake it into the
# public image. A catalog is offered only when its file is actually present, so a
# new one (e.g. Enamine) self-enables once its file is dropped on the box.
CATALOGS = [
    {
        "id": "emolecules-subset",
        "label": "eMolecules (subset)",
        "env": "RXNLAB_INVENTORY_PATH",
        "blurb": ("A ~1.7M-compound subset of the eMolecules aggregator "
                  "(~13M compounds total), the stock typically used in "
                  "retrosynthesis ML benchmarks (PaRoutes, syntheseus, "
                  "AiZynthFinder)."),
    },
    {
        "id": "enamine-global-stock",
        "label": "Enamine Global Stock",
        "env": "RXNLAB_ENAMINE_PATH",
        "blurb": ("About 1.1M in-stock, orderable building blocks from the vendor "
                  "Enamine, not a stock typically used in the ML literature."),
    },
    {
        "id": "test-fragments",
        "label": "Test fragments (small)",
        "env": "RXNLAB_TESTCATALOG_PATH",
        "blurb": ("A 50k subset of small eMolecules fragments, for testing the "
                  "catalog selector only, not a real vendor stock."),
    },
]
DEFAULT_CATALOG_ID = "emolecules-subset"


def _catalog_by_id(catalog_id: str) -> Optional[dict]:
    return next((c for c in CATALOGS if c["id"] == catalog_id), None)


def _catalog_path(catalog_id: str) -> Optional[Path]:
    cat = _catalog_by_id(catalog_id)
    if cat is None:
        return None
    raw = (os.environ.get(cat["env"]) or "").strip()
    if not raw:
        return None
    p = Path(raw)
    return p if p.exists() else None


def available_catalogs() -> list:
    """Catalogs whose backing file is present on disk, with UI metadata."""
    return [
        {"id": c["id"], "label": c["label"], "blurb": c["blurb"]}
        for c in CATALOGS if _catalog_path(c["id"]) is not None
    ]


def default_catalog_id() -> Optional[str]:
    ids = [c["id"] for c in available_catalogs()]
    if DEFAULT_CATALOG_ID in ids:
        return DEFAULT_CATALOG_ID
    return ids[0] if ids else None


def search_enabled(model_id: str, catalog_id: Optional[str] = None) -> bool:
    if model_id not in SEARCH_MODEL_IDS:
        return False
    if catalog_id is None:
        return bool(available_catalogs())
    return _catalog_path(catalog_id) is not None


def _iter_inventory_smiles(path: Path):
    """Yield SMILES from the inventory file.

    Supports a plain ``.txt`` (one SMILES per line) or the DESP ``{smiles: idx}``
    JSON dict (keys only). The .txt subset is what Phase 3a ships with.
    """
    if path.suffix == ".json":
        with open(path) as f:
            yield from json.load(f).keys()
    else:
        with open(path) as f:
            for line in f:
                s = line.strip()
                if s:
                    yield s


_INVENTORY_CACHE: dict = {}


def load_inventory(catalog_id: str):
    """Build a catalog's buyable-molecule inventory once, from disk. Cached per
    catalog for process life."""
    if catalog_id in _INVENTORY_CACHE:
        return _INVENTORY_CACHE[catalog_id]
    from syntheseus.search.mol_inventory import SmilesListInventory

    path = _catalog_path(catalog_id)
    if path is None:
        raise RuntimeError(f"Catalog {catalog_id!r} unavailable (env unset or file missing).")
    # canonicalize=False: the catalog files are already syntheseus-canonical (see
    # scripts/build_inventory_subset.py), so loading is just set construction — no
    # per-startup rdkit pass over millions of molecules.
    inv = SmilesListInventory(list(_iter_inventory_smiles(path)), canonicalize=False)
    _INVENTORY_CACHE[catalog_id] = inv
    return inv


def _build_algorithm(model_id: str, *, catalog_id: str, time_limit_s: float,
                     limit_iterations: int, limit_graph_nodes: int,
                     max_expansion_depth: int):
    from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch
    from syntheseus.search.node_evaluation.common import (
        ConstantNodeEvaluator,
        ReactionModelLogProbCost,
    )

    model = registry._instance(model_id)
    model.reset(use_cache=True)  # search re-queries the same molecules; cache within a run
    return RetroStarSearch(
        reaction_model=model,
        mol_inventory=load_inventory(catalog_id),
        and_node_cost_fn=ReactionModelLogProbCost(),  # -log model prob
        value_function=ConstantNodeEvaluator(0.0),     # uninformed heuristic (uniform-cost)
        time_limit_s=time_limit_s,
        limit_iterations=limit_iterations,
        limit_graph_nodes=limit_graph_nodes,
        max_expansion_depth=max_expansion_depth,
        stop_on_first_solution=False,
    )


def run_search(
    model_id: str,
    product_smiles: str,
    *,
    catalog_id: str = DEFAULT_CATALOG_ID,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    limit_iterations: int = DEFAULT_LIMIT_ITERATIONS,
    limit_graph_nodes: int = DEFAULT_LIMIT_GRAPH_NODES,
    max_routes: int = DEFAULT_MAX_ROUTES,
    max_expansion_depth: int = DEFAULT_MAX_EXPANSION_DEPTH,
) -> dict:
    """Run multi-step search and return a ``route-tree-v1`` result.

    Result shape (extends the Phase-0 schema; multi-step uses ``routes`` where
    single-step uses ``nodes``)::

        {schema, root, routes: [route_tree, ...], stats: {...}}
    """
    from syntheseus.interface.molecule import Molecule
    from syntheseus.search.analysis.route_extraction import iter_routes_cost_order

    algo = _build_algorithm(
        model_id,
        catalog_id=catalog_id,
        time_limit_s=time_limit_s,
        limit_iterations=limit_iterations,
        limit_graph_nodes=limit_graph_nodes,
        max_expansion_depth=max_expansion_depth,
    )

    t0 = time.monotonic()
    graph, _ = algo.run_from_mol(Molecule(smiles=product_smiles))
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    route_node_sets = list(iter_routes_cost_order(graph, max_routes=max_routes))
    routes = [_route_to_tree(graph, set(nodes)) for nodes in route_node_sets]

    # Describe every route before returning, so the UI renders them already
    # classified. Best-effort: a description failure must not sink the search.
    describe_ms = 0
    if routes:
        td = time.monotonic()
        try:
            for route, desc in zip(routes, describe_routes(routes)):
                route["description"] = desc
        except Exception:
            import traceback
            traceback.print_exc()
        describe_ms = int((time.monotonic() - td) * 1000)

    return {
        "schema": "route-tree-v1",
        "root": product_smiles,
        "routes": routes,
        "stats": {
            "model_id": model_id,
            "catalog_id": catalog_id,
            "n_routes": len(routes),
            "solved": bool(routes),
            "n_nodes": len(graph),
            "elapsed_ms": elapsed_ms,
            "describe_ms": describe_ms,
        },
    }


def _route_to_tree(graph, route_nodes: set) -> dict:
    """Convert one solution route (a set of graph nodes) into a nested tree,
    walking from the root molecule down through chosen reactions to buyable leaves.
    """
    from syntheseus.search.graph.and_or import AndNode

    def build_mol(or_node) -> dict:
        # The reaction chosen for this molecule in this route, if any.
        and_children = [
            c for c in graph.successors(or_node)
            if isinstance(c, AndNode) and c in route_nodes
        ]
        node = {
            "product": or_node.mol.smiles,
            "purchasable": not and_children,
            "reaction": None,
            "score": None,
            "children": [],
        }
        if and_children:
            and_node = and_children[0]
            rxn = and_node.reaction
            node["reaction"] = _reaction_smiles(rxn)
            node["score"] = rxn.metadata.get("probability")
            node["children"] = [
                build_mol(c) for c in graph.successors(and_node) if c in route_nodes
            ]
        return node

    return build_mol(graph.root_node)


def _reaction_smiles(rxn) -> str:
    reactants = ".".join(sorted(m.smiles for m in rxn.reactants))
    return f"{reactants}>>{rxn.product.smiles}"


def _is_unidentified(info: dict) -> bool:
    """RXN-Insight couldn't pin a reaction type (its catch-all is CLASS
    'Miscellaneous' / NAME 'Other')."""
    cls = (info.get("class") or "").lower()
    name = (info.get("name") or "").lower()
    return (not cls) or cls == "miscellaneous" or name == "other"


def _fg_transform(info: dict) -> str:
    """The functional-group change as a human-readable string, e.g.
    'Primary amine → Secondary amide'. Empty when RXN-Insight found none."""
    fgr = info.get("fg_reactants") or []
    fgp = info.get("fg_products") or []
    if not (fgr or fgp):
        return ""
    return f"{', '.join(fgr)} → {', '.join(fgp)}"


def _short_label(info: dict) -> str:
    """Concise label for the tree connector (the reaction class, or
    'unidentified reaction type')."""
    if _is_unidentified(info):
        return "unidentified reaction type"
    return info["class"]


def _reaction_label(info: dict) -> str:
    """Full human label: the class plus the specific named reaction in
    parentheses; for unidentified reactions, the functional-group change."""
    if _is_unidentified(info):
        fg = _fg_transform(info)
        return f"unidentified reaction type ({fg})" if fg else "unidentified reaction type"
    label = info["class"].lower()
    name = info.get("name")
    if name and name.lower() != info["class"].lower():
        label += f" ({name})"
    return label


def _article(word: str) -> str:
    return "an " if word[:1].lower() in "aeiou" else "a "


def _route_steps(route: dict) -> list:
    """Flatten a route tree into reaction steps in forward synthetic order
    (deepest first). Each step: ``{depth, product, precursors: [smiles]}``."""
    steps = []
    mols = set()

    def walk(node: dict, depth: int) -> None:
        mols.add(node["product"])
        kids = node.get("children") or []
        if kids and node.get("reaction"):
            steps.append({"depth": depth, "product": node["product"],
                          "precursors": [k["product"] for k in kids]})
        for k in kids:
            walk(k, depth + 1)

    walk(route, 0)
    steps.sort(key=lambda s: -s["depth"])
    return steps, mols


def _summarize(route: dict, steps: list, name_of) -> dict:
    target_name = name_of(route["product"])
    if not steps:
        summary = f"{target_name} is already a purchasable building block."
    elif len(steps) == 1:
        s = steps[0]
        precs = " and ".join(name_of(x) for x in s["precursors"])
        summary = f"Synthesize {target_name} by {_article(s['label'])}{s['label']} of {precs}."
    else:
        segs = []
        for s in steps:
            precs = " and ".join(name_of(x) for x in s["precursors"])
            prod = "the target" if s["depth"] == 0 else name_of(s["product"])
            segs.append(f"{_article(s['label'])}{s['label']} of {precs} to give {prod}")
        summary = f"Synthesize {target_name} via " + "; then ".join(segs) + "."

    out_steps = [{"class": s["info"].get("class"), "name": s["info"].get("name"),
                  "label": s["label"], "short_label": s["short_label"],
                  "fg_transform": _fg_transform(s["info"]),
                  "product": s["product"], "precursors": s["precursors"]} for s in steps]
    return {"summary": summary, "steps": out_steps}


def describe_routes(routes: list) -> list:
    """Classify and prose-summarize a batch of route trees.

    Fast path for the whole displayed set: every reaction and molecule is
    deduplicated across routes (routes share sub-trees), so each is classified /
    looked up exactly once. The network-bound PubChem batch runs in a background
    thread that overlaps the CPU-bound (and serialized — see ``_CLASSIFY_LOCK``)
    classification, so the two dominant costs run concurrently.

    Returns one ``{"summary", "steps"}`` dict per input route, in order.
    """
    import threading

    from app.rendering.classify import classify_reaction
    from app.rendering.pubchem import lookup_all_compounds

    per_route = [_route_steps(r) for r in routes]
    all_mols = set()
    rxn_info = {}  # (precursors_joined, product) -> classification dict
    for steps, mols in per_route:
        all_mols |= mols
        for s in steps:
            rxn_info[(".".join(s["precursors"]), s["product"])] = None

    profiles_box = {}

    def _fetch_pubchem():
        profiles_box["p"] = {
            p.get("smiles"): p for p in lookup_all_compounds(sorted(all_mols))
        }

    th = threading.Thread(target=_fetch_pubchem, daemon=True)
    th.start()

    for key in rxn_info:
        rxn_info[key] = classify_reaction(key[0], key[1])

    th.join()
    profiles = profiles_box.get("p", {})

    def name_of(smi: str) -> str:
        p = profiles.get(smi) or {}
        if p.get("found"):
            names = p.get("short_names") or []
            return names[0] if names else (p.get("iupac") or smi)
        return smi

    out = []
    for route, (steps, _mols) in zip(routes, per_route):
        for s in steps:
            info = rxn_info[(".".join(s["precursors"]), s["product"])]
            s["info"] = info
            s["label"] = _reaction_label(info)
            s["short_label"] = _short_label(info)
        out.append(_summarize(route, steps, name_of))
    return out


def describe_route(route: dict) -> dict:
    """Single-route description (the on-demand ``/api/search/describe`` path)."""
    return describe_routes([route])[0]


def warm_classifier() -> None:
    """Pre-load rxn_insight + RXNMapper (a ~7s one-time import) so the first
    search / description doesn't pay it. Best-effort; safe to fail."""
    try:
        from app.rendering.classify import classify_reaction
        classify_reaction("CC(=O)O.NC", "CC(=O)NC")
    except Exception:
        import traceback
        traceback.print_exc()
