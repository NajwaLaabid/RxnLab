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
from functools import lru_cache
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


def search_enabled(model_id: str) -> bool:
    return model_id in SEARCH_MODEL_IDS and _inventory_path() is not None


def _inventory_path() -> Optional[Path]:
    raw = (os.environ.get("RXNLAB_INVENTORY_PATH") or "").strip()
    if not raw:
        return None
    p = Path(raw)
    return p if p.exists() else None


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


@lru_cache(maxsize=1)
def load_inventory():
    """Build the buyable-molecule inventory once, from disk. Cached for process life."""
    from syntheseus.search.mol_inventory import SmilesListInventory

    path = _inventory_path()
    if path is None:
        raise RuntimeError("RXNLAB_INVENTORY_PATH unset or missing; cannot run search.")
    # canonicalize=False: the subset file is already syntheseus-canonical (see
    # scripts/build_inventory_subset.py), so loading is just set construction — no
    # per-startup rdkit pass over ~1.7M molecules.
    return SmilesListInventory(list(_iter_inventory_smiles(path)), canonicalize=False)


def _build_algorithm(model_id: str, *, time_limit_s: float, limit_iterations: int,
                     limit_graph_nodes: int, max_expansion_depth: int):
    from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch
    from syntheseus.search.node_evaluation.common import (
        ConstantNodeEvaluator,
        ReactionModelLogProbCost,
    )

    model = registry._instance(model_id)
    model.reset(use_cache=True)  # search re-queries the same molecules; cache within a run
    return RetroStarSearch(
        reaction_model=model,
        mol_inventory=load_inventory(),
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

    return {
        "schema": "route-tree-v1",
        "root": product_smiles,
        "routes": routes,
        "stats": {
            "model_id": model_id,
            "n_routes": len(routes),
            "solved": bool(routes),
            "n_nodes": len(graph),
            "elapsed_ms": elapsed_ms,
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


def describe_route(route: dict) -> dict:
    """Classify each reaction in a route tree (RXN-Insight) and build a prose
    summary naming the target and the reaction types, in forward synthetic order.

    Returns ``{"summary": str, "steps": [{class, name, product, precursors}, ...]}``.
    On-demand only (atom-mapping + classification is seconds per reaction).
    """
    from app.rendering.classify import classify_reaction
    from app.rendering.pubchem import lookup_all_compounds

    steps = []   # each: {depth, product, precursors: [smiles]}
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
    steps.sort(key=lambda s: -s["depth"])  # forward order: deepest step first

    profiles = {p.get("smiles"): p for p in lookup_all_compounds(sorted(mols))}

    def name_of(smi: str) -> str:
        p = profiles.get(smi) or {}
        if p.get("found"):
            names = p.get("short_names") or []
            return names[0] if names else (p.get("iupac") or smi)
        return smi

    for s in steps:
        info = classify_reaction(".".join(s["precursors"]), s["product"])
        s["info"] = info
        s["label"] = _reaction_label(info)
        s["short_label"] = _short_label(info)

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
