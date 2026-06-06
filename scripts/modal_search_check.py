"""Step-4 check: DiffAlign multi-step search over Modal (lean /get-reactions path).

Builds Retro* with the lean DiffAlign proxy + a tiny inline inventory and searches a
simple target, so it doesn't need the box's on-disk catalog. Reports routes found and
per-expansion latency (the metric that decides whether per-RPC search is viable).

    export RXNLAB_MODAL_DIFFALIGN_URL='https://najwalaabid--rxnlab-diffalign'
    export RXNLAB_PROXY_TOKEN="$(cat ~/.config/rxnlab/coolify-token >/dev/null; cat /tmp/rxnlab_proxy_token)"
    python scripts/modal_search_check.py
"""
import os
import time

from app.registry import DEFAULT_MODEL_ID, registry


def main() -> None:
    assert os.environ.get("RXNLAB_MODAL_DIFFALIGN_URL"), "set RXNLAB_MODAL_DIFFALIGN_URL"

    from syntheseus.interface.molecule import Molecule
    from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch
    from syntheseus.search.analysis.route_extraction import iter_routes_cost_order
    from syntheseus.search.mol_inventory import SmilesListInventory
    from syntheseus.search.node_evaluation.common import (
        ConstantNodeEvaluator,
        ReactionModelLogProbCost,
    )

    target = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin — 1-step from salicylic acid + an acetyl source
    buyables = ["O=C(O)c1ccccc1O", "CC=O", "CC(=O)Cl", "CC(=O)O", "CC(=O)OC(C)=O", "Cl"]

    model = registry.search_instance(DEFAULT_MODEL_ID)
    print(f"search instance: {type(model).__name__} rich={getattr(model, 'rich', 'n/a')}")
    model.reset(use_cache=True)

    algo = RetroStarSearch(
        reaction_model=model,
        mol_inventory=SmilesListInventory(buyables, canonicalize=True),
        and_node_cost_fn=ReactionModelLogProbCost(),
        value_function=ConstantNodeEvaluator(0.0),
        time_limit_s=90,
        limit_iterations=15,
        limit_graph_nodes=3000,
        max_expansion_depth=4,
        stop_on_first_solution=False,
    )

    t0 = time.monotonic()
    graph, _ = algo.run_from_mol(Molecule(smiles=target))
    dt = time.monotonic() - t0

    routes = list(iter_routes_cost_order(graph, max_routes=3))
    calls = model.num_calls()
    print(f"\nelapsed={dt:.1f}s  model_calls(expansions)={calls}  graph_nodes={len(graph)}")
    if calls:
        print(f"per-expansion ≈ {dt / calls:.1f}s")
    print(f"routes found: {len(routes)}")

    if routes:
        from app.search import _route_to_tree

        tree = _route_to_tree(graph, set(routes[0]))

        def show(node, d=0):
            tag = "[buyable]" if node["purchasable"] else ""
            print("   " * d + f"- {node['product']} {tag}")
            for c in node["children"]:
                show(c, d + 1)

        print("top route:")
        show(tree)
    print("\nStep-4 check complete.")


if __name__ == "__main__":
    main()
