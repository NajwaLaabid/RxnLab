"""Subset the DESP/eMolecules building-block catalog into a small SMILES list
for the Phase-3a multi-step search inventory.

The source is the 1.2GB ``{canonical_smiles: idx}`` JSON from multiguide
(~13M compounds). The full set OOMs a 4GB box as an in-memory ``SmilesListInventory``,
so we keep only small molecules (a SMILES-length proxy for low heavy-atom count —
the common building blocks that bottom out short routes) and cap the count.

Output is rdkit-canonicalized with the exact ``MolFromSmiles``→``MolToSmiles``
sequence syntheseus' ``Molecule`` uses, so ``SmilesListInventory(..., canonicalize=False)``
can load it as a plain set (no per-startup rdkit pass). Unparseable entries are dropped.

License note: this catalog is eMolecules-derived. The OUTPUT subset is for runtime
use on our own box only — do NOT commit it to the public repo or bake it into the
public image. See notes/syntheseus-upgrade.md Phase 3.

Usage:
    python scripts/build_inventory_subset.py \
        --source ~/laabidn1/multiguide/data/desp_data/canon_building_block_mol2idx_no_isotope.json \
        --out /tmp/rxnlab_inventory.txt --max-len 38 --cap 400000
"""
import argparse
import re
from pathlib import Path

# Matches a JSON string key followed by `: <int>` — i.e. one `"smiles": idx` entry.
# The string body allows escaped chars so SMILES with `\` (cis/trans) survive.
_ENTRY = re.compile(r'"((?:[^"\\]|\\.)*)"\s*:\s*\d+')
_CHUNK = 1 << 24  # 16 MB


def _iter_keys(path: Path):
    """Stream JSON dict keys without loading the whole file."""
    import json

    tail = ""
    with open(path, "r") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            buf = tail + chunk
            last = 0
            for m in _ENTRY.finditer(buf):
                yield json.loads(f'"{m.group(1)}"')  # unescape
                last = m.end()
            tail = buf[last:]  # carry the partial entry across the boundary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-len", type=int, default=38,
                    help="keep SMILES no longer than this many chars (size proxy)")
    ap.add_argument("--cap", type=int, default=400_000, help="max compounds to keep")
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    seen = set()
    n_scanned = n_bad = 0
    with open(args.out, "w") as out:
        for smi in _iter_keys(args.source):
            n_scanned += 1
            if len(smi) <= args.max_len:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    n_bad += 1
                else:
                    canon = Chem.MolToSmiles(mol)
                    if canon not in seen:
                        seen.add(canon)
                        out.write(canon + "\n")
                        if len(seen) >= args.cap:
                            break
            if n_scanned % 1_000_000 == 0:
                print(f"scanned {n_scanned:,}  kept {len(seen):,}  bad {n_bad:,}", flush=True)

    print(f"done: scanned {n_scanned:,}  kept {len(seen):,}  dropped {n_bad:,} -> {args.out}")


if __name__ == "__main__":
    main()
