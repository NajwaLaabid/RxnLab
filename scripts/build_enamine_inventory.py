"""Convert an Enamine building-block catalog (SDF or SMILES) into the canonical
SMILES list the multi-step search loads as its buyable inventory.

Enamine ships building-block catalogs as SD files (often gzipped) from
enamine.net/building-blocks/building-blocks-catalog. Download the tranche you
want (e.g. Global Stock, ~1.1M) to the box's private disk, then run this to emit
one canonical SMILES per line. Output uses the exact ``MolFromSmiles`` →
``MolToSmiles`` round-trip syntheseus' ``Molecule`` uses, so the app loads it
with ``SmilesListInventory(..., canonicalize=False)`` — a plain set, no per-start
rdkit pass. Unparseable / duplicate entries are dropped.

License: Enamine catalog data is licensed for use, not redistribution — keep the
output on the box's private path (point ``RXNLAB_ENAMINE_PATH`` at it); never
commit it or bake it into the public image. Same rule as the eMolecules subset.

Usage:
    python scripts/build_enamine_inventory.py \
        --source ~/data/enamine_global_stock.sdf.gz \
        --out ~/data/rxnlab_enamine.txt
"""
import argparse
import gzip
from pathlib import Path


def _open_maybe_gz(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode)
    return open(path, mode)


def _iter_smiles(path: Path):
    """Yield raw SMILES strings from an SDF or a SMILES/.smi/.txt file."""
    from rdkit import Chem

    suffixes = {s.lower() for s in path.suffixes}
    if ".sdf" in suffixes:
        with _open_maybe_gz(path, "rb") as f:
            supplier = Chem.ForwardSDMolSupplier(f, sanitize=True)
            for mol in supplier:
                if mol is not None:
                    yield Chem.MolToSmiles(mol)
    else:  # .smi / .smiles / .txt — first whitespace token per line is the SMILES
        with _open_maybe_gz(path, "rt") as f:
            for line in f:
                tok = line.split()
                if tok:
                    yield tok[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    seen = set()
    n_scanned = n_bad = 0
    with open(args.out, "w") as out:
        for smi in _iter_smiles(args.source):
            n_scanned += 1
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                n_bad += 1
                continue
            canon = Chem.MolToSmiles(mol)
            if canon not in seen:
                seen.add(canon)
                out.write(canon + "\n")
            if n_scanned % 200_000 == 0:
                print(f"scanned {n_scanned:,}  kept {len(seen):,}  bad {n_bad:,}", flush=True)

    print(f"done: scanned {n_scanned:,}  kept {len(seen):,}  dropped {n_bad:,} -> {args.out}")


if __name__ == "__main__":
    main()
