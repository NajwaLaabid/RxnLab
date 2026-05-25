"""SVG rendering for RDKit molecules — no X11 dependency."""
import json

from rdkit.Chem.Draw import rdMolDraw2D


def mol_to_svg(mol, width=250, height=200):
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def mols_to_svg(mols, width=300, height=150):
    n_mols = len(mols)
    mols_per_row = min(n_mols, 3)
    n_rows = (n_mols + mols_per_row - 1) // mols_per_row

    cell_width = width // mols_per_row
    cell_height = 120

    drawer = rdMolDraw2D.MolDraw2DSVG(width, cell_height * n_rows, cell_width, cell_height)
    drawer.DrawMolecules(mols)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def serialize_results_json(results):
    safe = []
    for r in results:
        safe.append({
            'precursors': r['precursors'],
            'score': r['score'],
            'sample_data': r.get('sample_data'),
            'atom_mapping': r.get('atom_mapping'),
            'mapped_rxn': r.get('mapped_rxn'),
            'formula': r.get('formula'),
            'reaction_info': r.get('reaction_info'),
        })
    return json.dumps(safe).replace('</', '<\\/')
