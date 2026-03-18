"""Reaction classification using rxn_insight."""


def classify_reaction(precursors_smi: str, product_smi: str) -> dict:
    """Classify a single retrosynthetic step using rxn_insight.

    Args:
        precursors_smi: dot-separated precursor SMILES (e.g. "A.B")
        product_smi: product SMILES

    Returns:
        dict with keys: name, class, success, error
    """
    try:
        from rxn_insight.reaction import Reaction
    except Exception:
        return {
            'name': None,
            'class': None,
            'success': False,
            'error': 'rxn_insight not available',
        }

    rxn_smi = f"{precursors_smi}>>{product_smi}"
    try:
        rxn = Reaction(rxn_smi)
        ri = rxn.get_reaction_info()
        return {
            'name': ri.get('NAME'),
            'class': ri.get('CLASS'),
            'success': True,
            'error': None,
        }
    except Exception as e:
        return {
            'name': None,
            'class': None,
            'success': False,
            'error': str(e),
        }


def classify_reactions(predictions: list, product_smi: str) -> list:
    """Annotate a list of prediction dicts with reaction_info in-place.

    Each prediction dict should have a 'precursors' key with dot-separated SMILES.
    A 'reaction_info' key is added to each dict.

    Returns the same list for convenience.
    """
    for pred in predictions:
        pred['reaction_info'] = classify_reaction(pred['precursors'], product_smi)
    return predictions
