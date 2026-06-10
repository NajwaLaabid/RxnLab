"""RxnLab platform landing page."""
from flask import Blueprint, current_app, render_template

from app.registry import registry
from app.search import SEARCH_MODEL_IDS

bp = Blueprint('landing', __name__)


def _model_cards() -> list[dict]:
    """One card per registered model, derived live from the registry so adding
    a model auto-updates the landing page."""
    cards = []
    for s in registry.list_specs():
        meta = s.metadata or {}
        cards.append({
            'model_id': s.model_id,
            'display_name': s.display_name,
            'description': s.description,
            'arch': meta.get('arch', ''),
            'training': meta.get('training', ''),
            'paper': meta.get('paper', ''),
            'multistep': s.model_id in SEARCH_MODEL_IDS,
        })
    return cards


@bp.route('/')
def landing():
    return render_template(
        'landing.html',
        models=_model_cards(),
        ux_feedback_form_url=current_app.config.get('UX_FEEDBACK_FORM_URL', ''),
    )
