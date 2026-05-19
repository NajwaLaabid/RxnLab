"""RxnLab platform landing page."""
from flask import Blueprint, current_app, render_template

bp = Blueprint('landing', __name__)


@bp.route('/')
def landing():
    return render_template(
        'landing.html',
        ux_feedback_form_url=current_app.config.get('UX_FEEDBACK_FORM_URL', ''),
    )
