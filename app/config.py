"""Application configuration, read from environment."""
import os


class Config:
    DATABASE_URL = (os.environ.get('DATABASE_URL') or '').strip() or None
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-insecure-change-in-production'
    SESSION_COOKIE_NAME = os.environ.get('RXNLAB_SESSION_COOKIE', 'rxnlab_session')
    SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 365  # 1 year
    STATS_TOKEN = (os.environ.get('RXNLAB_STATS_TOKEN') or '').strip() or None
    UX_FEEDBACK_FORM_URL = (
        os.environ.get('UX_FEEDBACK_FORM_URL')
        or 'https://docs.google.com/forms/d/e/1FAIpQLSeudYjnKoWLspvSxd9tFjlrfOASsHLH4pBnEgGPeUO0rD2AhA/viewform'
    ).strip()
    # Reject oversized request bodies before they're read into memory. Flask honors
    # MAX_CONTENT_LENGTH natively (→ 413). Generous enough for inpaint's
    # previous_sample_data blob; a guard against multi-hundred-MB abuse.
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    # Longest accepted target identifier (SMILES/name/InChI/CAS). Rejected before
    # any RDKit parse or PubChem lookup, so a giant string can't burn CPU/network.
    MAX_SMILES_LEN = 2000
