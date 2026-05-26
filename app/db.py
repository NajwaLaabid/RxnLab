"""SQLAlchemy engine, session helpers, and on-startup table creation."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

_engine = None
_SessionLocal: Optional[sessionmaker] = None


def init_db(database_url: str) -> None:
    global _engine, _SessionLocal
    _engine = create_engine(database_url, future=True, pool_pre_ping=True)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)

    from app.models_db import Base
    Base.metadata.create_all(_engine)
    _seed_models()


def _seed_models() -> None:
    from app.models_db import Model
    from app.registry import registry

    assert _SessionLocal is not None
    with _SessionLocal() as session:
        changed = False
        for spec in registry.list_specs():
            if session.get(Model, spec.model_id) is None:
                session.add(Model(
                    model_id=spec.model_id,
                    display_name=spec.display_name,
                    version=spec.version,
                    description=spec.description,
                    supports_inpainting=spec.supports_inpainting,
                    metadata_={**spec.metadata, 'backend': spec.backend},
                    added_at=datetime.now(timezone.utc),
                ))
                changed = True
        if changed:
            session.commit()


def get_session() -> Optional[Session]:
    if _SessionLocal is None:
        return None
    return _SessionLocal()


def db_enabled() -> bool:
    return _SessionLocal is not None
