import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from sqlalchemy import DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent.parent / "actuarial_risk_model.db"
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

# Render (and most managed Postgres providers) hand out "postgres://", but
# SQLAlchemy 2.x only accepts the "postgresql://" scheme.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
elif DATABASE_URL.startswith("postgresql") and "sslmode" not in DATABASE_URL:
    # Managed Postgres providers (Supabase included) require SSL.
    connect_args = {"sslmode": "require"}
else:
    connect_args = {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    input_json: Mapped[str] = mapped_column(Text)
    result_json: Mapped[str] = mapped_column(Text)

    def to_detail_dict(self) -> dict:
        return {
            'id': self.id,
            'kind': self.kind,
            'name': self.name,
            'created_at': self.created_at,
            'input': json.loads(self.input_json),
            'result': json.loads(self.result_json),
        }


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
