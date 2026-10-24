import sqlite3
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool


class DBManager:
    def __init__(self, db_path: str = "sqlite:///data/context.sqlite3"):
        url = make_url(db_path)
        if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:":
            raise ValueError("Provide a SQLite URL pointing to an existing context database")
        path = Path(url.database).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Context database not found: {path}")
        uri = path.as_uri() + "?mode=ro"
        self.engine = create_engine(
            "sqlite://", poolclass=NullPool,
            creator=lambda: sqlite3.connect(uri, uri=True, check_same_thread=False),
        )
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        return self.Session()
