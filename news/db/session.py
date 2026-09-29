import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Put it in .env, e.g. "
        "postgresql+psycopg://newsdigest:<password>@localhost:5434/newsdigest"
    )

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
