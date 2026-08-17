from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql+psycopg://newsdigest:qwerty123@localhost:5434/newsdigest"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
