from datetime import datetime, timezone

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from alembic import command
from news.repositories.article_repository import ArticleRepository
from news.repositories.feedback_repository import FeedbackRepository
from news.repositories.user_repository import UserRepository

TEST_DATABASE_URL = (
    "postgresql+psycopg://newsdigest:qwerty123@localhost:5434/newsdigest_test"
)
ADMIN_DATABASE_URL = "postgresql+psycopg://newsdigest:qwerty123@localhost:5434/postgres"


@pytest.fixture(scope="session")
def db():
    # ── ФАЗА SETUP (выполняется один раз за весь прогон) ──
    # try/except защищает ТОЛЬКО setup: если Postgres недоступен —
    # весь набор тестов уйдёт в skip, а не в fail.
    try:
        # admin_engine — подключение к служебной базе "postgres" (она есть всегда).
        # AUTOCOMMIT обязателен: CREATE DATABASE нельзя выполнить внутри транзакции,
        # а psycopg открывает транзакцию неявно.
        admin_engine = create_engine(ADMIN_DATABASE_URL, isolation_level="AUTOCOMMIT")
        with admin_engine.connect() as conn:  # with сам закроет соединение
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = 'newsdigest_test'")
            ).scalar()
            if not exists:  # у CREATE DATABASE нет IF NOT EXISTS — проверяем вручную
                conn.execute(text("CREATE DATABASE newsdigest_test"))

        # alembic сам читает URL из alembic.ini, но мы подменяем его на
        # тестовую базу — чтобы миграции не тронули рабочую newsdigest.
        cfg = Config("alembic.ini")
        cfg.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)

        # Накатываем схему из репо на пустую тестовую базу (то же, что alembic upgrade head)
        command.upgrade(cfg, "head")

        # Движок для самих тестов — уже на готовой тестовой базе
        engine = create_engine(TEST_DATABASE_URL)
        with engine.connect():  # короткая проверка живости — и сразу закрывается
            pass
    except OperationalError:
        pytest.skip("Postgres not available")

    # ── ФАЗА ОТДАЧИ ──
    # yield превращает фикстуру в генератор: управление отдаётся тестам,
    # всё, что после yield, выполнится ПОСЛЕ их завершения.
    SessionLocal = sessionmaker(bind=engine)
    yield SessionLocal

    # ── ФАЗА TEARDOWN (после последнего теста) ──
    engine.dispose()  # освобождаем пул соединений


@pytest.fixture
def session(db):
    s = db()
    s.execute(
        text(
            "TRUNCATE users, articles, digests, feedback, user_profile, articles_digests RESTART IDENTITY CASCADE"
        )
    )
    s.commit()
    yield s
    s.close()


def test_insert_user(session):
    inserted_id = UserRepository(session).get_by_telegram_id(987654321)

    row = session.execute(
        text("SELECT id, telegram_id FROM users WHERE id = :id"),
        {"id": inserted_id},
    ).one()

    assert row.id == inserted_id
    assert row.telegram_id == 987654321


def test_upsert_feedback(session):
    user_id = UserRepository(session).get_by_telegram_id(987654321)
    article_id = ArticleRepository(session).get_or_create(
        url="https://example.com",
        title="title",
        published_at=datetime.now(timezone.utc),
        source="example.com",
    )

    params = {
        "user_id": user_id,
        "article_id": article_id,
        "rating": 5,
        "comment": "отлично",
    }
    feedback_repo = FeedbackRepository(session)
    feedback_repo.upsert_feedback(**params)

    params["rating"] = 3
    params["comment"] = "обновлённая оценка"
    feedback_repo.upsert_feedback(**params)
    session.commit()

    rows = session.execute(text("SELECT * FROM feedback")).fetchall()
    assert len(rows) == 1
    assert rows[0].rating == 3


def test_join_returns_user_and_article(session):
    user_id = UserRepository(session).get_by_telegram_id(987654321)
    article_id = ArticleRepository(session).get_or_create(
        url="https://example.com",
        title="title",
        published_at=datetime.now(timezone.utc),
        source="example.com",
    )

    params = {
        "user_id": user_id,
        "article_id": article_id,
        "rating": 4,
        "comment": "норм",
    }
    feedback_repo = FeedbackRepository(session)
    feedback_repo.upsert_feedback(**params)

    session.commit()

    rows = feedback_repo.get_feedback_with_articles()
    assert len(rows) == 1
    row = rows[0]

    assert row.telegram_id == 987654321
    assert row.title == "title"
    assert row.rating == 4


def test_delete_article(session):
    repo = ArticleRepository(session)
    article_id = repo.get_or_create(
        url="https://example.com",
        title="title",
        published_at=datetime.now(timezone.utc),
        source="example.com",
    )
    session.commit()

    assert repo.delete_by_id(article_id) is True
    session.commit()

    rows = session.execute(text("SELECT * FROM articles")).fetchall()
    assert len(rows) == 0

    assert repo.delete_by_id(article_id) is False


def test_delete_article_by_url(session):
    repo = ArticleRepository(session)
    repo.get_or_create(
        url="https://example.com",
        title="title",
        published_at=datetime.now(timezone.utc),
        source="example.com",
    )
    session.commit()

    assert repo.delete_by_url("https://example.com") is True
    session.commit()

    rows = session.execute(text("SELECT * FROM articles")).fetchall()
    assert len(rows) == 0

    assert repo.delete_by_url("https://example.com") is False


def test_get_or_create_dedupes_normalized_url(session):
    repo = ArticleRepository(session)
    first_id = repo.get_or_create(
        url="https://example.com/article?utm_source=rss",
        title="Original title",
        published_at=datetime.now(timezone.utc),
        source="feed-a.com",
    )
    session.commit()

    second_id = repo.get_or_create(
        url="https://EXAMPLE.com/article/",
        title="Same article from another feed",
        published_at=datetime.now(timezone.utc),
        source="feed-b.com",
    )
    session.commit()

    assert second_id == first_id

    rows = session.execute(text("SELECT url, title, source FROM articles")).fetchall()
    assert len(rows) == 1
    assert rows[0].url == "https://example.com/article"


def test_delete_user(session):
    repo = UserRepository(session)
    repo.get_by_telegram_id(987654321)
    session.commit()

    assert repo.delete_by_telegram_id(987654321) is True
    session.commit()

    rows = session.execute(text("SELECT * FROM users")).fetchall()
    assert len(rows) == 0

    assert repo.delete_by_telegram_id(987654321) is False


def test_delete_feedback(session):
    user_id = UserRepository(session).get_by_telegram_id(987654321)
    article_id = ArticleRepository(session).get_or_create(
        url="https://example.com",
        title="title",
        published_at=datetime.now(timezone.utc),
        source="example.com",
    )

    feedback_repo = FeedbackRepository(session)
    feedback_repo.upsert_feedback(
        user_id=user_id, article_id=article_id, rating=4, comment="норм"
    )
    session.commit()

    assert feedback_repo.delete_feedback(user_id, article_id) is True
    session.commit()

    rows = session.execute(text("SELECT * FROM feedback")).fetchall()
    assert len(rows) == 0

    assert feedback_repo.delete_feedback(user_id, article_id) is False
