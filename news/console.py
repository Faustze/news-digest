from db import SessionLocal
from sqlalchemy import text


with SessionLocal() as session:
    session.execute(
        text("INSERT INTO users (telegram_id) VALUES (:telegram_id)"),
        {"telegram_id": 987654321},
    )
    # session.execute(
    #    text(
    #        "INSERT INTO feedback (user_id, article_id, rating, comment) "
    #        "VALUES (:user_id, :article_id, :rating, :comment)"
    #    ),
    #    {"user_id": 1, "article_id": 1, "rating": 5, "comment": "отлично"},
    # )
    session.execute(
        text("UPDATE feedback SET rating = 3 WHERE user_id = 1 AND article_id = 1")
    )
    session.commit()

with SessionLocal() as session:
    session.execute(
        text(
            "INSERT INTO articles (url, title, published_at, source) "
            "VALUES (:url , :title, NOW(), :source) "
            "ON CONFLICT (url) DO NOTHING"
        ),
        {
            "url": "https://example.com/news/2",
            "title": "Вторая новость",
            "source": "example.com",
        },
    )
    session.execute(
        text(
            "INSERT INTO feedback (user_id, article_id, rating, comment) "
            "VALUES (:user_id, :article_id, :rating, :comment) "
            "ON CONFLICT (user_id, article_id) "
            "DO UPDATE SET rating = :rating, comment = :comment"
        ),
        {"user_id": 1, "article_id": 1, "rating": 5, "comment": "отлично"},
    )
    session.commit()

with SessionLocal() as session:
    result = session.execute(
        text(
            "SELECT f.id, u.telegram_id, a.title, f.rating "
            "FROM feedback f "
            "JOIN users u ON u.id = f.user_id "
            "JOIN articles a ON a.id = f.article_id"
        )
    )
    for row in result:
        print(row)
