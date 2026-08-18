from datetime import datetime, timezone

from news.db import SessionLocal
from news.repositories import ArticleRepository, FeedbackRepository, UserRepository


def main():
    with SessionLocal() as session:
        user_repo = UserRepository(session)
        feedback_repo = FeedbackRepository(session)
        article_repo = ArticleRepository(session)

        user_id = user_repo.get_by_telegram_id(987654321)
        article_id = article_repo.get_or_create(
            url="https://example.com",
            title="title",
            published_at=datetime.now(timezone.utc),
            source="example.com",
        )
        feedback_repo.upsert_feedback(
            user_id=user_id, article_id=article_id, rating=5, comment="отлично"
        )
        result = feedback_repo.get_feedback_with_articles()
        for row in result:
            print(row)
        session.commit()


if __name__ == "__main__":
    main()
