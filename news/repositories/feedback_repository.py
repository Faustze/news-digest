from sqlalchemy import text


class FeedbackRepository:
    """Все операции с таблицей feedback. SQL живёт только здесь."""

    def __init__(self, session):
        self.session = session

    def upsert_feedback(
        self, user_id: int, article_id: int, rating: int, comment: str | None = None
    ) -> None:
        """Вставляет оценку или обновляет существующую для пары (user_id, article_id)."""
        self.session.execute(
            text(
                "INSERT INTO feedback (user_id, article_id, rating, comment) "
                "VALUES (:user_id, :article_id, :rating, :comment) "
                "ON CONFLICT (user_id, article_id) "
                "DO UPDATE SET rating = :rating, comment = :comment"
            ),
            {
                "user_id": user_id,
                "article_id": article_id,
                "rating": rating,
                "comment": comment,
            },
        )

    def get_feedback(self, user_id: int, article_id: int):
        """Возвращает строку фидбека для пары (user_id, article_id) или None."""
        return self.session.execute(
            text(
                "SELECT * FROM feedback "
                "WHERE user_id = :user_id AND article_id = :article_id"
            ),
            {"user_id": user_id, "article_id": article_id},
        ).one_or_none()

    def get_user_feedback(self, user_id: int):
        """Возвращает все оценки пользователя."""
        return self.session.execute(
            text("SELECT * FROM feedback WHERE user_id = :user_id"),
            {"user_id": user_id},
        ).fetchall()

    def get_feedback_with_articles(self):
        return self.session.execute(
            text(
                "SELECT f.id, u.telegram_id, a.title, f.rating FROM feedback f "
                "JOIN articles a ON a.id = f.article_id "
                "JOIN users u ON u.id = f.user_id"
            )
        ).fetchall()
