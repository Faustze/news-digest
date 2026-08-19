from sqlalchemy import text


class ArticleRepository:
    """Все операции с таблицей articles. SQL живёт только здесь."""

    def __init__(self, session):
        self.session = session

    def get_or_create(self, url, title, published_at, source):
        result = self.session.execute(
            text(
                "INSERT INTO articles (url, title, published_at, source) "
                "VALUES (:url, :title, :published_at, :source) "
                "ON CONFLICT (url) "
                "DO NOTHING RETURNING id"
            ),
            {
                "url": url,
                "title": title,
                "published_at": published_at,
                "source": source,
            },
        ).scalar_one_or_none()
        if result is None:
            result = self.session.execute(
                text("SELECT id FROM articles WHERE url = :url"), {"url": url}
            ).scalar_one()
        return result

    def delete_by_url(self, url: str) -> bool:
        """Удаляет статью по URL. Возвращает True, если строка была удалена."""
        result = self.session.execute(
            text("DELETE FROM articles WHERE url = :url RETURNING id"),
            {"url": url},
        ).scalar_one_or_none()
        return result is not None

    def delete_by_id(self, article_id: int) -> bool:
        """Удаляет статью по id. Возвращает True, если строка была удалена."""
        result = self.session.execute(
            text("DELETE FROM articles WHERE id = :id RETURNING id"),
            {"id": article_id},
        ).scalar_one_or_none()
        return result is not None
