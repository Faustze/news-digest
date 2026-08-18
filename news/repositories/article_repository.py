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
