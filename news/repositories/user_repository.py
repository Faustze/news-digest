from sqlalchemy import text


class UserRepository:
    """Все операции с таблицей users. SQL живёт только здесь."""

    def __init__(self, session):
        self.session = session

    def create(self, telegram_id: int) -> int:
        result = self.session.execute(
            text("INSERT INTO users (telegram_id) VALUES (:telegram_id) RETURNING id"),
            {"telegram_id": telegram_id},
        )
        return result.scalar_one()

    def get_by_telegram_id(self, telegram_id: int):
        row = self.session.execute(
            text("SELECT * FROM users WHERE telegram_id = :telegram_id"),
            {"telegram_id": telegram_id},
        ).one_or_none()
        if row is None:
            return self.create(telegram_id)
        return row.id
