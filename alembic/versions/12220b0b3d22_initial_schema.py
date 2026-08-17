"""initial_schema

Revision ID: 12220b0b3d22
Revises:
Create Date: 2026-08-17 15:39:16.970694

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "12220b0b3d22"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # родители создаются первыми (на них ссылаются дети)
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("telegram_id", sa.BigInteger(), nullable=False),
    )
    op.create_table(
        "articles",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("url", sa.Text(), nullable=False, unique=True),
        sa.Column("title", sa.VARCHAR(255), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.VARCHAR(255)),
    )
    op.create_table(
        "digests",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("title", sa.VARCHAR(255), nullable=False),
    )
    # дети создаются после родителей
    op.create_table(
        "user_profile",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("preferences", sa.VARCHAR(255), nullable=False),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id"),
            nullable=False,
            unique=True,
        ),
    )
    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column(
            "article_id",
            sa.Integer(),
            sa.ForeignKey("articles.id"),
            nullable=False,
        ),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("comment", sa.VARCHAR(255)),
    )
    op.create_check_constraint(
        "ck_feedback_rating_range",
        "feedback",
        "rating BETWEEN 1 AND 5",
    )
    op.create_table(
        "articles_digests",
        sa.Column("article_id", sa.Integer(), sa.ForeignKey("articles.id"), nullable=False),
        sa.Column("digest_id", sa.Integer(), sa.ForeignKey("digests.id"), nullable=False),
        sa.PrimaryKeyConstraint("article_id", "digest_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    # downgrade = upgrade наоборот: сначала дети, потом родители
    op.drop_table("articles_digests")
    op.drop_table("feedback")
    op.drop_table("user_profile")
    op.drop_table("digests")
    op.drop_table("articles")
    op.drop_table("users")
