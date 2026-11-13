"""Phase 4A: add users, refresh_tokens, processing_jobs, outbox_events, and ownership/job links.

Revision ID: 004_phase4a_auth_jobs_outbox
Revises: 003_add_document_review_tables
Create Date: 2026-11-13
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "004_phase4a_auth_jobs_outbox"
down_revision: Union[str, None] = "003_add_document_review_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=120), nullable=True),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False, server_default="REVIEWER"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_id", "users", ["id"], unique=False)
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_role", "users", ["role"], unique=False)
    op.create_index("ix_users_is_active", "users", ["is_active"], unique=False)

    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("jti", sa.String(length=64), nullable=False),
        sa.Column("token_hash", sa.String(length=128), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("replaced_by_jti", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_refresh_tokens_id", "refresh_tokens", ["id"], unique=False)
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"], unique=False)
    op.create_index("ix_refresh_tokens_jti", "refresh_tokens", ["jti"], unique=True)
    op.create_index("ix_refresh_tokens_token_hash", "refresh_tokens", ["token_hash"], unique=True)
    op.create_index("ix_refresh_tokens_expires_at", "refresh_tokens", ["expires_at"], unique=False)
    op.create_index("ix_refresh_tokens_revoked_at", "refresh_tokens", ["revoked_at"], unique=False)

    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.add_column(sa.Column("owner_id", sa.String(length=36), nullable=True))
        batch_op.create_foreign_key(
            "fk_documents_owner_id_users",
            "users",
            ["owner_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index("ix_documents_owner_id", ["owner_id"], unique=False)

    op.create_table(
        "processing_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("task_id", sa.String(length=64), nullable=True),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("extraction_method", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="QUEUED"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default=sa.text("3")),
        sa.Column("request_review", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_processing_jobs_id", "processing_jobs", ["id"], unique=False)
    op.create_index("ix_processing_jobs_document_id", "processing_jobs", ["document_id"], unique=False)
    op.create_index("ix_processing_jobs_task_id", "processing_jobs", ["task_id"], unique=False)
    op.create_index("ix_processing_jobs_idempotency_key", "processing_jobs", ["idempotency_key"], unique=True)
    op.create_index("ix_processing_jobs_status", "processing_jobs", ["status"], unique=False)
    op.create_index("ix_processing_jobs_heartbeat_at", "processing_jobs", ["heartbeat_at"], unique=False)
    op.create_index("ix_processing_jobs_created_at", "processing_jobs", ["created_at"], unique=False)

    op.create_table(
        "outbox_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("aggregate_type", sa.String(length=50), nullable=False),
        sa.Column("aggregate_id", sa.String(length=36), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("publish_attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("last_error", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_outbox_events_id", "outbox_events", ["id"], unique=False)
    op.create_index("ix_outbox_events_aggregate_id", "outbox_events", ["aggregate_id"], unique=False)
    op.create_index("ix_outbox_events_published_at", "outbox_events", ["published_at"], unique=False)
    op.create_index("ix_outbox_events_created_at", "outbox_events", ["created_at"], unique=False)

    with op.batch_alter_table("extraction_results", schema=None) as batch_op:
        batch_op.add_column(sa.Column("job_id", sa.String(length=36), nullable=True))
        batch_op.create_foreign_key(
            "fk_extraction_results_job_id_processing_jobs",
            "processing_jobs",
            ["job_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index("ix_extraction_results_job_id", ["job_id"], unique=True)


def downgrade() -> None:
    with op.batch_alter_table("extraction_results", schema=None) as batch_op:
        batch_op.drop_index("ix_extraction_results_job_id")
        batch_op.drop_constraint("fk_extraction_results_job_id_processing_jobs", type_="foreignkey")
        batch_op.drop_column("job_id")

    op.drop_index("ix_outbox_events_created_at", table_name="outbox_events")
    op.drop_index("ix_outbox_events_published_at", table_name="outbox_events")
    op.drop_index("ix_outbox_events_aggregate_id", table_name="outbox_events")
    op.drop_index("ix_outbox_events_id", table_name="outbox_events")
    op.drop_table("outbox_events")

    op.drop_index("ix_processing_jobs_created_at", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_heartbeat_at", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_status", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_idempotency_key", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_task_id", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_document_id", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_id", table_name="processing_jobs")
    op.drop_table("processing_jobs")

    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.drop_index("ix_documents_owner_id")
        batch_op.drop_constraint("fk_documents_owner_id_users", type_="foreignkey")
        batch_op.drop_column("owner_id")

    op.drop_index("ix_refresh_tokens_revoked_at", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_expires_at", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_token_hash", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_jti", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_user_id", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_id", table_name="refresh_tokens")
    op.drop_table("refresh_tokens")

    op.drop_index("ix_users_is_active", table_name="users")
    op.drop_index("ix_users_role", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_id", table_name="users")
    op.drop_table("users")

