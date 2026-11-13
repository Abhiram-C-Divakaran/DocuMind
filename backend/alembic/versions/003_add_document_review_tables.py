"""Add human-in-the-loop review, field correction, and audit history tables

Revision ID: 003_add_document_review_tables
Revises: 002_add_ocr_fields
Create Date: 2026-11-13 09:45:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '003_add_document_review_tables'
down_revision: Union[str, None] = '002_add_ocr_fields'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add review columns on documents using batch_alter_table for SQLite + Postgres compatibility
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('review_status', sa.String(length=30), nullable=False, server_default='NOT_REQUIRED')
        )
        batch_op.add_column(
            sa.Column('needs_review', sa.Boolean(), nullable=False, server_default=sa.false())
        )
        batch_op.add_column(
            sa.Column('review_reasons', sa.JSON(), nullable=False, server_default='[]')
        )
        batch_op.create_index('ix_documents_review_status', ['review_status'], unique=False)
        batch_op.create_index('ix_documents_needs_review', ['needs_review'], unique=False)

    # 2. document_reviews
    op.create_table(
        'document_reviews',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('document_id', sa.String(length=36), nullable=False),
        sa.Column('extraction_result_id', sa.String(length=36), nullable=True),
        sa.Column('review_status', sa.String(length=30), nullable=False, server_default='PENDING_REVIEW'),
        sa.Column('corrected_data', sa.JSON(), nullable=True),
        sa.Column('changed_fields', sa.JSON(), nullable=False),
        sa.Column('reviewer_identifier', sa.String(length=120), nullable=False, server_default='local-anonymous'),
        sa.Column('reviewer_notes', sa.Text(), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('approved_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['extraction_result_id'], ['extraction_results.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('document_id', name='uq_document_reviews_document_id'),
    )
    op.create_index(op.f('ix_document_reviews_id'), 'document_reviews', ['id'], unique=False)
    op.create_index(op.f('ix_document_reviews_document_id'), 'document_reviews', ['document_id'], unique=True)
    op.create_index(op.f('ix_document_reviews_extraction_result_id'), 'document_reviews', ['extraction_result_id'], unique=False)
    op.create_index(op.f('ix_document_reviews_review_status'), 'document_reviews', ['review_status'], unique=False)
    op.create_index(op.f('ix_document_reviews_created_at'), 'document_reviews', ['created_at'], unique=False)

    # 3. field_corrections
    op.create_table(
        'field_corrections',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('review_id', sa.String(length=36), nullable=False),
        sa.Column('field_name', sa.String(length=64), nullable=False),
        sa.Column('original_value', sa.Text(), nullable=True),
        sa.Column('corrected_value', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['review_id'], ['document_reviews.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_field_corrections_id'), 'field_corrections', ['id'], unique=False)
    op.create_index(op.f('ix_field_corrections_review_id'), 'field_corrections', ['review_id'], unique=False)
    op.create_index(op.f('ix_field_corrections_field_name'), 'field_corrections', ['field_name'], unique=False)

    # 4. review_audit_events
    op.create_table(
        'review_audit_events',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('review_id', sa.String(length=36), nullable=False),
        sa.Column('action', sa.String(length=50), nullable=False),
        sa.Column('from_status', sa.String(length=30), nullable=True),
        sa.Column('to_status', sa.String(length=30), nullable=True),
        sa.Column('changed_fields', sa.JSON(), nullable=False),
        sa.Column('actor_identifier', sa.String(length=120), nullable=False, server_default='local-anonymous'),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('event_timestamp', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['review_id'], ['document_reviews.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_review_audit_events_id'), 'review_audit_events', ['id'], unique=False)
    op.create_index(op.f('ix_review_audit_events_review_id'), 'review_audit_events', ['review_id'], unique=False)
    op.create_index(op.f('ix_review_audit_events_action'), 'review_audit_events', ['action'], unique=False)
    op.create_index(op.f('ix_review_audit_events_event_timestamp'), 'review_audit_events', ['event_timestamp'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_review_audit_events_event_timestamp'), table_name='review_audit_events')
    op.drop_index(op.f('ix_review_audit_events_action'), table_name='review_audit_events')
    op.drop_index(op.f('ix_review_audit_events_review_id'), table_name='review_audit_events')
    op.drop_index(op.f('ix_review_audit_events_id'), table_name='review_audit_events')
    op.drop_table('review_audit_events')

    op.drop_index(op.f('ix_field_corrections_field_name'), table_name='field_corrections')
    op.drop_index(op.f('ix_field_corrections_review_id'), table_name='field_corrections')
    op.drop_index(op.f('ix_field_corrections_id'), table_name='field_corrections')
    op.drop_table('field_corrections')

    op.drop_index(op.f('ix_document_reviews_created_at'), table_name='document_reviews')
    op.drop_index(op.f('ix_document_reviews_review_status'), table_name='document_reviews')
    op.drop_index(op.f('ix_document_reviews_extraction_result_id'), table_name='document_reviews')
    op.drop_index(op.f('ix_document_reviews_document_id'), table_name='document_reviews')
    op.drop_index(op.f('ix_document_reviews_id'), table_name='document_reviews')
    op.drop_table('document_reviews')

    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.drop_index('ix_documents_needs_review')
        batch_op.drop_index('ix_documents_review_status')
        batch_op.drop_column('review_reasons')
        batch_op.drop_column('needs_review')
        batch_op.drop_column('review_status')

