"""Add OCR raw text and field confidences to extraction results

Revision ID: 002_add_ocr_fields
Revises: 001_initial_tables
Create Date: 2026-11-13 01:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '002_add_ocr_fields'
down_revision: Union[str, None] = '001_initial_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Use batch_alter_table for SQLite compatibility
    with op.batch_alter_table('extraction_results', schema=None) as batch_op:
        batch_op.add_column(sa.Column('raw_text', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('field_confidences', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('extraction_metadata', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('extraction_results', schema=None) as batch_op:
        batch_op.drop_column('extraction_metadata')
        batch_op.drop_column('field_confidences')
        batch_op.drop_column('raw_text')
