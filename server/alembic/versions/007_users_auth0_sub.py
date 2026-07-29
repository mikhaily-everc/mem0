"""Add users.auth0_sub for SSO-provisioned accounts

Revision ID: 007
Revises: 006
Create Date: 2026-07-29

Nullable because every pre-existing row is a password account with no Auth0
identity, and those must keep working. Unique because `auth0_sub` is the lookup
key for JIT provisioning — without the constraint, a race between two concurrent
first-time logins would insert two rows for the same person and silently split
their memories in half. Postgres treats NULLs as distinct in a unique index, so
the constraint costs the password rows nothing.

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("auth0_sub", sa.String(255), nullable=True))
    op.create_index("ix_users_auth0_sub", "users", ["auth0_sub"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_users_auth0_sub", table_name="users")
    op.drop_column("users", "auth0_sub")
