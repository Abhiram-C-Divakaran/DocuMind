import argparse
import asyncio
import getpass
import sys
from typing import Optional

from sqlalchemy import select

from backend.app.core.security import hash_password
from backend.app.database import AsyncSessionLocal, Base, engine
from backend.app.models.user import User, UserRole


async def _create_or_update_admin(
    email: str,
    password: str,
    full_name: Optional[str],
    role: UserRole = UserRole.ADMIN,
) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    normalized_email = email.strip().lower()
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.email == normalized_email))
        existing = result.scalar_one_or_none()
        if existing is not None:
            existing.password_hash = hash_password(password)
            existing.role = role.value
            existing.is_active = True
            if full_name:
                existing.full_name = full_name.strip()
            await db.commit()
            print(f"Updated existing user '{normalized_email}' with role '{role.value}'.")
            return

        user = User(
            email=normalized_email,
            full_name=(full_name or "").strip() or "DocuMind Administrator",
            password_hash=hash_password(password),
            role=role.value,
            is_active=True,
        )
        db.add(user)
        await db.commit()
        print(f"Created user '{normalized_email}' with role '{role.value}'.")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DocuMind administration CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    admin_cmd = sub.add_parser("create-admin", help="Provision or update an initial administrator account")
    admin_cmd.add_argument("--email", required=True, help="Admin email address")
    admin_cmd.add_argument("--password", required=False, help="Admin password (>=8 chars; prompts if omitted)")
    admin_cmd.add_argument("--full-name", default="DocuMind Administrator", help="Optional display name")
    admin_cmd.add_argument(
        "--role",
        default=UserRole.ADMIN.value,
        choices=[r.value for r in UserRole],
        help="Account role (defaults to ADMIN)",
    )

    args = parser.parse_args(argv)
    if args.command == "create-admin":
        pw = args.password
        if not pw:
            pw = getpass.getpass("Enter password (>=8 chars): ")
            confirm = getpass.getpass("Confirm password: ")
            if pw != confirm:
                print("Error: passwords do not match.", file=sys.stderr)
                return 1
        asyncio.run(
            _create_or_update_admin(
                email=args.email,
                password=pw,
                full_name=args.full_name,
                role=UserRole(args.role),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

