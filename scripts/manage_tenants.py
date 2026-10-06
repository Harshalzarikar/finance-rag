"""CLI for managing tenants without going through the HTTP API.

Useful for:
- Initial setup (create the first admin tenant before the API is accessible)
- Incident response (deactivate a compromised tenant instantly)
- Bulk operations

Usage
-----
    python scripts/manage_tenants.py create  --id acme-insurance --name "Acme Insurance" --plan pro
    python scripts/manage_tenants.py add-user --tenant acme-insurance --email user@acme.com --password 'Secret123!'
    python scripts/manage_tenants.py list
    python scripts/manage_tenants.py rotate-key --id acme-insurance
    python scripts/manage_tenants.py deactivate --id acme-insurance
    python scripts/manage_tenants.py activate   --id acme-insurance
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.config.settings import get_settings  # noqa: E402
from src.db.schema import init_db  # noqa: E402
from src.db.tenant_store import (  # noqa: E402
    create_tenant,
    list_tenants,
    rotate_api_key,
    set_tenant_prompt_settings,
)
from src.db.user_store import create_user  # noqa: E402


def _db_url() -> str:
    url = get_settings().database_url
    if not url:
        print("ERROR: DATABASE_URL is not set. Add it to .env and retry.", file=sys.stderr)
        sys.exit(1)
    return url


def cmd_create(args: argparse.Namespace) -> None:
    db = _db_url()
    init_db(db)
    raw_key, tenant = create_tenant(db, tenant_id=args.id, name=args.name, plan=args.plan)
    print("\n[OK] Tenant created successfully!")
    print(f"   ID       : {tenant.id}")
    print(f"   Name     : {tenant.name}")
    print(f"   Plan     : {tenant.plan}")
    print("\n[KEY] API KEY (shown ONCE - copy it now):")
    print(f"\n   {raw_key}\n")


def cmd_add_user(args: argparse.Namespace) -> None:
    db = _db_url()
    init_db(db)
    user = create_user(
        db,
        tenant_id=args.tenant,
        email=args.email,
        password=args.password,
        full_name=args.name or "",
    )
    print("\n[OK] User created — they can sign in at the web app with email + password.")
    print(f"   Tenant : {user.tenant_id}")
    print(f"   Email  : {user.email}")
    if user.full_name:
        print(f"   Name   : {user.full_name}")
    print()


def cmd_list(args: argparse.Namespace) -> None:  # noqa: ARG001
    db = _db_url()
    rows = list_tenants(db)
    if not rows:
        print("No tenants registered yet.")
        return
    print(f"\n{'ID':<30} {'NAME':<30} {'PLAN':<12} {'ACTIVE'}")
    print("-" * 82)
    for r in rows:
        print(f"{r.id:<30} {r.name:<30} {r.plan:<12} {r.is_active}")
    print()


def cmd_rotate(args: argparse.Namespace) -> None:
    db = _db_url()
    new_key = rotate_api_key(db, args.id)
    print(f"\n[ROTATED] Key rotated for tenant '{args.id}'. Old key is now invalid.")
    print("\n[KEY] NEW API KEY (shown ONCE - copy it now):")
    print(f"\n   {new_key}\n")


def cmd_deactivate(args: argparse.Namespace) -> None:
    db = _db_url()
    from src.db.schema import Tenant, get_session_factory

    Session = get_session_factory(db)
    with Session() as session:
        row = session.get(Tenant, args.id)
        if row is None:
            print(f"ERROR: Tenant '{args.id}' not found.", file=sys.stderr)
            sys.exit(1)
        row.is_active = False
        session.commit()
    print(f"[BLOCKED] Tenant '{args.id}' deactivated. Their API key is now rejected immediately.")


def cmd_set_prompt(args: argparse.Namespace) -> None:
    if args.instructions is None and args.pack is None:
        print("ERROR: pass --instructions and/or --pack.", file=sys.stderr)
        sys.exit(1)
    db = _db_url()
    init_db(db)
    tenant = set_tenant_prompt_settings(
        db,
        args.id,
        custom_instructions=args.instructions,
        prompt_pack_id=args.pack,
    )
    print(f"\n[OK] Prompt settings updated for tenant '{tenant.id}'.")
    if tenant.custom_instructions:
        print(f"   Custom instructions: {len(tenant.custom_instructions)} chars")
    if tenant.prompt_pack_id:
        print(f"   Prompt pack id     : {tenant.prompt_pack_id}")
    print("   (Clear with --instructions '' or omit --pack to leave unchanged.)")
    print()


def cmd_activate(args: argparse.Namespace) -> None:
    db = _db_url()
    from src.db.schema import Tenant, get_session_factory

    Session = get_session_factory(db)
    with Session() as session:
        row = session.get(Tenant, args.id)
        if row is None:
            print(f"ERROR: Tenant '{args.id}' not found.", file=sys.stderr)
            sys.exit(1)
        row.is_active = True
        session.commit()
    print(f"[OK] Tenant '{args.id}' activated.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage tenants for the Policy Intelligence SaaS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # create
    p_create = sub.add_parser("create", help="Register a new tenant")
    p_create.add_argument("--id", required=True, help="Unique slug, e.g. acme-insurance")
    p_create.add_argument("--name", required=True, help="Human-readable agency name")
    p_create.add_argument("--plan", default="free", choices=["free", "pro", "enterprise"])
    p_create.set_defaults(func=cmd_create)

    p_user = sub.add_parser("add-user", help="Create a login for an existing tenant")
    p_user.add_argument("--tenant", required=True, help="Tenant ID slug")
    p_user.add_argument("--email", required=True, help="User email (unique)")
    p_user.add_argument("--password", required=True, help="Initial password (min 8 chars)")
    p_user.add_argument("--name", default="", help="Display name")
    p_user.set_defaults(func=cmd_add_user)

    # list
    p_list = sub.add_parser("list", help="List all tenants")
    p_list.set_defaults(func=cmd_list)

    # rotate-key
    p_rotate = sub.add_parser("rotate-key", help="Rotate a tenant's API key")
    p_rotate.add_argument("--id", required=True, help="Tenant ID")
    p_rotate.set_defaults(func=cmd_rotate)

    p_prompt = sub.add_parser("set-prompt", help="Set org-specific generation instructions / prompt pack")
    p_prompt.add_argument("--id", required=True, help="Tenant ID")
    p_prompt.add_argument(
        "--instructions",
        default=None,
        help="Free-text org instructions (Layer D). Pass empty string to clear.",
    )
    p_prompt.add_argument(
        "--pack",
        default=None,
        help="Prompt pack id (file src/core/prompts/{id}.json). Omit to leave unchanged.",
    )
    p_prompt.set_defaults(func=cmd_set_prompt)

    # deactivate
    p_deact = sub.add_parser("deactivate", help="Block a tenant's API key")
    p_deact.add_argument("--id", required=True, help="Tenant ID")
    p_deact.set_defaults(func=cmd_deactivate)

    # activate
    p_act = sub.add_parser("activate", help="Re-enable a deactivated tenant")
    p_act.add_argument("--id", required=True, help="Tenant ID")
    p_act.set_defaults(func=cmd_activate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
