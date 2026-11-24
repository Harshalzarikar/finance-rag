"""Back up Postgres and Qdrant to a timestamped directory.

Run from the repository root so ``docker compose exec`` resolves the correct
project. Credentials are read from the environment (or ``.env`` via python-dotenv).

Usage:
    python scripts/backup.py --target ./backups --keep 7
    python scripts/backup.py --target /mnt/backups --skip-qdrant
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from dotenv import load_dotenv


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _pg_dump(target: Path) -> Path:
    user = _env("POSTGRES_USER", "raguser")
    db = _env("POSTGRES_DB", "ragdb")
    password = _env("POSTGRES_PASSWORD", "")

    cmd = ["docker", "compose", "exec", "-T", "postgres"]
    if password:
        cmd += ["env", f"PGPASSWORD={password}"]
    cmd += ["pg_dump", "-U", user, "-d", db, "--no-owner", "--no-privileges"]

    out = target / "postgres.dump"
    with open(out, "wb") as fh:
        subprocess.run(cmd, stdout=fh, check=True)
    return out


def _qdrant_snapshot(target: Path, url: str, collection: str) -> Path:
    base = url.rstrip("/")
    create_url = f"{base}/collections/{collection}/snapshots"

    with urlopen(Request(create_url, method="POST")) as resp:
        payload = json.load(resp)
    name = payload["result"]["name"]

    download_url = f"{base}/collections/{collection}/snapshots/{name}"
    out = target / "qdrant.snapshot"
    with urlopen(download_url) as resp, open(out, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    return out


def _prune(root: Path, keep: int) -> None:
    backups = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    for stale in backups[:-keep]:
        shutil.rmtree(stale, ignore_errors=True)


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Back up Postgres and Qdrant.")
    parser.add_argument("--target", default="./backups", help="Parent directory for backups")
    parser.add_argument("--keep", type=int, default=7, help="Number of recent backups to retain")
    parser.add_argument("--skip-qdrant", action="store_true", help="Skip the Qdrant snapshot")
    args = parser.parse_args()

    qdrant_url = _env("QDRANT_URL", "http://localhost:6333")
    collection = _env("QDRANT_COLLECTION_NAME", "finance_documents_v1")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = Path(args.target) / stamp
    target.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] Dumping Postgres -> {target / 'postgres.dump'}")
    _pg_dump(target)

    if not args.skip_qdrant:
        print(f"[2/2] Snapshotting Qdrant collection '{collection}'")
        _qdrant_snapshot(target, qdrant_url, collection)

    _prune(Path(args.target), args.keep)
    print(f"Backup complete: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
