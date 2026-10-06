#!/usr/bin/env python3
"""Print a reminder to use the full production setup script."""

from __future__ import annotations

import sys

if __name__ == "__main__":
    print(
        "This helper was replaced by scripts/setup_production_env.py\n\n"
        "  python scripts/setup_production_env.py --domain app.example.com --email ops@example.com --force\n",
        file=sys.stderr,
    )
    sys.exit(2)
