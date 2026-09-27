"""Backwards-compatible entry point (`python -m bot_social_network.main`)."""

from __future__ import annotations

import sys

from .cli import main


def run() -> None:
    sys.exit(main())


if __name__ == "__main__":
    run()
