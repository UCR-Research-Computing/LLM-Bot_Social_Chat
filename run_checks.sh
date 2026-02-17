#!/bin/bash
#
# This script runs all local checks (linting and unit tests)
# to ensure code quality before committing.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Running Linter (ruff) ---"
uv run ruff check .

echo ""
echo "--- Running Unit Tests (pytest) ---"
uv run pytest

echo ""
echo "✅ All checks passed!"
