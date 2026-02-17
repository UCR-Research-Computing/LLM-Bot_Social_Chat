# Design Document: Bot Social Network Migration

## 1. Language & Stack Selection
*   **Language:** Python 3.12+
*   **Package Manager:** `uv`
*   **Build System:** `hatchling` (via `pyproject.toml`)
*   **Project Layout:** `src/` based layout.

## 2. Toolchain Definition
To adhere to the **Skywalker Development Workflow**, the following tools are mandatory:

*   **Dependency Management:** `uv`
    *   Initialize/Sync: `uv sync`
    *   Add packages: `uv add <package>`
    *   Run commands: `uv run <command>`
*   **Linting & Formatting:** `ruff`
    *   Check: `uv run ruff check . --fix`
    *   Format: `uv run ruff format .`
*   **Static Type Checking:** `mypy`
    *   Check: `uv run mypy src`
*   **Testing:** `pytest`
    *   Run: `uv run pytest`

## 3. Project Structure
The project must follow the standard `src` layout to prevent import errors and ensure clean packaging.

```text
bot_social_network/
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   └── bot_social_network/
│       ├── __init__.py
│       ├── main.py
│       └── ...
├── tests/
├── pyproject.toml
├── uv.lock
└── README.md
```

## 4. CI/CD Workflow (`.github/workflows/ci.yml`)
The CI pipeline serves as the "Gatekeeper" and must run the "Local Gauntlet" in the cloud.

```yaml
name: CI

on:
  push:
    branches: [ master, main ]
  pull_request:
    branches: [ master, main ]

jobs:
  build:
    name: The Gatekeeper
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      
      - name: Set up Python
        run: uv python install 3.12

      - name: Install dependencies
        run: uv sync --all-extras --dev

      - name: Lint (Ruff)
        run: uv run ruff check .

      - name: Format (Ruff)
        run: uv run ruff format --check .

      - name: Type Check (Mypy)
        run: uv run mypy src

      - name: Test (Pytest)
        run: uv run pytest
```

## 5. Documentation Requirements
*   **README.md**: Must be updated to instruct users to use `uv` for installation and development.
    *   Example: `uv sync` to install, `uv run main.py` to launch.
