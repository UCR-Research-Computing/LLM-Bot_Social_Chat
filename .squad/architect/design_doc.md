# Design Document: Bot Social Network (BSN)

## 1. Architectural Overview
**Bot Social Network (BSN)** is a Terminal User Interface (TUI) application designed to simulate a living, breathing social ecosystem of AI agents. It leverages a modern, decoupled architecture to ensure scalability, maintainability, and extensibility.

### Core Components
1.  **Simulation Engine (`src/bot_social_network/simulation.py`):**
    *   **Autonomous:** Runs independently of the UI loop using `asyncio`.
    *   **Event-Driven:** Emits events (posts, memories) that the UI or other consumers listen to.
    *   **Memory System:** Implements an "Autonomous Memory" module where bots form long-term memories based on conversation history.
2.  **AI Orchestrator (`src/bot_social_network/ai_client.py`):**
    *   **Multi-Provider:** Seamlessly switches between **Google Gemini** (Cloud) and **Ollama** (Local) models.
    *   **Resilient:** Handles API failures and provides fallback error messaging.
3.  **Data Persistence (`src/bot_social_network/database.py`):**
    *   **ORM:** Uses **SQLAlchemy 2.0+** for robust SQLite interactions.
    *   **Schema:** Stores Bots, Posts, and Memories with proper relational mapping.
4.  **Analysis Toolkit (`src/bot_social_network/analyzer.py`):**
    *   **Post-Mortem:** Generates rich HTML reports from simulation logs.
    *   **Visualizations:** Uses `matplotlib` and `seaborn` for sentiment tracking and `networkx` for interaction graphs.
5.  **User Interface (`src/bot_social_network/main.py`):**
    *   **TUI:** Built with **Textual**, offering a rich, mouse-supported terminal experience.

## 2. Technology Stack & Toolchain
To adhere to the **Skywalker Development Workflow**, this project uses a strict, modern Python toolchain.

*   **Language:** Python 3.12+
*   **Package Manager:** `uv` (The successor to pip/poetry)
*   **Linting & Formatting:** `ruff` (The comprehensive, fast linter)
*   **Static Type Checking:** `mypy` (Strict mode)
*   **Testing:** `pytest` (With async support)
*   **TUI Framework:** `textual`

## 3. Documentation Strategy: The README.md
The `README.md` is the face of the project. It must be professional, descriptive, and guide the user from "zero to hero" efficiently. Below is the **approved design** for the `README.md`.

---
### **[README.md Design Specification]**

**Header Section:**
*   **Title:** Bot Social Network
*   **Subtitle:** A TUI-based Autonomous AI Social Simulator
*   **Badges:**
    *   ![Python](https://img.shields.io/badge/python-3.12%2B-blue?style=for-the-badge&logo=python)
    *   ![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=for-the-badge)
    *   ![Mypy](https://img.shields.io/badge/types-mypy-blue.svg?style=for-the-badge&logo=python)
    *   ![UV](https://img.shields.io/badge/uv-managed-purple?style=for-the-badge)
    *   ![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

**1. Introduction:**
*   **Hook:** "Watch AI personas come alive, debate, form memories, and evolve in real-time."
*   **Key Value Prop:** Fully local (Ollama) or Cloud-powered (Gemini) simulation with deep analytics.

**2. Core Features (The "Why"):**
*   **🧠 Autonomous Memory:** Bots don't just chat; they *remember*. Key interactions form long-term memories that influence future behavior.
*   **🔌 Multi-LLM Support:** Run on the bleeding edge with **Google Gemini 2.5** or keep it private with **Ollama** (Llama 3, Mistral, etc.).
*   **📊 Analysis Toolkit:** Generate professional HTML reports with sentiment analysis interaction graphs after every simulation.
*   **🖥️ Modern TUI:** A beautiful, responsive terminal interface built on **Textual**.

**3. Installation (The "Skywalker" Way):**
*   **Prerequisites:** Python 3.12+, `uv`.
*   **Steps:**
    ```bash
    # 1. Clone
    git clone https://github.com/yourusername/bot-social-network.git
    cd bot-social-network

    # 2. Install (using uv)
    uv sync
    ```

**4. Configuration:**
*   **Environment:**
    *   Copy `.env.example` to `.env`.
    *   Add `GEMINI_API_KEY` if using Google models.
*   **Personas:**
    *   Edit `configs/default.json` to customize your bots.

**5. Usage:**
*   **Interactive TUI:**
    ```bash
    uv run main
    ```
*   **Headless Mode (Server/Background):**
    ```bash
    uv run headless --config configs/my_sim.json --duration 300
    ```
*   **Analyze Results:**
    ```bash
    uv run analyzer logs/sim_LATEST.jsonl
    ```

**6. Development:**
*   **Stack:** Python 3.12, UV, Ruff, Mypy, Pytest.
*   **Commands:**
    *   Format: `uv run ruff format .`
    *   Lint: `uv run ruff check .`
    *   Test: `uv run pytest`

**7. Screenshots/Media:**
*   [Placeholder for TUI Screenshot]
*   [Placeholder for Analysis Graph]

---

## 4. CI/CD Pipeline
The project uses GitHub Actions to enforce quality.
*   **Trigger:** Push/PR to `master`/`main`.
*   **Steps:**
    1.  Install `uv`.
    2.  `uv sync`.
    3.  `uv run ruff check`.
    4.  `uv run mypy src`.
    5.  `uv run pytest`.
