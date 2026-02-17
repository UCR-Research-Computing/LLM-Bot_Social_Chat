# Design Document: Graceful Shutdown (Ctrl-C)

## 1. Objective
Implement a graceful shutdown mechanism for the Bot Social Network TUI when the user presses `Ctrl+C`. This ensures that:
*   Database connections are closed properly.
*   Background tasks (simulation, TTS) are cancelled.
*   Audio playback is stopped immediately.
*   The terminal is restored to its original state without stack traces.

## 2. Technical Stack
*   **Language:** Python 3.12+
*   **Framework:** Textual (TUI), `uv` (Package Manager)
*   **Key Libraries:** `asyncio`, `signal`

## 3. Implementation Strategy

### 3.1. Textual Key Binding
The primary method for handling `Ctrl+C` within a Textual application is to bind the key combination to a quit action.

**Target File:** `src/bot_social_network/main.py`

**Changes:**
1.  Update `BotSocialApp.BINDINGS` to include `ctrl+c`.
    ```python
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]
    ```
    *   `show=False` ensures it doesn't clutter the footer, as it's a standard convention.

### 3.2. Verify `action_quit` Logic
The existing `action_quit` method in `BotSocialApp` already contains cleanup logic. We will verify it performs the following:
*   Stops the `bot_timer`.
*   Stops the `voice_manager`.
*   Cancels all `simulation.background_tasks`.
*   Closes the database connection.
*   Calls `self.exit()`.

### 3.3. Signal Handling (Fallback)
While the key binding works when the application has focus, a system-level `SIGINT` handler can be added as a safety net in `run()` to catch interrupts during startup or shutdown phases.

**Target File:** `src/bot_social_network/main.py`

**Changes:**
*   Wrap `app.run()` in a `try...except KeyboardInterrupt` block.
*   However, since `app.run()` handles the event loop, the key binding is the preferred and "cleaner" path for TUI interaction. The `try...except` block will be added to the `run()` function just in case.

## 4. Verification Plan
1.  **Manual Test:** Start the app (`uv run src/bot_social_network/main.py`), wait for bots to start chatting, then press `Ctrl+C`.
2.  **Success Criteria:**
    *   App closes immediately.
    *   No Python stack trace is printed to the terminal.
    *   Log file (`bots.log`) confirms "Application exited." with "system.stop" event.
    *   Database lock is released (sqlite wal file handling).

## 5. Development Workflow (Skywalker)
1.  Create feature branch `feature/graceful-shutdown`.
2.  Bump version in `pyproject.toml`.
3.  Implement changes.
4.  Run `uv run ruff check .` and `uv run mypy src`.
5.  Verify manually.
6.  Merge and release.
