# Project Overview

Bot Social Network is a Textual TUI (plus a headless runner) where AI bots with personas chat
in a shared feed, form memories, and can speak with Gemini TTS. Models: Gemini 3.x and Gemma 4
through the Gemini API (google-genai). Local models (Ollama) were dropped in v0.5.0.

## Layout (`src/bot_social_network/`)

| Module | Role |
|---|---|
| `settings.py` | Paths (config/data dirs), model catalog with prices and thinking mode, legacy model map, voices |
| `database.py` | SQLAlchemy models; `Database` with one engine (WAL) and a session per operation; column migration for old DBs |
| `ai_client.py` | `GeminiClient` (shared client under a lock, SDK retries), prompts, `Reply`, `ModelError`, structured `MemoryNote` |
| `simulation.py` | Team loading/validation, turn-taking, failure benching, budget, memory scheduling |
| `dynamics.py` | Pure Python, no model calls: mention/question parsing, `ObligationLedger`, `FairScheduler`, `RepetitionGuard` (shingles + Jaccard), `BM25` memory selection, `sanitize_post`, `conversation_metrics` |
| `voice.py` | Gemini TTS to WAV, stable voice per bot, playback (pygame or paplay/aplay/afplay) |
| `tui.py` | Textual app |
| `headless.py` | Terminal runner with Rich output and a summary |
| `analyzer.py` | HTML report from `simulation.jsonl` |
| `cli.py` | `bot-social-network` entry point and subcommands |
| `configs/` | Bundled teams |

## Rules

- Tests never call a paid API; fake the clients. `tests/conftest.py` isolates settings paths
  and the key.
- Model ids must exist in the live model list; update `settings.GEMINI_MODELS` (with prices,
  thinking mode and thinking room) and `LEGACY_MODEL_MAP` together.
- Thinking tokens count against `max_output_tokens`; keep `think_room` for models that think,
  or replies are cut off mid-sentence.
- Bundled teams: description + topic, a distinct voice per bot from `settings.VOICE_INFO`, no
  temperature on Gemini 3 models (Google recommends the 1.0 default), Pro only for roles that
  need depth. `test_bundled_teams_all_valid_and_current` enforces this.
- Gemini TTS reads every word of its input aloud (style prefixes included) and rejects system
  instructions; pick the voice, do not prompt a style.
- Gauntlet: `uv run ruff check . --fix && uv run ruff format . && uv run mypy src && uv run pytest`.
