# Bot Social Network

**A group chat of AI bots, in your terminal.** Give each bot a persona and a model, drop in a
topic, and watch them argue, remember what was said, and (optionally) speak out loud.

[![Python](https://img.shields.io/badge/python-3.12%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=for-the-badge)](https://github.com/astral-sh/ruff)
[![Mypy](https://img.shields.io/badge/types-mypy-blue.svg?style=for-the-badge&logo=python)](https://mypy-lang.org/)
[![UV](https://img.shields.io/badge/uv-managed-purple?style=for-the-badge)](https://github.com/astral-sh/uv)

## Features

- **Current models.** Gemini 3.8 Flash (default), 3.5 Flash-Lite, 3.1 Pro, and the open Gemma 4
  models through the Gemini API, plus any local **Ollama** model. Teams can mix them.
- **Real conversations.** Bots answer whoever @mentions them, quiet bots get the floor, nobody
  talks twice in a row. Persona and memories go in the system instruction; the chat goes in
  the prompt.
- **Memory that means something.** Every few posts a bot decides (as structured JSON) whether
  it learned something worth keeping. Memories feed back into its next posts.
- **Voices.** Each bot gets one of 30 Gemini TTS voices, stable across runs (or pick one).
  Same API key as the chat; no gcloud login.
- **Robust.** Retries with backoff on 429/5xx, typed errors that say why a turn failed, a bot
  that keeps failing is benched instead of crashing the run, WAL-mode SQLite with one session
  per operation, and replies are never cut off by the model's thinking budget.
- **Cost you can see.** Tokens, latency and cost per post; a live total in the TUI; `--budget`
  stops or pauses a run at a dollar limit.
- **Analysis.** `analyze latest` writes an HTML report: activity and cost per bot, an @mention
  graph, and sentiment over time.
- **Works from any folder.** Key, saved teams, database and logs live in your home directory.

## Install

```bash
uv tool install git+https://github.com/UCR-Research-Computing/LLM-Bot_Social_Chat.git
mkdir -p ~/.config/bot-social-network
echo 'GEMINI_API_KEY=your-key' >> ~/.config/bot-social-network/.env
chmod 600 ~/.config/bot-social-network/.env
bot-social-network doctor
```

Ollama is optional; if it is running on `localhost:11434` (or `OLLAMA_HOST`), its models show up
in the model picker.

## Use

```bash
bot-social-network                                   # TUI with the default team
bot-social-network run --team fantasy_tavern --autostart --tts
bot-social-network headless --team ai_philosophy_club --max-posts 20 \
    --topic "Is memory identity?" --budget 0.10
bot-social-network teams                             # bundled and saved teams
bot-social-network models --check                    # test each Gemini model live
bot-social-network models --check --ollama gemma4:e4b
bot-social-network analyze latest                    # HTML report of the last run
```

### TUI keys

| Key | Action | Key | Action |
|---|---|---|---|
| `space` | start / stop | `l` / `s` | load / save team |
| `n` | one post now | `a` / `e` / `d` | new / edit / delete bot |
| `t` | type a topic | `c` | clear posts (asks) |
| `v` | voice on/off | `+` / `-` | faster / slower |
| `q` | quit | | |

## Teams

A team is a JSON list of bots:

```json
[
  {
    "name": "Dan",
    "persona": "HPC systems engineer. Pragmatic, cautious about unproven tech.",
    "model": "gemini-3.8-flash",
    "voice": "Charon",
    "temperature": 0.9,
    "memories": [{ "key": "Primary concern", "value": "Stability over peak FLOPS." }]
  }
]
```

`voice`, `temperature` and `memories` are optional. Bundled teams live in the package; teams you
save (`s` in the TUI) go to `~/.config/bot-social-network/configs/` and override bundled ones with
the same name. Retired model ids in old team files (Gemini 1.5, 2.x) are upgraded on load.

## Files

| Path | What |
|---|---|
| `~/.config/bot-social-network/.env` | `GEMINI_API_KEY` and optional overrides |
| `~/.config/bot-social-network/configs/` | your saved teams |
| `~/.local/share/bot-social-network/bots.db` | bots, posts, memories |
| `~/.local/share/bot-social-network/runs/sim_*/` | `simulation.jsonl`, audio, reports |

Optional environment variables: `BSN_MEMORY_MODEL` (default `gemini-3.5-flash-lite`),
`BSN_TTS_MODEL` (default `gemini-3.8-flash-lite-tts`), `BSN_MAX_POST_TOKENS` (400),
`OLLAMA_HOST`, `BSN_HOME` (data dir).

## Cost

Chat posts on Gemini 3.8 Flash cost about $0.001 each; a 20-post run is a few cents. Voice adds
about $0.004 per spoken post. Prices from Google's pricing page on 2026-09-26; Flash prices
double on 2027-01-01. Gemma 4 through the API is free-tier only.

## Development

```bash
uv sync
uv run ruff check . --fix && uv run ruff format . && uv run mypy src && uv run pytest
```

Tests never call a paid API: the Gemini and Ollama clients are faked, the TUI is driven with
Textual's pilot, and a fixture keeps tests away from your real key and data.
