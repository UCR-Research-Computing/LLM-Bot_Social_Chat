"""Paths, settings and the model catalog.

Everything user-specific lives outside the checkout so the installed CLI works from
any folder:

- ``~/.config/bot-social-network/.env``      API key and optional overrides
- ``~/.config/bot-social-network/configs/``  saved bot teams (bundled ones are read-only)
- ``~/.local/share/bot-social-network/``      bots.db and per-run logs/audio

``BSN_HOME`` overrides the data dir (tests use it).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

APP = "bot-social-network"

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP
DATA_DIR = Path(
    os.environ.get("BSN_HOME")
    or Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / APP
)
USER_CONFIGS = CONFIG_DIR / "configs"
BUNDLED_CONFIGS = Path(__file__).parent / "configs"
DB_PATH = DATA_DIR / "bots.db"
RUNS_DIR = DATA_DIR / "runs"


def load_env() -> None:
    """Shell export wins, then the user settings file, then ./.env (fills gaps only)."""
    load_dotenv(CONFIG_DIR / ".env")
    load_dotenv()


load_env()


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    provider: str  # "gemini" or "ollama"
    in_per_m: float = 0.0  # USD per 1M input tokens (0 = free / local)
    out_per_m: float = 0.0  # USD per 1M output tokens, thinking included
    # How to keep chat replies short and cheap. Measured 2026-09-26: 3.8 Flash
    # accepts thinking_budget=0 but still thinks ~300-650 tokens; Pro only thinks
    # ("low"); Flash-Lite and Gemma honour "minimal" (no thinking tokens).
    thinking: str = "none"  # "budget0" | "minimal" | "low" | "none"
    note: str = ""
    # Extra output tokens reserved for thinking (thinking counts against the cap).
    think_room: int = 0


# Checked against the live model list and ai.google.dev/gemini-api/docs/pricing on
# 2026-09-26 (Flash prices double on 2027-01-01).
GEMINI_MODELS: list[ModelInfo] = [
    ModelInfo(
        "gemini-3.8-flash",
        "Gemini 3.8 Flash",
        "gemini",
        0.75,
        3.75,
        "budget0",
        "default; best all-rounder",
        think_room=1536,
    ),
    ModelInfo(
        "gemini-3.5-flash-lite",
        "Gemini 3.5 Flash-Lite",
        "gemini",
        0.30,
        2.50,
        "minimal",
        "fastest, cheapest",
    ),
    ModelInfo(
        "gemini-3.1-pro-preview",
        "Gemini 3.1 Pro (preview)",
        "gemini",
        2.00,
        12.00,
        "low",
        "deepest; always thinks, slower",
        think_room=2048,
    ),
    ModelInfo(
        "gemma-4-31b-it",
        "Gemma 4 31B (API)",
        "gemini",
        0.0,
        0.0,
        "minimal",
        "open model; free tier only",
    ),
    ModelInfo(
        "gemma-4-26b-a4b-it",
        "Gemma 4 26B MoE (API)",
        "gemini",
        0.0,
        0.0,
        "minimal",
        "open model; free tier only",
    ),
]

DEFAULT_MODEL = "gemini-3.8-flash"
MEMORY_MODEL = os.environ.get("BSN_MEMORY_MODEL", "gemini-3.5-flash-lite")

# Old ids found in saved configs -> current replacement. 1.5 models are shut down;
# 2.5 still answers but is two generations behind.
LEGACY_MODEL_MAP = {
    "gemini-1.5-flash": "gemini-3.8-flash",
    "gemini-1.5-pro": "gemini-3.1-pro-preview",
    "gemini-2.0-flash": "gemini-3.8-flash",
    "gemini-2.5-flash": "gemini-3.8-flash",
    "gemini-2.5-flash-lite": "gemini-3.5-flash-lite",
    "gemini-2.5-pro": "gemini-3.1-pro-preview",
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
}

TTS_MODEL = os.environ.get("BSN_TTS_MODEL", "gemini-3.8-flash-lite-tts")
TTS_OUT_PER_M = 6.00  # audio tokens, 25 per second
TTS_IN_PER_M = 0.50

# Gemini TTS prebuilt voices (all 30 answered on 2026-09-26).
VOICES = (
    "Zephyr Puck Charon Kore Fenrir Leda Orus Aoede Callirrhoe Autonoe Enceladus "
    "Iapetus Umbriel Algieba Despina Erinome Algenib Rasalgethi Laomedeia Achernar "
    "Alnilam Schedar Gacrux Pulcherrima Achird Zubenelgenubi Vindemiatrix Sadachbia "
    "Sadaltager Sulafat"
).split()

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL


def model_info(model_id: str) -> ModelInfo:
    for m in GEMINI_MODELS:
        if m.id == model_id:
            return m
    if model_id.startswith(("gemini", "gemma-")):
        return ModelInfo(model_id, model_id, "gemini", thinking="none")
    return ModelInfo(model_id, model_id, "ollama")


def upgrade_model(model_id: str | None) -> str:
    if not model_id:
        return DEFAULT_MODEL
    return LEGACY_MODEL_MAP.get(model_id, model_id)


def ensure_dirs() -> None:
    for d in (CONFIG_DIR, USER_CONFIGS, DATA_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
