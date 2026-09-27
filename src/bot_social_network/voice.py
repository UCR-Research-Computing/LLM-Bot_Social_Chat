"""Text-to-speech with Gemini TTS (same API key as the chat; no gcloud login).

Each bot gets a stable voice: its own `voice` if set (teams pick one that fits
the character from the documented voice qualities), otherwise one picked from
its name with a stable hash (Python's hash() changes every run, so the old code
gave bots a different voice each launch). Playback uses pygame when available and
falls back to paplay/aplay/afplay.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import settings
from .ai_client import GeminiClient, ModelError, _short_error

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000


@dataclass
class Speech:
    path: Path
    seconds: float
    cost_usd: float


def voice_for(bot_name: str, preferred: str | None = None) -> str:
    if preferred and preferred in settings.VOICES:
        return preferred
    h = int(hashlib.sha256(bot_name.encode()).hexdigest(), 16)
    return settings.VOICES[h % len(settings.VOICES)]


def _to_wav(data: bytes, mime: str) -> bytes:
    """Gemini TTS returns WAV (RIFF) or raw 16-bit PCM (audio/L16;rate=24000)."""
    if data[:4] == b"RIFF":
        return data
    rate = SAMPLE_RATE
    if "rate=" in mime:
        try:
            rate = int(mime.split("rate=")[1].split(";")[0])
        except ValueError:
            pass
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data)
    return buf.getvalue()


def wav_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate() or SAMPLE_RATE)
    except Exception:
        return 0.0


class Voice:
    def __init__(self, gemini: GeminiClient, model: str = settings.TTS_MODEL):
        self.gemini = gemini
        self.model = model
        self._play_lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._stopped = False

    async def synthesize(
        self, text: str, voice: str, out_path: Path, style: str = ""
    ) -> Speech:
        from google.genai import types

        # Gemini TTS reads everything in `contents` aloud, including a
        # "Say in a gruff voice:" prefix (verified 2026-09-27 by transcribing the
        # audio back) and it rejects system instructions. The prebuilt voice is
        # the only reliable style control, so `style` is kept out of the audio.
        del style
        prompt = text
        client = self.gemini.client()
        try:
            resp = await client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice
                            )
                        )
                    ),
                ),
            )
        except Exception as e:
            raise ModelError(_short_error(e), self.model) from e
        part: Any = None
        for c in getattr(resp, "candidates", None) or []:
            for p in getattr(getattr(c, "content", None), "parts", None) or []:
                if getattr(p, "inline_data", None) is not None:
                    part = p.inline_data
                    break
        if part is None or not part.data:
            raise ModelError("no audio in reply", self.model)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(_to_wav(part.data, part.mime_type or ""))
        secs = wav_seconds(out_path)
        u = getattr(resp, "usage_metadata", None)
        tin = (getattr(u, "prompt_token_count", 0) or 0) if u else 0
        tout = (getattr(u, "candidates_token_count", 0) or 0) if u else int(secs * 25)
        cost = tin / 1e6 * settings.TTS_IN_PER_M + tout / 1e6 * settings.TTS_OUT_PER_M
        return Speech(out_path, secs, round(cost, 6))

    # ---- playback ----------------------------------------------------------
    def play(self, path: Path) -> None:
        """Blocking; run in a thread. One clip at a time."""
        with self._play_lock:
            self._stopped = False
            if self._play_pygame(path):
                return
            player = _cli_player()
            if not player:
                log.warning("No audio player found (install pygame or pulseaudio)")
                return
            self._proc = subprocess.Popen(
                [*player, str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._proc.wait()
            self._proc = None

    def _play_pygame(self, path: Path) -> bool:
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame
        except Exception:
            return False
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=SAMPLE_RATE)
            pygame.mixer.music.load(str(path))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and not self._stopped:
                time.sleep(0.05)
            return True
        except Exception as e:
            log.info("pygame playback failed (%s); trying a CLI player", e)
            return False

    def stop(self) -> None:
        self._stopped = True
        try:
            import pygame

            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception:
            pass
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


def _cli_player() -> list[str] | None:
    for cmd in (
        ["paplay"],
        ["aplay", "-q"],
        ["afplay"],
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    ):
        if shutil.which(cmd[0]):
            return cmd
    return None


async def speak_and_play(
    voice: Voice, text: str, name: str, preferred: str | None, out: Path
) -> Speech:
    speech = await voice.synthesize(text, voice_for(name, preferred), out)
    await asyncio.to_thread(voice.play, speech.path)
    return speech
