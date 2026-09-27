import asyncio
import io
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from bot_social_network import settings
from bot_social_network.ai_client import GeminiClient, ModelError
from bot_social_network.voice import Voice, _to_wav, voice_for, wav_seconds


def test_voice_is_stable_and_valid():
    v = voice_for("Captain Eva Rostova")
    assert v in settings.VOICES
    assert all(voice_for("Captain Eva Rostova") == v for _ in range(5))
    names = [f"Bot{i}" for i in range(40)]
    assert len({voice_for(n) for n in names}) > 5


def test_preferred_voice_wins_if_valid():
    assert voice_for("X", "Kore") == "Kore"
    assert voice_for("X", "NotAVoice") in settings.VOICES


def test_raw_pcm_is_wrapped_as_wav(tmp_path):
    pcm = b"\x00\x01" * 24000  # 1 s of 16-bit mono at 24 kHz
    data = _to_wav(pcm, "audio/L16;codec=pcm;rate=24000")
    assert data[:4] == b"RIFF"
    p = tmp_path / "a.wav"
    p.write_bytes(data)
    assert wav_seconds(p) == pytest.approx(1.0)
    assert _to_wav(data, "audio/wav") is data


def _wav_bytes(seconds=0.5):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x00" * int(24000 * seconds))
    return buf.getvalue()


class FakeModels:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    async def generate_content(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return self.resp


def _voice(models):
    g = GeminiClient(api_key="x")
    g._client = SimpleNamespace(aio=SimpleNamespace(models=models))
    return Voice(g)


def test_synthesize_writes_wav_and_costs(tmp_path):
    part = SimpleNamespace(
        inline_data=SimpleNamespace(data=_wav_bytes(0.5), mime_type="audio/wav")
    )
    resp = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))],
        usage_metadata=SimpleNamespace(
            prompt_token_count=10, candidates_token_count=13
        ),
    )
    models = FakeModels(resp)
    sp = asyncio.run(
        _voice(models).synthesize("hello", "Kore", tmp_path / "x" / "a.wav")
    )
    assert Path(sp.path).exists() and sp.seconds == pytest.approx(0.5)
    assert sp.cost_usd == pytest.approx(10 / 1e6 * 0.5 + 13 / 1e6 * 6.0)
    cfg = models.calls[0]["config"]
    assert cfg.speech_config.voice_config.prebuilt_voice_config.voice_name == "Kore"
    assert models.calls[0]["model"] == settings.TTS_MODEL


def test_synthesize_without_audio_raises(tmp_path):
    resp = SimpleNamespace(candidates=[], usage_metadata=None)
    with pytest.raises(ModelError, match="no audio"):
        asyncio.run(
            _voice(FakeModels(resp)).synthesize("x", "Kore", tmp_path / "a.wav")
        )
