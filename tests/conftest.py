import pytest


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """No test reads the real ~/.config key or writes to the real data dir."""
    from bot_social_network import settings

    monkeypatch.setattr(settings, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(settings, "USER_CONFIGS", tmp_path / "cfg" / "configs")
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(settings, "RUNS_DIR", tmp_path / "data" / "runs")
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "data" / "bots.db")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)


@pytest.fixture
def db():
    from bot_social_network.database import Database

    d = Database("sqlite:///:memory:")
    yield d
    d.close()
