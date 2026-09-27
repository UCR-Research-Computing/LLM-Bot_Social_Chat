"""Drive the real TUI with Textual's pilot (no terminal, no network)."""

import pytest

from bot_social_network.ai_client import Reply
from bot_social_network.tui import BotEditScreen, BotSocialApp, ConfirmScreen


class FakeAI:
    def __init__(self):
        from bot_social_network.ai_client import GeminiClient, OllamaClient

        self.gemini = GeminiClient(api_key="x")
        self.ollama = OllamaClient("http://127.0.0.1:9")  # nothing listens

    async def write_post(self, bot, others, recent, memories):
        return Reply(f"hello from {bot.name}", bot.model, 10, 5, 0.001, 20, "STOP")

    async def form_memory(self, bot, recent):
        return None, None


@pytest.fixture
def app(tmp_path):
    a = BotSocialApp(
        team="default", db_url=f"sqlite:///{tmp_path / 't.db'}", interval=60
    )
    a.ai = FakeAI()
    a.sim.ai = a.ai
    return a


async def test_loads_team_steps_and_counts(app):
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        assert app.names == ["Dan", "Steve", "Mike"]
        await pilot.press("n")
        await pilot.pause()
        await pilot.press("n")
        await pilot.pause()
        assert app.sim.stats.posts == 2
        status = str(app.query_one("#status").render())
        assert "posts 2" in status and "PAUSED" in status


async def test_topic_injection_and_toggle(app):
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        await pilot.press("t")
        await pilot.press(*"GPUs?")
        await pilot.press("enter")
        await pilot.pause()
        assert app.db.recent_posts(1)[0].content == "GPUs?"
        await pilot.press("escape")
        app.set_focus(None)
        await pilot.press("space")
        assert app.running
        await pilot.press("space")
        assert not app.running


async def test_new_bot_dialog_validates(app):
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        await pilot.press("a")
        await pilot.pause()
        assert isinstance(app.screen, BotEditScreen)
        await pilot.click("#ok")
        await pilot.pause()
        assert "required" in str(app.screen.query_one("#err").render())
        await pilot.click("#cancel")
        await pilot.pause()
        assert not isinstance(app.screen, BotEditScreen)


async def test_clear_asks_first(app):
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        await pilot.press("n")
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.click("#yes")
        await pilot.pause()
        assert app.db.recent_posts() == []
