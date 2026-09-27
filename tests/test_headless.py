"""Headless run end to end with a scripted AI (no network)."""

import asyncio

from bot_social_network import headless
from bot_social_network.ai_client import Reply


class LastWordAI:
    """Dan asks Mike a question on the last regular post; Mike answers."""

    def __init__(self):
        from bot_social_network.ai_client import GeminiClient

        self.gemini = GeminiClient(api_key="x")
        self.n = 0

    async def write_post(
        self, bot, others, recent, memories, questions=(), avoid=None, closing=False
    ):
        self.n += 1
        self.closing_seen = getattr(self, "closing_seen", []) + [closing]
        if questions:
            # Answers, but also asks back, as real models do.
            text = (
                f"@{questions[0].asker} the answer is Ceph on NVMe. "
                f"@{questions[0].asker} do you agree?"
            )
        elif self.n == 3:  # the last regular post asks someone a question
            target = next(o for o in others)
            text = f"@{target} what storage should we buy for the cluster?"
        else:
            text = f"{bot.name} musing {self.n}: racks, power and cooling {self.n * 7}"
        return Reply(text, bot.model, 10, 5, 0.001, 5, "STOP")

    async def form_memory(self, bot, recent):
        return None, None

    async def aclose(self):
        pass


def run(monkeypatch, tmp_path, wrap_up):
    monkeypatch.setattr(headless, "AIClient", LastWordAI)
    return asyncio.run(
        headless.run_headless(
            team="default",
            max_posts=3,
            duration=None,
            topic="New cluster",
            tts=False,
            deterministic=False,
            seed=1,
            delay=0,
            budget=None,
            keep_posts=False,
            db_url=f"sqlite:///{tmp_path / 'h.db'}",
            run_dir=tmp_path / "run",
            wrap_up=wrap_up,
        )
    )


def test_wrap_up_lets_open_question_be_answered(monkeypatch, tmp_path):
    s = run(monkeypatch, tmp_path, wrap_up=4)
    assert s["questions"] >= 1
    # the counter-question asked during wrap-up is not tracked, so the run
    # ends with nothing open instead of chasing its own tail
    assert s["open_questions"] == 0
    assert s["posts"] == 4 and s["reason"] == "max posts reached (+1 wrap-up)"


def test_wrap_up_zero_stops_exactly(monkeypatch, tmp_path):
    s = run(monkeypatch, tmp_path, wrap_up=0)
    assert s["posts"] == 3 and s["reason"] == "max posts reached"
