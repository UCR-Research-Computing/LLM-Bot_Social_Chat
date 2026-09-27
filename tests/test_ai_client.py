import asyncio
from types import SimpleNamespace

import httpx
import pytest

from bot_social_network import settings
from bot_social_network.ai_client import (
    AIClient,
    GeminiClient,
    MemoryNote,
    ModelError,
    OllamaClient,
    clean_post,
    conversation_prompt,
    system_prompt,
)
from bot_social_network.database import Bot, Memory, Post


@pytest.fixture
def bot():
    return Bot(name="Dan", persona="An HPC engineer.", model="gemini-3.8-flash")


# ---- prompts ---------------------------------------------------------------


def test_system_prompt_has_persona_members_and_memories(bot):
    s = system_prompt(bot, ["Steve", "Mike"], [Memory(key="goal", value="uptime")])
    assert "You are Dan" in s and "An HPC engineer." in s
    assert "@Steve, @Mike" in s
    assert "- goal: uptime" in s


def test_conversation_prompt_orders_oldest_first_and_skips_errors(bot):
    posts = [  # newest first, as stored
        Post(sender="Steve", content="second"),
        Post(sender="SYSTEM", content="Mike could not post", error="429"),
        Post(sender="Mike", content="first"),
    ]
    p = conversation_prompt(posts, "Dan")
    assert p.index("@Mike: first") < p.index("@Steve: second")
    assert "could not post" not in p
    assert p.endswith("Write Dan's next post.")


NAMES = ["Captain Eva Rostova", "Commander Jax", "Zeke", "Dan"]


@pytest.mark.parametrize(
    "text, want",
    [
        ("@Captain Eva Rostova, status?", ["Captain Eva Rostova"]),
        ("@Zeke @Commander Jax thoughts?", ["Zeke", "Commander Jax"]),
        ("@zeke lower case works", ["Zeke"]),
        ("@Danny is not Dan", []),
        ("email me at a@Dan.com", ["Dan"]),  # boundary is '.', acceptable
        ("no mentions here", []),
        ("@Zeke and again @Zeke", ["Zeke"]),
    ],
)
def test_mentions_handles_multiword_names(text, want):
    from bot_social_network.ai_client import mentions

    assert mentions(text, NAMES) == want


def test_inbox_collects_mentions_since_last_post():
    from bot_social_network.ai_client import inbox

    posts = [  # newest first
        Post(sender="Commander Jax", content="@Zeke how long is the flush?"),
        Post(sender="Dan", content="unrelated"),
        Post(sender="Captain Eva Rostova", content="@Zeke check the valves"),
        Post(sender="Zeke", content="valves recalibrated"),
        Post(sender="Dan", content="@Zeke old question already answered"),
    ]
    got = inbox(posts, "Zeke", NAMES)
    assert [p.sender for p in got] == ["Captain Eva Rostova", "Commander Jax"]


def test_prompt_tells_target_who_is_waiting(bot):
    posts = [
        Post(sender="Steve", content="@Dan can the racks take 8:1 GPU density?"),
        Post(sender="Mike", content="Storage is the bottleneck."),
    ]
    p = conversation_prompt(posts, "Dan", ["Dan", "Steve", "Mike"])
    assert "Addressed to you (Dan) since you last spoke:" in p
    assert "- @Steve: @Dan can the racks take 8:1 GPU density?" in p
    assert "Reply to @Steve first" in p


def test_prompt_without_mention_points_at_latest_post(bot):
    posts = [Post(sender="Mike", content="Storage is the bottleneck.")]
    p = conversation_prompt(posts, "Dan", ["Dan", "Steve", "Mike"])
    assert "Addressed to you" not in p
    assert "The latest post is from @Mike" in p


def test_conversation_prompt_empty_chat(bot):
    assert "chat is empty" in conversation_prompt([], "Dan")


@pytest.mark.parametrize(
    "raw, want",
    [
        ("Dan: hello @Steve", "hello @Steve"),
        ("@Dan: hi", "hi"),
        ('"quoted"', "quoted"),
        ("  plain  ", "plain"),
        ("Daniel: not me", "Daniel: not me"),
    ],
)
def test_clean_post(raw, want):
    assert clean_post(raw, "Dan") == want


# ---- gemini ----------------------------------------------------------------


class FakeModels:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    async def generate_content(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return self.resp


def fake_resp(text, finish="STOP", tin=100, out=20, think=0, parsed=None):
    return SimpleNamespace(
        text=text,
        parsed=parsed,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
        usage_metadata=SimpleNamespace(
            prompt_token_count=tin,
            candidates_token_count=out,
            thoughts_token_count=think,
        ),
        prompt_feedback=None,
    )


def gemini_with(models):
    g = GeminiClient(api_key="x")
    g._client = SimpleNamespace(aio=SimpleNamespace(models=models))
    return g


def test_gemini_reply_cost_and_config(bot):
    models = FakeModels(
        fake_resp("Hi @Steve", tin=1_000_000, out=100_000, think=100_000)
    )
    ai = AIClient(gemini=gemini_with(models))
    r = asyncio.run(ai.write_post(bot, ["Steve"], [], []))
    assert r.text == "Hi @Steve"
    # 3.8 Flash: $0.75 in, $3.75 out (thinking billed as output)
    assert r.cost_usd == pytest.approx(0.75 + 0.2 * 3.75)
    cfg = models.calls[0]["config"]
    assert cfg.thinking_config.thinking_budget == 0  # Flash: no thinking for chat
    assert "You are Dan" in cfg.system_instruction
    assert cfg.automatic_function_calling.disable is True


def test_pro_uses_low_thinking_and_extra_room(bot):
    bot.model = "gemini-3.1-pro-preview"
    models = FakeModels(fake_resp("ok"))
    asyncio.run(AIClient(gemini=gemini_with(models)).write_post(bot, [], [], []))
    cfg = models.calls[0]["config"]
    assert cfg.thinking_config.thinking_level.name == "LOW"
    assert cfg.max_output_tokens > 2000


def test_gemma_uses_minimal_thinking(bot):
    bot.model = "gemma-4-31b-it"
    models = FakeModels(fake_resp("ok"))
    asyncio.run(AIClient(gemini=gemini_with(models)).write_post(bot, [], [], []))
    assert models.calls[0]["config"].thinking_config.thinking_level.name == "MINIMAL"


class SeqModels(FakeModels):
    def __init__(self, resps):
        super().__init__()
        self.resps = list(resps)

    async def generate_content(self, **kw):
        self.calls.append(kw)
        return self.resps.pop(0)


def test_truncated_by_thinking_retries_with_more_room(bot):
    # Measured on 3.8 Flash: thinking used ~380 of 400 tokens, reply cut mid-sentence.
    cut = fake_resp(
        "Mortal monarchs love their", finish="MAX_TOKENS", out=11, think=385
    )
    full = fake_resp("Mortal monarchs love their seals.", out=20, think=300)
    models = SeqModels([cut, full])
    r = asyncio.run(AIClient(gemini=gemini_with(models)).write_post(bot, [], [], []))
    assert r.text == "Mortal monarchs love their seals."
    assert len(models.calls) == 2
    first, second = (c["config"].max_output_tokens for c in models.calls)
    assert second > first
    assert r.tokens_out == 11 + 385 + 20 + 300  # both attempts billed


def test_default_flash_reserves_thinking_room(bot):
    models = FakeModels(fake_resp("ok"))
    asyncio.run(AIClient(gemini=gemini_with(models)).write_post(bot, [], [], []))
    assert models.calls[0]["config"].max_output_tokens >= 1500


def test_empty_reply_raises_with_finish_reason(bot):
    ai = AIClient(gemini=gemini_with(FakeModels(fake_resp("", finish="SAFETY"))))
    with pytest.raises(ModelError, match="SAFETY"):
        asyncio.run(ai.write_post(bot, [], [], []))


def test_api_error_becomes_short_reason(bot):
    exc = Exception("quota")
    exc.code = 429  # type: ignore[attr-defined]
    ai = AIClient(gemini=gemini_with(FakeModels(exc=exc)))
    with pytest.raises(ModelError) as e:
        asyncio.run(ai.write_post(bot, [], [], []))
    assert "429 rate limited" in e.value.reason


def test_missing_key_is_a_clear_error(bot, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    ai = AIClient(gemini=GeminiClient())
    assert not ai.gemini.configured
    with pytest.raises(ModelError, match="GEMINI_API_KEY is not set"):
        asyncio.run(ai.write_post(bot, [], [], []))


def test_structured_memory(bot):
    note = MemoryNote(worth_keeping=True, key="Steve", value="wants TPUs")
    models = FakeModels(fake_resp('{"x":1}', parsed=note))
    ai = AIClient(gemini=gemini_with(models))
    got, reply = asyncio.run(
        ai.form_memory(bot, [Post(sender="Steve", content="TPUs!")])
    )
    assert got == note and reply is not None
    cfg = models.calls[0]["config"]
    assert models.calls[0]["model"] == settings.MEMORY_MODEL
    assert cfg.response_mime_type == "application/json"


def test_client_created_once_across_threads(monkeypatch):
    import threading
    import time

    from google import genai

    made = []

    class Slow:
        def __init__(self, **kw):
            time.sleep(0.05)
            made.append(self)

    monkeypatch.setattr(genai, "Client", Slow)
    g = GeminiClient(api_key="k")
    got = []
    ts = [threading.Thread(target=lambda: got.append(g.client())) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(made) == 1 and all(x is made[0] for x in got)


# ---- ollama ----------------------------------------------------------------


def ollama_with(handler):
    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient

    class C(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **kw):
            kw["transport"] = transport
            super().__init__(*a, **kw)

    return C


def test_ollama_chat(monkeypatch, bot):
    bot.model = "gemma4:e4b"
    seen = {}

    def handler(req):
        seen["body"] = req.content
        return httpx.Response(
            200,
            json={
                "message": {"content": "Dan: local hello"},
                "prompt_eval_count": 50,
                "eval_count": 7,
                "done_reason": "stop",
            },
        )

    monkeypatch.setattr(httpx, "AsyncClient", ollama_with(handler))
    r = asyncio.run(
        AIClient(ollama=OllamaClient("http://x")).write_post(bot, [], [], [])
    )
    assert r.text == "local hello" and r.cost_usd == 0 and r.tokens_out == 7
    assert b'"role":"system"' in seen["body"].replace(b" ", b"")


def test_ollama_retries_without_think_flag(monkeypatch, bot):
    bot.model = "llama3.2"
    calls = []

    def handler(req):
        calls.append(req.content)
        if b'"think"' in req.content:
            return httpx.Response(400, text='{"error":"does not support think"}')
        return httpx.Response(200, json={"message": {"content": "ok"}})

    monkeypatch.setattr(httpx, "AsyncClient", ollama_with(handler))
    r = asyncio.run(
        AIClient(ollama=OllamaClient("http://x")).write_post(bot, [], [], [])
    )
    assert r.text == "ok" and len(calls) == 2


def test_ollama_missing_model(monkeypatch, bot):
    bot.model = "nope"
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        ollama_with(lambda r: httpx.Response(404, text="not found")),
    )
    with pytest.raises(ModelError, match="ollama pull nope"):
        asyncio.run(
            AIClient(ollama=OllamaClient("http://x")).write_post(bot, [], [], [])
        )


def test_ollama_down(monkeypatch, bot):
    bot.model = "gemma4:e4b"

    def handler(req):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "AsyncClient", ollama_with(handler))
    with pytest.raises(ModelError, match="not running"):
        asyncio.run(
            AIClient(ollama=OllamaClient("http://x")).write_post(bot, [], [], [])
        )
