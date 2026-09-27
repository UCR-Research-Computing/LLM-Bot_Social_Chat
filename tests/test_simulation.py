import asyncio
import json

import pytest

from bot_social_network import settings
from bot_social_network.ai_client import MemoryNote, ModelError, Reply
from bot_social_network.database import Post
from bot_social_network.simulation import (
    BENCH_TURNS,
    MAX_FAILS,
    Simulation,
    TeamError,
    list_teams,
    load_team,
    resolve_team,
    save_team,
)

TEAM = [
    {"name": "Dan", "persona": "engineer", "model": "gemini-2.5-flash"},
    {"name": "Steve", "persona": "director", "model": "gemini-3.8-flash"},
    {"name": "Mike", "persona": "sysadmin", "model": "gemini-1.5-pro"},
]


class FakeAI:
    def __init__(self, fail_for=(), memory=None):
        self.fail_for = set(fail_for)
        self.memory = memory
        self.calls = []

    async def write_post(
        self, bot, others, recent, memories, questions=(), avoid=None, closing=False
    ):
        self.calls.append(bot.name)
        if bot.name in self.fail_for:
            raise ModelError("429 rate limited", bot.model)
        return Reply(f"{bot.name} says hi", bot.model, 10, 5, 0.001, 50, "STOP")

    async def form_memory(self, bot, recent):
        return self.memory, Reply("{}", "m", 5, 5, 0.0001)


def write_team(tmp_path, data=TEAM, name="t.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


# ---- teams -----------------------------------------------------------------


def test_load_team_upgrades_retired_models(tmp_path):
    bots, notes = load_team(write_team(tmp_path))
    assert [b["model"] for b in bots] == [
        "gemini-3.8-flash",
        "gemini-3.8-flash",
        "gemini-3.1-pro-preview",
    ]
    assert "Dan: gemini-2.5-flash -> gemini-3.8-flash" in notes
    assert len(notes) == 2


@pytest.mark.parametrize(
    "data, msg",
    [
        ([], "non-empty"),
        ([{"name": "A"}], "needs 'name' and 'persona'"),
        ([{"name": "A", "persona": "x"}, {"name": "A", "persona": "y"}], "duplicate"),
    ],
)
def test_load_team_validation(tmp_path, data, msg):
    with pytest.raises(TeamError, match=msg):
        load_team(write_team(tmp_path, data))


def test_bundled_teams_all_valid_and_current():
    teams = list_teams()
    assert len(teams) >= 8
    for name, path in teams:
        bots, notes = load_team(path)
        assert notes == [], f"{name} still pins a retired model: {notes}"
        for b in bots:
            info = settings.model_info(b["model"])
            if info.provider == "gemini":
                assert b["model"] in {m.id for m in settings.GEMINI_MODELS}, (name, b)


def test_resolve_team_by_name_and_user_override(tmp_path):
    assert resolve_team("default").name == "default.json"
    settings.USER_CONFIGS.mkdir(parents=True)
    (settings.USER_CONFIGS / "default.json").write_text(json.dumps(TEAM))
    assert resolve_team("default").parent == settings.USER_CONFIGS
    with pytest.raises(TeamError, match="not found"):
        resolve_team("nope")


def test_save_team_sanitizes_name(db):
    db.replace_team([{"name": "A", "persona": "p", "model": "m"}])
    p = save_team(db, "../evil name")
    assert p.parent == settings.USER_CONFIGS and p.name == "evil_name.json"


# ---- engine ----------------------------------------------------------------


def test_steps_post_and_track_cost(db, tmp_path):
    sim = Simulation(db, FakeAI(), deterministic=True, memory_every=0)
    sim.load_team(str(write_team(tmp_path)))
    for _ in range(3):
        asyncio.run(sim.step())
    posts = db.recent_posts()
    assert [p.sender for p in reversed(posts)] == ["Dan", "Steve", "Mike"]
    assert sim.stats.posts == 3 and sim.stats.cost_usd == pytest.approx(0.003)
    assert posts[0].tokens_in == 10 and posts[0].latency_ms == 50


def test_mentioned_bot_speaks_next(db, tmp_path):
    sim = Simulation(db, FakeAI(), seed=1, memory_every=0)
    sim.load_team(str(write_team(tmp_path)))
    bots = db.bots()
    recent = [Post(sender="Dan", content="What do you think, @Mike?")]
    picks = [sim.pick_speaker(bots, recent).name for _ in range(50)]
    assert picks.count("Mike") >= 30 and "Dan" not in picks


def test_owed_reply_beats_newest_mention(db, tmp_path):
    """Mike was asked first and never answered; he speaks before Steve (asked later)."""
    sim = Simulation(db, FakeAI(), seed=1, memory_every=0)
    sim.load_team(str(write_team(tmp_path)))
    bots = db.bots()
    recent = [  # newest first
        Post(sender="Dan", content="And @Steve, budget?"),
        Post(sender="Steve", content="thinking"),
        Post(sender="Dan", content="@Mike what's the storage plan?"),
    ]
    picks = [sim.pick_speaker(bots, recent).name for _ in range(40)]
    assert picks.count("Mike") >= 30


def test_multiword_mention_selects_target(db, tmp_path):
    team = [
        {
            "name": "Captain Eva Rostova",
            "persona": "captain",
            "model": "gemini-3.8-flash",
        },
        {"name": "Zeke", "persona": "ai", "model": "gemini-3.8-flash"},
        {"name": "Commander Jax", "persona": "xo", "model": "gemini-3.8-flash"},
    ]
    sim = Simulation(db, FakeAI(), seed=2, memory_every=0)
    sim.load_team(str(write_team(tmp_path, team)))
    recent = [Post(sender="Zeke", content="@Captain Eva Rostova, drive is ready.")]
    picks = [sim.pick_speaker(db.bots(), recent).name for _ in range(40)]
    assert picks.count("Captain Eva Rostova") >= 30


def test_never_same_speaker_twice(db, tmp_path):
    sim = Simulation(db, FakeAI(), seed=3, memory_every=0)
    sim.load_team(str(write_team(tmp_path)))

    async def go():
        for _ in range(20):
            await sim.step()

    asyncio.run(go())
    senders = [p.sender for p in reversed(db.recent_posts(20))]
    assert all(a != b for a, b in zip(senders, senders[1:]))


def test_failures_become_system_posts_then_bench(db, tmp_path):
    sim = Simulation(db, FakeAI(fail_for={"Dan"}), deterministic=True, memory_every=0)
    sim.load_team(str(write_team(tmp_path)))

    async def go():
        for _ in range(3 * MAX_FAILS):
            await sim.step()

    asyncio.run(go())
    errs = [p for p in db.recent_posts(50) if p.error]
    assert len(errs) == MAX_FAILS
    assert "benched" in errs[0].content and "429" in errs[0].error
    assert sim.state["Dan"].benched_until > sim.turn
    assert sim.state["Dan"].benched_until - sim.turn <= BENCH_TURNS


def test_memory_is_saved_when_worth_keeping(db, tmp_path):
    note = MemoryNote(worth_keeping=True, key="Steve", value="likes TPUs")
    sim = Simulation(db, FakeAI(memory=note), deterministic=True, memory_every=1)
    sim.load_team(str(write_team(tmp_path)))

    async def go():
        await sim.step()
        await sim.drain()

    asyncio.run(go())
    dan = db.bot("Dan")
    assert [(m.key, m.value) for m in db.memories(dan.id)] == [("Steve", "likes TPUs")]
    assert sim.stats.memories == 1


def test_memory_skipped_when_not_worth_keeping(db, tmp_path):
    note = MemoryNote(worth_keeping=False)
    sim = Simulation(db, FakeAI(memory=note), deterministic=True, memory_every=1)
    sim.load_team(str(write_team(tmp_path)))

    async def go():
        await sim.step()
        await sim.drain()

    asyncio.run(go())
    assert db.memories(db.bot("Dan").id) == []


def test_budget_stop(db, tmp_path):
    sim = Simulation(db, FakeAI(), deterministic=True, memory_every=0, budget_usd=0.002)
    sim.load_team(str(write_team(tmp_path)))
    asyncio.run(sim.step())
    assert not sim.over_budget
    asyncio.run(sim.step())
    assert sim.over_budget


def test_topic_injection(db, tmp_path):
    sim = Simulation(db, FakeAI(), memory_every=0)
    sim.load_team(str(write_team(tmp_path)))
    p = sim.inject_topic("  GPUs or TPUs?  ")
    assert p.sender == "SYSTEM" and p.content == "GPUs or TPUs?"


# ---- deterministic dynamics in the engine ------------------------------------


class ScriptAI(FakeAI):
    """Replies from a script: {bot: [text, text, ...]} consumed in order."""

    def __init__(self, script, memory=None):
        super().__init__(memory=memory)
        self.script = {k: list(v) for k, v in script.items()}
        self.seen = []  # (bot, questions, avoid)

    async def write_post(
        self, bot, others, recent, memories, questions=(), avoid=None, closing=False
    ):
        self.calls.append(bot.name)
        self.seen.append((bot.name, [q.question for q in questions], avoid))
        text = self.script[bot.name].pop(0) if self.script.get(bot.name) else "ok"
        return Reply(text, bot.model, 10, 5, 0.001, 50, "STOP")


def test_open_question_decides_next_speaker_and_is_passed_to_prompt(db, tmp_path):
    ai = ScriptAI(
        {
            "Dan": ["@Mike what is the storage plan for the new racks?"],
            "Mike": ["@Dan Ceph on NVMe for the storage plan."],
        }
    )
    sim = Simulation(db, ai, seed=3, memory_every=0)
    sim.load_team(str(write_team(tmp_path)))
    sim.pick_speaker = lambda bots, recent: next(b for b in bots if b.name == "Dan")
    asyncio.run(sim.step())
    assert sim.open_questions == 1
    del sim.pick_speaker  # back to the real scheduler
    picks = [sim.pick_speaker(db.bots(), db.recent_posts()).name for _ in range(40)]
    assert picks.count("Mike") >= 34  # ledger debtor goes first 95% of the time
    sim.rng.seed(0)
    asyncio.run(sim.step())
    name, questions, _ = ai.seen[-1]
    assert name == "Mike"
    assert questions == ["@Mike what is the storage plan for the new racks?"]
    assert sim.open_questions == 0
    assert sim.stats.questions == 1 and sim.stats.answered == 1


def test_repeat_is_regenerated_once(db, tmp_path):
    line = "We should put every GPU in rack four because it has the best cooling."
    ai = ScriptAI(
        {
            "Dan": [line],
            "Steve": [
                line.replace("best", "most"),
                "Storage is the real problem here.",
            ],
            "Mike": ["fine"],
        }
    )
    sim = Simulation(db, ai, seed=1, memory_every=0)
    sim.load_team(str(write_team(tmp_path)))
    order = iter(["Dan", "Steve"])

    def pick(bots, recent):
        want = next(order)
        return next(b for b in bots if b.name == want)

    sim.pick_speaker = pick
    asyncio.run(sim.step())
    post = asyncio.run(sim.step())
    assert post.content == "Storage is the real problem here."
    assert sim.stats.regenerated == 1
    assert ai.seen[-1][2] == line.replace("best", "most")  # the draft to avoid


def test_duplicate_memory_not_stored(db, tmp_path):
    note = MemoryNote(
        worth_keeping=True, key="Steve TPUs", value="Steve wants TPUs for training"
    )
    sim = Simulation(db, FakeAI(memory=note), seed=1, memory_every=0)
    sim.load_team(str(write_team(tmp_path)))
    bot = db.bots()[0]
    asyncio.run(sim.form_memory(bot))
    asyncio.run(sim.form_memory(bot))
    assert sim.stats.memories == 1 and sim.stats.dup_memories == 1
    assert [m.value for m in db.memories(bot.id)].count(
        "Steve wants TPUs for training"
    ) == 1


def test_relevant_memories_are_capped_and_ranked(db, tmp_path):
    from bot_social_network.simulation import MEMORY_SLOTS

    sim = Simulation(db, FakeAI(), seed=1, memory_every=0)
    sim.load_team(str(write_team(tmp_path)))
    bot = db.bots()[0]
    for i in range(30):
        db.add_memory(bot.id, f"note {i}", f"filler number {i}")
    db.add_memory(bot.id, "cooling", "rack four has liquid cooling")
    db.add_memory(bot.id, "late", "newest thing")
    recent = [Post(sender="Steve", content="does rack four have liquid cooling?")]
    mems = sim._relevant_memories(db.memories(bot.id, 200), recent)
    assert len(mems) == MEMORY_SLOTS
    assert "rack four has liquid cooling" in [m.value for m in mems]


def test_fair_share_spreads_turns(db, tmp_path):
    team = [{"name": n, "persona": "p", "model": "gemini-3.8-flash"} for n in "ABCD"]
    sim = Simulation(db, FakeAI(), seed=4, memory_every=0)
    sim.load_team(str(write_team(tmp_path, team)))
    for _ in range(40):
        asyncio.run(sim.step())
    counts = sorted(sim.stats.by_bot.values())
    assert counts[0] >= 8 and counts[-1] <= 12  # 40 posts over 4 bots, near even
