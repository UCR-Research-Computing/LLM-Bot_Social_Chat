"""The simulation engine shared by the TUI and headless runs.

The model only writes the words. Everything around it is deterministic code from
dynamics.py:

- Turn-taking: a ledger of open @mention questions is a priority queue; the bot
  with the oldest unanswered question speaks next. Otherwise a bot owed a reply
  (mentioned since it last spoke), otherwise the fair-share scheduler (fewest
  words spoken so far, then silent longest) picks, with a little randomness. Never the same
  bot twice in a row.
- Every reply is sanitized (fabricated lines, bad @names, length) and checked for
  near-duplicates (3-word shingles, Jaccard); a repeat is regenerated once.
- Memories shown to a bot are chosen by BM25 relevance to the recent chat, and
  near-duplicate memories are not stored.
- A failed model call is recorded as a SYSTEM post with the reason and never
  crashes the loop; a bot that fails 3 times in a row is benched for 5 turns.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import settings
from .ai_client import AIClient, ModelError, gather_limited, inbox
from .database import Bot, Database, Memory, Post
from .dynamics import (
    FairScheduler,
    ObligationLedger,
    RepetitionGuard,
    is_duplicate_memory,
    select_memories,
)

log = logging.getLogger(__name__)

MAX_FAILS = 3
BENCH_TURNS = 5
MEMORY_SLOTS = 8  # memories shown per post, picked by BM25


# --------------------------------------------------------------------------- teams


class TeamError(ValueError):
    pass


def list_teams() -> list[tuple[str, Path]]:
    """(name, path) for user teams first, then bundled ones not overridden."""
    seen: dict[str, Path] = {}
    for d in (settings.USER_CONFIGS, settings.BUNDLED_CONFIGS):
        if d.is_dir():
            for p in sorted(d.glob("*.json")):
                seen.setdefault(p.name, p)
    return sorted(seen.items())


def resolve_team(name_or_path: str) -> Path:
    p = Path(name_or_path).expanduser()
    if p.is_file():
        return p
    name = p.name if p.suffix == ".json" else p.name + ".json"
    for d in (settings.USER_CONFIGS, settings.BUNDLED_CONFIGS, Path("configs")):
        cand = d / name
        if cand.is_file():
            return cand
    raise TeamError(
        f"Team '{name_or_path}' not found. Available: "
        + ", ".join(n for n, _ in list_teams())
    )


def load_team(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate a team file. Returns (bots, notes) where notes list model upgrades."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise TeamError(f"{path.name}: {e}") from e
    if isinstance(data, dict):
        data = data.get("bots", [])
    if not isinstance(data, list) or not data:
        raise TeamError(f"{path.name}: expected a non-empty list of bots")
    bots, notes, names = [], [], set()
    for i, b in enumerate(data, 1):
        if not isinstance(b, dict) or not b.get("name") or not b.get("persona"):
            raise TeamError(f"{path.name}: bot #{i} needs 'name' and 'persona'")
        name = str(b["name"]).strip()
        if name in names:
            raise TeamError(f"{path.name}: duplicate bot name '{name}'")
        names.add(name)
        old = b.get("model")
        new = settings.upgrade_model(old)
        if old and old != new:
            notes.append(f"{name}: {old} -> {new}")
        mems = [
            {"key": str(m["key"]), "value": str(m["value"])}
            for m in b.get("memories") or []
            if isinstance(m, dict) and m.get("key") and m.get("value")
        ]
        temp = b.get("temperature")
        bots.append(
            {
                "name": name,
                "persona": str(b["persona"]),
                "model": new,
                "voice": b.get("voice"),
                "temperature": float(temp) if temp is not None else None,
                "memories": mems,
            }
        )
    return bots, notes


def save_team(db: Database, name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "team"
    settings.ensure_dirs()
    path = settings.USER_CONFIGS / f"{safe.removesuffix('.json')}.json"
    path.write_text(json.dumps(db.export_team(), indent=2) + "\n")
    return path


# --------------------------------------------------------------------------- engine


@dataclass
class BotState:
    fails: int = 0
    benched_until: int = 0
    last_turn: int = -1


@dataclass
class Stats:
    posts: int = 0
    errors: int = 0
    memories: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    tts_cost_usd: float = 0.0
    by_bot: dict[str, int] = field(default_factory=dict)
    regenerated: int = 0  # replies redone because they repeated an earlier post
    fixes: int = 0  # replies the sanitizer changed
    questions: int = 0
    answered: int = 0
    dup_memories: int = 0


class Simulation:
    def __init__(
        self,
        db: Database,
        ai: AIClient | None = None,
        deterministic: bool = False,
        seed: int | None = None,
        memory_every: int = 3,
        budget_usd: float | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        self.db = db
        self.ai = ai or AIClient()
        self.deterministic = deterministic
        self.rng = random.Random(seed)
        self.memory_every = max(0, memory_every)
        self.budget_usd = budget_usd
        self.on_event = on_event or (lambda _e, _d: None)
        self.turn = 0
        self.state: dict[str, BotState] = {}
        self.stats = Stats()
        self.background: set[asyncio.Task[Any]] = set()
        self._rr = 0
        self.ledger = ObligationLedger()
        self.fair = FairScheduler()
        self.repeats = RepetitionGuard()
        # Wrap-up mode: bots answer what they owe and close; new questions are
        # not tracked, so the run can end with nothing left hanging.
        self.closing = False

    # ---- setup ---------------------------------------------------------------
    def load_team(self, name_or_path: str) -> list[str]:
        path = resolve_team(name_or_path)
        bots, notes = load_team(path)
        self.db.replace_team(bots)
        self.state.clear()
        self.turn = 0
        self.reset_dynamics()
        log.info(
            "team loaded",
            extra={"event": "config.load.success", "config_filename": path.name},
        )
        return notes

    def reset_dynamics(self) -> None:
        """Forget open questions, fair-share credit and repetition history (for a
        new team or a cleared feed)."""
        self.ledger = ObligationLedger()
        self.fair = FairScheduler()
        self.repeats = RepetitionGuard()

    @property
    def open_questions(self) -> int:
        return len(self.ledger)

    def inject_topic(self, topic: str) -> Post:
        post = self.db.add_post(content=topic.strip(), sender="SYSTEM")
        log.info("topic", extra={"event": "topic.injected", "topic": topic})
        self.on_event("post", {"post": post})
        return post

    @property
    def over_budget(self) -> bool:
        return (
            self.budget_usd is not None
            and self.stats.cost_usd + self.stats.tts_cost_usd >= self.budget_usd
        )

    # ---- turn-taking ---------------------------------------------------------
    def pick_speaker(self, bots: list[Bot], recent: list[Post]) -> Bot | None:
        active = [
            b
            for b in bots
            if self.state.setdefault(b.name, BotState()).benched_until <= self.turn
        ]
        if not active:
            return None
        if self.deterministic:
            bot = active[self._rr % len(active)]
            self._rr += 1
            return bot
        last_sender = recent[0].sender if recent else None
        pool = [b for b in active if b.name != last_sender] or active
        by_name = {b.name: b for b in pool}
        self.fair.sync(b.name for b in bots)
        # 1. The oldest unanswered question in the ledger (a priority queue).
        for name, _turn in self.ledger.debtors():
            if name in by_name and self.rng.random() < 0.95:
                return by_name[name]
        if recent:
            # 2. Whoever was @mentioned since they last spoke (no question mark
            #    needed), oldest mention first.
            all_names = [b.name for b in bots]
            owed: list[tuple[int, Bot]] = []
            for b in pool:
                waiting = inbox(recent, b.name, all_names)
                if waiting:
                    age = next(i for i, p in enumerate(recent) if p is waiting[0])
                    owed.append((age, b))
            if owed and self.rng.random() < 0.9:
                owed.sort(key=lambda t: -t[0])  # largest index = oldest mention
                return owed[0][1]
        # 3. Fair share: fewest words spoken so far, then silent longest. Pick
        #    between the two fairest so the order is not rigid.
        ranked = self.fair.ranked([b.name for b in pool])
        top = ranked[:2] if len(ranked) > 1 else ranked
        weights = [2.0, 1.0][: len(top)]
        return by_name[self.rng.choices(top, weights=weights, k=1)[0]]

    # ---- one step ------------------------------------------------------------
    async def step(self) -> Post | None:
        bots = await asyncio.to_thread(self.db.bots)
        if not bots:
            return None
        recent = await asyncio.to_thread(self.db.recent_posts, 50)
        bot = self.pick_speaker(bots, recent)
        if bot is None:
            self.turn += 1
            return None
        st = self.state.setdefault(bot.name, BotState())
        others = [b.name for b in bots if b.name != bot.name]
        names = [b.name for b in bots]
        memories = self._relevant_memories(
            await asyncio.to_thread(self.db.memories, bot.id, 200), recent
        )
        owed = self.ledger.owed_by(bot.name)
        self.turn += 1
        try:
            reply = await self.ai.write_post(
                bot, others, recent, memories, questions=owed, closing=self.closing
            )
            score, _who = self.repeats.check(reply.text)
            if score >= self.repeats.threshold:
                # Near-duplicate of a recent post: regenerate once, keep the
                # less repetitive of the two.
                first = reply
                again = await self.ai.write_post(
                    bot,
                    others,
                    recent,
                    memories,
                    questions=owed,
                    avoid=first.text,
                    closing=self.closing,
                )
                self.stats.cost_usd += first.cost_usd
                self.stats.tokens_in += first.tokens_in
                self.stats.tokens_out += first.tokens_out
                self.stats.regenerated += 1
                reply = again if self.repeats.check(again.text)[0] < score else first
                log.info(
                    "regenerated repeat",
                    extra={
                        "event": "post.regenerated",
                        "bot_name": bot.name,
                        "similarity": round(score, 3),
                    },
                )
        except ModelError as e:
            st.fails += 1
            self.stats.errors += 1
            benched = st.fails >= MAX_FAILS
            if benched:
                st.benched_until = self.turn + BENCH_TURNS
                st.fails = 0
            msg = f"{bot.name} could not post ({e.reason})" + (
                f"; benched for {BENCH_TURNS} turns" if benched else ""
            )
            post = await asyncio.to_thread(
                self.db.add_post,
                content=msg,
                sender="SYSTEM",
                model=bot.model,
                error=e.reason,
            )
            log.warning(
                msg,
                extra={
                    "event": "post.generation.fail",
                    "bot_name": bot.name,
                    "error": e.reason,
                },
            )
            self.on_event("post", {"post": post})
            return post

        st.fails = 0
        st.last_turn = self.turn
        self._account(bot.name, reply.text, names)
        if reply.fixes:
            self.stats.fixes += 1
        post = await asyncio.to_thread(
            self.db.add_post,
            content=reply.text,
            sender=bot.name,
            bot_id=bot.id,
            model=reply.model,
            tokens_in=reply.tokens_in,
            tokens_out=reply.tokens_out,
            cost_usd=reply.cost_usd,
            latency_ms=reply.latency_ms,
        )
        s = self.stats
        s.posts += 1
        s.tokens_in += reply.tokens_in
        s.tokens_out += reply.tokens_out
        s.cost_usd += reply.cost_usd
        s.by_bot[bot.name] = s.by_bot.get(bot.name, 0) + 1
        log.info(
            "post",
            extra={
                "event": "post.generated",
                "bot_name": bot.name,
                "bot_model": reply.model,
                "post_content": reply.text,
                "tokens_in": reply.tokens_in,
                "tokens_out": reply.tokens_out,
                "cost_usd": reply.cost_usd,
                "latency_ms": reply.latency_ms,
                "fixes": reply.fixes,
                "open_questions": len(self.ledger),
            },
        )
        # Register questions asked in this post now that it has a post id.
        if not self.closing:
            new = self.ledger.record(post.id, bot.name, post.content, names, self.turn)
            self.stats.questions += len(new)
        self.on_event("post", {"post": post})
        if self.memory_every and s.by_bot[bot.name] % self.memory_every == 0:
            self._spawn(self.form_memory(bot))
        return post

    def _account(self, speaker: str, text: str, names: list[str]) -> None:
        """Deterministic bookkeeping for an accepted post (before it is stored)."""
        done = self.ledger.resolve(speaker, text, names, self.turn)
        self.stats.answered += len(done)
        self.ledger.expire(self.turn)
        self.fair.spend(speaker, len(text.split()))
        self.repeats.add(speaker, text)

    def _relevant_memories(
        self, memories: list[Memory], recent: list[Post]
    ) -> list[Memory]:
        """Up to MEMORY_SLOTS memories: the persona seeds, then BM25 matches for
        the last few posts, then the newest."""
        query = " ".join(p.content for p in recent[:4] if not p.error)
        idx = select_memories(
            [(m.key, m.value) for m in memories], query, k=MEMORY_SLOTS
        )
        return [memories[i] for i in idx]

    async def form_memory(self, bot: Bot) -> None:
        recent = await asyncio.to_thread(self.db.recent_posts, 6)
        try:
            note, reply = await self.ai.form_memory(bot, recent)
        except ModelError as e:
            log.info(
                "memory failed",
                extra={
                    "event": "memory.form.fail",
                    "bot_name": bot.name,
                    "error": e.reason,
                },
            )
            return
        if reply:
            self.stats.cost_usd += reply.cost_usd
        if note and note.worth_keeping and note.key.strip() and note.value.strip():
            existing = await asyncio.to_thread(self.db.memories, bot.id, 200)
            if is_duplicate_memory(
                (note.key, note.value), [(m.key, m.value) for m in existing]
            ):
                self.stats.dup_memories += 1
                log.info(
                    "duplicate memory skipped",
                    extra={
                        "event": "memory.form.duplicate",
                        "bot_name": bot.name,
                        "memory_key": note.key,
                    },
                )
                return
            await asyncio.to_thread(
                self.db.add_memory, bot.id, note.key.strip(), note.value.strip()
            )
            self.stats.memories += 1
            log.info(
                "memory",
                extra={
                    "event": "memory.form.success",
                    "bot_name": bot.name,
                    "memory_key": note.key,
                    "memory_value": note.value,
                },
            )
            self.on_event(
                "memory", {"bot": bot.name, "key": note.key, "value": note.value}
            )

    def _spawn(self, coro: Any) -> None:
        t = asyncio.create_task(coro)
        self.background.add(t)
        t.add_done_callback(self.background.discard)

    async def drain(self, timeout: float = 20) -> None:
        if self.background:
            await asyncio.wait(list(self.background), timeout=timeout)

    async def shutdown(self) -> None:
        for t in list(self.background):
            t.cancel()
        await asyncio.gather(*self.background, return_exceptions=True)


async def check_models(ai: AIClient, models: list[str]) -> dict[str, str]:
    """Tiny live call per model; returns {model: 'ok' | reason}."""

    async def one(m: str) -> str:
        fake = Bot(name="Probe", persona="A terse tester.", model=m)
        try:
            r = await ai.write_post(fake, [], [], [])
            return f"ok ({r.latency_ms} ms)"
        except ModelError as e:
            return e.reason

    gem = [m for m in models if settings.model_info(m).provider == "gemini"]
    loc = [m for m in models if m not in gem]
    # Gemini in parallel; local models one at a time (each loads into RAM, and
    # parallel loads on a laptop can swap for minutes).
    res_g = await gather_limited([one(m) for m in gem], limit=4)
    res_l = [await one(m) for m in loc]
    res = dict(zip(gem + loc, res_g + res_l))
    return {m: (r if isinstance(r, str) else str(r)) for m, r in res.items()}
