"""Headless runs for experiments: stream posts to the terminal, log everything."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from . import settings
from .ai_client import AIClient
from .database import Database, Post
from .logging_config import setup_logging, new_run_dir
from .simulation import Simulation
from .voice import Voice, voice_for

console = Console(highlight=False)

COLORS = [
    "cyan",
    "magenta",
    "green",
    "yellow",
    "blue",
    "red",
    "bright_cyan",
    "bright_magenta",
]


def color_for(name: str, names: list[str]) -> str:
    return COLORS[names.index(name) % len(COLORS)] if name in names else "white"


def print_post(post: Post, names: list[str]) -> None:
    if post.sender == "SYSTEM":
        style = "bold red" if post.error else "bold white"
        console.print(f"[{style}]SYSTEM[/] {post.content}")
        return
    meta = ""
    if post.latency_ms:
        meta = f" [dim]({post.model}, {post.latency_ms / 1000:.1f}s)[/]"
    console.print(
        f"[bold {color_for(post.sender or '', names)}]{post.sender}[/]{meta}\n  {post.content}"
    )


async def run_headless(
    team: str,
    max_posts: int | None,
    duration: int | None,
    topic: str | None,
    tts: bool,
    deterministic: bool,
    seed: int | None,
    delay: float,
    budget: float | None,
    keep_posts: bool,
    db_url: str | None = None,
    run_dir: Path | None = None,
    wrap_up: int = 4,
) -> dict[str, Any]:
    run = setup_logging(run_dir or new_run_dir())
    db = Database(db_url)
    ai = AIClient()
    sim = Simulation(db, ai, deterministic=deterministic, seed=seed, budget_usd=budget)
    notes = sim.load_team(team) if not keep_posts else []
    if keep_posts and not db.bots():
        notes = sim.load_team(team)
    names = [b.name for b in db.bots()]
    console.print(f"[dim]Run folder: {run}[/]")
    console.print(f"[bold]Team:[/] {team} ({len(names)} bots: {', '.join(names)})")
    for n in notes:
        console.print(f"[yellow]Model upgraded[/] {n}")
    if sim.team_info.description:
        console.print(f"[dim]{sim.team_info.description}[/]")
    topic = topic or (None if keep_posts else sim.team_info.topic or None)
    if topic:
        print_post(sim.inject_topic(topic), names)

    voice = Voice(ai.gemini) if tts else None

    async def speak(post: Post) -> None:
        """Read one post aloud; the loop waits for it, so voice never falls behind."""
        assert voice is not None
        bot = db.bot(post.sender or "")
        try:
            sp = await voice.synthesize(
                post.content,
                voice_for(post.sender or "", bot.voice if bot else None),
                run / "audio" / f"post_{post.id}.wav",
            )
            sim.stats.tts_cost_usd += sp.cost_usd
            await asyncio.to_thread(voice.play, sp.path)
        except Exception as e:
            console.print(f"[red]TTS failed:[/] {e}")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            pass

    start = time.monotonic()
    reason = "done"
    extra = 0
    try:
        while not stop.is_set():
            if max_posts and sim.stats.posts >= max_posts:
                # Wrap-up: let bots that still owe an answer give it, up to
                # `wrap_up` extra posts, so the run does not end mid-question.
                if not sim.open_questions or extra >= wrap_up:
                    reason = "max posts reached" + (
                        f" (+{extra} wrap-up)" if extra else ""
                    )
                    break
                sim.closing = True
                extra += 1
            if duration and time.monotonic() - start >= duration:
                reason = "duration reached"
                break
            if sim.over_budget:
                reason = f"budget ${budget:.2f} reached"
                break
            post = await sim.step()
            if post is None:
                await asyncio.sleep(1)
                continue
            print_post(post, names)
            if voice and post.sender != "SYSTEM":
                await speak(post)
            if delay:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
        else:
            reason = "interrupted"
    finally:
        if voice:
            voice.stop()
        await sim.drain(timeout=20)
        await sim.shutdown()
        db.close()

    s = sim.stats
    summary = {
        "reason": reason,
        "posts": s.posts,
        "errors": s.errors,
        "memories": s.memories,
        "tokens_in": s.tokens_in,
        "tokens_out": s.tokens_out,
        "cost_usd": round(s.cost_usd, 4),
        "tts_cost_usd": round(s.tts_cost_usd, 4),
        "questions": s.questions,
        "answered": s.answered,
        "open_questions": sim.open_questions,
        "regenerated": s.regenerated,
        "sanitized": s.fixes,
        "duplicate_memories": s.dup_memories,
        "seconds": round(time.monotonic() - start, 1),
        "run_dir": str(run),
    }
    logging.info("run finished", extra={"event": "sim.end", **summary})
    console.print(
        f"\n[bold]Stopped:[/] {reason}. {s.posts} posts, {s.errors} errors, "
        f"{s.memories} new memories, {s.tokens_in + s.tokens_out:,} tokens, "
        f"about ${s.cost_usd + s.tts_cost_usd:.4f}. Log: {run / 'simulation.jsonl'}"
    )
    console.print(
        f"[dim]Questions: {s.questions} asked, {s.answered} answered, "
        f"{sim.open_questions} open. Repeats regenerated: {s.regenerated}. "
        f"Replies cleaned up: {s.fixes}. Duplicate memories skipped: "
        f"{s.dup_memories}.[/]"
    )
    return summary


def main_headless(args: Any) -> int:
    if not args.max_posts and not args.duration:
        console.print("[red]Set --max-posts or --duration.[/]")
        return 2
    asyncio.run(
        run_headless(
            team=args.team,
            max_posts=args.max_posts,
            duration=args.duration,
            topic=args.topic,
            tts=args.tts,
            deterministic=args.deterministic,
            seed=args.seed,
            delay=args.delay,
            budget=args.budget,
            keep_posts=args.keep,
            wrap_up=args.wrap_up,
        )
    )
    return 0


__all__ = ["run_headless", "main_headless", "settings"]
