"""Command line: `bot-social-network [run|headless|teams|models|analyze|doctor]`."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from importlib.metadata import PackageNotFoundError, version

from . import settings


def _version() -> str:
    try:
        return version("bot-social-network")
    except PackageNotFoundError:
        return "dev"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bot-social-network",
        description="A group chat of AI bots with personas, memories and voices. "
        "Gemini 3.x and local Ollama models.",
        epilog=(
            "examples:\n"
            "  bot-social-network                         open the TUI with the default team\n"
            "  bot-social-network run --team fantasy_tavern --autostart --tts\n"
            "  bot-social-network headless --team ai_philosophy_club --max-posts 20 \\\n"
            '      --topic "Is memory identity?" --budget 0.10\n'
            "  bot-social-network teams                   list teams\n"
            "  bot-social-network models --check          test every model live\n"
            "  bot-social-network analyze latest          HTML report of the last run\n\n"
            f"key:  GEMINI_API_KEY in {settings.CONFIG_DIR / '.env'}\n"
            f"data: {settings.DATA_DIR}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="interactive TUI (default)")
    _common(r)
    r.add_argument("--autostart", action="store_true", help="start posting immediately")
    r.add_argument(
        "--interval", type=float, default=8.0, help="seconds between posts (default 8)"
    )
    r.add_argument(
        "--clear-db", action="store_true", help="reload the team and clear posts"
    )

    h = sub.add_parser("headless", help="run without the TUI and print posts")
    _common(h)
    h.add_argument("--max-posts", type=int, help="stop after N bot posts")
    h.add_argument("--duration", type=int, help="stop after N seconds")
    h.add_argument(
        "--delay", type=float, default=0.0, help="pause between posts in seconds"
    )
    h.add_argument(
        "--deterministic",
        action="store_true",
        help="round-robin speakers (reproducible)",
    )
    h.add_argument("--seed", type=int, help="random seed for speaker choice")
    h.add_argument(
        "--keep",
        action="store_true",
        help="continue the current chat instead of reloading the team",
    )

    sub.add_parser("teams", help="list bundled and saved teams")

    m = sub.add_parser("models", help="list models (and optionally test them live)")
    m.add_argument(
        "--check", action="store_true", help="one tiny live call per Gemini model"
    )
    m.add_argument(
        "--ollama",
        nargs="*",
        metavar="MODEL",
        help="with --check, also test these local models (no names = all; slow, "
        "each one loads into RAM)",
    )

    a = sub.add_parser("analyze", help="HTML report from a run log")
    a.add_argument(
        "log",
        nargs="?",
        default="latest",
        help="simulation.jsonl, run folder, or 'latest'",
    )
    a.add_argument("-o", "--output", help="output HTML path (default: next to the log)")

    sub.add_parser("doctor", help="check key, models, Ollama, audio and data paths")
    return p


def _common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--team",
        "--config",
        dest="team",
        default="default",
        help="team name or JSON path",
    )
    sp.add_argument("--topic", help="opening topic posted as SYSTEM")
    sp.add_argument("--tts", action="store_true", help="speak posts with Gemini TTS")
    sp.add_argument(
        "--budget",
        type=float,
        help="stop/pause when estimated spend reaches this many USD",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cmd = args.cmd or "run"
    if cmd == "run":
        from .tui import BotSocialApp

        BotSocialApp(
            team=getattr(args, "team", "default"),
            autostart=getattr(args, "autostart", False),
            clear_db=getattr(args, "clear_db", False),
            tts=getattr(args, "tts", False),
            topic=getattr(args, "topic", None),
            interval=getattr(args, "interval", 8.0),
            budget=getattr(args, "budget", None),
        ).run()
        return 0
    if cmd == "headless":
        from .headless import main_headless

        return main_headless(args)
    if cmd == "teams":
        return _teams()
    if cmd == "models":
        return _models(args.check, args.ollama)
    if cmd == "analyze":
        from .analyzer import analyze_cli

        return analyze_cli(args.log, args.output)
    if cmd == "doctor":
        return _doctor()
    return 1


def _teams() -> int:
    from rich.console import Console
    from rich.table import Table

    from .simulation import TeamError, list_teams, load_team

    t = Table("team", "bots", "models", "source")
    for name, path in list_teams():
        try:
            bots, _ = load_team(path)
            models = sorted({b["model"] for b in bots})
            t.add_row(
                name.removesuffix(".json"),
                ", ".join(b["name"] for b in bots),
                ", ".join(models),
                "saved" if path.parent == settings.USER_CONFIGS else "bundled",
            )
        except TeamError as e:
            t.add_row(name, f"[red]{e}[/]", "", "")
    Console().print(t)
    return 0


def _models(check: bool, ollama_check: list[str] | None = None) -> int:
    from rich.console import Console
    from rich.table import Table

    from .ai_client import AIClient
    from .simulation import check_models

    ai = AIClient()
    ollama = asyncio.run(ai.ollama.list_models())
    ids = [m.id for m in settings.GEMINI_MODELS]
    if ollama_check is not None:
        ids += ollama_check or ollama
    results = asyncio.run(check_models(ai, ids)) if check else {}
    t = Table(
        "model",
        "provider",
        "USD per 1M in/out",
        "notes",
        *(["live check"] if check else []),
    )
    for m in settings.GEMINI_MODELS:
        price = f"{m.in_per_m:.2f} / {m.out_per_m:.2f}" if m.in_per_m else "free tier"
        row = [m.id, "gemini", price, m.note]
        if check:
            row.append(results.get(m.id, ""))
        t.add_row(*row)
    for n in ollama:
        row = [n, "ollama", "local", ""]
        if check:
            row.append(results.get(n, "[dim]not tested (use --ollama)[/]"))
        t.add_row(*row)
    Console().print(t)
    Console().print(
        f"memory model: {settings.MEMORY_MODEL}   voice model: {settings.TTS_MODEL}"
    )
    return 0


def _doctor() -> int:
    import shutil

    from rich.console import Console

    from .ai_client import AIClient
    from .simulation import check_models

    c = Console()
    ok = True
    key = os.environ.get("GEMINI_API_KEY")
    c.print(f"config dir   {settings.CONFIG_DIR}")
    c.print(f"data dir     {settings.DATA_DIR}")
    c.print(f"API key      {'set' if key else '[red]missing[/]'}")
    ai = AIClient()
    if key:
        res = asyncio.run(
            check_models(ai, [settings.DEFAULT_MODEL, settings.MEMORY_MODEL])
        )
        for m, r in res.items():
            good = r.startswith("ok")
            ok &= good
            c.print(f"model        {m}: {'[green]' if good else '[red]'}{r}[/]")
    else:
        ok = False
    ollama = asyncio.run(ai.ollama.list_models())
    c.print(
        f"ollama       {len(ollama)} models at {settings.OLLAMA_URL}"
        if ollama
        else "ollama       not running (optional)"
    )
    try:
        import pygame  # noqa: F401

        player = "pygame"
    except Exception:
        player = next(
            (p for p in ("paplay", "aplay", "afplay", "ffplay") if shutil.which(p)), ""
        )
    c.print(
        f"audio        {player or '[yellow]no player found (voice will be silent)[/]'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
