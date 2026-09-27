"""Interactive Textual app: bot roster, live feed, stats bar, controls."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from rich.markup import escape
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Select,
    Static,
    TextArea,
)

from . import settings
from .ai_client import AIClient
from .database import Bot, Database, Post
from .logging_config import setup_logging
from .simulation import Simulation, TeamError, list_teams, save_team
from .voice import Voice, voice_for

COLORS = [
    "cyan",
    "magenta",
    "green",
    "yellow",
    "dodger_blue1",
    "orange1",
    "orchid",
    "spring_green1",
]


# --------------------------------------------------------------------------- dialogs


class TeamScreen(ModalScreen[str | None]):
    def __init__(self, teams: list[str]):
        super().__init__()
        self.teams = teams

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Load a team (replaces bots, memories and posts)")
            yield Select([(t, t) for t in self.teams], prompt="Team", id="team")
            with Horizontal(classes="buttons"):
                yield Button("Load", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    @on(Button.Pressed, "#ok")
    def ok(self) -> None:
        v = self.query_one("#team", Select).value
        if isinstance(v, str):
            self.dismiss(v)

    @on(Button.Pressed, "#cancel")
    def cancel(self) -> None:
        self.dismiss(None)


class SaveScreen(ModalScreen[str | None]):
    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(f"Save team to {settings.USER_CONFIGS}")
            yield Input(placeholder="name (e.g. my_team)", id="name")
            with Horizontal(classes="buttons"):
                yield Button("Save", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    @on(Button.Pressed, "#ok")
    @on(Input.Submitted)
    def ok(self) -> None:
        v = self.query_one("#name", Input).value.strip()
        if v:
            self.dismiss(v)

    @on(Button.Pressed, "#cancel")
    def cancel(self) -> None:
        self.dismiss(None)


class ConfirmScreen(ModalScreen[bool]):
    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self.message)
            with Horizontal(classes="buttons"):
                yield Button("Yes", variant="error", id="yes")
                yield Button("No", id="no")

    @on(Button.Pressed)
    def done(self, e: Button.Pressed) -> None:
        self.dismiss(e.button.id == "yes")


class BotEditScreen(ModalScreen[dict[str, Any] | None]):
    def __init__(self, bot: Bot | None, models: list[tuple[str, str]]):
        super().__init__()
        self.bot = bot
        self.models = models

    def compose(self) -> ComposeResult:
        b = self.bot
        model = b.model if b else settings.DEFAULT_MODEL
        opts = (
            self.models
            if any(m == model for _, m in self.models)
            else [(model, model), *self.models]
        )
        with VerticalScroll(id="dialog", classes="tall"):
            yield Label("Edit bot" if b else "New bot")
            yield Input(value=b.name if b else "", placeholder="Name", id="name")
            yield Label("Persona", classes="dim")
            yield TextArea(b.persona if b else "", id="persona")
            yield Select(opts, value=model, allow_blank=False, id="model")
            yield Select(
                [("Voice: automatic", "")] + [(v, v) for v in settings.VOICES],
                value=(b.voice or "") if b else "",
                allow_blank=False,
                id="voice",
            )
            yield Input(
                value="" if not b or b.temperature is None else str(b.temperature),
                placeholder="Temperature (blank = model default, 0-2)",
                id="temp",
            )
            yield Label("", id="err", classes="error")
            with Horizontal(classes="buttons"):
                yield Button("Save", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    @on(Button.Pressed, "#ok")
    def ok(self) -> None:
        name = self.query_one("#name", Input).value.strip()
        persona = self.query_one("#persona", TextArea).text.strip()
        model = self.query_one("#model", Select).value
        voice = self.query_one("#voice", Select).value
        t = self.query_one("#temp", Input).value.strip()
        err = self.query_one("#err", Label)
        if not name or not persona:
            err.update("Name and persona are required.")
            return
        temp: float | None = None
        if t:
            try:
                temp = float(t)
                assert 0 <= temp <= 2
            except (ValueError, AssertionError):
                err.update("Temperature must be a number from 0 to 2.")
                return
        self.dismiss(
            {
                "name": name,
                "persona": persona,
                "model": str(model),
                "voice": str(voice) or None,
                "temperature": temp,
            }
        )

    @on(Button.Pressed, "#cancel")
    def cancel(self) -> None:
        self.dismiss(None)


# --------------------------------------------------------------------------- app


class BotSocialApp(App[None]):
    CSS_PATH = "style.css"
    TITLE = "Bot Social Network"
    BINDINGS = [
        Binding("space", "toggle_run", "Start/Stop"),
        Binding("n", "step", "Next post"),
        Binding("t", "focus_topic", "Topic"),
        Binding("v", "toggle_tts", "Voice"),
        Binding("l", "load_team", "Load team"),
        Binding("s", "save_team", "Save team"),
        Binding("a", "new_bot", "New bot"),
        Binding("e", "edit_bot", "Edit bot"),
        Binding("d", "delete_bot", "Delete bot"),
        Binding("c", "clear", "Clear posts"),
        Binding("plus,equals_sign", "faster", "Faster", show=False),
        Binding("minus", "slower", "Slower", show=False),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        team: str = "default",
        autostart: bool = False,
        clear_db: bool = False,
        tts: bool = False,
        topic: str | None = None,
        interval: float = 8.0,
        budget: float | None = None,
        db_url: str | None = None,
    ):
        super().__init__()
        self.run_dir = setup_logging()
        self.db = Database(db_url)
        self.ai = AIClient()
        self.sim = Simulation(self.db, self.ai, budget_usd=budget)
        self.voice = Voice(self.ai.gemini)
        self.team = team
        self.autostart = autostart
        self.clear_db = clear_db
        self.tts = tts
        self.topic = topic
        self.interval = max(1.0, interval)
        self.running = False
        self.busy = False
        self.speaking = False
        self.selected: str | None = None
        self.names: list[str] = []
        self.models: list[tuple[str, str]] = [
            (f"{m.label}  ({m.note})", m.id) for m in settings.GEMINI_MODELS
        ]

    # ---- layout --------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="left"):
                yield ListView(id="bots")
                yield Static("", id="botinfo")
            with Vertical(id="right"):
                yield RichLog(id="feed", wrap=True, markup=True, auto_scroll=True)
                with Horizontal(id="topicbar"):
                    yield Input(
                        placeholder="Inject a topic or say something as SYSTEM (Enter)",
                        id="topic",
                    )
                yield Static("", id="status")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#bots", ListView).border_title = "Bots"
        self.query_one("#feed", RichLog).border_title = "Feed"
        self.query_one("#botinfo", Static).border_title = "Bot"
        if not self.ai.gemini.configured:
            self.notify(
                f"GEMINI_API_KEY not set. Put it in {settings.CONFIG_DIR / '.env'}",
                severity="error",
                timeout=15,
            )
        self.load_ollama_models()
        if self.clear_db:
            await asyncio.to_thread(self.db.clear_posts)
        if self.team and (self.clear_db or not await asyncio.to_thread(self.db.bots)):
            await self.do_load_team(self.team)
        else:
            await self.refresh_all()
        if self.topic:
            await asyncio.to_thread(self.sim.inject_topic, self.topic)
            await self.refresh_feed()
        self.timer = self.set_interval(self.interval, self.tick, pause=True)
        if self.autostart:
            self.action_toggle_run()
        self.update_status()

    @work(exclusive=True, group="ollama")
    async def load_ollama_models(self) -> None:
        names = await self.ai.ollama.list_models()
        self.models += [(f"{n}  (Ollama, local)", n) for n in names]

    # ---- rendering -----------------------------------------------------------
    def color(self, name: str | None) -> str:
        if name in self.names:
            return COLORS[self.names.index(name) % len(COLORS)]
        return "white"

    def render_post(self, p: Post) -> Text:
        ts = p.created_at.strftime("%H:%M") if p.created_at else ""
        if p.sender == "SYSTEM":
            style = "bold red" if p.error else "bold"
            return Text.from_markup(
                f"[dim]{ts}[/] [{style}]SYSTEM[/] {escape(p.content)}\n"
            )
        meta = (
            f" [dim]{p.model or ''} {p.latency_ms / 1000:.1f}s[/]"
            if p.latency_ms
            else ""
        )
        return Text.from_markup(
            f"[dim]{ts}[/] [bold {self.color(p.sender)}]{escape(p.sender or '?')}[/]{meta}\n{escape(p.content)}\n"
        )

    async def refresh_all(self) -> None:
        await self.refresh_bots()
        await self.refresh_feed()

    async def refresh_bots(self) -> None:
        bots = await asyncio.to_thread(self.db.bots)
        self.names = [b.name for b in bots]
        lv = self.query_one("#bots", ListView)
        await lv.clear()
        for b in bots:
            item = ListItem(
                Label(
                    Text.from_markup(f"[bold {self.color(b.name)}]{escape(b.name)}[/]")
                )
            )
            item.bot_name = b.name  # type: ignore[attr-defined]
            await lv.append(item)
        if self.selected not in self.names:
            self.selected = self.names[0] if self.names else None
        self.show_bot()

    async def refresh_feed(self) -> None:
        posts = await asyncio.to_thread(self.db.recent_posts, 200)
        log = self.query_one("#feed", RichLog)
        log.clear()
        for p in reversed(posts):
            log.write(self.render_post(p))

    def show_bot(self) -> None:
        info = self.query_one("#botinfo", Static)
        b = self.db.bot(self.selected) if self.selected else None
        if not b:
            info.update("No bot selected")
            return
        mems = self.db.memories(b.id, 8)
        st = self.sim.state.get(b.name)
        lines = [
            f"[bold {self.color(b.name)}]{escape(b.name)}[/]",
            f"[dim]model[/] {escape(b.model)}",
            f"[dim]voice[/] {voice_for(b.name, b.voice)}"
            + (f"  [dim]temp[/] {b.temperature}" if b.temperature is not None else ""),
            f"[dim]posts this run[/] {self.sim.stats.by_bot.get(b.name, 0)}"
            + ("  [red]benched[/]" if st and st.benched_until > self.sim.turn else ""),
            "",
            escape(b.persona[:400] + ("..." if len(b.persona) > 400 else "")),
        ]
        if mems:
            lines += ["", "[bold]Memories[/]"] + [
                f"- {escape(m.key)}: {escape(m.value)}" for m in mems
            ]
        info.update("\n".join(lines))

    def update_status(self) -> None:
        s = self.sim.stats
        state = "[green]RUNNING[/]" if self.running else "[yellow]PAUSED[/]"
        if self.speaking:
            state += " [dim](speaking...)[/]"
        elif self.busy:
            state += " [dim](writing...)[/]"
        budget = f" / ${self.sim.budget_usd:.2f}" if self.sim.budget_usd else ""
        self.query_one("#status", Static).update(
            f"{state}  every {self.interval:.0f}s   posts {s.posts}   errors {s.errors}   "
            f"memories {s.memories}   tokens {s.tokens_in + s.tokens_out:,}   "
            f"cost ${s.cost_usd + s.tts_cost_usd:.4f}{budget}   voice {'ON' if self.tts else 'off'}"
        )

    # ---- loop ----------------------------------------------------------------
    async def tick(self) -> None:
        if self.busy:
            return
        if self.sim.over_budget:
            self.action_toggle_run()
            self.notify("Budget reached; paused.", severity="warning")
            return
        self.busy = True
        self.update_status()
        try:
            post = await self.sim.step()
            if post:
                self.query_one("#feed", RichLog).write(self.render_post(post))
                if post.sender == self.selected or post.sender == "SYSTEM":
                    self.show_bot()
                # With voice on, speaking is part of the turn: the next post waits
                # until this one has been read out (busy stays set, so timer ticks
                # that fire meanwhile are skipped). No backlog can build up.
                if self.tts and post.sender != "SYSTEM":
                    await self.speak(post)
                    # Restart the interval so the gap after a spoken post is the
                    # full interval, not whatever was left on the timer.
                    self.timer.reset()
        except Exception as e:  # never let a bad turn kill the timer
            logging.exception("tick failed")
            self.notify(f"Turn failed: {e}", severity="error")
        finally:
            self.busy = False
            self.update_status()

    async def speak(self, post: Post) -> None:
        self.speaking = True
        self.update_status()
        try:
            b = await asyncio.to_thread(self.db.bot, post.sender or "")
            sp = await self.voice.synthesize(
                post.content,
                voice_for(post.sender or "", b.voice if b else None),
                Path(self.run_dir) / "audio" / f"post_{post.id}.wav",
            )
            self.sim.stats.tts_cost_usd += sp.cost_usd
            if self.tts:  # voice may have been switched off while synthesizing
                await asyncio.to_thread(self.voice.play, sp.path)
        except Exception as e:
            self.notify(f"Voice failed: {e}", severity="warning")
        finally:
            self.speaking = False

    # ---- events --------------------------------------------------------------
    @on(ListView.Highlighted, "#bots")
    def pick(self, e: ListView.Highlighted) -> None:
        if e.item is not None:
            self.selected = getattr(e.item, "bot_name", None)
            self.show_bot()

    @on(Input.Submitted, "#topic")
    async def topic_entered(self, e: Input.Submitted) -> None:
        if e.value.strip():
            post = await asyncio.to_thread(self.sim.inject_topic, e.value)
            self.query_one("#feed", RichLog).write(self.render_post(post))
            e.input.value = ""

    # ---- actions -------------------------------------------------------------
    def action_toggle_run(self) -> None:
        self.running = not self.running
        if self.running:
            self.timer.resume()
            self.run_worker(self.tick(), exclusive=False)
        else:
            self.timer.pause()
        self.update_status()

    def action_step(self) -> None:
        # Run as a worker so the UI stays responsive while a post is written/spoken.
        self.run_worker(self.tick(), exclusive=False)

    def action_focus_topic(self) -> None:
        self.query_one("#topic", Input).focus()

    def action_toggle_tts(self) -> None:
        if not self.ai.gemini.configured:
            self.notify("Voice needs GEMINI_API_KEY.", severity="error")
            return
        self.tts = not self.tts
        if not self.tts:
            self.voice.stop()
        self.update_status()

    def _set_interval(self, value: float) -> None:
        self.interval = min(120.0, max(1.0, value))
        was = self.running
        self.timer.stop()
        self.timer = self.set_interval(self.interval, self.tick, pause=not was)
        self.update_status()

    def action_faster(self) -> None:
        self._set_interval(self.interval / 1.5)

    def action_slower(self) -> None:
        self._set_interval(self.interval * 1.5)

    def action_load_team(self) -> None:
        async def done(name: str | None) -> None:
            if name:
                await self.do_load_team(name)

        self.push_screen(TeamScreen([n for n, _ in list_teams()]), done)

    async def do_load_team(self, name: str) -> None:
        try:
            notes = await asyncio.to_thread(self.sim.load_team, name)
        except TeamError as e:
            self.notify(str(e), severity="error", timeout=10)
            return
        self.team = name
        self.sub_title = name
        await self.refresh_all()
        for n in notes:
            self.notify(f"Model upgraded: {n}", timeout=6)

    def action_save_team(self) -> None:
        def done(name: str | None) -> None:
            if name:
                path = save_team(self.db, name)
                self.notify(f"Saved {path}")

        self.push_screen(SaveScreen(), done)

    def action_new_bot(self) -> None:
        async def done(data: dict[str, Any] | None) -> None:
            if not data:
                return
            if data["name"] in self.names:
                self.notify("A bot with that name exists.", severity="error")
                return
            await asyncio.to_thread(lambda: self.db.create_bot(**data))
            self.selected = data["name"]
            await self.refresh_bots()

        self.push_screen(BotEditScreen(None, self.models), done)

    def action_edit_bot(self) -> None:
        b = self.db.bot(self.selected) if self.selected else None
        if not b:
            return

        async def done(data: dict[str, Any] | None) -> None:
            if not data:
                return
            if data["name"] != b.name and data["name"] in self.names:
                self.notify("A bot with that name exists.", severity="error")
                return
            await asyncio.to_thread(lambda: self.db.update_bot(b.id, **data))
            self.selected = data["name"]
            await self.refresh_bots()

        self.push_screen(BotEditScreen(b, self.models), done)

    def action_delete_bot(self) -> None:
        b = self.db.bot(self.selected) if self.selected else None
        if not b:
            return

        async def done(yes: bool | None) -> None:
            if yes:
                await asyncio.to_thread(self.db.delete_bot, b.id)
                self.selected = None
                await self.refresh_all()

        self.push_screen(
            ConfirmScreen(f"Delete {b.name} and all their posts and memories?"), done
        )

    def action_clear(self) -> None:
        async def done(yes: bool | None) -> None:
            if yes:
                await asyncio.to_thread(self.db.clear_posts)
                await self.refresh_feed()

        self.push_screen(
            ConfirmScreen("Clear every post? Bots and memories stay."), done
        )

    async def action_quit(self) -> None:
        self.timer.stop()
        self.voice.stop()
        await self.sim.shutdown()
        self.db.close()
        logging.info("exit", extra={"event": "system.stop"})
        self.exit()
