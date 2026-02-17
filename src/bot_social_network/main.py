import json
import argparse
import logging
import asyncio
import os
import tempfile
from typing import Optional, List, Tuple

from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Header, Footer, ListView, ListItem, Label, Input, Button, Static, TextArea, Select
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.binding import Binding
from . import logging_config
from . import voice_manager

from .database import Bot, Post, session, close_database_connection, db_create_bot, db_edit_bot, db_delete_bot, db_clear_posts
from .simulation import Simulation, get_available_models

# --- Initialize Logging ---
log_filename = logging_config.setup_logging()

# --- Screens ---

class SaveConfigScreen(ModalScreen[str]):
    """A modal screen for saving a bot configuration."""

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="dialog"):
            yield Label("Save Bot Configuration")
            yield Input(placeholder="Filename (e.g., my_team)", id="config_name")
            with Horizontal(id="buttons"):
                yield Button("Save", variant="primary", id="save")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss()
        elif event.button.id == "save":
            filename = self.query_one("#config_name", Input).value
            if filename:
                self.dismiss(filename)

class LoadConfigScreen(ModalScreen[str]):
    """A modal screen for loading a bot configuration."""

    def __init__(self, config_files: List[str]):
        super().__init__()
        self.config_files = config_files

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="dialog"):
            yield Label("Load Bot Configuration")
            if self.config_files:
                yield Select([(f, f) for f in self.config_files], prompt="Select a file", id="config_select")
            else:
                yield Label("No configuration files found in 'configs/' directory.")
            with Horizontal(id="buttons"):
                yield Button("Load", variant="primary", id="load")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss()
        elif event.button.id == "load":
            select = self.query_one(Select)
            if select.value:
                self.dismiss(select.value)

class BotEditScreen(ModalScreen[dict]):
    """A modal screen for creating or editing a bot."""

    def __init__(self, bot_to_edit: Optional[Bot] = None, available_models: List[Tuple[str, str]] = []):
        super().__init__()
        self.bot_to_edit = bot_to_edit
        self.available_models = available_models

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="dialog"):
            yield Label("Edit Bot" if self.bot_to_edit else "Create New Bot")
            yield Input(
                value=self.bot_to_edit.name if self.bot_to_edit else "",
                placeholder="Name",
                id="bot_name"
            )
            yield TextArea(
                text=self.bot_to_edit.persona if self.bot_to_edit else "",
                id="bot_persona"
            )
            # Determine the initial value for the Select widget
            initial_model = None
            if self.bot_to_edit:
                initial_model = self.bot_to_edit.model
            elif self.available_models:
                initial_model = self.available_models[0][1]

            yield Select(
                self.available_models,
                value=initial_model,
                prompt="Select model",
                id="bot_model",
                allow_blank=False if self.available_models else True,
            )
            with Horizontal(id="buttons"):
                yield Button("Save", variant="primary", id="save")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss()
        elif event.button.id == "save":
            bot_data = {
                "name": self.query_one("#bot_name", Input).value,
                "persona": self.query_one("#bot_persona", TextArea).text,
                "model": self.query_one("#bot_model", Select).value,
            }
            if all(bot_data.values()):
                self.dismiss(bot_data)

# --- Widgets ---

class BotManager(ListView):
    """Widget to display and manage the list of bots."""
    def on_mount(self) -> None:
        self.border_title = "Bots"
        self.app.run_task(self.refresh_bots())

    async def refresh_bots(self):
        def _get_bots() -> List[Bot]:
            return session.query(Bot).all()
        bots = await asyncio.to_thread(_get_bots)
        self.app.bot_names = [bot.name for bot in bots]
        self.clear()
        for bot in bots:
            self.append(ListItem(Label(bot.name)))

class PostView(ListView):
    """Widget to display the chat history."""
    def on_mount(self) -> None:
        self.border_title = "Posts"
        self.app.run_task(self.refresh_posts())

    def add_post(self, post: Post):
        sender = post.sender or (post.bot.name if post.bot else "Unknown")
        self.mount(ListItem(Static(f"{sender}: {post.content}")), before=0)
        self.scroll_home(animate=False)

    async def refresh_posts(self):
        def _get_posts() -> List[Post]:
            return session.query(Post).order_by(Post.id.desc()).all()
        posts = await asyncio.to_thread(_get_posts)
        self.clear()
        for post in reversed(posts): # Display in chronological order
            self.add_post(post)

# --- Main App ---

class BotSocialApp(App):
    CSS_PATH = os.path.join(os.path.dirname(__file__), "style.css")
    BINDINGS = [Binding("q", "quit", "Quit")]

    def __init__(self, config_file: str = "default.json", autostart: bool = False, clear_db: bool = False, autostart_tts: bool = False, topic: Optional[str] = None):
        super().__init__()
        self.simulation = Simulation(
            config_file=config_file,
            autostart=autostart,
            tts_enabled=autostart_tts,
            clear_db=clear_db,
            topic=topic
        )
        self.selected_bot: Optional[Bot] = None
        self.available_models = get_available_models()
        logging.info("Application started.", extra={
            'event': 'system.start',
            'config_file': config_file,
            'autostart': autostart,
            'tts_enabled': autostart_tts,
            'clear_db': clear_db,
            'topic': topic
        })

    def compose(self) -> ComposeResult:
        yield Header()
        yield Horizontal(
            Vertical(
                BotManager(),
                Horizontal(Button("Create Bot", id="create_bot"), Button("Edit Bot", id="edit_bot")),
                Button("Delete Bot", id="delete_bot"),
                Horizontal(Button("Load Config", id="load_bots"), Button("Save Config", id="save_bots")),
                id="left-pane"
            ),
            Vertical(
                PostView(id="posts"),
                Horizontal(Button("Start", id="start_chat"), Button("Stop", id="stop_chat"), Button("Clear", id="clear_chat")),
                Horizontal(Input(placeholder="Topic", id="topic_input"), Button("Inject Topic", id="inject_topic")),
                Button("TTS: ON" if self.simulation.tts_enabled else "TTS: OFF", id="toggle_tts"),
                id="right-pane"
            )
        )
        yield Footer()

    def on_mount(self) -> None:
        self.run_task(self.simulation.initialize())
        self.run_task(self.speaker_worker())
        self.bot_timer = self.set_interval(15, self.run_bot_activity, pause=not self.simulation.autostart)

    def run_task(self, coro):
        self.simulation.run_task(coro)

    # --- Event Handlers ---

    def on_list_view_selected(self, event: ListView.Selected):
        if isinstance(event.list_view, BotManager):
            bot_name = event.item.children[0].renderable
            self.run_task(self.select_bot_by_name(str(bot_name)))
    
    async def select_bot_by_name(self, name: str):
        def _get_bot():
            return session.query(Bot).filter_by(name=name).first()
        self.selected_bot = await asyncio.to_thread(_get_bot)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "topic_input" and event.value:
            await self.action_inject_topic(event.value)
            event.input.value = ""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "create_bot":
            self.selected_bot = None
            self.push_screen(BotEditScreen(available_models=self.available_models), self.handle_bot_edit_result)
        elif event.button.id == "edit_bot":
            if self.selected_bot:
                self.push_screen(BotEditScreen(bot_to_edit=self.selected_bot, available_models=self.available_models), self.handle_bot_edit_result)
        elif event.button.id == "delete_bot":
            if self.selected_bot:
                self.run_task(self.action_delete_bot())
        elif event.button.id == "load_bots":
            self.push_screen(LoadConfigScreen(config_files=self.get_config_files()), self.handle_load_config)
        elif event.button.id == "save_bots":
            self.push_screen(SaveConfigScreen(), self.handle_save_config)
        elif event.button.id == "start_chat":
            self.bot_timer.resume()
        elif event.button.id == "stop_chat":
            self.bot_timer.pause()
        elif event.button.id == "clear_chat":
            self.run_task(self.action_clear_posts())
        elif event.button.id == "inject_topic":
            topic_input = self.query_one("#topic_input", Input)
            if topic_input.value:
                self.run_task(self.action_inject_topic(topic_input.value))
                topic_input.value = ""
        elif event.button.id == "toggle_tts":
            self.simulation.tts_enabled = not self.simulation.tts_enabled
            event.button.label = f"TTS: {'ON' if self.simulation.tts_enabled else 'OFF'}"

    # --- Action & Callback Methods ---

    def handle_bot_edit_result(self, bot_data: Optional[dict]):
        if bot_data:
            task = self.action_edit_bot(bot_data) if self.selected_bot else self.action_create_bot(bot_data)
            self.run_task(task)

    def handle_save_config(self, filename: Optional[str]):
        if filename:
            self.run_task(self.save_bots_to_json(filename))

    def handle_load_config(self, filename: Optional[str]):
        if filename:
            self.run_task(self.load_bots_from_json(filename))

    async def run_bot_activity(self):
        post = await self.simulation.run_bot_activity()
        if post and self.simulation.tts_enabled:
            await self.simulation.tts_queue.put({"post": post})
        elif post:
            self.query_one(PostView).add_post(post)

    async def speaker_worker(self):
        """The consumer task that plays audio from the queue."""
        while True:
            try:
                data = await self.simulation.tts_queue.get()
                post = data["post"]
                
                self.query_one(PostView).add_post(post)

                if self.simulation.tts_enabled and not post.sender == "SYSTEM":
                    voice_name = voice_manager.select_voice(post.sender)
                    if voice_name:
                        # In TUI mode, we save audio to a temporary file for playback
                        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
                            output_path = temp_audio_file.name
                        
                        success = await voice_manager.generate_voice_file(post.content, voice_name, output_path)
                        if success:
                            await asyncio.to_thread(voice_manager.play_audio_file, output_path)
                        
                        if os.path.exists(output_path):
                            os.remove(output_path)
                
                self.simulation.tts_queue.task_done()
            except asyncio.CancelledError:
                break

    # --- Core Actions ---

    async def action_create_bot(self, bot_data: dict):
        await asyncio.to_thread(db_create_bot, **bot_data)
        await self.query_one(BotManager).refresh_bots()

    async def action_edit_bot(self, bot_data: dict):
        await asyncio.to_thread(db_edit_bot, self.selected_bot, **bot_data)
        await self.query_one(BotManager).refresh_bots()

    async def action_delete_bot(self):
        await asyncio.to_thread(db_delete_bot, self.selected_bot)
        self.selected_bot = None
        await self.query_one(BotManager).refresh_bots()

    async def action_clear_posts(self):
        await asyncio.to_thread(db_clear_posts)
        await self.query_one(PostView).refresh_posts()

    async def action_inject_topic(self, topic: str):
        new_post = await self.simulation.inject_topic(topic)
        self.query_one(PostView).add_post(new_post)

    # --- Config File Methods ---

    def get_config_files(self) -> List[str]:
        config_dir = "configs"
        if not os.path.exists(config_dir):
            return []
        return [f for f in os.listdir(config_dir) if f.endswith(".json")]

    async def load_bots_from_json(self, filename: str):
        await self.simulation.load_bots_from_json(filename)
        await self.query_one(BotManager).refresh_bots()
        await self.query_one(PostView).refresh_posts()

    async def save_bots_to_json(self, filename: str):
        # This can remain in the App as it's a UI-specific action
        def _save():
            bots = session.query(Bot).all()
            bots_data = []
            for bot in bots:
                bot_dict = {"name": bot.name, "persona": bot.persona, "model": bot.model}
                memories = [{"key": m.key, "value": m.value} for m in bot.memories]
                if memories:
                    bot_dict["memories"] = memories
                bots_data.append(bot_dict)
            
            filepath = os.path.join("configs", f"{filename}.json")
            with open(filepath, "w") as f:
                json.dump(bots_data, f, indent=4)
        await asyncio.to_thread(_save)

    async def action_quit(self) -> None:
        self.log("Stopping bot timer...")
        self.bot_timer.stop()
        
        self.log("Stopping any playing audio...")
        voice_manager.stop_audio()

        self.log(f"Cancelling {len(self.simulation.background_tasks)} background tasks...")
        for task in self.simulation.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.simulation.background_tasks, return_exceptions=True)
        
        self.log("Closing database connection...")
        await asyncio.to_thread(close_database_connection)
        
        logging.info("Application exited.", extra={'event': 'system.stop'})
        self.log("Exiting application.")
        self.exit()

def run():
    parser = argparse.ArgumentParser(
        prog="Bot Social Network (Interactive TUI)",
        description="""
A fully interactive terminal application that simulates a social media feed for autonomous AI agents. 
Create bots, load configurations, and watch conversations unfold in real-time.

This is the main entry point for the interactive TUI. For automated, non-interactive runs, 
see headless.py.
""",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
--------------------------------------------------------------------------------
Usage Examples:
  # Run with default settings
  python3 main.py

  # Load a specific group of bots and inject a starting topic
  python3 main.py --config example_tinydolphin.json --topic "What do you think of the ocean?"

  # Start a fresh, voiced conversation
  python3 main.py --autostart --tts --clear-db
--------------------------------------------------------------------------------
"""
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="default.json", 
        help="The bot configuration file to load from the 'configs/' directory."
    )
    parser.add_argument(
        "--autostart", 
        action="store_true", 
        help="Start the bot chat automatically on launch."
    )
    parser.add_argument(
        "--clear-db", 
        action="store_true", 
        help="Clear the posts database on launch for a clean slate."
    )
    parser.add_argument(
        "--tts", 
        action="store_true", 
        help="Enable text-to-speech on launch. Requires Google Cloud authentication."
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="An initial topic to inject into the conversation."
    )
    args = parser.parse_args()

    app = BotSocialApp(
        config_file=args.config,
        autostart=args.autostart, 
        clear_db=args.clear_db, 
        autostart_tts=args.tts,
        topic=args.topic
    )
    app.run()

if __name__ == "__main__":
    run()




