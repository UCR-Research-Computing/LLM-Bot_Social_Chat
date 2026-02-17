import asyncio
import random
import logging
import os
import json
from typing import Optional, List, Tuple

from database import Bot, Post, Memory, session, clear_posts_table
import ai_client
import voice_manager
import subprocess

def get_available_models() -> List[Tuple[str, str]]:
    """Gets a list of available models from Gemini and Ollama."""
    
    gemini_models = [
        ("Gemini 1.5 Flash", "gemini-1.5-flash"),
        ("Gemini 1.5 Pro", "gemini-1.5-pro"),
        ("Gemini 2.5 Flash", "gemini-2.5-flash"),
        ("Gemini 2.5 Pro", "gemini-2.5-pro"),
    ]

    ollama_models = []
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            for line in lines[1:]:
                parts = line.split()
                if parts:
                    model_name = parts[0].split(':')[0]
                    ollama_models.append((model_name, model_name))
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logging.warning(f"Could not list Ollama models: {e}")

    return gemini_models + ollama_models

def _parse_memory_string(memory_string: Optional[str]) -> Optional[Tuple[str, str]]:

    """Parses a 'key: value' string into a tuple, handling errors."""
    if not memory_string or memory_string.lower().strip() == "none" or ":" not in memory_string:
        return None
    try:
        key, value = memory_string.split(":", 1)
        return key.strip(), value.strip()
    except ValueError:
        return None

class Simulation:
    def __init__(self, config_file: str, autostart: bool, tts_enabled: bool, clear_db: bool, 
                 max_posts: Optional[int] = None, duration: Optional[int] = None, 
                 topic: Optional[str] = None, deterministic: bool = False, audio_dir: Optional[str] = None):
        self.config_file = config_file
        self.autostart = autostart
        self.tts_enabled = tts_enabled
        self.clear_db = clear_db
        self.max_posts = max_posts
        self.duration = duration
        self.topic = topic
        self.deterministic = deterministic
        self.audio_dir = audio_dir

        self.tts_queue = asyncio.Queue(maxsize=1)
        self.background_tasks = set()
        self.bot_names: List[str] = []
        self.post_count = 0
        self.next_bot_index = 0

        if self.clear_db:
            clear_posts_table()

    def run_task(self, coro):
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def initialize(self):
        """Initializes the simulation, loading bots and voices."""
        await asyncio.to_thread(voice_manager.get_voices)
        await self.load_bots_from_json(self.config_file)
        if self.topic:
            await self.inject_topic(self.topic)

    async def run_bot_activity(self):
        """The main loop for generating bot posts."""
        bots = await asyncio.to_thread(session.query(Bot).all)
        if not bots:
            return

        if self.deterministic:
            bot_to_post = bots[self.next_bot_index]
            self.next_bot_index = (self.next_bot_index + 1) % len(bots)
        else:
            bot_to_post = random.choice(bots)

        other_bot_names = [b.name for b in bots if b.name != bot_to_post.name]
        
        recent_posts = await asyncio.to_thread(
            session.query(Post).order_by(Post.id.desc()).limit(50).all
        )
        
        memories = await asyncio.to_thread(
            session.query(Memory).filter_by(bot_id=bot_to_post.id).all
        )

        post_content, prompt = "", ""
        try:
            if bot_to_post.model.startswith("gemini"):
                post_content, prompt = await ai_client.generate_post_gemini(bot_to_post, other_bot_names, recent_posts, memories)
            else:
                post_content, prompt = await ai_client.generate_post_ollama(bot_to_post, other_bot_names, recent_posts, memories)
        except Exception as e:
            post_content = f"[SYSTEM Error: {e}]"
            logging.error("Error generating post", extra={'event': 'post.generation.fail', 'bot_name': bot_to_post.name, 'error': str(e)})

        sender_name = "SYSTEM" if post_content.startswith(("[Error", "[SYSTEM")) else bot_to_post.name
        new_post = await asyncio.to_thread(self.create_post, post_content, sender_name, bot_to_post)
        self.post_count += 1

        logging.info("Bot post generated", extra={
            'event': 'post.generated',
            'bot_name': bot_to_post.name,
            'bot_model': bot_to_post.model,
            'post_content': post_content,
            'prompt': prompt
        })

        # The calling script is now responsible for queuing.
        self.run_task(self.form_new_memory(bot_to_post))
        return new_post

    async def form_new_memory(self, bot: Bot):
        """Asks the AI to generate a new memory and saves it to the database."""
        recent_posts = await asyncio.to_thread(
            session.query(Post).order_by(Post.id.desc()).limit(5).all
        )
        
        new_memory_str = await ai_client.generate_new_memory(bot, recent_posts)
        
        parsed_memory = _parse_memory_string(new_memory_str)
        if parsed_memory:
            key, value = parsed_memory
            new_memory = Memory(key=key, value=value, bot=bot)
            await asyncio.to_thread(self.db_add_memory, new_memory)
            logging.info(f"New memory for {bot.name}", extra={
                'event': 'memory.form.success',
                'bot_name': bot.name,
                'memory_key': key,
                'memory_value': value
            })
        else:
            logging.info(f"No new memory formed for {bot.name}.", extra={
                'event': 'memory.form.fail',
                'bot_name': bot.name,
                'llm_response': new_memory_str
            })

    async def inject_topic(self, topic: str) -> Post:
        new_post = await asyncio.to_thread(self.create_post, topic, "SYSTEM")
        logging.info("Topic injected", extra={'event': 'topic.injected', 'topic': topic})
        return new_post


    def create_post(self, content, sender, bot=None):
        new_post = Post(content=content, sender=sender, bot=bot)
        session.add(new_post)
        session.commit()
        return new_post

    async def load_bots_from_json(self, filename: str):
        def _load():
            filepath = os.path.join("configs", filename)
            try:
                with open(filepath, "r") as f:
                    bots_data = json.load(f)
                session.query(Post).delete()
                session.query(Memory).delete()
                session.query(Bot).delete()
                for bot_data in bots_data:
                    memories = bot_data.pop("memories", [])
                    bot = Bot(**bot_data)
                    session.add(bot)
                    for memory_data in memories:
                        memory = Memory(key=memory_data["key"], value=memory_data["value"], bot=bot)
                        session.add(memory)
                session.commit()
                self.bot_names = [bot.name for bot in session.query(Bot).all()]
                return None
            except (FileNotFoundError, json.JSONDecodeError):
                return "Error"
        
        error = await asyncio.to_thread(_load)
        if error:
            logging.error(f"Failed to load config file: {filename}", extra={'event': 'config.load.fail', 'config_filename': filename})
        else:
            logging.info(f"Successfully loaded config file: {filename}", extra={'event': 'config.load.success', 'config_filename': filename})

    def db_add_memory(self, memory: Memory):
        session.add(memory)
        session.commit()
