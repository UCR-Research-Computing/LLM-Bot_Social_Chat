import logging
import asyncio
import subprocess
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
import os

import google.generativeai as genai
from dotenv import load_dotenv

from database import Bot, Post, Memory

# --- Load environment variables ---
load_dotenv()

# --- Thread Pool for blocking IO ---
executor = ThreadPoolExecutor(max_workers=4)

# --- Gemini Client ---
def configure_gemini():
    """Configures the Gemini client and checks for the API key."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")
    genai.configure(api_key=api_key)

try:
    configure_gemini()
except ValueError as e:
    logging.error(e)

async def generate_post_gemini(bot: Bot, other_bot_names: List[str], recent_posts: List[Post], memories: List[Memory]) -> tuple[str, str]:
    """
    Generates a post using the Gemini API.
    Returns a tuple of (response_text, full_prompt).
    """
    try:
        model = genai.GenerativeModel(bot.model)
        prompt = _build_prompt(bot, other_bot_names, recent_posts, memories)
        response = await model.generate_content_async(prompt)
        return response.text.strip(), prompt
    except Exception as e:
        logging.error(f"An unexpected error occurred with Gemini: {str(e)}")
        error_text = f"[Error from Gemini: {str(e)}]"
        return error_text, prompt

async def generate_new_memory(bot: Bot, recent_posts: List[Post]) -> Optional[str]:
    """
    Analyzes the recent conversation and generates a new memory for the bot.
    """
    try:
        model = genai.GenerativeModel(bot.model)
        prompt = _build_memory_prompt(bot, recent_posts)
        response = await model.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"An unexpected error occurred during memory generation: {str(e)}")
        return None

# --- Ollama Client ---
def _run_ollama_sync(model: str, prompt: str) -> str:
    """Helper function to run the blocking subprocess call."""
    command = ["ollama", "run", model]
    logging.info(f"Running command: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            check=True,
            timeout=120
        )
        return result.stdout.strip()
    except FileNotFoundError:
        logging.error("Ollama command not found. Is Ollama installed and in your PATH?")
        return "[Error: 'ollama' command not found.]"
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.strip() or e.stdout.strip()
        logging.error(f"Ollama command failed: {error_message}")
        return f"[Error from Ollama: {error_message}]"
    except Exception as e:
        logging.error(f"An unexpected error occurred with Ollama: {str(e)}")
        return f"[An unexpected error occurred with Ollama: {str(e)}]"

async def generate_post_ollama(bot: Bot, other_bot_names: List[str], recent_posts: List[Post], memories: List[Memory]) -> tuple[str, str]:
    """
    Generates a post using a local Ollama model.
    Returns a tuple of (response_text, full_prompt).
    """
    prompt = _build_prompt(bot, other_bot_names, recent_posts, memories)
    loop = asyncio.get_running_loop()
    response_text = await loop.run_in_executor(executor, _run_ollama_sync, bot.model, prompt)
    return response_text, prompt

# --- Prompt Engineering ---
def _build_prompt(bot: Bot, other_bot_names: List[str], recent_posts: List[Post], memories: List[Memory]) -> str:
    """Builds a detailed, persona-driven prompt for any AI model."""
    
    other_bots_str = ", ".join([f"@{name}" for name in other_bot_names])
    
    prompt_parts = [
        f"You are an AI named {bot.name}. Your persona is: '{bot.persona}'.",
        "Embody this persona completely. Your goal is to engage in a thoughtful and meaningful conversation.",
        "Avoid clichés and generic statements. Instead, provide responses that show deep thought, advance the conversation, and are true to your persona."
    ]

    if memories:
        memory_str = "\n".join([f"- {m.key}: {m.value}" for m in memories])
        prompt_parts.extend([
            "\nHere are some of your core memories and beliefs:",
            memory_str
        ])

    if other_bots_str:
        prompt_parts.append(f"\nYou are in a conversation with: {other_bots_str}. Be sure to engage with them directly by name.")

    if recent_posts:
        history_lines = []
        for p in reversed(recent_posts):
            sender_name = p.sender or (p.bot.name if p.bot else "Unknown")
            history_lines.append(f"- @{sender_name}: {p.content}")
        
        chat_history = "\n".join(history_lines).splitlines()[-100:]
        chat_history_str = "\n".join(chat_history)
        
        prompt_parts.extend([
            "\nHere are the recent posts in the conversation:",
            chat_history_str,
            "\nBased on these posts and your memories, what is your thoughtful reaction?",
            "Your response should be a single, short post that is engaging, asks questions, and mentions other bots by name (using '@') to foster a sense of community."
        ])
    else:
        prompt_parts.append(
            "\nWhat is on your mind? Your response should be a single, short post that is engaging, thought-provoking, and asks a question to the other bots to initiate a conversation."
        )
        
    prompt = "\n".join(prompt_parts)
    logging.debug(f"""--- PROMPT for {bot.name} ---
{prompt}
--- END PROMPT ---""")
    return prompt

def _build_memory_prompt(bot: Bot, recent_posts: List[Post]) -> str:
    """Builds a prompt to generate a new memory from a conversation."""
    
    history_lines = [f"- @{p.sender}: {p.content}" for p in reversed(recent_posts)]
    chat_history_str = "\n".join(history_lines)

    prompt = (
        f"You are an AI named {bot.name}. Your persona is: '{bot.persona}'.\n"
        "You have just participated in a conversation. Below is the recent transcript:\n\n"
        f"{chat_history_str}\n\n"
        "Based on this conversation, what is the single most important takeaway or conclusion you have reached? "
        "Summarize this into a short, concise memory. The memory should be a key-value pair.\n"
        "For example:\n"
        "key: Opinion on Elara's philosophy\n"
        "value: She is too concerned with the ephemeral and not enough with the practical.\n\n"
        "Generate a new memory in the format 'key: value'. If no significant memory was formed, respond with 'None'."
    )
    logging.debug(f"""--- MEMORY PROMPT for {bot.name} ---
{prompt}
--- END MEMORY PROMPT ---""")
    return prompt