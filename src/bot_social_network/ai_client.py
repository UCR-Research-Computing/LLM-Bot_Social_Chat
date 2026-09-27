"""Model calls: Gemini and Gemma 4 through the Gemini API (google-genai, async).

Every call returns a `Reply` with text, token usage, cost and latency, or raises
`ModelError` with a reason a person can act on. Nothing here writes to the database.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from pydantic import BaseModel, Field

from . import settings
from .database import Bot, Memory, Post
from .dynamics import Obligation, find_mentions, sanitize_post

log = logging.getLogger(__name__)

MAX_POST_TOKENS = int(os.environ.get("BSN_MAX_POST_TOKENS", "400"))
HISTORY_POSTS = 30
MAX_POST_WORDS = 120  # hard cap; the prompt asks for under 80


class ModelError(RuntimeError):
    """A model call failed; `reason` is short and user-facing."""

    def __init__(self, reason: str, model: str):
        super().__init__(f"{model}: {reason}")
        self.reason = reason
        self.model = model


@dataclass
class Reply:
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    finish: str = ""
    fixes: list[str] = field(default_factory=list)  # sanitizer changes


class MemoryNote(BaseModel):
    """Structured memory the model returns as JSON (no more 'key: value' parsing)."""

    worth_keeping: bool = Field(
        description="false if nothing in the transcript is worth remembering"
    )
    key: str = Field(default="", description="short label, under 8 words")
    value: str = Field(default="", description="the takeaway, one sentence")


# --------------------------------------------------------------------------- prompts


def system_prompt(bot: Bot, others: Sequence[str], memories: Sequence[Memory]) -> str:
    lines = [
        f"You are {bot.name}, one member of a small online group chat.",
        f"Your persona: {bot.persona.strip()}",
        "",
        "Stay fully in character. Write ONE short post (1-4 sentences, under 80 words).",
        "Advance the conversation: react to something specific, add a new idea or a "
        "concrete detail, and when it fits ask a direct question.",
        "Address people with @Name (their full name). Never write other members' "
        "lines, never add a name prefix like 'Name:' to your own post, no hashtags, "
        "no stage directions.",
    ]
    if others:
        lines.append("Other members: " + ", ".join(f"@{n}" for n in others) + ".")
    if memories:
        lines.append("")
        lines.append("Things you remember and believe:")
        lines.extend(f"- {m.key}: {m.value}" for m in memories)
    return "\n".join(lines)


def mentions(text: str, names: Sequence[str]) -> list[str]:
    """Member names @mentioned in text (see dynamics.find_mentions)."""
    return find_mentions(text, names)


def inbox(
    recent_posts: Sequence[Post], bot_name: str, names: Sequence[str]
) -> list[Post]:
    """Posts that @mention bot_name since bot_name last spoke (oldest first).

    recent_posts are newest first. This is what the bot owes a reply to.
    """
    out: list[Post] = []
    for p in recent_posts:
        if p.error:
            continue
        if p.sender == bot_name:
            break
        if bot_name in mentions(p.content, names):
            out.append(p)
    return list(reversed(out))


def conversation_prompt(
    recent_posts: Sequence[Post],
    bot_name: str,
    names: Sequence[str] = (),
    questions: Sequence[Obligation] = (),
    avoid: str | None = None,
    closing: bool = False,
) -> str:
    """recent_posts are newest first (as stored); render oldest first, then say
    exactly who is waiting on this bot and what the latest post is."""
    if not recent_posts:
        return (
            "The chat is empty. Open with something on your mind that the others "
            "will want to answer."
        )
    shown = [p for p in reversed(list(recent_posts)[:HISTORY_POSTS]) if not p.error]
    rows = [
        f"@{p.sender or (p.bot.name if p.bot else 'Unknown')}: {p.content}"
        for p in shown
    ]
    parts = ["Recent chat (oldest first):", *rows, ""]
    waiting = inbox(recent_posts, bot_name, names) if names else []
    if waiting:
        parts.append(f"Addressed to you ({bot_name}) since you last spoke:")
        parts += [f"- @{p.sender}: {p.content}" for p in waiting]
        askers = ", ".join(dict.fromkeys(f"@{p.sender}" for p in waiting))
        parts.append(
            f"Reply to {askers} first: answer their questions directly and refer to "
            "what they actually said. Then add your own angle or a question."
        )
    if questions:
        parts.append("Questions you still owe an answer to (answer each one):")
        parts += [f'- @{o.asker} asked: "{o.question}"' for o in questions]
    elif shown:
        last = shown[-1]
        parts.append(
            f"The latest post is from @{last.sender}. Respond to it or to an earlier "
            "point that still needs an answer, then move the conversation forward."
        )
    if avoid:
        parts.append(
            "Your draft repeated this earlier post too closely, so say something "
            f'new and use different wording: "{avoid}"'
        )
    if closing:
        parts.append(
            "The chat is wrapping up. Give your answer and a closing thought; do not "
            "ask anyone a new question."
        )
    parts.append(f"Write {bot_name}'s next post.")
    return "\n".join(parts)


def memory_prompt(bot: Bot, recent_posts: Sequence[Post]) -> str:
    rows = [
        f"@{p.sender}: {p.content}" for p in reversed(list(recent_posts)) if not p.error
    ]
    return (
        f"You are {bot.name}. Persona: {bot.persona.strip()}\n\n"
        "Transcript of the last few posts:\n" + "\n".join(rows) + "\n\n"
        f"Is there one new fact, opinion or relationship insight {bot.name} should "
        "remember from this? Only keep something specific and new; otherwise set "
        "worth_keeping to false."
    )


def clean_post(text: str, bot_name: str) -> str:
    """Strip a leading 'Name:' / '@Name:' the model sometimes adds, and quotes."""
    t = (text or "").strip()
    for prefix in (f"@{bot_name}:", f"{bot_name}:", f"**{bot_name}:**"):
        if t.lower().startswith(prefix.lower()):
            t = t[len(prefix) :].strip()
    if len(t) >= 2 and t[0] == t[-1] == '"':
        t = t[1:-1].strip()
    return t


# --------------------------------------------------------------------------- gemini


class GeminiClient:
    """One shared google-genai client with SDK-level retries on 408/429/5xx."""

    def __init__(self, api_key: str | None = None):
        self._key = api_key
        self._client: Any = None
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return bool(self._key or os.environ.get("GEMINI_API_KEY"))

    def client(self) -> Any:
        with self._lock:
            if self._client is None:
                from google import genai
                from google.genai import types

                key = self._key or os.environ.get("GEMINI_API_KEY")
                if not key:
                    raise ModelError(
                        "GEMINI_API_KEY is not set (put it in "
                        f"{settings.CONFIG_DIR / '.env'})",
                        "gemini",
                    )
                self._client = genai.Client(
                    api_key=key,
                    http_options=types.HttpOptions(
                        timeout=90_000,
                        retry_options=types.HttpRetryOptions(
                            attempts=4, initial_delay=1.0, max_delay=20.0
                        ),
                    ),
                )
            return self._client

    def _config(self, info: settings.ModelInfo, **extra: Any) -> Any:
        from google.genai import types

        cfg: dict[str, Any] = {
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            **extra,
        }
        if info.thinking == "budget0":
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        elif info.thinking in ("minimal", "low"):
            level = (
                types.ThinkingLevel.MINIMAL
                if info.thinking == "minimal"
                else types.ThinkingLevel.LOW
            )
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=level)
        return types.GenerateContentConfig(**cfg)

    async def generate(
        self,
        model: str,
        system: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int = MAX_POST_TOKENS,
        schema: type[BaseModel] | None = None,
    ) -> tuple[Reply, Any]:
        info = settings.model_info(model)
        extra: dict[str, Any] = {"system_instruction": system}
        # Thinking tokens count against max_output_tokens, so reserve room for them
        # or replies get cut off mid-sentence (finish MAX_TOKENS).
        extra["max_output_tokens"] = max_tokens + info.think_room
        if temperature is not None:
            extra["temperature"] = temperature
        if schema is not None:
            extra["response_mime_type"] = "application/json"
            extra["response_schema"] = schema
        t0 = time.monotonic()
        client = self.client()  # keep a reference for the whole call
        resp = await self._call(client, model, prompt, self._config(info, **extra))
        if (
            schema is None
            and _finish(resp) == "MAX_TOKENS"
            and not _has_room(resp, max_tokens)
        ):
            # Thinking ate the budget before the answer finished; retry once with room.
            extra["max_output_tokens"] = extra["max_output_tokens"] * 2 + 1024
            resp2 = await self._call(client, model, prompt, self._config(info, **extra))
            resp = _merge_usage(resp2, resp)
        ms = int((time.monotonic() - t0) * 1000)
        u = getattr(resp, "usage_metadata", None)
        tin = (getattr(u, "prompt_token_count", 0) or 0) if u else 0
        tout = (
            (getattr(u, "candidates_token_count", 0) or 0)
            + (getattr(u, "thoughts_token_count", 0) or 0)
            if u
            else 0
        )
        cand = (getattr(resp, "candidates", None) or [None])[0]
        finish = _finish(resp)
        text = (resp.text or "").strip() if cand else ""
        cost = tin / 1e6 * info.in_per_m + tout / 1e6 * info.out_per_m
        reply = Reply(text, model, tin, tout, round(cost, 6), ms, finish)
        if schema is None and not text:
            fb = getattr(resp, "prompt_feedback", None)
            why = finish or (
                f"blocked: {getattr(fb, 'block_reason', '')}" if fb else "empty"
            )
            raise ModelError(f"no text in reply ({why})", model)
        parsed = getattr(resp, "parsed", None) if schema is not None else None
        return reply, parsed

    async def _call(self, client: Any, model: str, prompt: str, config: Any) -> Any:
        try:
            return await client.aio.models.generate_content(
                model=model, contents=prompt, config=config
            )
        except ModelError:
            raise
        except Exception as e:  # google.genai.errors.APIError and transport errors
            raise ModelError(_short_error(e), model) from e


def _finish(resp: Any) -> str:
    cand = (getattr(resp, "candidates", None) or [None])[0]
    return getattr(getattr(cand, "finish_reason", None), "name", "") or ""


def _has_room(resp: Any, max_tokens: int) -> bool:
    """True if the visible answer itself hit the post limit (a genuinely long post)."""
    u = getattr(resp, "usage_metadata", None)
    return bool(u and (getattr(u, "candidates_token_count", 0) or 0) >= max_tokens)


def _merge_usage(new: Any, old: Any) -> Any:
    """Bill both attempts: add the first call's tokens to the retry's usage."""
    nu, ou = getattr(new, "usage_metadata", None), getattr(old, "usage_metadata", None)
    if nu is not None and ou is not None:
        for f in (
            "prompt_token_count",
            "candidates_token_count",
            "thoughts_token_count",
        ):
            try:
                setattr(nu, f, (getattr(nu, f, 0) or 0) + (getattr(ou, f, 0) or 0))
            except Exception:
                pass
    return new


def _short_error(e: Exception) -> str:
    code = getattr(e, "code", None)
    msg = getattr(e, "message", None) or str(e)
    msg = " ".join(str(msg).split())[:200]
    hints = {
        400: "bad request",
        401: "API key rejected",
        403: "API key not allowed for this model",
        404: "model not found",
        429: "rate limited / quota exhausted",
    }
    if code in hints:
        return f"{code} {hints[code]}: {msg}"
    return f"{code} {msg}" if code else msg


# --------------------------------------------------------------------------- facade


class AIClient:
    def __init__(self, gemini: GeminiClient | None = None):
        self.gemini = gemini or GeminiClient()

    async def write_post(
        self,
        bot: Bot,
        others: Sequence[str],
        recent_posts: Sequence[Post],
        memories: Sequence[Memory],
        questions: Sequence[Obligation] = (),
        avoid: str | None = None,
        closing: bool = False,
    ) -> Reply:
        names = [bot.name, *others]
        system = system_prompt(bot, others, memories)
        prompt = conversation_prompt(
            recent_posts, bot.name, names, questions, avoid, closing
        )
        reply, _ = await self.gemini.generate(
            bot.model, system, prompt, bot.temperature
        )
        cleaned = sanitize_post(reply.text, bot.name, names, max_words=MAX_POST_WORDS)
        reply.text = cleaned.text
        reply.fixes = cleaned.fixes
        if not reply.text:
            raise ModelError("reply was only a name prefix", bot.model)
        return reply

    async def form_memory(
        self, bot: Bot, recent_posts: Sequence[Post]
    ) -> tuple[MemoryNote | None, Reply | None]:
        """Structured JSON memory from a cheap Gemini model."""
        prompt = memory_prompt(bot, recent_posts)
        system = "You extract durable memories for a chat character. Reply in JSON."
        if not self.gemini.configured:
            return None, None
        reply, parsed = await self.gemini.generate(
            settings.MEMORY_MODEL, system, prompt, max_tokens=200, schema=MemoryNote
        )
        note = parsed if isinstance(parsed, MemoryNote) else None
        if note is None and reply.text:
            try:
                note = MemoryNote.model_validate_json(_json_block(reply.text))
            except Exception:
                note = None
        return note, reply


def _json_block(text: str) -> str:
    t = text.strip()
    if "```" in t:
        t = t.split("```")[1]
        t = t[4:] if t.startswith("json") else t
    a, b = t.find("{"), t.rfind("}")
    return t[a : b + 1] if a >= 0 and b > a else t


async def gather_limited(coros: Sequence[Any], limit: int = 4) -> list[Any]:
    sem = asyncio.Semaphore(limit)

    async def run(c: Any) -> Any:
        async with sem:
            return await c

    return await asyncio.gather(*(run(c) for c in coros), return_exceptions=True)
