"""Model calls: Gemini (google-genai, async) and Ollama (HTTP API).

Every call returns a `Reply` with text, token usage, cost and latency, or raises
`ModelError` with a reason a person can act on. Nothing here writes to the database.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

import httpx
from pydantic import BaseModel, Field

from . import settings
from .database import Bot, Memory, Post

log = logging.getLogger(__name__)

MAX_POST_TOKENS = int(os.environ.get("BSN_MAX_POST_TOKENS", "400"))
HISTORY_POSTS = 30


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
        "Address people with @Name. Never write other members' lines, never add a "
        "name prefix like 'Name:' to your own post, no hashtags, no stage directions.",
    ]
    if others:
        lines.append("Other members: " + ", ".join(f"@{n}" for n in others) + ".")
    if memories:
        lines.append("")
        lines.append("Things you remember and believe:")
        lines.extend(f"- {m.key}: {m.value}" for m in memories)
    return "\n".join(lines)


def conversation_prompt(recent_posts: Sequence[Post], bot_name: str) -> str:
    """recent_posts are newest first (as stored); render oldest first."""
    if not recent_posts:
        return (
            "The chat is empty. Open with something on your mind that the others "
            "will want to answer."
        )
    rows = [
        f"@{p.sender or (p.bot.name if p.bot else 'Unknown')}: {p.content}"
        for p in reversed(list(recent_posts)[:HISTORY_POSTS])
        if not p.error
    ]
    return (
        "Recent chat (oldest first):\n"
        + "\n".join(rows)
        + f"\n\nWrite {bot_name}'s next post."
    )


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


# --------------------------------------------------------------------------- ollama


class OllamaClient:
    def __init__(self, base_url: str = settings.OLLAMA_URL):
        self.base_url = base_url

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(f"{self.base_url}/api/tags")
                r.raise_for_status()
                return sorted(m["name"] for m in r.json().get("models", []))
        except Exception as e:
            log.info("Ollama not reachable: %s", e)
            return []

    async def generate(
        self,
        model: str,
        system: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int = MAX_POST_TOKENS,
    ) -> Reply:
        body: dict[str, Any] = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {"num_predict": max_tokens},
            "think": False,
        }
        if temperature is not None:
            body["options"]["temperature"] = temperature
        t0 = time.monotonic()
        data: dict[str, Any] = {}
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=180) as c:
                    r = await c.post(f"{self.base_url}/api/chat", json=body)
                if r.status_code == 400 and "think" in r.text and attempt == 0:
                    body.pop("think", None)  # model without a thinking switch
                    continue
                if r.status_code == 404:
                    raise ModelError(
                        f"not pulled locally (run: ollama pull {model})", model
                    )
                r.raise_for_status()
                data = r.json()
                break
            except ModelError:
                raise
            except httpx.ConnectError as e:
                raise ModelError(
                    f"Ollama is not running at {self.base_url}", model
                ) from e
            except Exception as e:
                raise ModelError(_short_error(e), model) from e
        text = ((data.get("message") or {}).get("content") or "").strip()
        if not text:
            raise ModelError("no text in reply", model)
        return Reply(
            text,
            model,
            int(data.get("prompt_eval_count") or 0),
            int(data.get("eval_count") or 0),
            0.0,
            int((time.monotonic() - t0) * 1000),
            str(data.get("done_reason") or ""),
        )


# --------------------------------------------------------------------------- facade


class AIClient:
    def __init__(
        self, gemini: GeminiClient | None = None, ollama: OllamaClient | None = None
    ):
        self.gemini = gemini or GeminiClient()
        self.ollama = ollama or OllamaClient()

    async def write_post(
        self,
        bot: Bot,
        others: Sequence[str],
        recent_posts: Sequence[Post],
        memories: Sequence[Memory],
    ) -> Reply:
        system = system_prompt(bot, others, memories)
        prompt = conversation_prompt(recent_posts, bot.name)
        info = settings.model_info(bot.model)
        if info.provider == "ollama":
            reply = await self.ollama.generate(
                bot.model, system, prompt, bot.temperature
            )
        else:
            reply, _ = await self.gemini.generate(
                bot.model, system, prompt, bot.temperature
            )
        reply.text = clean_post(reply.text, bot.name)
        if not reply.text:
            raise ModelError("reply was only a name prefix", bot.model)
        return reply

    async def form_memory(
        self, bot: Bot, recent_posts: Sequence[Post]
    ) -> tuple[MemoryNote | None, Reply | None]:
        """Structured JSON memory. Uses a cheap Gemini model; Ollama bots use their own."""
        prompt = memory_prompt(bot, recent_posts)
        system = "You extract durable memories for a chat character. Reply in JSON."
        info = settings.model_info(bot.model)
        if info.provider == "ollama":
            reply = await self.ollama.generate(
                bot.model, system + " Keys: worth_keeping, key, value.", prompt
            )
            try:
                return MemoryNote.model_validate_json(_json_block(reply.text)), reply
            except Exception:
                return None, reply
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
