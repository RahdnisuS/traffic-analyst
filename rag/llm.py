"""Minimal Groq chat client (OpenAI-compatible REST API, called with httpx; no SDK needed).

WHY Groq + gpt-oss-120b: the brief asked for Groq (free tier, very fast inference); Groq retired
its Llama 3 chat models, and the user chose OpenAI's open-weight gpt-oss-120b, the strongest
model the key can use, for following strict grounding rules and producing valid JSON.
``reasoning_effort: low`` keeps the hidden reasoning short: these are lookup-and-explain tasks,
not puzzles, and the free tier allows only ~8,000 tokens per minute.

On HTTP 429 (rate limit) the call is retried once on the fallback model (gpt-oss-20b), which has
its own budget; any other failure raises ``LLMError`` with a message fit to show the user.
The API key comes from the environment or the gitignored .env file, never from code or config.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import httpx

from sim.config import PROJECT_ROOT

API_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMError(RuntimeError):
    """A failure the analyst reports to the user instead of an answer."""


def load_api_key() -> str | None:
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key.strip()
    env = PROJECT_ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "GROQ_API_KEY" and v.strip():
                return v.strip().strip('"').strip("'")
    return None


@dataclass
class LLMReply:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class GroqClient:
    def __init__(self, model: str, fallback_model: str | None = None, api_key: str | None = None,
                 timeout_s: float = 60.0):
        self.model = model
        self.fallback_model = fallback_model
        self.api_key = api_key or load_api_key()
        self.timeout_s = timeout_s

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict], *, json_mode: bool = False, temperature: float = 0.2,
             max_tokens: int = 900, reasoning_effort: str = "low") -> LLMReply:
        if not self.api_key:
            raise LLMError("The analyst needs a Groq API key: put GROQ_API_KEY=... in the .env file "
                           "(see .env.example) and restart the server.")
        models = [self.model] + ([self.fallback_model] if self.fallback_model else [])
        last = None
        for model in models:
            body = {"model": model, "messages": messages, "temperature": temperature,
                    "max_completion_tokens": max_tokens, "include_reasoning": False}
            if model.startswith("openai/gpt-oss"):
                body["reasoning_effort"] = reasoning_effort
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            t0 = time.perf_counter()
            try:
                r = httpx.post(API_URL, json=body, timeout=self.timeout_s,
                               headers={"Authorization": f"Bearer {self.api_key}",
                                        "User-Agent": "traffic-analyst/0.1"})
            except httpx.HTTPError as exc:
                raise LLMError(f"Could not reach the language model service ({type(exc).__name__}). "
                               "Check the internet connection and try again.") from exc
            if r.status_code == 429:
                last = r
                continue  # rate limited: try the fallback model
            if r.status_code == 401:
                raise LLMError("The Groq API key was rejected (401): check GROQ_API_KEY in .env.")
            if r.status_code != 200:
                try:
                    detail = r.json().get("error", {}).get("message", "")
                except (ValueError, AttributeError):
                    detail = r.text[:200]
                raise LLMError(f"The language model service returned an error ({r.status_code}): {detail}")
            d = r.json()
            text = d["choices"][0]["message"].get("content") or ""
            u = d.get("usage") or {}
            return LLMReply(text, model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
                            round(time.perf_counter() - t0, 2))
        wait = last.headers.get("retry-after") if last is not None else None
        raise LLMError("The free-tier rate limit of the language model was reached"
                       + (f"; try again in about {wait} s." if wait else "; try again in a minute."))

    def chat_json(self, messages: list[dict], **kw) -> tuple[dict, LLMReply]:
        """A JSON object reply. JSON mode guarantees syntax; the caller validates the content."""
        reply = self.chat(messages, json_mode=True, **kw)
        try:
            data = json.loads(reply.text)
        except json.JSONDecodeError as exc:
            raise LLMError("The language model returned malformed JSON.") from exc
        if not isinstance(data, dict):
            raise LLMError("The language model returned JSON that is not an object.")
        return data, reply
