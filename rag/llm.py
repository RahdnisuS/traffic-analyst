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

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from sim.config import PROJECT_ROOT

API_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMError(RuntimeError):
    """A failure the analyst reports to the user instead of an answer."""


class RateLimited(LLMError):
    """Every model's rate limit is exhausted (e.g. the free tier's daily cap)."""


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
                 timeout_s: float = 60.0, cache_dir: str | Path | None = None, wait_on_limit_s: float = 0.0):
        """``cache_dir``: store every reply on disk, keyed by the request (evaluation). ``wait_on_limit_s``:
        on a rate limit whose retry-after is at most this long, wait and retry the same model instead of
        falling back (evaluation: answers must all come from one model)."""
        self.model = model
        self.fallback_model = fallback_model
        self.api_key = api_key or load_api_key()
        self.timeout_s = timeout_s
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.wait_on_limit_s = wait_on_limit_s

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict], *, json_mode: bool = False, temperature: float = 0.2,
             max_tokens: int = 900, reasoning_effort: str = "low") -> LLMReply:
        if not self.api_key:
            raise LLMError("The analyst needs a Groq API key: put GROQ_API_KEY=... in the .env file "
                           "(see .env.example) and restart the server.")
        key = None
        if self.cache_dir is not None:
            # Evaluation: the same request always gets the same (stored) reply, so a run interrupted by
            # the daily rate limit resumes where it stopped, and re-scoring never re-pays for answers.
            key = hashlib.sha256(json.dumps([self.model, messages, json_mode, temperature, max_tokens,
                                             reasoning_effort], sort_keys=True).encode()).hexdigest()
            hit = self.cache_dir / f"{key}.json"
            if hit.is_file():
                d = json.loads(hit.read_text(encoding="utf-8"))
                return LLMReply(d["text"], d["model"], d["prompt_tokens"], d["completion_tokens"], 0.0)
        reply = self._call(messages, json_mode, temperature, max_tokens, reasoning_effort)
        if key is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.json").write_text(json.dumps(reply.__dict__), encoding="utf-8")
        return reply

    def _call(self, messages, json_mode, temperature, max_tokens, reasoning_effort) -> LLMReply:
        models = [self.model] + ([self.fallback_model] if self.fallback_model else [])
        last = None
        waits = 0
        i = 0
        while i < len(models):
            model = models[i]
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
                try:
                    retry = float(r.headers.get("retry-after", "60"))
                except ValueError:
                    retry = 60.0
                if self.wait_on_limit_s and retry <= self.wait_on_limit_s and waits < 20:
                    waits += 1  # per-minute budget: wait it out on the SAME model (evaluation)
                    time.sleep(retry + 1.0)
                    continue
                i += 1  # rate limited: try the fallback model
                continue
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
        try:  # which limit (per minute, per day, requests): kept for diagnosis
            detail = (last.json().get("error") or {}).get("message", "") if last is not None else ""
        except (ValueError, AttributeError):
            detail = ""
        exc = RateLimited("The free-tier rate limit of the language model was reached"
                          + (f"; try again in about {wait} s." if wait else "; try again in a minute."))
        exc.detail = detail
        raise exc

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
