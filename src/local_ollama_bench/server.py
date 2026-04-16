from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, AsyncIterator, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError
from pydantic import BaseModel, Field, field_validator

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

app = FastAPI(
    title="Local Ollama Assistant",
    description="Thin proxy + simple UI for streaming Ollama /api/chat (optional JSON / JSON Schema via Ollama `format`).",
    version="0.4.0",
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)


class ChatStreamBody(BaseModel):
    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    # Ollama: "json" for any JSON object, or a JSON Schema object for structured outputs.
    format: str | dict[str, Any] | None = Field(
        default=None,
        description='Structured output: "json" or a JSON Schema object (Ollama `format`).',
    )
    # Ollama generation options, e.g. temperature, seed, num_predict.
    options: dict[str, Any] | None = Field(default=None)
    # When format is "json" or a JSON Schema dict, validate the assistant reply and retry with feedback.
    max_validation_attempts: int = Field(default=3, ge=1, le=12)

    @field_validator("format")
    @classmethod
    def _format_not_empty_string(cls, v: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
        if isinstance(v, str) and not v.strip():
            return None
        return v


def _extract_json_text(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def _validate_structured_reply(raw: str, fmt: str | dict[str, Any]) -> tuple[bool, str]:
    text = _extract_json_text(raw)
    if not text:
        return False, "empty reply"
    try:
        inst = json.loads(text)
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}"
    if fmt == "json":
        return True, ""
    if isinstance(fmt, dict):
        try:
            Draft7Validator(fmt).validate(inst)
        except ValidationError as e:
            return False, e.message
        return True, ""
    return True, ""


def _structured_validation_enabled(fmt: str | dict[str, Any] | None) -> bool:
    if fmt is None:
        return False
    if fmt == "json":
        return True
    return isinstance(fmt, dict)


def _static_index_path() -> Path:
    return Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    path = _static_index_path()
    if not path.is_file():
        raise HTTPException(status_code=500, detail="UI bundle missing (static/index.html).")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """App liveness plus whether Ollama responds at OLLAMA_BASE."""
    out: dict[str, Any] = {"status": "ok", "ollama_base": OLLAMA_BASE}
    url = f"{OLLAMA_BASE}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(url)
    except httpx.RequestError as e:
        out["ollama_reachable"] = False
        out["ollama_error"] = str(e)
        return out
    if r.status_code >= 400:
        out["ollama_reachable"] = False
        out["ollama_error"] = f"HTTP {r.status_code}: {r.text[:400]}"
        return out
    out["ollama_reachable"] = True
    return out


@app.get("/api/models")
async def list_models() -> dict[str, list[str]]:
    url = f"{OLLAMA_BASE}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Ollama unreachable at {OLLAMA_BASE}: {e}. "
                "Start the Ollama app (or run `ollama serve`), or set OLLAMA_BASE_URL if it runs elsewhere."
            ),
        ) from e
    if r.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Ollama returned {r.status_code}: {r.text[:500]}. "
                f"Check that Ollama is healthy at {OLLAMA_BASE}."
            ),
        )
    data = r.json()
    names = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    names.sort()
    return {"models": names}


async def _ollama_chat_stream(payload: dict) -> AsyncIterator[bytes]:
    url = f"{OLLAMA_BASE}/api/chat"
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    err = body.decode("utf-8", errors="replace")[:2000]
                    yield (json.dumps({"error": f"Ollama HTTP {resp.status_code}: {err}"}) + "\n").encode(
                        "utf-8"
                    )
                    return
                async for chunk in resp.aiter_bytes():
                    yield chunk
        except httpx.RequestError as e:
            yield (json.dumps({"error": str(e)}) + "\n").encode("utf-8")


async def _ollama_chat_complete(payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{OLLAMA_BASE}/api/chat"
    req = {**payload, "stream": False}
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            r = await client.post(url, json=req)
        except httpx.RequestError as e:
            return {"error": str(e)}
    if r.status_code >= 400:
        return {"error": f"Ollama HTTP {r.status_code}: {r.text[:2000]}"}
    try:
        return r.json()
    except json.JSONDecodeError:
        return {"error": f"Ollama returned non-JSON body: {r.text[:500]}"}


async def _ollama_chat_stream_validated(body: ChatStreamBody) -> AsyncIterator[bytes]:
    fmt = body.format
    assert fmt is not None and _structured_validation_enabled(fmt)

    max_attempts = body.max_validation_attempts
    messages_work: list[dict[str, str]] = [m.model_dump() for m in body.messages]
    last_err = ""

    for attempt in range(1, max_attempts + 1):
        payload: dict[str, Any] = {
            "model": body.model,
            "messages": messages_work,
            "stream": False,
            "format": fmt,
        }
        if body.options:
            payload["options"] = body.options

        data = await _ollama_chat_complete(payload)
        if err := data.get("error"):
            yield (json.dumps({"error": str(err)}) + "\n").encode("utf-8")
            return

        msg = data.get("message") or {}
        content = (msg.get("content") or "").strip()
        ok, val_err = _validate_structured_reply(content, fmt)
        if ok:
            parsed = json.loads(_extract_json_text(content))
            normalized = json.dumps(parsed, ensure_ascii=False)
            line = json.dumps(
                {
                    "model": data.get("model", body.model),
                    "message": {"role": "assistant", "content": normalized},
                    "done": True,
                    "validation_attempts": attempt,
                }
            )
            yield (line + "\n").encode("utf-8")
            return

        last_err = val_err
        messages_work.append({"role": "assistant", "content": content or "(empty)"})
        fix = (
            f"Your previous reply was invalid: {val_err}. "
            "Respond with ONLY valid JSON"
            + (" matching the requested schema, with no markdown fences or extra text." if isinstance(fmt, dict) else ", with no markdown fences or extra text.")
        )
        messages_work.append({"role": "system", "content": fix})

    yield (
        json.dumps(
            {
                "error": f"Validation failed after {max_attempts} attempts: {last_err}",
                "validation_attempts": max_attempts,
            }
        )
        + "\n"
    ).encode("utf-8")


@app.post("/api/chat/stream")
async def chat_stream(body: ChatStreamBody) -> StreamingResponse:
    payload: dict[str, Any] = {
        "model": body.model,
        "messages": [m.model_dump() for m in body.messages],
        "stream": True,
    }
    if body.format is not None:
        payload["format"] = body.format
    if body.options:
        payload["options"] = body.options

    if _structured_validation_enabled(body.format):
        return StreamingResponse(
            _ollama_chat_stream_validated(body),
            media_type="application/x-ndjson; charset=utf-8",
        )

    return StreamingResponse(
        _ollama_chat_stream(payload),
        media_type="application/x-ndjson; charset=utf-8",
    )
