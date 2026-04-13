from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, AsyncIterator, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

app = FastAPI(
    title="Local Ollama Assistant",
    description="Thin proxy + simple UI for streaming Ollama /api/chat (optional JSON / JSON Schema via Ollama `format`).",
    version="0.3.0",
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

    @field_validator("format")
    @classmethod
    def _format_not_empty_string(cls, v: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
        if isinstance(v, str) and not v.strip():
            return None
        return v


def _static_index_path() -> Path:
    return Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    path = _static_index_path()
    if not path.is_file():
        raise HTTPException(status_code=500, detail="UI bundle missing (static/index.html).")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "ollama_base": OLLAMA_BASE}


@app.get("/api/models")
async def list_models() -> dict[str, list[str]]:
    url = f"{OLLAMA_BASE}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {e}") from e
    if r.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned {r.status_code}: {r.text[:500]}",
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
    return StreamingResponse(
        _ollama_chat_stream(payload),
        media_type="application/x-ndjson; charset=utf-8",
    )
