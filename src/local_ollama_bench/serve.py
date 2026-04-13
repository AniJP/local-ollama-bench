from __future__ import annotations

import argparse


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="Run Local Ollama Assistant (FastAPI + UI).")
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default: %(default)s)")
    p.add_argument("--port", type=int, default=8765, help="Bind port (default: %(default)s)")
    p.add_argument("--reload", action="store_true", help="Dev auto-reload")
    args = p.parse_args()

    uvicorn.run(
        "local_ollama_bench.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
