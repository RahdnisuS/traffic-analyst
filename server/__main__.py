"""``python -m server [--host 0.0.0.0] [--port 8000]``. PORT env var wins (for hosting later)."""

import argparse
import os

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    args = ap.parse_args()
    uvicorn.run("server.main:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
