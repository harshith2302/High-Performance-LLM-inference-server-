"""
main.py — entry point. Launches the FastAPI app with uvicorn.

OWNED BY: Ajay (M1).

Run from the repo root:
    set PYTHONPATH=src        (Windows)
    python -m server.main

Then open http://127.0.0.1:8000/docs in a browser for the interactive API.
"""

from __future__ import annotations

import uvicorn


def run() -> None:
    uvicorn.run(
        "server.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    run()
