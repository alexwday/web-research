"""
FastAPI application factory and background thread management
for the Deep Research Agent web dashboard.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .routes import router
from src.service import get_service

# ---------------------------------------------------------------------------
# Thin wrappers — delegate to ResearchService so that routes.py and tests
# that patch ``src.web.app.is_research_running`` etc. keep working.
# ---------------------------------------------------------------------------


def is_research_running() -> bool:
    """Check if a research thread is currently active."""
    return get_service().is_running()


def start_research_background(query: str, overrides: dict = None,
                              refined_brief: str = None,
                              refinement_qa: str = None) -> bool:
    """
    Launch a research run in a daemon thread.
    Returns False if research is already running.
    """
    result = get_service().start_run(
        query,
        overrides=overrides,
        refined_brief=refined_brief,
        refinement_qa=refinement_qa,
        blocking=False,
    )
    return result.get("status") == "started"


def get_current_phase() -> str:
    """Return the current orchestrator phase, or 'idle' if not running."""
    return get_service().get_current_phase()


def stop_research() -> bool:
    """
    Gracefully stop the running research by setting is_running = False.
    Returns True if a running orchestrator was signalled.
    """
    result = get_service().cancel_run()
    return result.get("status") == "cancelling"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Deep Research Agent", docs_url="/docs")

    # Static files
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Routes
    app.include_router(router)

    return app
