"""
FastAPI application factory and background thread management
for the Deep Research Agent web dashboard.
"""
import io
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .routes import router

# ---------------------------------------------------------------------------
# Background research thread state
# ---------------------------------------------------------------------------
_research_thread: threading.Thread | None = None
_research_lock = threading.Lock()
_current_orchestrator = None  # holds ResearchOrchestrator ref for stop


def is_research_running() -> bool:
    """Check if a research thread is currently active."""
    with _research_lock:
        return _research_thread is not None and _research_thread.is_alive()


def start_research_background(query: str, overrides: dict = None) -> bool:
    """
    Launch a research run in a daemon thread.
    Returns False if research is already running.
    """
    global _research_thread, _current_orchestrator

    with _research_lock:
        if _research_thread is not None and _research_thread.is_alive():
            return False

        def _worker():
            global _current_orchestrator
            # Suppress Rich console output so it doesn't garble the
            # terminal that uvicorn owns.  File logging still works.
            import src.logger as _logger_mod
            _logger_mod.console = type(_logger_mod.console)(file=io.StringIO())

            # Enable RBC SSL certs if available (needed in worker thread)
            from src.utils.rbc_security import configure_rbc_security_certs
            configure_rbc_security_certs()

            from src.config import get_config, set_config, apply_overrides
            from src.orchestrator import ResearchOrchestrator

            original = get_config()
            if overrides:
                overridden = apply_overrides(original, overrides)
                set_config(overridden)

            orchestrator = ResearchOrchestrator(register_signals=False)
            _current_orchestrator = orchestrator
            try:
                orchestrator.run(query)
            except Exception as e:
                from src.logger import get_logger
                get_logger(__name__).exception(
                    "Background research worker failed: %s", e
                )
            finally:
                _current_orchestrator = None
                if overrides:
                    set_config(original)

        _research_thread = threading.Thread(target=_worker, daemon=True)
        _research_thread.start()
        return True


def get_current_phase() -> str:
    """Return the current orchestrator phase, or 'idle' if not running."""
    if _current_orchestrator is not None:
        return _current_orchestrator.phase
    return "idle"


def stop_research() -> bool:
    """
    Gracefully stop the running research by setting is_running = False.
    Returns True if a running orchestrator was signalled.
    """
    global _current_orchestrator
    if _current_orchestrator is not None:
        _current_orchestrator.is_running = False
        return True
    return False


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
