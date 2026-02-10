"""
ResearchService — facade for research run lifecycle.

Single API for CLI, web, and future MCP adapters.  Owns background-thread
management so that callers never construct orchestrators directly.
"""
import io
import threading
from typing import Optional

from .config import get_config, set_config, apply_overrides, RESEARCH_PRESETS
from .database import get_database
from .llm_client import get_token_tracker
from .logger import get_logger

logger = get_logger(__name__)


class ResearchService:
    """Facade for research run lifecycle."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._orchestrator = None  # ResearchOrchestrator | None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(
        self,
        query: str,
        mode: Optional[str] = None,
        overrides: Optional[dict] = None,
        refined_brief: Optional[str] = None,
        refinement_qa: Optional[str] = None,
        resume: bool = False,
        blocking: bool = False,
    ) -> dict:
        """Start a research run.

        Args:
            query: Research topic / question.
            mode: Optional preset name (e.g. "quick", "standard").
            overrides: Dotted-key config overrides layered *on top* of preset.
            refined_brief: Enhanced research brief from query refinement.
            refinement_qa: JSON string of Q&A pairs from refinement.
            resume: Resume an existing session instead of starting fresh.
            blocking: If True, run synchronously and return the result dict.

        Returns:
            ``{"status": "started"}`` for background runs, or the full
            result dict (output_files, statistics, duration) for blocking runs.
        """
        # Merge preset overrides + explicit overrides
        merged_overrides = self._merge_overrides(mode, overrides)

        if blocking:
            return self._run_blocking(
                query,
                merged_overrides=merged_overrides,
                refined_brief=refined_brief,
                refinement_qa=refinement_qa,
                resume=resume,
            )

        return self._run_background(
            query,
            merged_overrides=merged_overrides,
            refined_brief=refined_brief,
            refinement_qa=refinement_qa,
        )

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def get_current_phase(self) -> str:
        if self._orchestrator is not None:
            return self._orchestrator.phase
        return "idle"

    def get_run_status(self, session_id: Optional[int] = None) -> dict:
        """Return status dict for the given (or current) session."""
        db = get_database()
        session = self._resolve_session(db, session_id)
        if not session:
            return {"session_id": None, "status": "no_session", "running": self.is_running()}

        stats = db.get_statistics(session_id=session.id)
        return {
            "session_id": session.id,
            "status": session.status,
            "phase": self.get_current_phase(),
            "running": self.is_running(),
            "statistics": stats,
            "costs": get_token_tracker().get_stats(),
        }

    def cancel_run(self) -> dict:
        if self._orchestrator is not None:
            self._orchestrator._cancel_requested = True
            self._orchestrator.is_running = False
            session_id = self._orchestrator.session_id
            if session_id is not None:
                from datetime import datetime, timezone
                import json
                db = get_database()
                now = datetime.now(timezone.utc)
                db.update_session(session_id, cancel_requested_at=now)
                db.add_run_event(
                    session_id=session_id,
                    event_type="cancellation_requested",
                    phase=self._orchestrator.phase,
                    severity="warning",
                    payload_json=json.dumps({
                        "cancelled_at": now.isoformat(),
                        "phase": self._orchestrator.phase,
                    }),
                )
            return {"status": "cancelling"}
        return {"status": "not_running"}

    def get_run_result(self, session_id: Optional[int] = None) -> dict:
        """Return result artifacts for a completed session."""
        db = get_database()
        session = self._resolve_session(db, session_id)
        if not session:
            return {"session_id": None, "status": "no_session"}

        sections = db.get_all_sections(session_id=session.id)
        section_list = [
            {"title": s.title, "position": s.position, "word_count": s.word_count or 0}
            for s in sections
        ]

        return {
            "session_id": session.id,
            "status": session.status,
            "artifacts": {
                "markdown_path": session.report_markdown_path,
                "html_path": session.report_html_path,
            },
            "executive_summary": session.executive_summary,
            "conclusion": session.conclusion,
            "sections": section_list,
        }

    def list_presets(self) -> dict:
        return RESEARCH_PRESETS

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_overrides(mode: Optional[str], overrides: Optional[dict]) -> Optional[dict]:
        merged: dict = {}
        if mode and mode in RESEARCH_PRESETS:
            merged.update(RESEARCH_PRESETS[mode]["overrides"])
        if overrides:
            merged.update(overrides)
        return merged or None

    @staticmethod
    def _resolve_session(db, session_id: Optional[int]):
        if session_id is not None:
            return db.get_session_by_id(session_id)
        session = db.get_current_session()
        if session:
            return session
        return db.get_most_recent_session()

    def _run_blocking(self, query, merged_overrides, refined_brief, refinement_qa, resume):
        from .orchestrator import ResearchOrchestrator

        original = get_config()
        if merged_overrides:
            overridden = apply_overrides(original, merged_overrides)
            set_config(overridden)

        orchestrator = ResearchOrchestrator(register_signals=True)
        self._orchestrator = orchestrator
        try:
            result = orchestrator.run(
                query,
                resume=resume,
                refined_brief=refined_brief,
                refinement_qa=refinement_qa,
            )
            return result
        finally:
            self._orchestrator = None
            if merged_overrides:
                set_config(original)

    def _run_background(self, query, merged_overrides, refined_brief, refinement_qa):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {"status": "already_running"}

            def _worker():
                # Suppress Rich console output so it doesn't garble uvicorn's terminal.
                import src.logger as _logger_mod
                _logger_mod.console = type(_logger_mod.console)(file=io.StringIO())

                # Enable RBC SSL certs if available (needed in worker thread)
                from .utils.rbc_security import configure_rbc_security_certs
                configure_rbc_security_certs()

                from .orchestrator import ResearchOrchestrator

                original = get_config()
                if merged_overrides:
                    overridden = apply_overrides(original, merged_overrides)
                    set_config(overridden)

                orchestrator = ResearchOrchestrator(register_signals=False)
                self._orchestrator = orchestrator
                try:
                    orchestrator.run(
                        query,
                        refined_brief=refined_brief,
                        refinement_qa=refinement_qa,
                    )
                except Exception as e:
                    logger.exception("Background research worker failed: %s", e)
                finally:
                    self._orchestrator = None
                    if merged_overrides:
                        set_config(original)

            self._thread = threading.Thread(target=_worker, daemon=True)
            self._thread.start()
            return {"status": "started"}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_service: Optional[ResearchService] = None
_service_lock = threading.Lock()


def get_service() -> ResearchService:
    """Return the global ResearchService singleton."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = ResearchService()
    return _service


def reset_service() -> None:
    """Reset the singleton (for testing)."""
    global _service
    with _service_lock:
        _service = None
