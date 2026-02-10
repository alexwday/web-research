"""MCP server for Deep Research Agent.

Exposes research tools via the Model Context Protocol (stdio transport).
All logic delegates to ResearchService.
"""
from datetime import datetime, timezone
from typing import Optional

from mcp.server.fastmcp import FastMCP

from src.service import get_service

mcp = FastMCP(
    name="deep-research",
    instructions="Deep Research Agent — run and monitor comprehensive web research sessions.",
)


@mcp.tool()
def research_presets() -> dict:
    """Return available research presets (quick, standard, deep, exhaustive).

    Each preset includes a label, description, and configuration overrides
    that control research depth, task counts, and output length.
    """
    return {"presets": get_service().list_presets()}


@mcp.tool()
def research_status(run_id: Optional[int] = None) -> dict:
    """Return current status and metrics for a research run.

    Args:
        run_id: Session ID to query. If omitted, returns the most recent session.

    Returns a dict with:
        - run_id: session ID
        - status: running/completed/partial/cancelled/failed/no_session
        - phase: current pipeline phase (idle, pre_planning, researching, etc.)
        - running: whether research is actively executing
        - progress: task completion counts and percentage
        - timing: start/end times and elapsed_seconds (int)
        - counts: sources, words, failed tasks
        - costs: token usage and cost breakdown
        - cancel_requested_at: ISO timestamp if cancellation was requested, else null
    """
    service = get_service()
    raw = service.get_run_status(session_id=run_id)

    session_id = raw.get("session_id")
    if session_id is None:
        return raw

    from src.database import get_database

    db = get_database()
    session = db.get_session_by_id(session_id)

    stats = raw.get("statistics", {})
    total = stats.get("total_tasks", 0)
    completed = stats.get("completed_tasks", 0)
    pct = int((completed / total) * 100) if total > 0 else 0

    # Compute elapsed seconds
    elapsed_seconds = None
    if session and session.started_at:
        started = session.started_at
        # Ensure timezone-aware comparison
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        end = session.ended_at if session.ended_at else datetime.now(timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        elapsed_seconds = int((end - started).total_seconds())

    result = {
        "run_id": session_id,
        "status": raw.get("status"),
        "phase": raw.get("phase"),
        "running": raw.get("running", False),
        "progress": {
            "completed_tasks": completed,
            "total_tasks": total,
            "pct": pct,
        },
        "timing": {
            "started_at": session.started_at.isoformat() if session and session.started_at else None,
            "ended_at": session.ended_at.isoformat() if session and session.ended_at else None,
            "elapsed_seconds": elapsed_seconds,
        },
        "counts": {
            "sources": stats.get("total_sources", 0),
            "words": stats.get("total_words", 0),
            "failed_tasks": stats.get("failed_tasks", 0),
        },
        "costs": raw.get("costs", {}),
        "cancel_requested_at": (
            session.cancel_requested_at.isoformat()
            if session and session.cancel_requested_at else None
        ),
    }
    return result


@mcp.tool()
def research_start(
    query: str,
    preset: Optional[str] = None,
) -> dict:
    """Start a new research session.

    Args:
        query: The research topic or question to investigate.
        preset: Optional preset name (quick, standard, deep, exhaustive)
                that controls research depth and output length.
                Call research_presets() to see available options.

    Returns a dict with:
        - status: "started" or "already_running"
        - run_id: session ID (when started), for use with research_status()
    """
    if not query or not query.strip():
        return {"status": "error", "error": "Query is required"}

    service = get_service()
    result = service.start_run(query.strip(), mode=preset, blocking=False)

    if result.get("status") == "already_running":
        return {"status": "already_running"}

    # Return run_id so the caller can poll research_status()
    from src.database import get_database

    db = get_database()
    session = db.get_current_session()
    run_id = session.id if session else None

    return {"status": "started", "run_id": run_id}


@mcp.tool()
def research_cancel() -> dict:
    """Cancel the currently running research session.

    Sends a graceful stop signal. The session will finish its current task
    and then stop. The session's terminal status will be "cancelled".
    Use research_status() to confirm it has stopped.

    Returns a dict with:
        - status: "cancelling" or "not_running"
        - run_id: session ID of the cancelled run (when cancelling)
    """
    service = get_service()
    run_id = None
    if service._orchestrator is not None:
        run_id = service._orchestrator.session_id
    result = service.cancel_run()
    if run_id is not None:
        result["run_id"] = run_id
    return result


# Entry point for standalone execution: python -m src.mcp_server
if __name__ == "__main__":
    mcp.run()
