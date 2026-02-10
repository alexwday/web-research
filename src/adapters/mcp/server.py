"""MCP server for Deep Research Agent.

Exposes research tools via the Model Context Protocol (stdio transport).
All logic delegates to ResearchService.
"""
import json
from datetime import datetime, timezone
from typing import Optional

from mcp.server.fastmcp import FastMCP

from src.pipeline.service import get_service

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

    from src.infra._database import get_database

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
    from src.infra._database import get_database

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


@mcp.tool()
def research_events(
    run_id: Optional[int] = None,
    cursor: Optional[str] = None,
    limit: int = 100,
) -> dict:
    """Return a page of run events (queries, results, phase changes, errors).

    Uses cursor-based pagination. Pass the returned ``next_cursor`` value
    back as ``cursor`` to fetch the next page.

    Args:
        run_id: Session ID to query. If omitted, uses the most recent session.
        cursor: Opaque pagination cursor from a previous response.
        limit: Maximum events per page (1-500, default 100).

    Returns a dict with:
        - session_id: resolved session ID
        - events: list of event dicts with event_id, ts, type, task_id,
          query_group, and a payload object
        - next_cursor: pagination cursor for the next page (null if no more)
    """
    return get_service().get_run_events_page(
        session_id=run_id, cursor=cursor, limit=limit,
    )


@mcp.tool()
def research_result(run_id: Optional[int] = None) -> dict:
    """Return the final result of a research run, including report artifacts,
    summary, sections with citation counts, and all sources.

    Args:
        run_id: Session ID to query. If omitted, uses the most recent session.

    Returns a dict with:
        - run_id: session ID
        - status: session status
        - artifacts: markdown_path and html_path
        - summary: executive_summary, conclusion, and sections list
        - sources: list of source dicts with url, title, domain, quality_score,
          is_academic, snippet, and task_ids
    """
    service = get_service()
    base = service.get_run_result(session_id=run_id)

    session_id = base.get("session_id")
    if session_id is None:
        return base

    from src.infra._database import get_database

    db = get_database()

    # Enrich sections with citation_count
    sections = db.get_all_sections(session_id=session_id)
    section_list = [
        {
            "title": s.title,
            "position": s.position,
            "word_count": s.word_count or 0,
            "citation_count": s.citation_count or 0,
        }
        for s in sections
    ]

    # Add sources
    sources = db.get_sources_for_session(session_id)
    source_list = [
        {
            "url": s.url,
            "title": s.title,
            "domain": s.domain,
            "quality_score": s.quality_score,
            "is_academic": s.is_academic,
            "snippet": s.snippet,
            "task_ids": s.task_ids or [],
        }
        for s in sources
    ]

    return {
        "run_id": session_id,
        "status": base.get("status"),
        "artifacts": base.get("artifacts", {}),
        "summary": {
            "executive_summary": base.get("executive_summary"),
            "conclusion": base.get("conclusion"),
            "sections": section_list,
        },
        "sources": source_list,
    }


# =========================================================================
# MCP Resources — read-only data access
# =========================================================================

@mcp.resource("research://runs")
def resource_runs() -> str:
    """List all research runs."""
    from src.infra._database import get_database

    db = get_database()
    sessions = db.get_all_sessions()
    return json.dumps([
        {
            "run_id": s.id,
            "query": s.query,
            "status": s.status,
            "started_at": s.started_at.isoformat() if s.started_at else None,
            "ended_at": s.ended_at.isoformat() if s.ended_at else None,
        }
        for s in sessions
    ])


@mcp.resource("research://runs/{run_id}/status")
def resource_run_status(run_id: int) -> str:
    """Status for a specific run."""
    return json.dumps(research_status(run_id=run_id))


@mcp.resource("research://runs/{run_id}/events")
def resource_run_events(run_id: int) -> str:
    """Events for a specific run (first page)."""
    return json.dumps(research_events(run_id=run_id))


@mcp.resource("research://runs/{run_id}/tasks")
def resource_run_tasks(run_id: int) -> str:
    """Tasks for a specific run."""
    from src.infra._database import get_database

    db = get_database()
    tasks = db.get_all_tasks(session_id=run_id)
    return json.dumps([
        {
            "task_id": t.id,
            "topic": t.topic,
            "status": t.status.value if hasattr(t.status, "value") else t.status,
            "section_id": t.section_id,
            "priority": t.priority,
            "depth": t.depth,
            "word_count": t.word_count or 0,
            "citation_count": t.citation_count or 0,
        }
        for t in tasks
    ])


@mcp.resource("research://runs/{run_id}/sources")
def resource_run_sources(run_id: int) -> str:
    """Sources for a specific run."""
    from src.infra._database import get_database

    db = get_database()
    sources = db.get_sources_for_session(run_id)
    return json.dumps([
        {
            "url": s.url,
            "title": s.title,
            "domain": s.domain,
            "snippet": s.snippet,
            "quality_score": s.quality_score,
            "is_academic": s.is_academic,
            "task_ids": s.task_ids or [],
        }
        for s in sources
    ])


@mcp.resource("research://runs/{run_id}/sections")
def resource_run_sections(run_id: int) -> str:
    """Sections for a specific run."""
    from src.infra._database import get_database

    db = get_database()
    sections = db.get_all_sections(session_id=run_id)
    return json.dumps([
        {
            "section_id": s.id,
            "title": s.title,
            "position": s.position,
            "status": s.status.value if hasattr(s.status, "value") else s.status,
            "word_count": s.word_count or 0,
            "citation_count": s.citation_count or 0,
        }
        for s in sections
    ])


@mcp.resource("research://runs/{run_id}/artifacts")
def resource_run_artifacts(run_id: int) -> str:
    """Report artifacts for a specific run."""
    return json.dumps(research_result(run_id=run_id))


@mcp.resource("research://runs/{run_id}/costs")
def resource_run_costs(run_id: int) -> str:
    """Cost breakdown for a specific run."""
    status = research_status(run_id=run_id)
    return json.dumps({
        "run_id": status.get("run_id"),
        "costs": status.get("costs", {}),
    })


# Entry point for standalone execution: python -m src.mcp_server
if __name__ == "__main__":
    mcp.run()
