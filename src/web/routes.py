"""
Route handlers for the Deep Research Agent web dashboard.

Covers:
- Full HTML pages (dashboard, tasks, sources, report, sessions)
- JSON API endpoints
- HTMX fragment endpoints (polled every 5s)
"""
from datetime import datetime
from pathlib import Path
from typing import Optional
from collections import OrderedDict
from urllib.parse import urlparse

import markdown as md
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates

from src.config import TaskStatus, ResearchSession, RESEARCH_PRESETS
from src.database import get_database
from src.llm_client import get_token_tracker
from src.tools import read_file

router = APIRouter()

_templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))


# =========================================================================
# Helpers
# =========================================================================

def _db():
    return get_database()


def _app():
    """Lazy import to avoid circular dependency with app.py."""
    from . import app as app_module
    return app_module


def _is_running() -> bool:
    return _app().is_research_running()


def _resolve_session(session_param: Optional[int] = None) -> Optional[ResearchSession]:
    """Resolve session: explicit ID > running session > most recent."""
    db = _db()
    if session_param is not None:
        return db.get_session_by_id(session_param)
    # When research is running, show the running session
    running = db.get_current_session()
    if running:
        return running
    return db.get_most_recent_session()


def _empty_activity_log_context() -> dict:
    return {
        "phase_groups": [],
        "total_queries": 0,
        "total_results": 0,
        "unique_domains": 0,
    }


def _build_activity_log_context(session_id: Optional[int]) -> dict:
    """Build nested activity data: phase -> query -> result."""
    if not session_id:
        return _empty_activity_log_context()

    db = _db()
    events = db.get_search_events(session_id)
    if not events:
        return _empty_activity_log_context()

    task_topics = {t.id: t.topic for t in db.get_all_tasks(session_id=session_id)}
    phase_map = OrderedDict(
        [
            (
                "planning",
                {
                    "phase_key": "planning",
                    "phase_label": "Planning Phase",
                    "queries": [],
                    "query_count": 0,
                    "result_count": 0,
                },
            ),
            (
                "research",
                {
                    "phase_key": "research",
                    "phase_label": "Research Phase",
                    "queries": [],
                    "query_count": 0,
                    "result_count": 0,
                },
            ),
            (
                "compilation",
                {
                    "phase_key": "compilation",
                    "phase_label": "Compilation Phase",
                    "queries": [],
                    "query_count": 0,
                    "result_count": 0,
                },
            ),
        ]
    )

    query_map = {}
    total_queries = 0
    total_results = 0
    domains = set()

    for ev in events:
        # Agent actions (rewrite, summary, conclusion) go to compilation phase
        if ev.event_type == "agent_action":
            phase_group = phase_map["compilation"]
            phase_group["query_count"] += 1
            total_queries += 1
            phase_group["queries"].append({
                "query_group": ev.query_group,
                "query_text": (ev.query_text or "").strip(),
                "results": [],
                "task_id": ev.task_id,
                "task_topic": task_topics.get(ev.task_id, "") if ev.task_id else "",
            })
            continue

        phase_key = "planning" if ev.task_id is None else "research"
        phase_group = phase_map[phase_key]
        query_key = f"{phase_key}:{ev.task_id or 0}:{ev.query_group}"

        if ev.event_type == "query":
            total_queries += 1
            phase_group["query_count"] += 1
            q_entry = {
                "query_group": ev.query_group,
                "query_text": (ev.query_text or "").strip() or "(query unavailable)",
                "results": [],
                "task_id": ev.task_id,
                "task_topic": task_topics.get(ev.task_id, "") if ev.task_id else "",
            }
            query_map[query_key] = q_entry
            phase_group["queries"].append(q_entry)
            continue

        if ev.event_type != "result":
            continue

        total_results += 1
        phase_group["result_count"] += 1
        if ev.url:
            try:
                netloc = urlparse(ev.url).netloc
                if netloc:
                    domains.add(netloc)
            except Exception:
                pass

        q_entry = query_map.get(query_key)
        if not q_entry:
            q_entry = {
                "query_group": ev.query_group,
                "query_text": "(query unavailable)",
                "results": [],
                "task_id": ev.task_id,
                "task_topic": task_topics.get(ev.task_id, "") if ev.task_id else "",
            }
            query_map[query_key] = q_entry
            phase_group["queries"].append(q_entry)

        q_entry["results"].append(
            {
                "url": ev.url or "",
                "title": ev.title or "Untitled",
                "snippet": ev.snippet or "",
                "quality_score": ev.quality_score,
            }
        )

    phase_groups = [
        g for g in phase_map.values() if g["queries"] or g["query_count"] or g["result_count"]
    ]
    return {
        "phase_groups": phase_groups,
        "total_queries": total_queries,
        "total_results": total_results,
        "unique_domains": len(domains),
    }


# =========================================================================
# Full HTML pages
# =========================================================================

@router.get("/", response_class=HTMLResponse)
async def index():
    return RedirectResponse(url="/dashboard", status_code=302)


@router.get("/sessions", response_class=HTMLResponse)
async def sessions_page(request: Request):
    db = _db()
    all_sessions = db.get_all_sessions()
    # Enrich with task counts
    session_rows = []
    for s in all_sessions:
        stats = db.get_statistics(session_id=s.id)
        session_rows.append({"session": s, "stats": stats})
    return templates.TemplateResponse("sessions.html", {
        "request": request,
        "session_rows": session_rows,
        "page": "sessions",
    })


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, session: Optional[int] = None):
    db = _db()
    resolved = _resolve_session(session)
    sid = resolved.id if resolved else None
    stats = db.get_statistics(session_id=sid)
    running = _is_running()

    # progress percentage
    total = stats.get("total_tasks", 0)
    completed = stats.get("completed_tasks", 0)
    progress_pct = int((completed / total) * 100) if total > 0 else 0

    # Phase, current task, elapsed time
    phase = _app().get_current_phase()
    current_tasks = db.get_in_progress_tasks(session_id=sid) if sid else []
    elapsed_seconds = 0
    if resolved and resolved.started_at:
        elapsed_seconds = int((datetime.now() - resolved.started_at).total_seconds())
    activity_log = _build_activity_log_context(sid)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "session": resolved,
        "session_id": sid,
        "stats": stats,
        "running": running,
        "progress_pct": progress_pct,
        "page": "dashboard",
        "token_stats": get_token_tracker().get_stats(),
        "phase": phase,
        "current_tasks": current_tasks,
        "elapsed_seconds": elapsed_seconds,
        **activity_log,
    })


@router.get("/tasks", response_class=HTMLResponse)
async def tasks_page(request: Request, status: Optional[str] = None, session: Optional[int] = None):
    db = _db()
    resolved = _resolve_session(session)
    sid = resolved.id if resolved else None

    if status and status != "all":
        try:
            task_status = TaskStatus(status)
        except ValueError:
            task_status = None
        task_list = db.get_all_tasks(task_status, session_id=sid) if task_status else db.get_all_tasks(session_id=sid)
    else:
        task_list = db.get_all_tasks(session_id=sid)

    return templates.TemplateResponse("tasks.html", {
        "request": request,
        "tasks": task_list,
        "current_filter": status or "all",
        "session": resolved,
        "session_id": sid,
        "page": "tasks",
    })


@router.get("/sources", response_class=HTMLResponse)
async def sources_page(request: Request, session: Optional[int] = None):
    db = _db()
    resolved = _resolve_session(session)
    sid = resolved.id if resolved else None

    if sid is not None:
        source_list = db.get_sources_for_session(sid)
    else:
        source_list = db.get_all_sources()
    # sort by quality_score descending
    source_list.sort(key=lambda s: s.quality_score, reverse=True)

    return templates.TemplateResponse("sources.html", {
        "request": request,
        "sources": source_list,
        "session": resolved,
        "session_id": sid,
        "page": "sources",
    })


def _build_report_sections(session_id: int = None):
    """Build sorted report sections with globally-remapped citations.

    Returns (sections, global_sources) where citations in each section's
    HTML match the global bibliography numbering.
    """
    import re
    _CITATION_RE = re.compile(r'(?<!\])\[(\d+)\](?!\()')

    db = _db()
    tasks = db.get_all_tasks(TaskStatus.COMPLETED, session_id=session_id)

    # First pass: read content and collect per-task sources
    raw_sections = []
    for task in tasks:
        content = read_file(task.file_path)
        if content:
            raw_sections.append({
                "id": task.id,
                "topic": task.topic,
                "content": content,
                "word_count": task.word_count,
                "citation_count": task.citation_count,
                "priority": task.priority,
                "file_path": task.file_path,
            })

    # Match compiler ordering (numeric prefix in file path).
    raw_sections.sort(key=lambda s: s["file_path"])

    # Build global deduplicated source list and remap citations
    global_sources = []
    url_to_global = {}  # url -> 1-indexed global number

    sections = []
    for section in raw_sections:
        task_sources = db.get_sources_for_task(section["id"])

        # Build local-to-global mapping for this section
        local_to_global = {}
        for local_idx, source in enumerate(task_sources, 1):
            if source.url not in url_to_global:
                global_sources.append(source)
                url_to_global[source.url] = len(global_sources)
            local_to_global[local_idx] = url_to_global[source.url]

        # Remap citations in markdown content before converting to HTML
        content = section["content"]
        if local_to_global:
            def _replace(match, mapping=local_to_global):
                local_num = int(match.group(1))
                return f"[{mapping.get(local_num, local_num)}]"
            content = _CITATION_RE.sub(_replace, content)

        html = md.markdown(content, extensions=["tables", "fenced_code", "toc"])
        sections.append({
            "id": section["id"],
            "topic": section["topic"],
            "html": html,
            "word_count": section["word_count"],
            "citation_count": section["citation_count"],
            "priority": section["priority"],
        })

    return sections, global_sources


@router.get("/report", response_class=HTMLResponse)
async def report_page(
    request: Request,
    view: Optional[str] = None,
    page: Optional[int] = None,
    session: Optional[int] = None,
):
    db = _db()
    resolved = _resolve_session(session)
    sid = resolved.id if resolved else None

    sections, global_sources = _build_report_sections(session_id=sid)
    stats = db.get_statistics(session_id=sid)

    # Use globally-ordered sources for bibliography (matches citation numbers)
    sources = global_sources

    # Executive summary & conclusion from session
    exec_summary_html = None
    conclusion_html = None
    has_compiled_report = False
    if resolved:
        if resolved.executive_summary:
            exec_summary_html = md.markdown(resolved.executive_summary)
        if resolved.conclusion:
            conclusion_html = md.markdown(resolved.conclusion)
        if resolved.report_html_path and Path(resolved.report_html_path).exists():
            has_compiled_report = True

    # Clamp page index
    if view == "paged" and sections:
        page = max(0, min(page or 0, len(sections) - 1))
    else:
        page = 0

    return templates.TemplateResponse("report.html", {
        "request": request,
        "sections": sections,
        "stats": stats,
        "sources": sources,
        "exec_summary_html": exec_summary_html,
        "conclusion_html": conclusion_html,
        "has_compiled_report": has_compiled_report,
        "session": resolved,
        "session_id": sid,
        "page": "report",
        "view": view or "scroll",
        "current_page": page,
    })


@router.get("/report/compiled", response_class=HTMLResponse)
async def report_compiled(session: Optional[int] = None):
    """Serve the standalone compiled HTML report."""
    resolved = _resolve_session(session)
    if not resolved or not resolved.report_html_path:
        raise HTTPException(404, "No compiled report found")
    path = Path(resolved.report_html_path)
    if not path.exists():
        raise HTTPException(404, "Report file not found on disk")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@router.get("/report/download/{fmt}")
async def report_download(fmt: str, session: Optional[int] = None):
    """Download compiled report file."""
    resolved = _resolve_session(session)
    if not resolved:
        raise HTTPException(404, "No session found")

    if fmt == "markdown":
        file_path = resolved.report_markdown_path
        media_type = "text/markdown"
    elif fmt == "html":
        file_path = resolved.report_html_path
        media_type = "text/html"
    else:
        raise HTTPException(400, f"Unsupported format: {fmt}")

    if not file_path or not Path(file_path).exists():
        raise HTTPException(404, f"No {fmt} report file found")

    return FileResponse(
        file_path,
        media_type=media_type,
        filename=Path(file_path).name,
    )


# =========================================================================
# JSON API
# =========================================================================

@router.get("/api/status")
async def api_status(session: Optional[int] = None):
    db = _db()
    resolved = _resolve_session(session)
    sid = resolved.id if resolved else None
    stats = db.get_statistics(session_id=sid)
    return {
        "running": _is_running(),
        "session": resolved.model_dump() if resolved else None,
        "statistics": stats,
    }


@router.get("/api/tasks")
async def api_tasks(status: Optional[str] = None, session: Optional[int] = None):
    db = _db()
    resolved = _resolve_session(session)
    sid = resolved.id if resolved else None

    if status:
        try:
            task_status = TaskStatus(status)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status}")
        task_list = db.get_all_tasks(task_status, session_id=sid)
    else:
        task_list = db.get_all_tasks(session_id=sid)
    return [t.model_dump() for t in task_list]


@router.get("/api/tasks/{task_id}")
async def api_task_detail(task_id: int):
    db = _db()
    task = db.get_task_by_id(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    data = task.model_dump()
    if task.status == "completed" and task.file_path:
        data["content"] = read_file(task.file_path)
    return data


@router.get("/api/sources")
async def api_sources(session: Optional[int] = None):
    db = _db()
    if session is not None:
        return [s.model_dump() for s in db.get_sources_for_session(session)]
    return [s.model_dump() for s in db.get_all_sources()]


@router.get("/api/glossary")
async def api_glossary():
    db = _db()
    return [g.model_dump() for g in db.get_all_glossary_terms()]


@router.get("/api/costs")
async def api_costs():
    return get_token_tracker().get_stats()


@router.get("/api/presets")
async def api_presets():
    """Return available research presets."""
    return RESEARCH_PRESETS


@router.post("/api/research/start")
async def api_research_start(request: Request):
    content_type = request.headers.get("content-type", "")
    if "json" in content_type:
        body = await request.json()
        query = body.get("query", "").strip()
        preset_name = body.get("preset", "")
        raw_fields = body
    else:
        form = await request.form()
        query = form.get("query", "").strip()
        preset_name = form.get("preset", "")
        # Collect all dotted-key fields (research.*, search.*)
        raw_fields = {}
        for key in form.keys():
            if "." in key:
                # Take last value for each key (hidden-field checkbox trick)
                raw_fields[key] = form.getlist(key)[-1]

    if not query:
        raise HTTPException(400, "Query is required")

    # Build overrides: start from preset, layer individual fields on top
    overrides = {}
    if preset_name and preset_name in RESEARCH_PRESETS:
        overrides.update(RESEARCH_PRESETS[preset_name]["overrides"])

    # Layer any explicit field overrides on top of preset
    for key, value in raw_fields.items():
        if "." in key and key.split(".")[0] in ("research", "search"):
            overrides[key] = value

    started = _app().start_research_background(query, overrides=overrides or None)
    if not started:
        raise HTTPException(409, "Research is already running")

    return {"status": "started", "query": query}


@router.post("/api/research/stop")
async def api_research_stop():
    stopped = _app().stop_research()
    return {"status": "stopping" if stopped else "not_running"}


# =========================================================================
# HTMX fragment endpoints (polled every 5 s)
# =========================================================================

@router.get("/fragments/status-badge", response_class=HTMLResponse)
async def fragment_status_badge(request: Request):
    return templates.TemplateResponse("fragments/status_badge.html", {
        "request": request,
        "running": _is_running(),
    })


@router.get("/fragments/stats", response_class=HTMLResponse)
async def fragment_stats(request: Request, session: Optional[int] = None):
    db = _db()
    resolved = _resolve_session(session)
    sid = resolved.id if resolved else None
    stats = db.get_statistics(session_id=sid)
    return templates.TemplateResponse("fragments/stats.html", {
        "request": request,
        "stats": stats,
        "token_stats": get_token_tracker().get_stats(),
    })


@router.get("/fragments/task-list", response_class=HTMLResponse)
async def fragment_task_list(request: Request, status: Optional[str] = None, session: Optional[int] = None):
    db = _db()
    resolved = _resolve_session(session)
    sid = resolved.id if resolved else None

    if status and status != "all":
        try:
            task_status = TaskStatus(status)
        except ValueError:
            task_status = None
        task_list = db.get_all_tasks(task_status, session_id=sid) if task_status else db.get_all_tasks(session_id=sid)
    else:
        task_list = db.get_all_tasks(session_id=sid)

    return templates.TemplateResponse("fragments/task_list.html", {
        "request": request,
        "tasks": task_list,
    })


@router.get("/fragments/report-page", response_class=HTMLResponse)
async def fragment_report_page(request: Request, page: int = 0, session: Optional[int] = None):
    resolved = _resolve_session(session)
    sid = resolved.id if resolved else None
    sections, _ = _build_report_sections(session_id=sid)
    total = len(sections)

    if total == 0:
        return HTMLResponse("<p>No completed sections yet.</p>")

    page = max(0, min(page, total - 1))
    section = sections[page]

    return templates.TemplateResponse("fragments/report_page.html", {
        "request": request,
        "section": section,
        "page": page,
        "total": total,
        "sections": sections,
        "session_id": sid,
        "has_prev": page > 0,
        "has_next": page < total - 1,
    })


@router.get("/fragments/progress", response_class=HTMLResponse)
async def fragment_progress(request: Request, session: Optional[int] = None):
    db = _db()
    resolved = _resolve_session(session)
    sid = resolved.id if resolved else None
    stats = db.get_statistics(session_id=sid)
    running = _is_running()
    total = stats.get("total_tasks", 0)
    completed = stats.get("completed_tasks", 0)
    progress_pct = int((completed / total) * 100) if total > 0 else 0

    # Phase and current tasks
    phase = _app().get_current_phase()
    current_tasks = db.get_in_progress_tasks(session_id=sid) if sid else []

    # Elapsed time (seconds since session started)
    elapsed_seconds = 0
    if resolved and resolved.started_at:
        elapsed_seconds = int((datetime.now() - resolved.started_at).total_seconds())

    return templates.TemplateResponse("fragments/progress.html", {
        "request": request,
        "running": running,
        "progress_pct": progress_pct,
        "completed": completed,
        "total": total,
        "phase": phase,
        "current_tasks": current_tasks,
        "elapsed_seconds": elapsed_seconds,
    })


@router.get("/fragments/activity", response_class=HTMLResponse)
async def fragment_activity(request: Request, session: Optional[int] = None):
    db = _db()
    resolved = _resolve_session(session)
    sid = resolved.id if resolved else None
    recent = db.get_recent_completed_tasks(limit=5, session_id=sid) if sid else []
    return templates.TemplateResponse("fragments/activity.html", {
        "request": request,
        "recent_tasks": recent,
    })


@router.get("/fragments/search-activity", response_class=HTMLResponse)
async def fragment_search_activity(request: Request, session: Optional[int] = None):
    resolved = _resolve_session(session)
    sid = resolved.id if resolved else None
    activity_log = _build_activity_log_context(sid)
    return templates.TemplateResponse("fragments/search_activity.html", {
        "request": request,
        **activity_log,
    })
