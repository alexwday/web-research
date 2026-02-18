"""
Route handlers for the Deep Research Agent web dashboard.

Covers:
- Full HTML pages (dashboard, tasks, sources, report, sessions)
- JSON API endpoints
- HTMX fragment endpoints (polled every 5s)
"""
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from collections import OrderedDict
from urllib.parse import urlparse

import markdown as md
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates

from src.config.types import TaskStatus, SectionStatus, ResearchSession
from src.config.presets import RESEARCH_PRESETS
from src.config.settings import get_config
from src.infra._database import get_database
from src.infra.llm import get_token_tracker
from src.pipeline._tools import read_file

router = APIRouter()

_templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))


# =========================================================================
# Token store for query refinement flow (in-memory, transient)
# =========================================================================

_refine_tokens: dict = {}
_REFINE_TOKEN_TTL_MINUTES = 10


def _cleanup_refine_tokens():
    """Remove expired refinement tokens (lazy cleanup)."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=_REFINE_TOKEN_TTL_MINUTES)
    expired = [k for k, v in _refine_tokens.items() if v.get("created_at", cutoff) < cutoff]
    for k in expired:
        del _refine_tokens[k]


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
        "task_groups": [],
        "total_queries": 0,
        "total_results": 0,
        "unique_domains": 0,
    }


def _build_activity_log_context(session_id: Optional[int]) -> dict:
    """Build nested activity data grouped by task, with status resolution.

    Returns task_groups (ordered list) each containing queries with nested results.
    Each result has status flags: is_processed, is_skipped, is_planning_complete.
    """
    if not session_id:
        return _empty_activity_log_context()

    db = _db()
    events = db.get_run_events(session_id)
    if not events:
        return _empty_activity_log_context()

    all_tasks = db.get_all_tasks(session_id=session_id)
    task_topics = {t.id: t.topic for t in all_tasks}
    task_statuses = {t.id: (t.status.value if hasattr(t.status, 'value') else t.status)
                     for t in all_tasks}
    has_tasks = len(task_statuses) > 0

    # Resolve which URLs were saved as sources per-task
    processed_by_task = db.get_processed_urls_by_task(session_id)
    processed_any = set()
    for urls in processed_by_task.values():
        processed_any.update(urls)

    # Ordered dict of task groups keyed by task_id (None = planning, "compilation" = synthesis)
    task_group_map = OrderedDict()
    query_map = {}
    total_queries = 0
    total_results = 0
    domains = set()

    def _get_task_group(task_id, phase):
        """Get or create a task group."""
        key = task_id if task_id is not None else f"__{phase}"
        if key not in task_group_map:
            if phase == "compilation":
                label = "Synthesis & Compilation"
                icon = "compilation"
            elif task_id is None:
                label = "Planning"
                icon = "planning"
            else:
                label = task_topics.get(task_id, f"Task {task_id}")
                icon = "research"
            task_group_map[key] = {
                "task_id": task_id,
                "task_topic": label,
                "phase": phase,
                "icon": icon,
                "queries": [],
                "query_count": 0,
                "result_count": 0,
            }
        return task_group_map[key]

    for ev in events:
        if ev.event_type == "agent_action":
            group = _get_task_group(ev.task_id, "compilation")
            group["query_count"] += 1
            total_queries += 1
            group["queries"].append({
                "query_group": ev.query_group,
                "query_text": (ev.query_text or "").strip(),
                "is_action": True,
                "results": [],
            })
            continue

        phase = "planning" if ev.task_id is None else "research"
        group = _get_task_group(ev.task_id, phase)
        query_key = f"{phase}:{ev.task_id or 0}:{ev.query_group}"

        if ev.event_type == "query":
            total_queries += 1
            group["query_count"] += 1
            q_entry = {
                "query_group": ev.query_group,
                "query_text": (ev.query_text or "").strip() or "(query unavailable)",
                "is_action": False,
                "results": [],
            }
            query_map[query_key] = q_entry
            group["queries"].append(q_entry)
            continue

        if ev.event_type != "result":
            continue

        total_results += 1
        group["result_count"] += 1
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
                "is_action": False,
                "results": [],
            }
            query_map[query_key] = q_entry
            group["queries"].append(q_entry)

        # Build result entry with status flags
        result = {
            "url": ev.url or "",
            "title": ev.title or "Untitled",
            "snippet": ev.snippet or "",
            "quality_score": ev.quality_score,
            "is_processed": False,
            "is_skipped": False,
            "is_planning_complete": False,
        }

        url = ev.url or ""
        task_id = ev.task_id
        if task_id is not None:
            if url and url in processed_by_task.get(task_id, set()):
                result["is_processed"] = True
            elif task_statuses.get(task_id) in ("completed", "failed"):
                result["is_processed"] = True
                result["is_skipped"] = True
        elif url and url in processed_any:
            result["is_processed"] = True
        elif has_tasks:
            result["is_processed"] = True
            result["is_planning_complete"] = True

        q_entry["results"].append(result)

    task_groups = list(task_group_map.values())
    return {
        "task_groups": task_groups,
        "total_queries": total_queries,
        "total_results": total_results,
        "unique_domains": len(domains),
    }


def _build_flat_activity_context(session_id: Optional[int]) -> dict:
    """Build flat chronological activity feed (reverse order)."""
    empty = {
        "feed_entries": [],
        "total_queries": 0,
        "total_results": 0,
        "unique_domains": 0,
    }
    if not session_id:
        return empty

    db = _db()
    events = db.get_run_events(session_id)
    if not events:
        return empty

    task_topics = {t.id: t.topic for t in db.get_all_tasks(session_id=session_id)}

    entries = []
    total_queries = 0
    total_results = 0
    domains = set()

    for ev in events:
        if ev.event_type == "agent_action":
            total_queries += 1
            entries.append({
                "type": "action",
                "text": (ev.query_text or "").strip(),
                "task_id": ev.task_id,
                "task_topic": task_topics.get(ev.task_id, "") if ev.task_id else "",
                "url": None,
                "quality_score": None,
                "created_at": getattr(ev, "created_at", None),
            })
        elif ev.event_type == "query":
            total_queries += 1
            entries.append({
                "type": "search",
                "text": (ev.query_text or "").strip() or "(query unavailable)",
                "task_id": ev.task_id,
                "task_topic": task_topics.get(ev.task_id, "") if ev.task_id else "",
                "url": None,
                "quality_score": None,
                "created_at": getattr(ev, "created_at", None),
            })
        elif ev.event_type == "result":
            total_results += 1
            if ev.url:
                try:
                    netloc = urlparse(ev.url).netloc
                    if netloc:
                        domains.add(netloc)
                except Exception:
                    pass
            entries.append({
                "type": "result",
                "text": ev.title or "Untitled",
                "task_id": ev.task_id,
                "task_topic": task_topics.get(ev.task_id, "") if ev.task_id else "",
                "url": ev.url or "",
                "quality_score": ev.quality_score,
                "created_at": getattr(ev, "created_at", None),
                "is_processed": False,  # will be resolved below
            })

    # Resolve processing status for result entries.
    # A result is "processed" when either:
    #   1. Its URL was saved as a source (in processed_by_task), OR
    #   2. Its task has moved past in_progress (search phase done; URL was skipped/filtered), OR
    #   3. It's a planning-phase result (task_id=None) and tasks have been created.
    processed_by_task = db.get_processed_urls_by_task(session_id)
    processed_any = set()
    for urls in processed_by_task.values():
        processed_any.update(urls)

    # Build set of task statuses to detect finished tasks
    task_statuses = {t.id: (t.status.value if hasattr(t.status, 'value') else t.status)
                     for t in db.get_all_tasks(session_id=session_id)}
    has_tasks = len(task_statuses) > 0

    for entry in entries:
        if entry["type"] != "result":
            continue
        url = entry.get("url")
        if not url:
            continue
        task_id = entry.get("task_id")
        if task_id is not None:
            if url in processed_by_task.get(task_id, set()):
                entry["is_processed"] = True
            elif task_statuses.get(task_id) in ("completed", "failed"):
                # Task finished — URL was seen but not saved as source (filtered out)
                entry["is_processed"] = True
                entry["is_skipped"] = True
        elif url in processed_any:
            entry["is_processed"] = True
        elif has_tasks:
            # Planning-phase result: tasks have been created, planning is done
            entry["is_processed"] = True
            entry["is_planning_complete"] = True

    # Reverse for newest-first
    entries.reverse()

    return {
        "feed_entries": entries,
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
    all_sessions = db.get_all_sessions()
    session_rows = []
    for s in all_sessions:
        session_rows.append({
            "session": s,
            "stats": db.get_statistics(session_id=s.id),
        })
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
        end = resolved.ended_at if resolved.ended_at else datetime.now()
        elapsed_seconds = max(0, int((end - resolved.started_at).total_seconds()))
    activity_log = _build_activity_log_context(sid)

    refinement_enabled = get_config().query_refinement.enabled

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
        "session_rows": session_rows,
        "refinement_enabled": refinement_enabled,
        **activity_log,
    })


@router.get("/tasks", response_class=HTMLResponse)
async def tasks_page(request: Request, session: Optional[int] = None):
    sq = f"?session={session}" if session else ""
    return RedirectResponse(url=f"/research{sq}", status_code=302)


@router.get("/sources", response_class=HTMLResponse)
async def sources_page(request: Request, session: Optional[int] = None):
    sq = f"?session={session}" if session else ""
    return RedirectResponse(url=f"/research{sq}", status_code=302)


def _build_source_groups(db, sid, task_id=None, include_rejected=False):
    """Build grouped source list for research/sources pages.

    If task_id is given, return only sources for that task.
    If include_rejected is True, append quality-rejected search results.
    """
    if sid is not None:
        source_list = db.get_sources_for_session(sid)
    else:
        source_list = db.get_all_sources()

    task_topics = {}
    if sid:
        for t in db.get_all_tasks(session_id=sid):
            task_topics[t.id] = t.topic

    # Optionally filter to a single task
    if task_id is not None:
        source_list = [s for s in source_list if task_id in (s.task_ids or [])]

    task_groups = OrderedDict()
    ungrouped = []
    for s in source_list:
        if s.task_ids:
            first_task = min(s.task_ids)
            task_groups.setdefault(first_task, []).append(s)
        else:
            ungrouped.append(s)

    # Add rejected search results (from search events not saved as sources)
    if include_rejected and sid:
        rejected = db.get_rejected_results(sid)
        for r in rejected:
            if task_id is not None and r["task_id"] != task_id:
                continue
            # Use SimpleNamespace for template attribute access compatibility
            domain = ""
            try:
                domain = urlparse(r["url"]).netloc
            except Exception:
                pass
            rejected_source = SimpleNamespace(
                url=r["url"],
                title=r["title"],
                domain=domain,
                snippet=r["snippet"] or "",
                quality_score=r["quality_score"] or 0.0,
                is_academic=False,
                is_rejected=True,
                extracted_content=None,
                full_content=None,
            )
            tid = r["task_id"]
            if tid is not None:
                task_groups.setdefault(tid, []).append(rejected_source)
            else:
                ungrouped.append(rejected_source)

    ordered_groups = []
    for tid in sorted(task_groups.keys()):
        sources = task_groups[tid]
        # Sort: accepted first (by quality desc), then rejected (by quality desc)
        sources.sort(key=lambda s: (not getattr(s, 'is_rejected', False), s.quality_score), reverse=True)
        ordered_groups.append({
            "task_id": tid,
            "task_topic": task_topics.get(tid, f"Task {tid}"),
            "sources": sources,
        })
    if ungrouped:
        ungrouped.sort(key=lambda s: (not getattr(s, 'is_rejected', False), s.quality_score), reverse=True)
        ordered_groups.append({
            "task_id": None,
            "task_topic": "Ungrouped",
            "sources": ungrouped,
        })

    total = sum(len(g["sources"]) for g in ordered_groups)
    return ordered_groups, total


@router.get("/research", response_class=HTMLResponse)
async def research_page(request: Request, session: Optional[int] = None):
    db = _db()
    resolved = _resolve_session(session)
    sid = resolved.id if resolved else None

    task_list = db.get_all_tasks(session_id=sid)

    # Build source count per task (accepted only for the count)
    source_groups_accepted, _ = _build_source_groups(db, sid)
    task_source_counts = {}
    for g in source_groups_accepted:
        if g["task_id"] is not None:
            task_source_counts[g["task_id"]] = len(g["sources"])

    # Build full source groups including rejected for display
    source_groups, total_sources = _build_source_groups(db, sid, include_rejected=True)

    return templates.TemplateResponse("research.html", {
        "request": request,
        "tasks": task_list,
        "task_source_counts": task_source_counts,
        "source_groups": source_groups,
        "total_sources": total_sources,
        "session": resolved,
        "session_id": sid,
        "page": "research",
    })


def _build_report_sections(session_id: int = None):
    """Build sorted report sections with globally-remapped citations.

    Supports both section-based sessions (new pipeline with synthesized
    sections) and task-based sessions (legacy). Returns (sections,
    global_sources) where citations in each section's HTML match the
    global bibliography numbering.
    """
    import re
    _CITATION_RE = re.compile(r'(?<!\])\[(\d+)\](?!\()')

    db = _db()

    # Check for section-based session first (new pipeline)
    db_sections = db.get_all_sections(session_id=session_id) if session_id else []
    synthesized = [s for s in db_sections if s.synthesized_content]

    if synthesized:
        # Section-based: use synthesized content from sections
        global_sources = []
        url_to_global = {}

        sections = []
        for sec in sorted(synthesized, key=lambda s: s.position):
            section_sources = db.get_sources_for_section(sec.id)

            local_to_global = {}
            for local_idx, source in enumerate(section_sources, 1):
                if source.url not in url_to_global:
                    global_sources.append(source)
                    url_to_global[source.url] = len(global_sources)
                local_to_global[local_idx] = url_to_global[source.url]

            content = sec.synthesized_content
            if local_to_global:
                def _replace(match, mapping=local_to_global):
                    local_num = int(match.group(1))
                    return f"[{mapping.get(local_num, local_num)}]"
                content = _CITATION_RE.sub(_replace, content)

            html = md.markdown(content, extensions=["tables", "fenced_code", "toc"])
            sections.append({
                "id": sec.id,
                "topic": sec.title,
                "html": html,
                "word_count": sec.word_count or 0,
                "citation_count": sec.citation_count or 0,
                "priority": sec.position,
            })

        return sections, global_sources

    # Fallback: task-based (legacy sessions)
    tasks = db.get_all_tasks(TaskStatus.COMPLETED, session_id=session_id)

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

    raw_sections.sort(key=lambda s: s["file_path"])

    global_sources = []
    url_to_global = {}

    sections = []
    for section in raw_sections:
        task_sources = db.get_sources_for_task(section["id"])

        local_to_global = {}
        for local_idx, source in enumerate(task_sources, 1):
            if source.url not in url_to_global:
                global_sources.append(source)
                url_to_global[source.url] = len(global_sources)
            local_to_global[local_idx] = url_to_global[source.url]

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
# Query Refinement endpoints
# =========================================================================

@router.post("/api/research/refine")
async def api_research_refine(request: Request):
    """Accept query + settings, generate questions via LLM.

    Returns JSON when Accept: application/json, otherwise redirects to /refine.
    """
    _cleanup_refine_tokens()

    content_type = request.headers.get("content-type", "")
    accept = request.headers.get("accept", "")
    want_json = "application/json" in accept or "json" in content_type

    if "json" in content_type:
        body = await request.json()
        query = body.get("query", "").strip()
        preset_name = body.get("preset", "")
        raw_fields = body
    else:
        form = await request.form()
        query = form.get("query", "").strip()
        preset_name = form.get("preset", "")
        raw_fields = {}
        for key in form.keys():
            if "." in key:
                raw_fields[key] = form.getlist(key)[-1]

    if not query:
        raise HTTPException(400, "Query is required")

    # Build overrides
    overrides = {}
    if preset_name and preset_name in RESEARCH_PRESETS:
        overrides.update(RESEARCH_PRESETS[preset_name]["overrides"])
    for key, value in raw_fields.items():
        if "." in key and key.split(".")[0] in ("research", "search"):
            overrides[key] = value

    # Generate questions via LLM
    from src.pipeline._stages import QueryRefinementAgent
    agent = QueryRefinementAgent()
    questions = agent.generate_questions(query)

    # Store in token
    token = uuid.uuid4().hex
    _refine_tokens[token] = {
        "query": query,
        "preset": preset_name,
        "overrides": overrides,
        "questions": questions,
        "answers": None,
        "brief": None,
        "created_at": datetime.now(timezone.utc),
    }

    if want_json:
        return JSONResponse({"token": token, "questions": questions})

    return RedirectResponse(url=f"/refine?token={token}", status_code=303)


@router.get("/refine", response_class=HTMLResponse)
async def refine_page(request: Request, token: str = "", step: Optional[str] = None):
    """Render questions (default) or brief review based on token state."""
    if not token or token not in _refine_tokens:
        return RedirectResponse(url="/dashboard", status_code=302)

    data = _refine_tokens[token]

    # Allow explicit step override (e.g. "Back to Questions" link)
    if step == "questions":
        step = "questions"
    elif data.get("brief"):
        step = "brief"
    else:
        step = "questions"

    return templates.TemplateResponse("refine.html", {
        "request": request,
        "token": token,
        "query": data["query"],
        "questions": data.get("questions", []),
        "brief": data.get("brief", ""),
        "step": step,
        "page": "dashboard",
    })


@router.post("/api/research/brief")
async def api_research_brief(request: Request):
    """Accept answers, generate brief via LLM.

    Returns JSON when Accept: application/json, otherwise redirects to /refine.
    """
    accept = request.headers.get("accept", "")
    content_type = request.headers.get("content-type", "")
    want_json = "application/json" in accept or "json" in content_type

    if "json" in content_type:
        body = await request.json()
        token = body.get("token", "")
        answers = body.get("answers", [])
    else:
        form = await request.form()
        token = form.get("token", "")
        answers = None

    if not token or token not in _refine_tokens:
        raise HTTPException(400, "Invalid or expired refinement token")

    data = _refine_tokens[token]
    questions = data.get("questions", [])

    # Collect answers
    if answers is not None:
        # JSON path: answers is a list of {question, answer} dicts
        qa_pairs = answers
    else:
        # Form path: answers come from form fields
        qa_pairs = []
        for i, q in enumerate(questions):
            answer = form.get(f"answer_{i}", "").strip()
            custom = form.get(f"custom_{i}", "").strip()
            final_answer = custom if custom else answer
            if final_answer:
                qa_pairs.append({
                    "question": q["question"],
                    "answer": final_answer,
                })

    # Store answers
    data["answers"] = qa_pairs

    # Generate brief via LLM
    from src.pipeline._stages import QueryRefinementAgent
    agent = QueryRefinementAgent()
    brief = agent.synthesize_brief(data["query"], qa_pairs)
    data["brief"] = brief

    if want_json:
        return JSONResponse({"brief": brief})

    return RedirectResponse(url=f"/refine?token={token}", status_code=303)


@router.post("/api/research/start-refined")
async def api_research_start_refined(request: Request):
    """Accept final brief, start research, redirect to dashboard."""
    form = await request.form()
    token = form.get("token", "")

    if not token or token not in _refine_tokens:
        raise HTTPException(400, "Invalid or expired refinement token")

    data = _refine_tokens[token]
    # Allow user to edit the brief in the textarea
    brief = form.get("brief", data.get("brief", "")).strip()
    query = data["query"]
    overrides = data.get("overrides") or None

    # Build refinement_qa JSON string
    refinement_qa = json.dumps(data.get("answers") or [])

    started = _app().start_research_background(
        query,
        overrides=overrides,
        refined_brief=brief,
        refinement_qa=refinement_qa,
    )

    # Clean up token
    _refine_tokens.pop(token, None)

    if not started:
        raise HTTPException(409, "Research is already running")

    return RedirectResponse(url="/dashboard", status_code=303)


# =========================================================================
# HTMX fragment endpoints (polled every 5 s)
# =========================================================================

@router.get("/fragments/session-info", response_class=HTMLResponse)
async def fragment_session_info(request: Request, session: Optional[int] = None):
    db = _db()
    resolved = _resolve_session(session)
    sid = resolved.id if resolved else None
    stats = db.get_statistics(session_id=sid)
    running = _is_running()
    phase = _app().get_current_phase()
    elapsed_seconds = 0
    if resolved and resolved.started_at:
        end = resolved.ended_at if resolved.ended_at else datetime.now()
        elapsed_seconds = max(0, int((end - resolved.started_at).total_seconds()))
    return templates.TemplateResponse("fragments/session_info.html", {
        "request": request,
        "session": resolved,
        "session_id": sid,
        "stats": stats,
        "token_stats": get_token_tracker().get_stats(),
        "running": running,
        "phase": phase,
        "elapsed_seconds": elapsed_seconds,
    })


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
    running = _is_running()
    phase = _app().get_current_phase()
    elapsed_seconds = 0
    if resolved and resolved.started_at:
        end = resolved.ended_at if resolved.ended_at else datetime.now()
        elapsed_seconds = max(0, int((end - resolved.started_at).total_seconds()))
    return templates.TemplateResponse("fragments/stats.html", {
        "request": request,
        "stats": stats,
        "token_stats": get_token_tracker().get_stats(),
        "running": running,
        "phase": phase,
        "elapsed_seconds": elapsed_seconds,
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

    task_queries = db.get_run_queries_by_task(sid) if sid else {}

    return templates.TemplateResponse("fragments/task_list.html", {
        "request": request,
        "tasks": task_list,
        "task_queries": task_queries,
    })


@router.get("/fragments/research-tasks", response_class=HTMLResponse)
async def fragment_research_tasks(request: Request, session: Optional[int] = None):
    db = _db()
    resolved = _resolve_session(session)
    sid = resolved.id if resolved else None
    task_list = db.get_all_tasks(session_id=sid)

    # Source count per task
    source_groups, _ = _build_source_groups(db, sid)
    task_source_counts = {}
    for g in source_groups:
        if g["task_id"] is not None:
            task_source_counts[g["task_id"]] = len(g["sources"])

    return templates.TemplateResponse("fragments/research_tasks.html", {
        "request": request,
        "tasks": task_list,
        "task_source_counts": task_source_counts,
    })


@router.get("/fragments/research-sources", response_class=HTMLResponse)
async def fragment_research_sources(request: Request, session: Optional[int] = None, task: Optional[int] = None):
    db = _db()
    resolved = _resolve_session(session)
    sid = resolved.id if resolved else None
    source_groups, _ = _build_source_groups(db, sid, task_id=task, include_rejected=True)

    return templates.TemplateResponse("fragments/research_sources.html", {
        "request": request,
        "source_groups": source_groups,
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
        end = resolved.ended_at if resolved.ended_at else datetime.now()
        elapsed_seconds = max(0, int((end - resolved.started_at).total_seconds()))

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
