"""Typer CLI for Tavily enterprise demo."""
from datetime import date
from typing import Optional

import typer
from rich.console import Console

from src.config.logger import print_error, print_info, print_success, print_warning
from src.pipeline.service import get_service

app = typer.Typer(
    name="tavily-demo",
    help="Standalone Tavily enterprise use-case demo",
    add_completion=False,
)
console = Console()


@app.command()
def validate() -> None:
    """Validate config and environment for local/work execution."""
    result = get_service().validate_environment()

    if result["tavily_key"]:
        print_success("TAVILY_API_KEY is configured")
    else:
        print_error("TAVILY_API_KEY is missing")

    if result["auth_configured"]:
        print_success("OpenAI/OAuth auth pattern is configured")
    else:
        print_warning("No OpenAI/OAuth auth configured (optional for this demo)")

    print_info(f"Banks configured: {result['bank_count']}")
    print_info(f"Search depth: {result['search_depth']}")
    print_info(f"Output directory: {result['output_directory']}")

    if not result["tavily_key"]:
        raise typer.Exit(1)


@app.command("quarterly-docs")
def quarterly_docs(
    period: str = typer.Option(..., "--period", "-p", help="Quarter identifier, e.g. 2025Q4"),
    loop: bool = typer.Option(False, "--loop", help="Keep polling until all banks are complete"),
    poll_seconds: Optional[int] = typer.Option(
        None,
        "--poll-seconds",
        help="Polling interval in seconds (loop mode)",
    ),
    max_iterations: Optional[int] = typer.Option(
        None,
        "--max-iterations",
        help="Max polling iterations (loop mode)",
    ),
    no_download: bool = typer.Option(False, "--no-download", help="Skip file downloads"),
) -> None:
    """Use case 1: find shareholder and Pillar 3 disclosures and save files."""
    result = get_service().run_quarterly_disclosures(
        period=period,
        run_loop=loop,
        poll_seconds=poll_seconds,
        max_iterations=max_iterations,
        download_files=not no_download,
    )

    print_info(f"Status: {result['status']}")
    print_info(f"Complete banks: {result['complete_banks']}/{result['bank_count']}")
    print_info(f"Iterations: {result['iterations']}")
    print_success(f"Manifest: {result['manifest_path']}")
    print_success(f"Summary: {result['summary_path']}")


@app.command("lcr-metrics")
def lcr_metrics(
    period: str = typer.Option(..., "--period", "-p", help="Quarter identifier, e.g. 2025Q4"),
) -> None:
    """Use case 2: web search for LCR values when internal data is unavailable."""
    result = get_service().run_lcr_metrics(period=period)

    print_info(f"Period: {result['period']}")
    print_info(f"LCR values found: {result['found']}/{result['total_banks']}")
    print_success(f"JSON: {result['json_path']}")
    print_success(f"CSV: {result['csv_path']}")
    print_success(f"Summary: {result['summary_path']}")


@app.command("headlines")
def headlines(
    start_date: Optional[str] = typer.Option(
        None,
        "--start-date",
        help="Start date (YYYY-MM-DD)",
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end-date",
        help="End date (YYYY-MM-DD)",
    ),
    recency_days: int = typer.Option(1, "--recency-days", help="Fallback day window when no dates are passed"),
    topics: Optional[str] = typer.Option(
        None,
        "--topics",
        help="Comma-separated finance topics",
    ),
) -> None:
    """Use case 3: latest finance and Big-6 headline digest."""
    parsed_start = _parse_date(start_date, "start-date") if start_date else None
    parsed_end = _parse_date(end_date, "end-date") if end_date else None
    topic_list = [t.strip() for t in topics.split(",") if t.strip()] if topics else None

    result = get_service().run_headlines(
        start_date=parsed_start,
        end_date=parsed_end,
        recency_days=recency_days,
        topics=topic_list,
    )

    print_info(f"Timeframe: {result['timeframe']}")
    print_info(f"Headline count: {result['headline_count']}")
    print_success(f"JSON: {result['json_path']}")
    print_success(f"Digest: {result['summary_path']}")


@app.command("deep-research")
def deep_research(
    query: str = typer.Argument(..., help="Research query"),
    max_sources: Optional[int] = typer.Option(None, "--max-sources", help="Cap number of ranked sources"),
) -> None:
    """Use case 4: search, rank, scrape, and synthesize a research brief."""
    result = get_service().run_deep_research(query=query, max_sources=max_sources)

    print_info(f"Query: {result['query']}")
    print_info(f"Sources used: {result['source_count']}")
    print_success(f"Payload: {result['json_path']}")
    print_success(f"Brief: {result['summary_path']}")


@app.command("internal-check")
def internal_check(
    sample_query: Optional[str] = typer.Option(
        "latest canadian bank earnings release",
        "--sample-query",
        help="Optional Tavily probe query",
    ),
) -> None:
    """Use case 5: verify internal-network readiness patterns."""
    result = get_service().run_internal_check(sample_query=sample_query)

    print_info(f"Auth mode: {result['auth_mode']}")
    print_info(f"Has Tavily key: {result['has_tavily_key']}")
    print_info(f"Has OAuth: {result['has_oauth']}")

    probe = result["search_probe"]
    if probe["attempted"]:
        if probe["ok"]:
            print_success(f"Search probe ok. Results: {probe['result_count']}")
        else:
            print_warning(f"Search probe failed: {probe['error']}")

    print_success(f"Output: {result['output_path']}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind"),
) -> None:
    """Launch interactive web UI with modal walkthroughs."""
    import uvicorn
    from src.adapters.web.app import create_app

    print_info(f"Starting interactive demo UI: http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port)


def _parse_date(raw: str, name: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        print_error(f"Invalid {name}: {raw}. Expected YYYY-MM-DD")
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
