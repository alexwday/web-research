"""
CLI Interface for Deep Research Agent
"""
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Prompt, Confirm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.rbc_security import configure_rbc_security_certs
from src.orchestrator import run_research
from src.database import get_database, reset_database
from src.config import load_config, get_env_settings, set_config
from src.llm_client import get_llm_client
from src.logger import print_header, print_info, print_error, print_success, print_statistics_table

# Enable RBC SSL certificates early (no-op if package not installed)
configure_rbc_security_certs()

app = typer.Typer(
    name="deep-research",
    help="24-Hour Deep Research Agent - Comprehensive automated research",
    add_completion=False
)

console = Console()


@app.command()
def research(
    query: Optional[str] = typer.Argument(
        None,
        help="The research query/topic. If not provided, will prompt for input."
    ),
    resume: bool = typer.Option(
        False,
        "--resume", "-r",
        help="Resume an existing research session"
    ),
    config_file: str = typer.Option(
        "config.yaml",
        "--config", "-c",
        help="Path to configuration file"
    )
):
    """
    Start or resume a deep research session.
    
    The agent will comprehensively research the given topic for up to 24 hours,
    producing a detailed report with citations.
    """
    # Validate environment
    settings = get_env_settings()
    if not _validate_api_keys(settings):
        raise typer.Exit(1)
    
    # Get query if not provided
    if not query and not resume:
        query = Prompt.ask(
            "[bold cyan]Enter your research query[/bold cyan]",
            console=console
        )
        
        if not query.strip():
            print_error("Query cannot be empty")
            raise typer.Exit(1)
    
    # Confirm if this is a long-running task
    if not resume:
        console.print("\n[yellow]Note:[/yellow] Deep research can take several hours.")
        console.print("The agent will search, read, and synthesize information comprehensively.")
        
        if not Confirm.ask("Start research?", default=True):
            raise typer.Abort()
    
    # Run research
    try:
        result = run_research(query or "", resume=resume)
        
        if "error" in result:
            print_error(f"Research failed: {result['error']}")
            raise typer.Exit(1)
        
        print_success("Research completed successfully!")
        
    except KeyboardInterrupt:
        print_info("\nResearch interrupted. Progress has been saved.")
        print_info("Use --resume to continue later.")
        raise typer.Exit(0)
    
    except Exception as e:
        print_error(f"An error occurred: {e}")
        raise typer.Exit(1)


@app.command()
def status():
    """
    Show the status of the current research session.
    """
    db = get_database()
    
    session = db.get_current_session()
    if not session:
        session = db.get_most_recent_session()
        if not session:
            print_info("No research session found.")
            return
        print_info("No active session. Showing most recent session.")
    
    print_header(
        "Research Session Status",
        f"Session #{session.id}"
    )
    
    query_display = session.query[:200] + ("..." if len(session.query) > 200 else "")
    console.print(f"\n[cyan]Query:[/cyan] {query_display}")
    console.print(f"[cyan]Started:[/cyan] {session.started_at}")
    console.print(f"[cyan]Status:[/cyan] {session.status}")
    
    stats = db.get_statistics(session_id=session.id)
    print_statistics_table(stats)


@app.command()
def reset(
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Skip confirmation prompt"
    )
):
    """
    Reset the research database and clear all progress.
    """
    if not force:
        if not Confirm.ask(
            "[red]This will delete all research progress. Are you sure?[/red]",
            default=False
        ):
            raise typer.Abort()
    
    # Delete database file
    config = load_config()
    db_path = Path(config.database.path)
    
    if db_path.exists():
        db_path.unlink()
        print_success("Database reset successfully.")
    else:
        print_info("No database found.")
    
    # Reset global instance
    reset_database()


@app.command()
def export(
    export_format: str = typer.Option(
        "all",
        "--format", "-f",
        help="Export format: markdown, html, pdf, or all"
    ),
    output_dir: str = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for exported files"
    )
):
    """
    Export the current research to different formats.
    """
    from src.compiler import ReportCompiler

    db = get_database()
    session = db.get_current_session() or db.get_most_recent_session()

    if not session:
        print_error("No research session found to export.")
        raise typer.Exit(1)

    config = load_config()

    # Override output directory if specified
    if output_dir:
        config.output.directory = output_dir

    # Set formats based on option
    if export_format == "all":
        config.output.formats = ["markdown", "html"]
    else:
        config.output.formats = [export_format]

    # Ensure compiler uses the same runtime config for this command.
    set_config(config)
    
    compiler = ReportCompiler()
    
    try:
        output_files = compiler.compile_report(
            session.query,
            None,
            None,
            0,
            session_id=session.id
        )
        
        print_success("Export completed!")
        for fmt, path in output_files.items():
            console.print(f"  • {fmt}: {path}")
            
    except Exception as e:
        print_error(f"Export failed: {e}")
        raise typer.Exit(1)


@app.command()
def validate():
    """
    Validate configuration and API keys.
    """
    print_header("Configuration Validation", "Checking setup...")
    
    # Check config file
    config_path = Path("config.yaml")
    if config_path.exists():
        print_success("config.yaml found")
    else:
        example_path = Path("config.example.yaml")
        if example_path.exists():
            print_info("config.yaml not found, using config.example.yaml")
        else:
            print_error("No configuration file found")
    
    # Load and validate config
    try:
        config = load_config()
        print_success("Configuration loaded successfully")
    except Exception as e:
        print_error(f"Configuration error: {e}")
        return
    
    # Check API keys
    settings = get_env_settings()
    _validate_api_keys(settings, verbose=True)
    
    # Check directories
    output_dir = Path(config.output.directory)
    if output_dir.exists():
        print_success(f"Output directory exists: {output_dir}")
    else:
        print_info(f"Output directory will be created: {output_dir}")
    
    print_success("\nValidation complete!")


@app.command("model-smoke")
def model_smoke(
    models: str = typer.Option(
        "gpt-4.1,gpt-5-mini,gpt-5,gpt-5.1",
        "--models",
        help="Comma-separated model names to probe",
    ),
    skip_tool_calling: bool = typer.Option(
        False,
        "--skip-tool-calling",
        help="Skip tool/function-calling probe",
    ),
):
    """Run a quick model compatibility smoke test for completion + tool calling."""
    from rich.table import Table

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        print_error("No models provided.")
        raise typer.Exit(1)

    settings = get_env_settings()
    has_llm_auth = bool(settings.openai_api_key) or all([
        settings.oauth_url, settings.oauth_client_id, settings.oauth_client_secret
    ])
    if not has_llm_auth:
        print_error("No LLM auth configured: set OPENAI_API_KEY or OAuth credentials.")
        raise typer.Exit(1)

    client = get_llm_client()
    table = Table(title="Model Smoke Test", border_style="cyan")
    table.add_column("Model", style="cyan")
    table.add_column("Text", justify="center")
    table.add_column("Tool", justify="center")
    table.add_column("Notes", style="dim")

    any_fail = False

    for model in model_list:
        text_status = "fail"
        tool_status = "skip" if skip_tool_calling else "fail"
        notes = []

        try:
            text = client.complete(
                prompt="Reply with exactly OK.",
                system="You are a probe. Output only OK.",
                max_tokens=20,
                temperature=0.1,
                model=model,
            )
            if (text or "").strip():
                text_status = "pass"
            else:
                notes.append("empty text response")
        except Exception as e:
            notes.append(f"text error: {str(e)[:80]}")

        if not skip_tool_calling:
            try:
                payload = client.complete_with_function(
                    prompt="Call the function with status='ok' and message='smoke'.",
                    system="You are a probe. Use the provided function.",
                    function_name="emit_probe",
                    function_description="Emit a probe payload",
                    function_parameters={
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["ok"]},
                            "message": {"type": "string"},
                        },
                        "required": ["status", "message"],
                        "additionalProperties": False,
                    },
                    max_tokens=120,
                    temperature=0.1,
                    model=model,
                    require_tool_call=True,
                )
                if isinstance(payload, dict) and payload.get("status") == "ok":
                    tool_status = "pass"
                else:
                    tool_status = "fail"
                    notes.append(f"bad tool payload: {str(payload)[:80]}")
            except Exception as e:
                tool_status = "error"
                notes.append(f"tool error: {str(e)[:80]}")

        if text_status != "pass" or (not skip_tool_calling and tool_status != "pass"):
            any_fail = True

        table.add_row(model, text_status, tool_status, "; ".join(notes) if notes else "-")

    console.print(table)

    if any_fail:
        print_error("One or more model probes failed.")
        raise typer.Exit(1)

    print_success("All model probes passed.")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
):
    """Start the web dashboard."""
    import uvicorn
    from src.web.app import create_app

    print_info(f"Starting web dashboard at http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port)


def _validate_api_keys(settings, verbose: bool = False) -> bool:
    """Validate that required API keys are set"""
    valid = True

    # Check OpenAI / OAuth credentials
    has_api_key = bool(settings.openai_api_key)
    has_oauth = all([settings.oauth_url, settings.oauth_client_id, settings.oauth_client_secret])

    if has_api_key:
        if verbose:
            print_success("OPENAI_API_KEY is set (local mode)")
    elif has_oauth:
        if verbose:
            print_success("OAuth credentials set (corporate mode)")
    else:
        print_error("No LLM auth configured: set OPENAI_API_KEY or OAuth credentials (OAUTH_URL, CLIENT_ID, CLIENT_SECRET)")
        valid = False

    # Check Tavily key
    if settings.tavily_api_key:
        if verbose:
            print_success("TAVILY_API_KEY is set")
    else:
        print_error("TAVILY_API_KEY is not set")
        valid = False

    return valid


if __name__ == "__main__":
    app()
