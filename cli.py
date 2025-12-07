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

from src.orchestrator import run_research
from src.database import get_database, reset_database
from src.config import load_config, get_env_settings
from src.logger import print_header, print_info, print_error, print_success, print_statistics_table

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
        print_info("No active research session found.")
        return
    
    print_header(
        "Research Session Status",
        f"Session #{session.id}"
    )
    
    console.print(f"\n[cyan]Query:[/cyan] {session.query[:200]}...")
    console.print(f"[cyan]Started:[/cyan] {session.started_at}")
    console.print(f"[cyan]Status:[/cyan] {session.status}")
    
    stats = db.get_statistics()
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
    format: str = typer.Option(
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
    session = db.get_current_session()
    
    if not session:
        print_error("No research session found to export.")
        raise typer.Exit(1)
    
    config = load_config()
    
    # Override output directory if specified
    if output_dir:
        config.output.directory = output_dir
    
    # Set formats based on option
    if format == "all":
        config.output.formats = ["markdown", "html"]
    else:
        config.output.formats = [format]
    
    compiler = ReportCompiler()
    
    try:
        output_files = compiler.compile_report(
            session.query,
            None,
            None,
            0
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


def _validate_api_keys(settings, verbose: bool = False) -> bool:
    """Validate that required API keys are set"""
    config = load_config()
    valid = True
    
    # Check LLM provider key
    provider = config.llm.provider.value
    
    if provider == "anthropic":
        if settings.anthropic_api_key:
            if verbose:
                print_success("ANTHROPIC_API_KEY is set")
        else:
            print_error("ANTHROPIC_API_KEY is not set")
            valid = False
            
    elif provider == "openai":
        if settings.openai_api_key:
            if verbose:
                print_success("OPENAI_API_KEY is set")
        else:
            print_error("OPENAI_API_KEY is not set")
            valid = False
            
    elif provider == "openrouter":
        if settings.openrouter_api_key:
            if verbose:
                print_success("OPENROUTER_API_KEY is set")
        else:
            print_error("OPENROUTER_API_KEY is not set")
            valid = False
    
    # Check search provider key
    search_provider = config.search.provider.value
    
    if search_provider == "tavily":
        if settings.tavily_api_key:
            if verbose:
                print_success("TAVILY_API_KEY is set")
        else:
            print_error("TAVILY_API_KEY is not set")
            valid = False
            
    elif search_provider == "serper":
        if settings.serper_api_key:
            if verbose:
                print_success("SERPER_API_KEY is set")
        else:
            print_error("SERPER_API_KEY is not set")
            valid = False
            
    elif search_provider == "brave":
        if settings.brave_api_key:
            if verbose:
                print_success("BRAVE_API_KEY is set")
        else:
            print_error("BRAVE_API_KEY is not set")
            valid = False
    
    return valid


if __name__ == "__main__":
    app()
