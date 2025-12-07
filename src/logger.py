"""
Logging Module for Deep Research Agent
Provides rich console output and file logging
"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live

from .config import get_config

# Rich console for beautiful output
console = Console()


class ResearchLogger:
    """Custom logger with rich formatting"""
    
    def __init__(self, name: str = "research_agent"):
        self.config = get_config()
        self.name = name
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, self.config.logging.level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers"""
        
        # Rich console handler
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True
        )
        console_handler.setLevel(getattr(logging, self.config.logging.level.upper()))
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = Path(self.config.logging.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.logging.max_file_size * 1024 * 1024,
            backupCount=self.config.logging.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
    
    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        self.logger.exception(msg, *args, **kwargs)


# Need to import after logging is configured
import logging.handlers

# Logger cache
_loggers = {}


def get_logger(name: str = "research_agent") -> ResearchLogger:
    """Get or create a logger instance"""
    if name not in _loggers:
        _loggers[name] = ResearchLogger(name)
    return _loggers[name]


# =============================================================================
# RICH OUTPUT HELPERS
# =============================================================================

def print_header(title: str, subtitle: str = None):
    """Print a styled header"""
    text = Text()
    text.append(f"🔬 {title}", style="bold cyan")
    if subtitle:
        text.append(f"\n{subtitle}", style="dim")
    
    console.print(Panel(text, border_style="cyan"))


def print_success(message: str):
    """Print a success message"""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str):
    """Print an error message"""
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str):
    """Print a warning message"""
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_info(message: str):
    """Print an info message"""
    console.print(f"[blue]ℹ[/blue] {message}")


def print_task_start(task_topic: str, task_id: int):
    """Print task start indicator"""
    console.print(f"\n[bold magenta]━━━ Task #{task_id}: {task_topic} ━━━[/bold magenta]")


def print_search(query: str):
    """Print search indicator"""
    console.print(f"  [cyan]🔍 Searching:[/cyan] {query}")


def print_scrape(url: str):
    """Print scrape indicator"""
    # Truncate long URLs
    display_url = url[:60] + "..." if len(url) > 60 else url
    console.print(f"  [cyan]🌐 Scraping:[/cyan] {display_url}")


def print_write(filepath: str, words: int):
    """Print write indicator"""
    console.print(f"  [green]💾 Saved:[/green] {filepath} ({words:,} words)")


def print_statistics_table(stats: dict):
    """Print a statistics table"""
    table = Table(title="Research Statistics", border_style="cyan")
    
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    
    table.add_row("Total Tasks", f"{stats.get('total_tasks', 0):,}")
    table.add_row("Completed", f"{stats.get('completed_tasks', 0):,}")
    table.add_row("Pending", f"{stats.get('pending_tasks', 0):,}")
    table.add_row("Failed", f"{stats.get('failed_tasks', 0):,}")
    table.add_row("─" * 15, "─" * 10)
    table.add_row("Total Sources", f"{stats.get('total_sources', 0):,}")
    table.add_row("Total Words", f"{stats.get('total_words', 0):,}")
    table.add_row("Glossary Terms", f"{stats.get('glossary_terms', 0):,}")
    
    console.print(table)


def print_task_table(tasks: list, limit: int = 10):
    """Print a table of tasks"""
    table = Table(title=f"Research Tasks (showing {min(len(tasks), limit)} of {len(tasks)})", border_style="cyan")
    
    table.add_column("ID", style="dim", width=4)
    table.add_column("Topic", style="cyan", max_width=40)
    table.add_column("Status", width=12)
    table.add_column("Words", justify="right", width=8)
    table.add_column("Depth", justify="right", width=5)
    
    status_colors = {
        "pending": "yellow",
        "in_progress": "blue",
        "completed": "green",
        "failed": "red",
        "skipped": "dim"
    }
    
    for task in tasks[:limit]:
        status = task.status if isinstance(task.status, str) else task.status.value
        color = status_colors.get(status, "white")
        
        table.add_row(
            str(task.id),
            task.topic[:40],
            f"[{color}]{status}[/{color}]",
            f"{task.word_count:,}" if task.word_count else "-",
            str(task.depth)
        )
    
    console.print(table)


def create_progress_bar() -> Progress:
    """Create a research progress bar"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    )


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_completion_summary(
    total_tasks: int,
    completed_tasks: int,
    total_words: int,
    total_sources: int,
    duration_seconds: float
):
    """Print a completion summary"""
    completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    summary = f"""
[bold green]Research Complete![/bold green]

[cyan]Tasks:[/cyan] {completed_tasks}/{total_tasks} ({completion_rate:.1f}% complete)
[cyan]Words Written:[/cyan] {total_words:,}
[cyan]Sources Used:[/cyan] {total_sources:,}
[cyan]Duration:[/cyan] {format_duration(duration_seconds)}
[cyan]Avg Words/Task:[/cyan] {total_words // max(completed_tasks, 1):,}
"""
    
    console.print(Panel(summary, title="Summary", border_style="green"))
