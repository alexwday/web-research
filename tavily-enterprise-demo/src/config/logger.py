"""Logging utilities for Tavily enterprise demo."""
import logging
import logging.handlers
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from src.config.settings import get_config

console = Console()


class DemoLogger:
    def __init__(self, name: str = "tavily_demo"):
        cfg = get_config()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            console_handler = RichHandler(console=console, show_time=True, show_path=False)
            console_handler.setLevel(logging.INFO)
            self.logger.addHandler(console_handler)

            log_file = Path(cfg.output.directory) / "demo.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
            )
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)

    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        self.logger.exception(msg, *args, **kwargs)


_loggers: dict[str, DemoLogger] = {}


def get_logger(name: str = "tavily_demo") -> DemoLogger:
    if name not in _loggers:
        _loggers[name] = DemoLogger(name)
    return _loggers[name]


def print_info(message: str) -> None:
    console.print(f"[blue][INFO][/blue] {message}")


def print_success(message: str) -> None:
    console.print(f"[green][OK][/green] {message}")


def print_warning(message: str) -> None:
    console.print(f"[yellow][WARN][/yellow] {message}")


def print_error(message: str) -> None:
    console.print(f"[red][ERR][/red] {message}")
