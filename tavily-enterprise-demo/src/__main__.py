#!/usr/bin/env python3
"""CLI entrypoint for Tavily enterprise demo."""

from src.adapters.cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
