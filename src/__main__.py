#!/usr/bin/env python3
"""
Deep Research Agent - Main Entry Point

A 24-hour automated research system that produces comprehensive,
book-length reports with citations on any topic.

Usage:
    python3 -m src "Your research query here"
    python3 -m src --resume
    python3 -m src --help
"""

from src.adapters.cli import app


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()
