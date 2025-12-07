#!/usr/bin/env python3
"""
Deep Research Agent - Main Entry Point

A 24-hour automated research system that produces comprehensive,
book-length reports with citations on any topic.

Usage:
    python main.py "Your research query here"
    python main.py --resume
    python main.py --help
"""

import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent))

from cli import app


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()
