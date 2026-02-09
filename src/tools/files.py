"""File operations: reading, writing, path generation."""
import re
from pathlib import Path
from typing import Optional

from ..logger import get_logger

logger = get_logger(__name__)


def ensure_directory(path: str) -> Path:
    """Ensure a directory exists"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_markdown(filepath: str, content: str, append: bool = True) -> bool:
    """
    Save content to a markdown file
    """
    try:
        path = Path(filepath)
        ensure_directory(path.parent)

        mode = 'a' if append and path.exists() else 'w'
        with open(path, mode, encoding='utf-8') as f:
            f.write(content)
            if not content.endswith('\n'):
                f.write('\n')

        logger.debug(f"Saved content to {filepath}")
        return True

    except Exception as e:
        logger.error(f"Failed to save to {filepath}: {e}")
        return False


def read_file(filepath: str) -> Optional[str]:
    """Read content from a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.error(f"Failed to read {filepath}: {e}")
        return None


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be used as a filename"""
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    # Remove invalid characters
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    # Limit length
    name = name[:100]
    return name


def generate_file_path(topic: str, output_dir: str = "report", index: int = None) -> str:
    """Generate a file path for a topic"""
    safe_name = sanitize_filename(topic)

    if index is not None:
        filename = f"{index:02d}_{safe_name}.md"
    else:
        filename = f"{safe_name}.md"

    return str(Path(output_dir) / filename)
