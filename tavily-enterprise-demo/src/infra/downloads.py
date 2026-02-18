"""HTTP download helper for file artifacts."""
import re
from pathlib import Path
from typing import Optional

import requests

from src.config.logger import get_logger

logger = get_logger(__name__)

_CONTENT_TYPE_TO_EXT: dict[str, str] = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xls",
}


class FileDownloader:
    """Downloads file artifacts from URLs."""

    def __init__(self, timeout_seconds: int = 45, user_agent: str = ""):
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent

    def download(self, url: str, destination: Path) -> tuple[bool, Optional[str]]:
        headers = {"User-Agent": self.user_agent} if self.user_agent else {}
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            with requests.get(url, headers=headers, timeout=self.timeout_seconds, stream=True) as response:
                response.raise_for_status()
                with open(destination, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True, None
        except Exception as exc:
            logger.warning("Download failed for %s: %s", url, exc)
            return False, str(exc)

    def probe_content_type(self, url: str) -> tuple[Optional[str], Optional[str]]:
        """HEAD request to detect Content-Type and filename.

        Returns (detected_extension, filename_from_header).
        """
        headers = {"User-Agent": self.user_agent} if self.user_agent else {}
        try:
            resp = requests.head(
                url, headers=headers, timeout=15, allow_redirects=True
            )
            resp.raise_for_status()

            ct = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
            cd = resp.headers.get("Content-Disposition", "")

            ext = _CONTENT_TYPE_TO_EXT.get(ct)

            # Try to get filename from Content-Disposition
            filename: Optional[str] = None
            if cd:
                match = re.search(r"filename[*]?=[\"']?([^\"';]+)", cd)
                if match:
                    filename = match.group(1).strip()
                    # Infer extension from filename if Content-Type didn't match
                    if ext is None and filename:
                        for suffix, e in ((".xlsx", "xlsx"), (".xls", "xls"), (".pdf", "pdf")):
                            if filename.lower().endswith(suffix):
                                ext = e
                                break

            return ext, filename
        except Exception as exc:
            logger.debug("HEAD probe failed for %s: %s", url, exc)
            return None, None

    def fetch_page_html(self, url: str) -> Optional[str]:
        """Fetch raw HTML from a URL (for IR page link discovery)."""
        headers = {"User-Agent": self.user_agent} if self.user_agent else {}
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            ct = resp.headers.get("Content-Type", "")
            if "text/html" not in ct and "text/xhtml" not in ct:
                return None
            return resp.text
        except Exception as exc:
            logger.debug("HTML fetch failed for %s: %s", url, exc)
            return None
