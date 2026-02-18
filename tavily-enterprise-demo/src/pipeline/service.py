"""Service facade mirroring the main project's adapter->service pattern."""
import threading
from datetime import date
from typing import Callable, Optional

from src.pipeline.orchestrator import DemoOrchestrator


class TavilyDemoService:
    def __init__(self):
        self._orchestrator = DemoOrchestrator()

    def validate_environment(self) -> dict:
        return self._orchestrator.validate_environment()

    def run_quarterly_disclosures(
        self,
        period: str,
        bank_codes: Optional[list[str]] = None,
        run_loop: bool = False,
        poll_seconds: Optional[int] = None,
        max_iterations: Optional[int] = None,
        download_files: bool = True,
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        return self._orchestrator.run_quarterly_disclosure_scan(
            period=period,
            bank_codes=bank_codes,
            run_loop=run_loop,
            poll_seconds=poll_seconds,
            max_iterations=max_iterations,
            download_files=download_files,
            progress_cb=progress_cb,
        )

    def run_lcr_metrics(self, period: str, progress_cb: Optional[Callable] = None) -> dict:
        return self._orchestrator.run_lcr_metric_scan(period=period, progress_cb=progress_cb)

    def run_headlines(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        recency_days: int = 1,
        topics: Optional[list[str]] = None,
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        return self._orchestrator.run_headline_digest(
            start_date=start_date,
            end_date=end_date,
            recency_days=recency_days,
            topics=topics,
            progress_cb=progress_cb,
        )

    def run_deep_research(
        self,
        query: str,
        max_sources: Optional[int] = None,
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        return self._orchestrator.run_deep_research(
            query=query,
            max_sources=max_sources,
            progress_cb=progress_cb,
        )

    def run_internal_check(
        self,
        sample_query: Optional[str] = None,
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        return self._orchestrator.run_internal_readiness_check(
            sample_query=sample_query,
            progress_cb=progress_cb,
        )


_service: Optional[TavilyDemoService] = None
_service_lock = threading.Lock()


def get_service() -> TavilyDemoService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = TavilyDemoService()
    return _service


def reset_service() -> None:
    global _service
    with _service_lock:
        _service = None
