import json
from pathlib import Path

from src.config.types import BankConfig, Config
from src.pipeline.orchestrator import DemoOrchestrator


class FakeSearchTool:
    def search(self, query, max_results, include_domains=None, exclude_domains=None):
        return [
            {
                "url": "https://news.example.com/story-1",
                "title": "Big six banks face new liquidity stress test",
                "snippet": "Major Canadian banks react to new rule.",
                "raw_content": "Major Canadian banks react to new rule with updated liquidity plans.",
                "score": 0.88,
            },
            {
                "url": "https://news.example.com/story-1",
                "title": "Duplicate should dedupe",
                "snippet": "Duplicate",
                "raw_content": "Duplicate",
                "score": 0.20,
            },
            {
                "url": "https://finance.example.com/story-2",
                "title": "Canadian finance outlook improves",
                "snippet": "Markets close higher.",
                "raw_content": "Markets close higher as rates expectations shift.",
                "score": 0.72,
            },
        ]


class NoopDownloader:
    def download(self, url, destination):
        return True, None


def _build_cfg(tmp_path):
    cfg = Config()
    cfg.output.directory = str(tmp_path / "report")
    cfg.banks = [
        BankConfig(
            code="rbc",
            name="Royal Bank of Canada",
            aliases=["RBC", "Royal Bank"],
            primary_domains=["rbc.com"],
        )
    ]
    return cfg


def test_headline_digest_deduplicates_urls(tmp_path):
    orchestrator = DemoOrchestrator(
        config=_build_cfg(tmp_path),
        search_tool=FakeSearchTool(),
        downloader=NoopDownloader(),
    )

    result = orchestrator.run_headline_digest(recency_days=1)
    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

    total = payload["total_headlines"]
    assert total == 2


def test_deep_research_outputs_findings(tmp_path):
    orchestrator = DemoOrchestrator(
        config=_build_cfg(tmp_path),
        search_tool=FakeSearchTool(),
        downloader=NoopDownloader(),
    )

    result = orchestrator.run_deep_research("Canadian bank liquidity strategy", max_sources=5)

    assert result["source_count"] >= 1
    summary = Path(result["summary_path"]).read_text(encoding="utf-8")
    assert "Deep Research Brief" in summary
    assert "Canadian bank liquidity strategy" in summary
