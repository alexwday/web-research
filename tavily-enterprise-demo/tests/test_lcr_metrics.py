import json
from pathlib import Path

from src.config.types import BankConfig, Config
from src.pipeline.orchestrator import DemoOrchestrator


class FakeSearchTool:
    def search(self, query, max_results, include_domains=None, exclude_domains=None):
        if include_domains:
            return [
                {
                    "url": "https://rbc.com/investor-relations/rbc_q4_liquidity.pdf",
                    "title": "RBC Q4 2025 Pillar 3 Report",
                    "snippet": "Liquidity Coverage Ratio (LCR) was 132%.",
                    "raw_content": "In Q4 2025, Liquidity Coverage Ratio was 132% and stable.",
                    "score": 0.80,
                }
            ]
        return [
            {
                "url": "https://blog.example.com/rbc-liquidity",
                "title": "Unofficial discussion",
                "snippet": "LCR near 118%.",
                "raw_content": "LCR near 118% based on estimates.",
                "score": 0.40,
            }
        ]


class NoopDownloader:
    def download(self, url, destination):
        return True, None


def test_lcr_metric_prefers_primary_source(tmp_path):
    cfg = Config()
    cfg.output.directory = str(tmp_path / "report")
    cfg.banks = [
        BankConfig(
            code="rbc",
            name="Royal Bank of Canada",
            aliases=["RBC"],
            primary_domains=["rbc.com"],
        )
    ]

    orchestrator = DemoOrchestrator(
        config=cfg,
        search_tool=FakeSearchTool(),
        downloader=NoopDownloader(),
    )
    result = orchestrator.run_lcr_metric_scan(period="2025Q4")

    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    row = payload["rows"][0]

    assert row["lcr_value"] == "132%"
    assert row["source_type"] == "primary"
    assert "rbc.com" in row["source_url"]
