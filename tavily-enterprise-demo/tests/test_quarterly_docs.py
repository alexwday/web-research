from pathlib import Path

from src.config.types import BankConfig, Config
from src.pipeline.orchestrator import DemoOrchestrator


class FakeSearchTool:
    def __init__(self):
        self.calls = []

    def search(self, query, max_results, include_domains=None, exclude_domains=None):
        self.calls.append(
            {
                "query": query,
                "include_domains": include_domains,
                "exclude_domains": exclude_domains,
            }
        )

        if include_domains:
            return [
                {
                    "url": "https://rbc.com/investor-relations/rbc_q4_2025_report.pdf",
                    "title": "RBC Q4 2025 Report to Shareholders PDF",
                    "snippet": "Quarterly report package",
                    "raw_content": "",
                    "score": 0.91,
                },
                {
                    "url": "https://rbc.com/investor-relations/rbc_q4_2025_pillar3.pdf",
                    "title": "RBC Q4 2025 Pillar 3 Disclosure PDF",
                    "snippet": "Pillar 3 capital and liquidity",
                    "raw_content": "",
                    "score": 0.88,
                },
                {
                    "url": "https://rbc.com/investor-relations/rbc_q4_2025_pillar3.xlsx",
                    "title": "RBC Q4 2025 Pillar 3 Disclosure XLSX",
                    "snippet": "Pillar 3 liquidity and capital data tables",
                    "raw_content": "",
                    "score": 0.87,
                },
                {
                    "url": "https://rbc.com/investor-relations/rbc_q4_2025_supp_fin.xlsx",
                    "title": "RBC Q4 2025 Supplementary Financial Information",
                    "snippet": "Supplementary financial data package",
                    "raw_content": "",
                    "score": 0.85,
                },
            ]

        return [
            {
                "url": "https://newswire.example.com/rbc-quarterly-results.pdf",
                "title": "RBC results release",
                "snippet": "Secondary source",
                "raw_content": "",
                "score": 0.70,
            }
        ]


class FakeDownloader:
    def __init__(self):
        self.calls = []

    def download(self, url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(b"demo")
        self.calls.append((url, str(destination)))
        return True, None


def test_quarterly_docs_scan_downloads_files(tmp_path):
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

    search_tool = FakeSearchTool()
    downloader = FakeDownloader()
    orchestrator = DemoOrchestrator(config=cfg, search_tool=search_tool, downloader=downloader)

    result = orchestrator.run_quarterly_disclosure_scan(period="2025Q4", run_loop=False)

    assert result["status"] == "complete"
    assert result["all_complete"] is True
    assert len(downloader.calls) >= 4  # pdf+pdf+xlsx+xlsx for 3 doc types
    assert Path(result["manifest_path"]).exists()
    assert Path(result["summary_path"]).exists()
    assert "downloaded_files" in result
    assert len(result["downloaded_files"]) >= 4
    for f in result["downloaded_files"]:
        assert f["bank"] == "Royal Bank of Canada"
        assert f["path"]
        assert f["extension"] in ("pdf", "xlsx")


def test_quarterly_docs_bank_filter(tmp_path):
    """bank_codes filters to only the selected bank."""
    cfg = Config()
    cfg.output.directory = str(tmp_path / "report")
    cfg.banks = [
        BankConfig(code="rbc", name="Royal Bank of Canada", aliases=["RBC"], primary_domains=["rbc.com"]),
        BankConfig(code="td", name="Toronto-Dominion Bank", aliases=["TD"], primary_domains=["td.com"]),
    ]

    search_tool = FakeSearchTool()
    downloader = FakeDownloader()
    orchestrator = DemoOrchestrator(config=cfg, search_tool=search_tool, downloader=downloader)

    result = orchestrator.run_quarterly_disclosure_scan(period="2025Q4", bank_codes=["rbc"])

    assert result["bank_count"] == 1
    assert result["status"] == "complete"
    for f in result["downloaded_files"]:
        assert f["bank"] == "Royal Bank of Canada"


def test_quarterly_docs_all_banks_when_no_filter(tmp_path):
    """When bank_codes is None, all banks are scanned."""
    cfg = Config()
    cfg.output.directory = str(tmp_path / "report")
    cfg.banks = [
        BankConfig(code="rbc", name="Royal Bank of Canada", aliases=["RBC"], primary_domains=["rbc.com"]),
        BankConfig(code="td", name="Toronto-Dominion Bank", aliases=["TD"], primary_domains=["td.com"]),
    ]

    search_tool = FakeSearchTool()
    downloader = FakeDownloader()
    orchestrator = DemoOrchestrator(config=cfg, search_tool=search_tool, downloader=downloader)

    result = orchestrator.run_quarterly_disclosure_scan(period="2025Q4", bank_codes=None)

    assert result["bank_count"] == 2


def test_quarterly_docs_dry_run(tmp_path):
    """With download_files=False, no downloads occur but candidates are reported."""
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

    search_tool = FakeSearchTool()
    downloader = FakeDownloader()
    orchestrator = DemoOrchestrator(config=cfg, search_tool=search_tool, downloader=downloader)

    result = orchestrator.run_quarterly_disclosure_scan(
        period="2025Q4", run_loop=False, download_files=False
    )

    assert len(downloader.calls) == 0  # No downloads in dry run
    assert Path(result["manifest_path"]).exists()
