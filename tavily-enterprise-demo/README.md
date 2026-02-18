# Tavily Enterprise Demo (Standalone)

This folder is a standalone demo project that mirrors the main project's runtime style:

- `python -m src` entrypoint
- Typer CLI commands
- `.env` + `config.yaml` configuration
- `service` facade over an `orchestrator`
- reusable Tavily `_tools` layer

It is designed for manager demos showing practical enterprise Tavily use cases.

## Setup

```bash
cd tavily-enterprise-demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill TAVILY_API_KEY in .env
```

## Validate Environment

```bash
python -m src validate
```

## Interactive Popup Interface

Launch a web UI where each use case opens in a walkthrough modal with:

- description + step-by-step flow
- run controls
- live execution logs
- result payload/artifact paths

```bash
python -m src serve --host 127.0.0.1 --port 8000
```

Open: `http://127.0.0.1:8000`

## Use Case Commands

### 1) Quarterly disclosure monitoring + PDF/XLSX download

Find latest report to shareholders and Pillar 3 disclosures for each Big-6 bank.
Primary source is each bank IR domain, with secondary web discovery included.

```bash
python -m src quarterly-docs --period 2025Q4
```

Loop mode (production-style polling):

```bash
python -m src quarterly-docs --period 2025Q4 --loop --poll-seconds 900 --max-iterations 96
```

### 2) LCR metric discovery from web sources

Search for bank LCR values when internal line-item data is missing.

```bash
python -m src lcr-metrics --period 2025Q4
```

### 3) Latest finance + Big-6 headlines digest

```bash
python -m src headlines --recency-days 1
python -m src headlines --start-date 2026-02-01 --end-date 2026-02-07
```

Optional custom topics:

```bash
python -m src headlines --recency-days 7 --topics "bank earnings,capital markets,liquidity"
```

### 4) Deep research workflow (search -> rank -> synthesize)

```bash
python -m src deep-research "How are Canadian banks adjusting liquidity strategy under elevated rates?"
```

### 5) Internal-network readiness proof

Validates local/corporate auth pattern and optionally runs Tavily probe query.

```bash
python -m src internal-check
```

## Output Artifacts

Artifacts are written under `report/`:

- `report/quarterly/<period>/<run_tag>/manifest.json`
- `report/lcr/<period>/<run_tag>/lcr_metrics.csv`
- `report/headlines/<run_tag>/HEADLINES.md`
- `report/deep_research/<run_tag>/RESEARCH_BRIEF.md`
- `report/internal_check/internal_check_<timestamp>.json`

## Notes

- `config.yaml` is pre-created and editable.
- For strict source controls, edit each bank `primary_domains` and search settings in `config.yaml`.
