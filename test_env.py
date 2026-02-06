#!/usr/bin/env python3
"""
Environment Compatibility Test Script
======================================
Tests whether this machine can access the Tavily API for web search.

Usage:
    pip install tavily-python python-dotenv
    cp .env.example .env   # fill in TAVILY_API_KEY
    python test_env.py
"""

import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
results = []


def record(name: str, status: str, detail: str = ""):
    results.append((name, status, detail))
    msg = f"  {status} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)


# =========================================================================
# 1. RBC Security (optional — enables SSL certs in corporate environments)
# =========================================================================
print("\n=== RBC Security ===")
try:
    import rbc_security

    rbc_security.enable_certs()
    record("rbc_security", PASS, "SSL certificates enabled")
except ImportError:
    record("rbc_security", SKIP, "Package not available (OK outside RBC network)")
except Exception as e:
    record("rbc_security", FAIL, str(e))


# =========================================================================
# 2. Python version & dependencies
# =========================================================================
print("\n=== Python Environment ===")
py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
if sys.version_info >= (3, 9):
    record("python_version", PASS, py_ver)
else:
    record("python_version", FAIL, f"{py_ver} (need 3.9+)")

for import_name, pip_name in [("dotenv", "python-dotenv"), ("tavily", "tavily-python")]:
    try:
        __import__(import_name)
        record(f"dep:{pip_name}", PASS)
    except ImportError:
        record(f"dep:{pip_name}", FAIL, f"pip install {pip_name}")


# =========================================================================
# 3. Tavily API key
# =========================================================================
print("\n=== API Key ===")
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

tavily_key = os.getenv("TAVILY_API_KEY", "")
if tavily_key:
    record("TAVILY_API_KEY", PASS, f"set (length {len(tavily_key)})")
else:
    record("TAVILY_API_KEY", FAIL, "not set in environment or .env file")


# =========================================================================
# 4. Tavily API connectivity
# =========================================================================
print("\n=== Tavily API Connectivity ===")
if tavily_key:
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=tavily_key)

        # Basic search test
        result = client.search("test query python programming", max_results=2)
        num_results = len(result.get("results", []))
        record("tavily_basic_search", PASS, f"returned {num_results} results")

        if num_results > 0:
            first = result["results"][0]
            print(f"         Sample: {first.get('title', 'N/A')[:60]}")
            print(f"         URL:    {first.get('url', 'N/A')[:80]}")

        # Advanced search with raw content (this is what the research agent uses)
        result_adv = client.search(
            "artificial intelligence overview",
            max_results=1,
            search_depth="advanced",
            include_raw_content=True,
        )
        adv_results = result_adv.get("results", [])
        has_content = bool(adv_results and adv_results[0].get("raw_content", ""))
        content_len = len(adv_results[0].get("raw_content", "")) if adv_results else 0
        record(
            "tavily_advanced_search",
            PASS if has_content else FAIL,
            f"raw_content {'works' if has_content else 'empty'} ({content_len:,} chars)",
        )

    except Exception as e:
        record("tavily_basic_search", FAIL, str(e))
else:
    record("tavily_basic_search", SKIP, "no API key")


# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
passes = sum(1 for _, s, _ in results if s == PASS)
fails = sum(1 for _, s, _ in results if s == FAIL)
skips = sum(1 for _, s, _ in results if s == SKIP)
print(f"  {PASS} {passes}   {FAIL} {fails}   {SKIP} {skips}")

if fails:
    print("\nFailed checks:")
    for name, status, detail in results:
        if status == FAIL:
            print(f"  - {name}: {detail}")
    sys.exit(1)
else:
    print("\nAll checks passed! Tavily API is accessible from this environment.")
    sys.exit(0)
