#!/usr/bin/env python3
"""
Environment Compatibility Test Script
======================================
Tests whether this machine can access the Tavily API for web search
and the OpenAI API (direct key or OAuth2).

Usage:
    pip install tavily-python python-dotenv openai requests
    cp .env.example .env   # fill in keys
    python3 test_env.py
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

for import_name, pip_name in [("dotenv", "python-dotenv"), ("tavily", "tavily-python"), ("openai", "openai"), ("requests", "requests")]:
    try:
        __import__(import_name)
        record(f"dep:{pip_name}", PASS)
    except ImportError:
        record(f"dep:{pip_name}", FAIL, f"pip install {pip_name}")


# =========================================================================
# 3. API Keys
# =========================================================================
print("\n=== API Keys ===")
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

openai_key = os.getenv("OPENAI_API_KEY", "")
oauth_url = os.getenv("OAUTH_URL", "")
client_id = os.getenv("CLIENT_ID", "")
client_secret = os.getenv("CLIENT_SECRET", "")
azure_base_url = os.getenv("AZURE_BASE_URL", "")

if openai_key:
    record("OPENAI_API_KEY", PASS, f"set (length {len(openai_key)}) — local mode")
elif all([oauth_url, client_id, client_secret]):
    record("OAuth credentials", PASS, f"OAUTH_URL, CLIENT_ID, CLIENT_SECRET all set — corporate mode")
else:
    record("LLM auth", FAIL, "Set OPENAI_API_KEY or (OAUTH_URL + CLIENT_ID + CLIENT_SECRET)")

if azure_base_url:
    record("AZURE_BASE_URL", PASS, azure_base_url)
else:
    record("AZURE_BASE_URL", SKIP, "not set — will use https://api.openai.com/v1")


# =========================================================================
# 4. OpenAI / LLM API Connectivity
# =========================================================================
print("\n=== OpenAI API Connectivity ===")
auth_token = None
auth_method = None
base_url = azure_base_url or "https://api.openai.com/v1"

if openai_key:
    auth_token = openai_key
    auth_method = "api_key_local"
elif all([oauth_url, client_id, client_secret]):
    try:
        import requests as req
        import time

        payload = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
        resp = req.post(oauth_url, data=payload, timeout=30)
        resp.raise_for_status()
        auth_token = resp.json().get("access_token")
        if not auth_token:
            record("oauth_token_fetch", FAIL, "access_token not in response")
        else:
            record("oauth_token_fetch", PASS, f"token acquired (length {len(auth_token)})")
            auth_method = "oauth"
    except Exception as e:
        record("oauth_token_fetch", FAIL, str(e))

if auth_token:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=auth_token, base_url=base_url)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=10,
        )
        reply = response.choices[0].message.content.strip()
        record(
            "openai_api_call",
            PASS,
            f"auth={auth_method}, base_url={base_url}, reply={reply!r}",
        )
    except Exception as e:
        record("openai_api_call", FAIL, f"auth={auth_method}, base_url={base_url}, error={e}")
else:
    record("openai_api_call", SKIP, "no auth token available")


# =========================================================================
# 5. Tavily API connectivity
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
    print("\nAll checks passed! Environment is ready.")
    sys.exit(0)
