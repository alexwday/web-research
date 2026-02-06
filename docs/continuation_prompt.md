# Continuation Prompt: Add Dual OpenAI Endpoint Support (Local + RBC OAuth2)

Copy everything below this line into a new Claude Code chat.

---

## Task

Add dual OpenAI endpoint support to the web-research project so it can run in two environments:

1. **Local/personal** — Uses `OPENAI_API_KEY` directly with `https://api.openai.com/v1`
2. **RBC corporate** — Uses OAuth2 client credentials flow to get a token, then connects to a custom Azure base URL with RBC SSL certificates

The project is at `/Users/alexwday/Projects/testing/web-research` and the remote repo is `https://github.com/alexwday/web-research`.

## Current Architecture

The LLM client is in `src/llm_client.py`. It has a single `OpenAIClient` class that:
- Reads `OPENAI_API_KEY` from env via `src/config.py` (`EnvSettings` Pydantic model)
- Creates an `OpenAI(api_key=...)` client
- Has `complete()` and `complete_with_messages()` methods used throughout the codebase
- Already skips temperature for reasoning models (`o1`, `o3`, `o4`, `gpt-5-mini`, `gpt-5-nano`)
- Already uses `max_completion_tokens` instead of `max_tokens` for newer models

The key callers are:
- `src/agents.py` — PlannerAgent, ResearcherAgent, EditorAgent all call `get_llm_client().complete()`
- `src/orchestrator.py` — Coordinates agents
- `src/web/app.py` — Background thread runs the orchestrator

## What Needs to Change

### 1. Add RBC Security SSL support

There's an existing pattern from the iris-project. Add `src/utils/rbc_security.py`:

```python
"""RBC Security certificate setup. Uses optional rbc_security for SSL when available."""
import logging
from typing import Optional

try:
    import rbc_security
    _RBC_SECURITY_AVAILABLE = True
except ImportError:
    _RBC_SECURITY_AVAILABLE = False

logger = logging.getLogger(__name__)

def configure_rbc_security_certs() -> Optional[str]:
    if not _RBC_SECURITY_AVAILABLE:
        logger.info("rbc_security not available, continuing without SSL certificates")
        return None
    logger.info("Enabling RBC Security certificates...")
    rbc_security.enable_certs()
    logger.info("RBC Security certificates enabled")
    return "rbc_security"
```

Call `configure_rbc_security_certs()` early in application startup (in `cli.py` before the server starts, and at the top of the background worker in `src/web/app.py`).

### 2. Add OAuth2 token provider

Create `src/oauth.py` following the iris-project pattern:

```python
"""Token Provider - API authentication for LLM access.
If OPENAI_API_KEY is set, use it directly. Otherwise, use OAuth2 client credentials."""
```

Key behavior:
- Check `os.getenv("OPENAI_API_KEY")` first — if set, return it directly (local mode)
- Otherwise, read `OAUTH_URL`, `CLIENT_ID`, `CLIENT_SECRET` from env
- POST to `OAUTH_URL` with `grant_type=client_credentials`, extract `access_token`
- Retry up to 3 times with 2s delay
- Return `(token, auth_info_dict)`

### 3. Modify `src/llm_client.py` OpenAIClient

Currently the constructor does:
```python
def __init__(self):
    settings = get_env_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    from openai import OpenAI
    self.client = OpenAI(api_key=settings.openai_api_key)
```

Change to:
- Import `fetch_oauth_token` from `src/oauth`
- Import `configure_rbc_security_certs` from `src/utils/rbc_security`
- Call `configure_rbc_security_certs()` once
- Call `fetch_oauth_token()` to get the token (works for both local API key and OAuth)
- Read `BASE_URL` from env: `os.getenv("AZURE_BASE_URL") or "https://api.openai.com/v1"`
- Create client: `OpenAI(api_key=token, base_url=base_url)`
- Log which auth method and endpoint are being used

### 4. Update `src/config.py` EnvSettings

Add these optional env vars:
```python
oauth_url: Optional[str] = Field(default=None, alias="OAUTH_URL")
oauth_client_id: Optional[str] = Field(default=None, alias="CLIENT_ID")
oauth_client_secret: Optional[str] = Field(default=None, alias="CLIENT_SECRET")
azure_base_url: Optional[str] = Field(default=None, alias="AZURE_BASE_URL")
```

### 5. Update `.env.example`

Add the new env vars with comments:
```
# OpenAI Configuration (local development)
OPENAI_API_KEY=sk-your-key-here

# RBC Corporate Environment (overrides OPENAI_API_KEY when set)
# AZURE_BASE_URL=https://your-corporate-endpoint/v1
# OAUTH_URL=https://your-oauth-server/token
# CLIENT_ID=your-client-id
# CLIENT_SECRET=your-client-secret
```

### 6. Update `test_env.py`

Add a section that tests OpenAI connectivity using the same dual-mode logic:
- If `OPENAI_API_KEY` is set → test direct OpenAI call
- If `OAUTH_URL` + `CLIENT_ID` + `CLIENT_SECRET` are set → test OAuth token fetch, then test OpenAI call with that token and `AZURE_BASE_URL`
- Report which auth method was used

## Reference: iris-project OAuth implementation

Here's the exact OAuth pattern from `/Users/alexwday/Projects/iris-project/services/src/connections/oauth.py`:

```python
def fetch_oauth_token() -> tuple[str, dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key, {"method": "api_key_local", "source": "OPENAI_API_KEY"}

    oauth_url = config.OAUTH_URL
    client_id = config.OAUTH_CLIENT_ID
    client_secret = config.OAUTH_CLIENT_SECRET

    if not all([oauth_url, client_id, client_secret]):
        raise ValueError("Missing OPENAI_API_KEY or OAuth settings")

    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }

    with requests.Session() as session:
        for attempt_num in range(1, 4):
            try:
                response = session.post(oauth_url, data=payload, timeout=180)
                response.raise_for_status()
                token = response.json().get("access_token")
                if not token:
                    raise ValueError("OAuth token not found in response")
                return str(token), {"method": "oauth", "client_id": client_id}
            except (requests.exceptions.RequestException, ValueError) as exc:
                if attempt_num == 3:
                    raise
                time.sleep(2)
```

## Reference: iris-project LLM client endpoint switching

From `/Users/alexwday/Projects/iris-project/services/src/connections/llm.py`:
```python
base_url = config.BASE_URL  # = os.getenv("AZURE_BASE_URL") or "https://api.openai.com/v1"
client = OpenAI(api_key=oauth_token, base_url=base_url)
```

## Reference: iris-project env config

From `/Users/alexwday/Projects/iris-project/services/src/utils/env_config.py`:
```python
BASE_URL: str = os.getenv("AZURE_BASE_URL") or "https://api.openai.com/v1"
OAUTH_URL: str = os.getenv("OAUTH_URL", "")
OAUTH_CLIENT_ID: str = os.getenv("CLIENT_ID", "")
OAUTH_CLIENT_SECRET: str = os.getenv("CLIENT_SECRET", "")
```

## Important Notes

- `python` doesn't work on this machine; use `python3`
- Use `py_compile` for syntax checks (dependencies may not be importable)
- The project uses Pydantic models for config (`src/config.py`) and SQLAlchemy for DB (`src/database.py`)
- The web server is FastAPI with HTMX templates
- After making changes, commit and push to `origin main` so the user can pull from their work computer
- Don't change any agent logic, orchestrator logic, or web dashboard — only the auth/connection layer
