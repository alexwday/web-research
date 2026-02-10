"""Token Provider - API authentication for LLM access.

If OPENAI_API_KEY is set, use it directly. Otherwise, use OAuth2 client credentials
flow with OAUTH_URL, CLIENT_ID, and CLIENT_SECRET environment variables.
"""
import os
import time
import logging

import requests

logger = logging.getLogger(__name__)


def fetch_oauth_token() -> tuple[str, dict]:
    """Fetch an API token for OpenAI / Azure LLM access.

    Returns:
        (token, auth_info) where auth_info is a dict describing the method used.

    Raises:
        ValueError: If neither OPENAI_API_KEY nor complete OAuth credentials are set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        logger.info("Using OPENAI_API_KEY for authentication (local mode)")
        return api_key, {"method": "api_key_local", "source": "OPENAI_API_KEY"}

    oauth_url = os.getenv("OAUTH_URL", "")
    client_id = os.getenv("CLIENT_ID", "")
    client_secret = os.getenv("CLIENT_SECRET", "")

    if not all([oauth_url, client_id, client_secret]):
        raise ValueError(
            "Missing OPENAI_API_KEY or OAuth credentials (OAUTH_URL, CLIENT_ID, CLIENT_SECRET)"
        )

    logger.info("Using OAuth2 client credentials flow for authentication")

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
                logger.info("OAuth token acquired successfully (client_id=%s)", client_id)
                return str(token), {"method": "oauth", "client_id": client_id}
            except (requests.exceptions.RequestException, ValueError) as exc:
                logger.warning(
                    "OAuth attempt %d/3 failed: %s", attempt_num, exc
                )
                if attempt_num == 3:
                    raise
                time.sleep(2)

    # Unreachable, but keeps type checkers happy
    raise RuntimeError("OAuth token fetch failed")
