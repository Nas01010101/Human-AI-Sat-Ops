"""
Brave Search integration — fetches live space news and related resources.

Used in the dashboard to surface relevant conjunction events, space debris
news, and similar open-source tools.
"""

import os
import requests
from typing import List, Dict

BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"


def _get_api_key() -> str | None:
    key = os.getenv("BRAVE_SEARCH_API_KEY", "")
    return key if key and key != "your_brave_api_key_here" else None


def search(query: str, count: int = 5) -> List[Dict]:
    """
    Run a Brave Search query. Returns list of {title, url, description}.
    Returns empty list if API key is not configured.
    """
    api_key = _get_api_key()
    if not api_key:
        return []

    try:
        resp = requests.get(
            BRAVE_API_URL,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
            params={"q": query, "count": count},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("web", {}).get("results", [])
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "description": r.get("description", "")}
            for r in results
        ]
    except Exception:
        return []


def get_space_news() -> List[Dict]:
    return search("satellite conjunction collision avoidance space debris 2025", count=5)


def get_similar_tools() -> List[Dict]:
    return search(
        "site:github.com satellite conjunction assessment collision avoidance decision support tool",
        count=5,
    )
