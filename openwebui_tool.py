"""
title: Web Search & URL Fetch
author: Christopher Gray
description: Search the web via SearXNG and fetch full URL content. Works with any model (Gemma, Qwen, Ollama, vLLM).
version: 1.1.0
updated: 5/16/2026

Download from:  https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/openwebui_tool.py

license: MIT
requirements: requests, beautifulsoup4, lxml
"""

import requests
from bs4 import BeautifulSoup
from typing import Optional
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        searxng_url: str = Field(
            default="http://localhost:8080",
            description="Base URL of your SearXNG instance",
        )
        max_results: int = Field(
            default=5,
            description="Maximum number of search results to return",
        )
        fetch_timeout: int = Field(
            default=15,
            description="Timeout in seconds for URL fetch requests",
        )
        max_content_lines: int = Field(
            default=400,
            description="Maximum lines of page content to return to the model",
        )

    def __init__(self):
        self.valves = self.Valves()

    def web_search(self, query: str) -> str:
        """
        Search the web using SearXNG and return titles, URLs, and snippets.
        :param query: The search query string
        :return: Formatted search results
        """
        try:
            resp = requests.get(
                f"{self.valves.searxng_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "safesearch": 0,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])[: self.valves.max_results]
            if not results:
                return "No search results found."

            lines = [f"Search results for: {query}\n"]
            for i, r in enumerate(results, 1):
                lines.append(
                    f"[{i}] {r.get('title', 'No title')}\n"
                    f"    URL: {r.get('url', '')}\n"
                    f"    {r.get('content', '').strip()}\n"
                )
            return "\n".join(lines)

        except requests.exceptions.ConnectionError:
            return (
                "Error: Cannot reach SearXNG at "
                f"{self.valves.searxng_url}. "
                "Make sure the SearXNG container is running."
            )
        except Exception as e:
            return f"Search error: {type(e).__name__}: {e}"

    def fetch_url(self, url: str) -> str:
        """
        Fetch and extract the readable text content from any URL.
        :param url: Full URL to fetch (e.g. https://www.cnn.com/business)
        :return: Extracted page text content
        """
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/150.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
            resp = requests.get(
                url, headers=headers, timeout=self.valves.fetch_timeout
            )
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "lxml")

            # Strip non-content tags
            for tag in soup(["script", "style", "nav", "footer", "header",
                              "aside", "form", "noscript", "iframe"]):
                tag.decompose()

            # Prefer article/main content if available
            main = soup.find("article") or soup.find("main") or soup.body
            text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)

            lines = [l for l in text.splitlines() if l.strip()]
            truncated = lines[: self.valves.max_content_lines]

            result = f"Content from: {url}\n{'='*60}\n" + "\n".join(truncated)
            if len(lines) > self.valves.max_content_lines:
                result += f"\n\n[Content truncated — {len(lines)} lines total, showing first {self.valves.max_content_lines}]"
            return result

        except requests.exceptions.ConnectionError:
            return f"Error: Could not connect to {url}"
        except requests.exceptions.Timeout:
            return f"Error: Request to {url} timed out after {self.valves.fetch_timeout}s"
        except requests.exceptions.HTTPError as e:
            return f"Error: HTTP {e.response.status_code} from {url}"
        except Exception as e:
            return f"Fetch error: {type(e).__name__}: {e}"
