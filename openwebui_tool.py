"""
title: Web Search & URL Fetch
author: Christopher Gray
description: Search the web via SearXNG and fetch full URL content. Works with any model (Gemma, Qwen, Ollama, vLLM).
version: 1.1.5
updated: 5/16/2026

Download from:  https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/openwebui_tool.py

license: MIT
requirements: requests, beautifulsoup4, lxml
"""

import re
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
        search_engines: str = Field(
            default="",
            description="Comma-separated SearXNG engines to use (e.g. 'google,bing'). Leave blank to use all enabled engines.",
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
            params = {"q": query, "format": "json", "safesearch": 0}
            if self.valves.search_engines.strip():
                params["engines"] = self.valves.search_engines.strip()
            resp = requests.get(
                f"{self.valves.searxng_url}/search",
                params=params,
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

    def get_weather(self, location: str) -> str:
        """
        Get current weather and a 3-day forecast for any location — zip code, city name, airport code, or coordinates.
        :param location: Location to get weather for (e.g. "80203", "Denver CO", "LAX", "48.85,2.35")
        :return: Current conditions and forecast as plain text
        """
        try:
            resp = requests.get(
                f"https://wttr.in/{requests.utils.quote(location)}",
                params={"format": "j1"},
                headers={"User-Agent": "curl/7.68.0"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            current = data["current_condition"][0]
            area = data.get("nearest_area", [{}])[0]

            city = area.get("areaName", [{}])[0].get("value", "")
            region = area.get("region", [{}])[0].get("value", "")
            country = area.get("country", [{}])[0].get("value", "")
            location_label = ", ".join(filter(None, [city, region, country])) or location

            temp_f = current.get("temp_F", "?")
            temp_c = current.get("temp_C", "?")
            feels_f = current.get("FeelsLikeF", "?")
            feels_c = current.get("FeelsLikeC", "?")
            humidity = current.get("humidity", "?")
            wind_mph = current.get("windspeedMiles", "?")
            wind_dir = current.get("winddir16Point", "?")
            visibility = current.get("visibility", "?")
            desc = current.get("weatherDesc", [{}])[0].get("value", "?")
            uv = current.get("uvIndex", "?")

            lines = [
                f"Weather for: {location_label}",
                f"Conditions:  {desc}",
                f"Temperature: {temp_f}°F / {temp_c}°C  (feels like {feels_f}°F / {feels_c}°C)",
                f"Humidity:    {humidity}%",
                f"Wind:        {wind_mph} mph {wind_dir}",
                f"Visibility:  {visibility} miles",
                f"UV Index:    {uv}",
                "",
                "3-Day Forecast:",
            ]

            for day in data.get("weather", []):
                date = day.get("date", "")
                hi_f = day.get("maxtempF", "?")
                lo_f = day.get("mintempF", "?")
                hi_c = day.get("maxtempC", "?")
                lo_c = day.get("mintempC", "?")
                day_desc = day.get("hourly", [{}])[4].get("weatherDesc", [{}])[0].get("value", "")
                sunrise = day.get("astronomy", [{}])[0].get("sunrise", "")
                sunset = day.get("astronomy", [{}])[0].get("sunset", "")
                lines.append(
                    f"  {date}: {day_desc} | Hi {hi_f}°F/{hi_c}°C  Lo {lo_f}°F/{lo_c}°C"
                    + (f" | Sunrise {sunrise}  Sunset {sunset}" if sunrise else "")
                )

            return "\n".join(lines)

        except requests.exceptions.Timeout:
            return f"Error: Weather request timed out for '{location}'"
        except requests.exceptions.HTTPError as e:
            return f"Error: Could not fetch weather for '{location}' (HTTP {e.response.status_code})"
        except Exception as e:
            return f"Weather error: {type(e).__name__}: {e}"

    def get_stock_price(self, ticker: str) -> str:
        """
        Get real-time stock price, daily stats, and key metrics for any stock, ETF, index, or crypto.
        Accepts a ticker symbol or a company name. Uses Yahoo Finance with Stooq as a fallback.
        :param ticker: Ticker symbol (e.g. "AAPL", "BTC-USD", "SPY") or company name (e.g. "Apple", "Tesla")
        :return: Current price, daily change, volume, 52-week range, market cap, and pre/after-hours price if available
        """
        original = ticker.strip()
        symbol = original.upper()

        # If it looks like a company name (has spaces or is longer than a typical ticker), resolve it
        if re.search(r"\s", original) or len(original) > 6:
            resolved = self._resolve_ticker(original)
            if resolved:
                symbol = resolved

        result = self._yahoo_quote(symbol)
        if result and not result.startswith("Error:"):
            return result + f"\nSee also:    https://www.cnn.com/markets/stocks/{symbol}"

        result = self._cnn_quote(symbol)
        if result and not result.startswith("Error:"):
            return result

        return self._stooq_quote(symbol)

    def _resolve_ticker(self, name: str) -> str:
        try:
            resp = requests.get(
                "https://query2.finance.yahoo.com/v1/finance/search",
                params={"q": name, "quotesCount": 5, "newsCount": 0},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=8,
            )
            quotes = resp.json().get("quotes", [])
            for q in quotes:
                if q.get("quoteType") in ("EQUITY", "ETF"):
                    return q.get("symbol", "")
            return quotes[0].get("symbol", "") if quotes else ""
        except Exception:
            return ""

    def _yahoo_quote(self, symbol: str) -> str:
        try:
            resp = requests.get(
                f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}",
                params={"modules": "price,summaryDetail"},
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json",
                },
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()["quoteSummary"]["result"][0]
            p = result["price"]
            sd = result.get("summaryDetail", {})

            name = p.get("longName") or p.get("shortName", symbol)
            currency = p.get("currency", "USD")
            market_state = p.get("marketState", "REGULAR")
            exchange = p.get("exchangeName", "")

            current = p.get("regularMarketPrice", {}).get("raw")
            if current is None:
                return f"Error: No price data returned for {symbol}"

            change = p.get("regularMarketChange", {}).get("raw", 0)
            pct = p.get("regularMarketChangePercent", {}).get("raw", 0)
            sign = "+" if change >= 0 else ""

            def f(v):
                return f"${v:,.2f}" if v is not None else "N/A"

            open_raw = p.get("regularMarketOpen", {}).get("raw")
            prev_raw = p.get("regularMarketPreviousClose", {}).get("raw")
            hi_raw = p.get("regularMarketDayHigh", {}).get("raw")
            lo_raw = p.get("regularMarketDayLow", {}).get("raw")
            vol_fmt = p.get("regularMarketVolume", {}).get("fmt", "N/A")
            avg_vol = p.get("averageDailyVolume3Month", {}).get("fmt", "N/A")
            mktcap = p.get("marketCap", {}).get("fmt")
            wk52_hi = sd.get("fiftyTwoWeekHigh", {}).get("raw")
            wk52_lo = sd.get("fiftyTwoWeekLow", {}).get("raw")

            lines = [
                f"Stock: {name} ({symbol})  |  {exchange}  |  {market_state}  |  {currency}",
                f"Price:       {f(current)}  ({sign}{change:.2f} / {sign}{pct * 100:.2f}% today)",
            ]

            if market_state == "PRE":
                pre = p.get("preMarketPrice", {}).get("raw")
                pre_chg = p.get("preMarketChange", {}).get("raw", 0)
                pre_pct = p.get("preMarketChangePercent", {}).get("raw", 0)
                if pre:
                    s = "+" if pre_chg >= 0 else ""
                    lines.append(f"Pre-Market:  {f(pre)}  ({s}{pre_chg:.2f} / {s}{pre_pct * 100:.2f}%)")
            elif market_state in ("POST", "POSTPOST"):
                post = p.get("postMarketPrice", {}).get("raw")
                post_chg = p.get("postMarketChange", {}).get("raw", 0)
                post_pct = p.get("postMarketChangePercent", {}).get("raw", 0)
                if post:
                    s = "+" if post_chg >= 0 else ""
                    lines.append(f"After-Hours: {f(post)}  ({s}{post_chg:.2f} / {s}{post_pct * 100:.2f}%)")

            lines += [
                f"Open:        {f(open_raw)}   Prev Close: {f(prev_raw)}",
                f"Day Range:   {f(lo_raw)} – {f(hi_raw)}",
            ]
            if wk52_hi and wk52_lo:
                lines.append(f"52-Wk Range: {f(wk52_lo)} – {f(wk52_hi)}")
            lines.append(f"Volume:      {vol_fmt}   Avg Volume: {avg_vol}")
            if mktcap:
                lines.append(f"Market Cap:  {mktcap}")

            return "\n".join(lines)

        except Exception as e:
            return f"Error: Yahoo Finance: {type(e).__name__}: {e}"

    def _cnn_quote(self, symbol: str) -> str:
        try:
            resp = requests.get(
                "https://production.dataviz.cnn.io/v2/ticker/list",
                params={"tickers": symbol},
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json",
                    "Referer": f"https://www.cnn.com/markets/stocks/{symbol}",
                },
                timeout=10,
            )
            resp.raise_for_status()
            payload = resp.json()
            items = payload if isinstance(payload, list) else payload.get("data", [])
            if not items:
                return f"Error: CNN returned no data for {symbol}"

            q = items[0]

            def _raw(key, *aliases):
                for k in (key, *aliases):
                    v = q.get(k)
                    if v is not None:
                        return v
                return None

            def f(v):
                return f"${v:,.2f}" if isinstance(v, (int, float)) else "N/A"

            def fmt_vol(v):
                if not isinstance(v, (int, float)):
                    return "N/A"
                for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
                    if v >= threshold:
                        return f"{v / threshold:.2f}{suffix}"
                return str(int(v))

            def fmt_cap(v):
                if not isinstance(v, (int, float)):
                    return "N/A"
                for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
                    if v >= threshold:
                        return f"${v / threshold:.2f}{suffix}"
                return f"${v:,.0f}"

            name = _raw("name", "company_name") or symbol
            price = _raw("price", "last", "regularMarketPrice")
            if price is None:
                return f"Error: CNN returned no price for {symbol}"

            change = _raw("change", "priceChange") or 0
            pct = _raw("changePercent", "pct_change", "changePercent1Day") or 0
            open_p = _raw("open", "openPrice")
            high = _raw("high", "dayHigh", "regularMarketDayHigh")
            low = _raw("low", "dayLow", "regularMarketDayLow")
            prev = _raw("previousClose", "prevClose")
            volume = _raw("volume", "regularMarketVolume")
            mktcap = _raw("marketCap", "market_cap")
            wk52_hi = _raw("52weekHigh", "week52High", "fiftyTwoWeekHigh")
            wk52_lo = _raw("52weekLow", "week52Low", "fiftyTwoWeekLow")

            sign = "+" if change >= 0 else ""
            lines = [
                f"Stock: {name} ({symbol})  |  CNN Markets",
                f"Price:       {f(price)}  ({sign}{change:.2f} / {sign}{pct:.2f}% today)",
                f"Open:        {f(open_p)}   Prev Close: {f(prev)}",
                f"Day Range:   {f(low)} – {f(high)}",
            ]
            if wk52_hi is not None and wk52_lo is not None:
                lines.append(f"52-Wk Range: {f(wk52_lo)} – {f(wk52_hi)}")
            lines.append(f"Volume:      {fmt_vol(volume)}")
            if mktcap is not None:
                lines.append(f"Market Cap:  {fmt_cap(mktcap)}")
            lines.append(f"Source:      https://www.cnn.com/markets/stocks/{symbol}")
            return "\n".join(lines)

        except Exception as e:
            return f"Error: CNN: {type(e).__name__}: {e}"

    def _stooq_quote(self, symbol: str) -> str:
        stooq_sym = symbol.lower() if "." in symbol else f"{symbol.lower()}.us"
        try:
            resp = requests.get(
                "https://stooq.com/q/l/",
                params={"s": stooq_sym, "f": "sd2t2ohlcv", "h": "", "e": "csv"},
                timeout=10,
            )
            resp.raise_for_status()
            rows = [l for l in resp.text.strip().splitlines() if l]
            if len(rows) < 2:
                return f"No data found for '{symbol}' on Stooq."

            d = dict(zip(rows[0].split(","), rows[1].split(",")))
            close = float(d.get("Close", 0))
            open_p = float(d.get("Open", 0))
            high = float(d.get("High", 0))
            low = float(d.get("Low", 0))
            volume = d.get("Volume", "N/A")
            date = d.get("Date", "")
            time_str = d.get("Time", "")

            change = close - open_p
            pct = (change / open_p * 100) if open_p else 0
            sign = "+" if change >= 0 else ""

            return "\n".join([
                f"Stock: {symbol} (Stooq fallback) — {date} {time_str}",
                f"Price:     ${close:,.2f}  ({sign}{change:.2f} / {sign}{pct:.2f}% vs open)",
                f"Open:      ${open_p:,.2f}",
                f"Day Range: ${low:,.2f} – ${high:,.2f}",
                f"Volume:    {volume}",
            ])
        except Exception as e:
            return f"Error: Stooq: {type(e).__name__}: {e}"
