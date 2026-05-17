"""
title: Web Search & URL Fetch
author: Christopher Gray
description: Search the web via SearXNG and fetch full URL content. Works with any model (Gemma, Qwen, Ollama, vLLM).
version: 1.3.0
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
        Get current weather conditions and a 7-day forecast using the National Weather Service API (weather.gov). US locations only.
        :param location: Zip code, city name, or "city, state" (e.g. "80203", "Denver", "Chicago, IL")
        :return: Current conditions and 7-day forecast from NWS
        """
        NWS_HEADERS = {
            "User-Agent": "OpenWebUI-WeatherTool/1.0 (openwebui-tool)",
            "Accept": "application/geo+json",
        }

        # Step 1: Geocode location to lat/lon
        lat, lon, place_name = self._geocode_location(location)
        if lat is None:
            return f"Error: Could not find '{location}'. Try a zip code, city name, or 'City, ST'."

        # Step 2: NWS grid lookup
        try:
            resp = requests.get(
                f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}",
                headers=NWS_HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
            grid = resp.json()["properties"]
            forecast_url  = grid["forecast"]
            stations_url  = grid["observationStations"]
            rel           = grid.get("relativeLocation", {}).get("properties", {})
            city          = rel.get("city", "")
            state         = rel.get("state", "")
            if city and state:
                place_name = f"{city}, {state}"
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return f"Error: '{location}' is outside NWS coverage. The NWS API only covers US locations."
            return f"Error: NWS grid lookup failed ({e.response.status_code})"
        except Exception as e:
            return f"Error: NWS grid lookup failed: {e}"

        # Step 3: Current observations from nearest station
        current_lines = [f"Weather for: {place_name}  (source: weather.gov)"]
        try:
            resp = requests.get(stations_url, headers=NWS_HEADERS, timeout=10)
            resp.raise_for_status()
            features = resp.json().get("features", [])
            if features:
                station_id = features[0]["properties"]["stationIdentifier"]
                obs_resp = requests.get(
                    f"https://api.weather.gov/stations/{station_id}/observations/latest",
                    headers=NWS_HEADERS,
                    timeout=10,
                )
                obs_resp.raise_for_status()
                obs = obs_resp.json()["properties"]

                def c_to_f(c):
                    return round(c * 9 / 5 + 32) if c is not None else None

                def deg_to_dir(d):
                    if d is None:
                        return ""
                    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
                    return dirs[round(d / 22.5) % 16]

                temp_c     = obs.get("temperature", {}).get("value")
                feels_c    = (obs.get("windChill", {}) or obs.get("heatIndex", {})).get("value")
                humidity   = obs.get("relativeHumidity", {}).get("value")
                wind_ms    = obs.get("windSpeed", {}).get("value")
                wind_deg   = obs.get("windDirection", {}).get("value")
                vis_m      = obs.get("visibility", {}).get("value")
                desc       = obs.get("textDescription", "")

                temp_f   = c_to_f(temp_c)
                feels_f  = c_to_f(feels_c)
                wind_mph = round(wind_ms * 2.237) if wind_ms is not None else None
                vis_mi   = round(vis_m / 1609.34, 1) if vis_m is not None else None

                if desc:
                    current_lines.append(f"Conditions:  {desc}")
                if temp_f is not None:
                    feels_str = f"  (feels like {feels_f}°F)" if feels_f is not None else ""
                    current_lines.append(f"Temperature: {temp_f}°F / {temp_c:.0f}°C{feels_str}")
                if humidity is not None:
                    current_lines.append(f"Humidity:    {humidity:.0f}%")
                if wind_mph is not None:
                    current_lines.append(f"Wind:        {wind_mph} mph {deg_to_dir(wind_deg)}")
                if vis_mi is not None:
                    current_lines.append(f"Visibility:  {vis_mi} miles")
        except Exception:
            current_lines.append("(Current observations unavailable)")

        # Step 4: 7-day forecast
        forecast_lines = ["", "7-Day Forecast:"]
        try:
            resp = requests.get(forecast_url, headers=NWS_HEADERS, timeout=10)
            resp.raise_for_status()
            periods = resp.json()["properties"]["periods"]
            for p in periods:
                name   = p.get("name", "")
                temp   = p.get("temperature", "?")
                unit   = p.get("temperatureUnit", "F")
                wind   = p.get("windSpeed", "")
                wdir   = p.get("windDirection", "")
                short  = p.get("shortForecast", "")
                precip = (p.get("probabilityOfPrecipitation") or {}).get("value")
                rain   = f"  Rain {precip}%" if precip is not None else ""
                forecast_lines.append(f"  {name:<22} {temp}°{unit}  {wind} {wdir:<3}  {short}{rain}")
        except Exception as e:
            forecast_lines.append(f"  (Forecast unavailable: {e})")

        return "\n".join(current_lines + forecast_lines)

    def _geocode_location(self, location: str):
        """Convert a location string to (lat, lon, display_name) via Nominatim. Returns (None, None, None) on failure."""
        try:
            resp = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": location, "format": "json", "limit": 1, "countrycodes": "us"},
                headers={"User-Agent": "OpenWebUI-WeatherTool/1.0"},
                timeout=8,
            )
            resp.raise_for_status()
            results = resp.json()
            if results:
                r = results[0]
                return float(r["lat"]), float(r["lon"]), r.get("display_name", location)
        except Exception:
            pass
        return None, None, None

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
        # Yahoo Finance now requires a cookie + crumb for API access.
        # We establish a session, grab the crumb, then call v7/finance/quote.
        try:
            ua = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            )
            session = requests.Session()

            # Step 1: seed the session cookie
            session.get("https://fc.yahoo.com", headers={"User-Agent": ua}, timeout=8)

            # Step 2: fetch crumb
            crumb_resp = session.get(
                "https://query2.finance.yahoo.com/v1/test/getcrumb",
                headers={"User-Agent": ua, "Accept": "*/*"},
                timeout=8,
            )
            crumb = crumb_resp.text.strip()
            if not crumb or "<" in crumb:
                return "Error: Yahoo Finance: could not obtain auth crumb"

            # Step 3: fetch quote
            resp = session.get(
                "https://query1.finance.yahoo.com/v7/finance/quote",
                params={"symbols": symbol, "crumb": crumb},
                headers={"User-Agent": ua, "Accept": "application/json"},
                timeout=10,
            )
            resp.raise_for_status()

            results = resp.json().get("quoteResponse", {}).get("result", [])
            if not results:
                return f"Error: Yahoo Finance returned no data for {symbol}"

            q = results[0]

            name         = q.get("longName") or q.get("shortName", symbol)
            currency     = q.get("currency", "USD")
            exchange     = q.get("fullExchangeName", "")
            market_state = q.get("marketState", "REGULAR")
            current      = q.get("regularMarketPrice")
            if current is None:
                return f"Error: Yahoo Finance returned no price for {symbol}"

            change = q.get("regularMarketChange", 0)
            pct    = q.get("regularMarketChangePercent", 0)
            sign   = "+" if change >= 0 else ""

            def f(v):
                return f"${v:,.2f}" if v is not None else "N/A"

            def fmt_vol(v):
                if v is None:
                    return "N/A"
                for thresh, sfx in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
                    if v >= thresh:
                        return f"{v / thresh:.2f}{sfx}"
                return str(int(v))

            def fmt_cap(v):
                if v is None:
                    return "N/A"
                for thresh, sfx in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
                    if v >= thresh:
                        return f"${v / thresh:.2f}{sfx}"
                return f"${v:,.0f}"

            lines = [
                f"Stock: {name} ({symbol})  |  {exchange}  |  {market_state}  |  {currency}",
                f"Price:       {f(current)}  ({sign}{change:.2f} / {sign}{pct:.2f}% today)",
            ]

            if market_state == "PRE":
                pre     = q.get("preMarketPrice")
                pre_chg = q.get("preMarketChange", 0)
                pre_pct = q.get("preMarketChangePercent", 0)
                if pre:
                    s = "+" if pre_chg >= 0 else ""
                    lines.append(f"Pre-Market:  {f(pre)}  ({s}{pre_chg:.2f} / {s}{pre_pct:.2f}%)")
            elif market_state in ("POST", "POSTPOST"):
                post     = q.get("postMarketPrice")
                post_chg = q.get("postMarketChange", 0)
                post_pct = q.get("postMarketChangePercent", 0)
                if post:
                    s = "+" if post_chg >= 0 else ""
                    lines.append(f"After-Hours: {f(post)}  ({s}{post_chg:.2f} / {s}{post_pct:.2f}%)")

            lines += [
                f"Open:        {f(q.get('regularMarketOpen'))}   Prev Close: {f(q.get('regularMarketPreviousClose'))}",
                f"Day Range:   {f(q.get('regularMarketDayLow'))} – {f(q.get('regularMarketDayHigh'))}",
            ]
            wk52_hi = q.get("fiftyTwoWeekHigh")
            wk52_lo = q.get("fiftyTwoWeekLow")
            if wk52_hi and wk52_lo:
                lines.append(f"52-Wk Range: {f(wk52_lo)} – {f(wk52_hi)}")
            lines.append(f"Volume:      {fmt_vol(q.get('regularMarketVolume'))}   Avg Volume: {fmt_vol(q.get('averageDailyVolume3Month'))}")
            if q.get("marketCap"):
                lines.append(f"Market Cap:  {fmt_cap(q.get('marketCap'))}")

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
