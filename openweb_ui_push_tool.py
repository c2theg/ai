#!/usr/bin/env python3
"""
Auther: Christopher Gray
Version: 0.0.2
Updated: 5/16/2026

Updated from:  https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/openweb_ui_push_tool.py

Push openwebui_tool.py to a running Open WebUI instance via its REST API.
No browser required — run this after every edit.

Setup (one time):
    export OWUI_URL="http://your-server:3000"
    export OWUI_API_KEY="sk-..."      # Admin Panel → Settings → Account → API Keys

Usage:
    python3 push_tool.py              # push once
    python3 push_tool.py --watch      # push automatically on every file save
"""

import os
import re
import sys
import time
import argparse
import requests

TOOL_FILE = os.path.join(os.path.dirname(__file__), "openwebui_tool.py")
OWUI_URL  = os.getenv("OWUI_URL",    "http://localhost:3000")
OWUI_KEY  = os.getenv("OWUI_API_KEY", "")


def extract_meta(content: str) -> dict:
    def _get(field):
        m = re.search(rf"^{field}:\s*(.+)", content, re.MULTILINE)
        return m.group(1).strip() if m else ""

    title   = _get("title")   or "Web Search & URL Fetch"
    version = _get("version") or "1.0.0"
    desc    = _get("description") or ""
    tool_id = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return {"id": tool_id, "name": title, "version": version, "description": desc}


def push(content: str, meta: dict) -> None:
    if not OWUI_KEY:
        print(
            "Error: OWUI_API_KEY is not set.\n"
            "  Get one from: Admin Panel → Settings → Account → API Keys\n"
            "  Then run:  export OWUI_API_KEY='sk-...'"
        )
        sys.exit(1)

    headers = {"Authorization": f"Bearer {OWUI_KEY}", "Content-Type": "application/json"}
    base    = OWUI_URL.rstrip("/")

    # Find an existing tool with the same name or id
    resp = requests.get(f"{base}/api/v1/tools/", headers=headers, timeout=10)
    resp.raise_for_status()
    tools    = resp.json()
    existing = next(
        (t for t in tools if t.get("id") == meta["id"] or t.get("name") == meta["name"]),
        None,
    )

    payload = {
        "id":      meta["id"],
        "name":    meta["name"],
        "content": content,
        "meta": {
            "description": meta["description"],
            "manifest":    {"version": meta["version"]},
        },
    }

    if existing:
        eid  = existing["id"]
        resp = requests.post(f"{base}/api/v1/tools/{eid}/update", headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        print(f"[{time.strftime('%H:%M:%S')}] Updated  '{meta['name']}' (id={eid})  v{meta['version']}")
    else:
        resp = requests.post(f"{base}/api/v1/tools/add", headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        print(f"[{time.strftime('%H:%M:%S')}] Created  '{meta['name']}' (id={meta['id']})  v{meta['version']}")


def push_file() -> None:
    with open(TOOL_FILE, encoding="utf-8") as f:
        content = f.read()
    meta = extract_meta(content)
    push(content, meta)


def watch() -> None:
    print(f"Watching {TOOL_FILE} — press Ctrl+C to stop")
    last_mtime = None
    while True:
        try:
            mtime = os.path.getmtime(TOOL_FILE)
            if mtime != last_mtime:
                last_mtime = mtime
                if last_mtime is not None:  # skip the very first read
                    try:
                        push_file()
                    except Exception as e:
                        print(f"[{time.strftime('%H:%M:%S')}] Error: {e}")
                else:
                    # prime the mtime on first loop without pushing
                    pass
            time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push openwebui_tool.py to Open WebUI")
    parser.add_argument("--watch", action="store_true", help="Re-push on every file save")
    args = parser.parse_args()

    if args.watch:
        watch()
    else:
        try:
            push_file()
        except requests.exceptions.ConnectionError:
            print(f"Error: Cannot connect to Open WebUI at {OWUI_URL}")
            sys.exit(1)
        except requests.exceptions.HTTPError as e:
            print(f"Error: HTTP {e.response.status_code} — {e.response.text[:200]}")
            sys.exit(1)
