#!/usr/bin/env python3
"""
Auther: Christopher Gray
Version: 0.1.4
Updated: 5/16/2026

Updated from: https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/openweb_ui_push_tool.py

Push openwebui_tool.py to a running Open WebUI instance via its REST API.
No browser required — run this after every edit.

Auth (use one — email/password is easiest if you can't find your API key):
    export OWUI_URL="http://your-server:3000"

    Option A — email + password (works with any Open WebUI version):
        export OWUI_EMAIL="you@example.com"
        export OWUI_PASSWORD="yourpassword"

    Option B — API key (Admin Panel > Settings > Account > API Keys):
        export OWUI_API_KEY="sk-..."

Usage:
    python3 openweb_ui_push_tool.py              # push once
    python3 openweb_ui_push_tool.py --watch      # push automatically on every file save
    python3 openweb_ui_push_tool.py --probe      # discover working API endpoints (run this if push fails)
"""

import os
import re
import sys
import time
import argparse
import requests

TOOL_FILE    = os.path.join(os.path.dirname(__file__), "openwebui_tool.py")
OWUI_URL     = os.getenv("OWUI_URL",      "http://localhost:3000")
OWUI_KEY     = os.getenv("OWUI_API_KEY",  "")
OWUI_EMAIL   = os.getenv("OWUI_EMAIL",    "")
OWUI_PASSWORD= os.getenv("OWUI_PASSWORD", "")


def get_token() -> str:
    """Return a Bearer token — from env API key, or by signing in with email/password."""
    if OWUI_KEY:
        return OWUI_KEY

    if OWUI_EMAIL and OWUI_PASSWORD:
        base = OWUI_URL.rstrip("/")
        try:
            resp = requests.post(
                f"{base}/api/v1/auths/signin",
                json={"email": OWUI_EMAIL, "password": OWUI_PASSWORD},
                timeout=10,
            )
            resp.raise_for_status()
            token = resp.json().get("token")
            if not token:
                print(f"Error: Sign-in succeeded but no token returned: {resp.text[:200]}")
                sys.exit(1)
            return token
        except requests.exceptions.HTTPError as e:
            print(f"Error: Login failed (HTTP {e.response.status_code}) — check OWUI_EMAIL and OWUI_PASSWORD")
            sys.exit(1)
        except requests.exceptions.ConnectionError:
            print(f"Error: Cannot connect to Open WebUI at {OWUI_URL}")
            sys.exit(1)

    print(
        "Error: No credentials set. Use one of:\n"
        "\n"
        "  Option A — email + password:\n"
        "    export OWUI_EMAIL='you@example.com'\n"
        "    export OWUI_PASSWORD='yourpassword'\n"
        "\n"
        "  Option B — API key:\n"
        "    export OWUI_API_KEY='sk-...'\n"
        "\n"
        "  Also set:  export OWUI_URL='http://your-server:3000'"
    )
    sys.exit(1)


def extract_meta(content: str) -> dict:
    def _get(field):
        m = re.search(rf"^{field}:\s*(.+)", content, re.MULTILINE)
        return m.group(1).strip() if m else ""

    title   = _get("title")       or "Web Search & URL Fetch"
    version = _get("version")     or "1.0.0"
    desc    = _get("description") or ""
    tool_id = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return {"id": tool_id, "name": title, "version": version, "description": desc}


def _call(method: str, url: str, headers: dict, payload: dict) -> requests.Response:
    resp = requests.request(method, url, headers=headers, json=payload, timeout=10)
    return resp


def push(content: str, meta: dict) -> None:
    token   = get_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    base    = OWUI_URL.rstrip("/")

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
        # No working PUT/PATCH endpoint found in this OWUI version — delete then recreate.
        eid = existing["id"]
        deleted = False
        for method, url in [
            ("DELETE", f"{base}/api/v1/tools/{eid}"),
            ("POST",   f"{base}/api/v1/tools/{eid}/delete"),
            ("DELETE", f"{base}/api/v1/tools/{eid}/delete"),
            ("DELETE", f"{base}/api/tools/{eid}"),
        ]:
            resp = _call(method, url, headers, {})
            if resp.status_code not in (404, 405):
                resp.raise_for_status()
                deleted = True
                break

        if not deleted:
            # Last resort: try standard update verbs before giving up
            for method, url in [
                ("PUT",   f"{base}/api/v1/tools/{eid}"),
                ("PATCH", f"{base}/api/v1/tools/{eid}"),
                ("POST",  f"{base}/api/v1/tools/{eid}/update"),
            ]:
                resp = _call(method, url, headers, payload)
                if resp.status_code not in (404, 405):
                    resp.raise_for_status()
                    print(f"[{time.strftime('%H:%M:%S')}] Updated  '{meta['name']}' (id={eid})  v{meta['version']}  [{method} {url.split(base)[1]}]")
                    return
            print("All update/delete endpoints failed. Run with --probe to diagnose.")
            raise RuntimeError("Could not update tool — no working endpoint found")

    # Create (or recreate after delete)
    resp = _call("POST", f"{base}/api/v1/tools/create", headers, payload)
    if resp.ok:
        action = "Updated" if existing else "Created"
        print(f"[{time.strftime('%H:%M:%S')}] {action}  '{meta['name']}' (id={meta['id']})  v{meta['version']}  [DELETE+POST /api/v1/tools/create]")
        return
    print(f"POST /api/v1/tools/create failed ({resp.status_code}): {resp.text[:300]}")
    raise RuntimeError("Could not create tool — see error above")


def probe() -> None:
    """Try every known endpoint pattern and print the HTTP status for each."""
    token   = get_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    base    = OWUI_URL.rstrip("/")

    # Fetch real tool id to use in probes
    resp = requests.get(f"{base}/api/v1/tools/", headers=headers, timeout=10)
    resp.raise_for_status()
    tools = resp.json()
    eid   = tools[0]["id"] if tools else "test_id"
    dummy = {"id": eid, "name": "probe", "content": "", "meta": {}}

    print(f"Open WebUI: {base}")
    print(f"Using tool id for probe: {eid}")
    print(f"{'METHOD':<8} {'ENDPOINT':<55} STATUS")
    print("-" * 75)

    # Try to fetch the OpenAPI spec so we can see all real endpoints
    for spec_path in ("/openapi.json", "/docs/openapi.json", "/api/openapi.json"):
        try:
            r = requests.get(base + spec_path, headers=headers, timeout=6)
            if r.ok:
                paths = list(r.json().get("paths", {}).keys())
                tool_paths = [p for p in paths if "tool" in p.lower()]
                print(f"OpenAPI spec found at {spec_path}")
                print(f"  Tool-related paths: {tool_paths}")
                break
        except Exception:
            pass

    probe_urls = [
        # list
        ("GET",    f"/api/v1/tools/"),
        ("GET",    f"/api/tools/"),
        # read single
        ("GET",    f"/api/v1/tools/{eid}"),
        # update
        ("PUT",    f"/api/v1/tools/{eid}"),
        ("PATCH",  f"/api/v1/tools/{eid}"),
        ("POST",   f"/api/v1/tools/{eid}"),
        ("POST",   f"/api/v1/tools/{eid}/update"),
        ("PUT",    f"/api/v1/tools/id/{eid}"),
        ("POST",   f"/api/v1/tools/update"),
        # delete
        ("DELETE", f"/api/v1/tools/{eid}"),
        ("DELETE", f"/api/v1/tools/{eid}/delete"),
        ("POST",   f"/api/v1/tools/{eid}/delete"),
        ("DELETE", f"/api/tools/{eid}"),
        # create
        ("POST",   f"/api/v1/tools/"),
        ("POST",   f"/api/v1/tools/add"),
        ("POST",   f"/api/v1/tools/create"),
        ("PUT",    f"/api/v1/tools/"),
        ("POST",   f"/api/tools/"),
        ("POST",   f"/api/tools/add"),
    ]

    for method, path in probe_urls:
        url  = base + path
        try:
            r = requests.request(method, url, headers=headers, json=dummy, timeout=8)
            status = r.status_code
            allow  = r.headers.get("Allow", "")
            note   = ""
            if status < 300:
                note = " <-- WORKS"
            elif status == 401:
                note = " (auth error)"
            elif status == 404:
                note = " (not found)"
            elif status == 405:
                note = f" (wrong method — allowed: {allow})" if allow else " (wrong method)"
            elif status == 422:
                note = " (endpoint exists, payload rejected)"
            print(f"{method:<8} {path:<55} {status}{note}")
        except Exception as e:
            print(f"{method:<8} {path:<55} ERROR: {e}")

    print("\nShare the output above to identify which endpoint to use.")


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
                if last_mtime is not None:
                    try:
                        push_file()
                    except Exception as e:
                        print(f"[{time.strftime('%H:%M:%S')}] Error: {e}")
            time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push openwebui_tool.py to Open WebUI")
    parser.add_argument("--watch", action="store_true", help="Re-push on every file save")
    parser.add_argument("--probe", action="store_true", help="Discover working API endpoints")
    args = parser.parse_args()

    if args.probe:
        probe()
    elif args.watch:
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
