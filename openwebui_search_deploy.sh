#!/bin/bash
#  Christopher Gray
#    Version 0.0.20
#    Updated: 5/24/2026
#
#  *** ENTRY POINT ***
#
# Self-contained deploy script for SearXNG + Open WebUI tool.
# Run directly on the server as root — no other files needed.
#
# Download and run in one command:
#   curl -fsSL https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/openwebui_search_deploy.sh -o openwebui_search_deploy.sh && bash openwebui_search_deploy.sh

set -e

SEARXNG_CONFIG_DIR="/opt/models/searxng"
TOOL_URL="https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/openwebui_tool.py"
TOOL_DEST="/opt/models/openwebui_tool.py"
PUSH_URL="https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/openweb_ui_push_tool.py"
PUSH_DEST="/opt/models/openweb_ui_push_tool.py"

# ---------------------------------------------------------------------------
# 1. SearXNG settings.yml
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETTINGS_LOCAL="${SCRIPT_DIR}/searxng_settings.yml"
SETTINGS_URL="https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/searxng_settings.yml"

echo "==> Generating SearXNG secret key..."
SECRET=$(openssl rand -hex 32)

mkdir -p "$SEARXNG_CONFIG_DIR"

if [ -f "$SETTINGS_LOCAL" ]; then
    echo "==> Copying local searxng_settings.yml..."
    cp "$SETTINGS_LOCAL" "$SEARXNG_CONFIG_DIR/settings.yml"
else
    echo "==> Downloading searxng_settings.yml..."
    curl -fsSL "$SETTINGS_URL" -o "$SEARXNG_CONFIG_DIR/settings.yml"
fi

if [ -n "$SECRET" ]; then
    echo "==> Injecting generated secret key..."
    sed -i \
        -e '/^  secret_key: /d' \
        -e "s|  #secret_key: \"\${SECRET}\"|  secret_key: \"${SECRET}\"|" \
        "$SEARXNG_CONFIG_DIR/settings.yml"
else
    echo "==> No secret generated — using default key from settings file."
fi

echo "==> settings.yml written."

# ---------------------------------------------------------------------------
# 2. Start SearXNG container
# ---------------------------------------------------------------------------

# Remove existing container if present so we can re-apply the new config
if docker ps -a --format '{{.Names}}' | grep -q '^searxng$'; then
    echo "==> Removing existing searxng container..."
    docker rm -f searxng
fi

echo "==> Starting SearXNG container..."
docker run -d \
  --name searxng \
  --restart always \
  -p 8080:8080 \
  -v "$SEARXNG_CONFIG_DIR/settings.yml:/etc/searxng/settings.yml:ro" \
  searxng/searxng

echo "==> Waiting for SearXNG to be ready..."
for i in $(seq 1 15); do
    if curl -sf "http://localhost:8080/search?q=test&format=json" > /dev/null 2>&1; then
        echo "==> SearXNG is up."
        break
    fi
    echo "    Attempt $i/15 — waiting..."
    sleep 2
done

# ---------------------------------------------------------------------------
# 3. Test JSON API
# ---------------------------------------------------------------------------
echo ""
echo "==> Testing SearXNG JSON API..."
curl -s "http://localhost:8080/search?q=hello+world&format=json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
r = d.get('results', [])
print(f'    OK — got {len(r)} results')
if r:
    print(f'    First result: {r[0].get(\"title\", \"\")}')
" || echo "    WARNING: JSON API test failed — check logs: docker logs searxng"

# ---------------------------------------------------------------------------
# 4. Download Open WebUI tool
# ---------------------------------------------------------------------------
echo ""
echo "==> Downloading Open WebUI tool to ${TOOL_DEST}..."
curl -fsSL "$TOOL_URL" -o "$TOOL_DEST"
echo "==> Tool saved."

# ---------------------------------------------------------------------------
# 5. Download push_tool.py (auto-deploy without the browser)
# ---------------------------------------------------------------------------
echo ""
echo "==> Downloading push_tool.py to ${PUSH_DEST}..."
curl -fsSL "$PUSH_URL" -o "$PUSH_DEST"
chmod +x "$PUSH_DEST"
echo "==> push_tool.py saved and marked executable."

# Check python3 + requests are available for push_tool.py
if command -v python3 &>/dev/null; then
    if ! python3 -c "import requests" 2>/dev/null; then
        echo "==> Installing requests for push_tool.py..."
        python3 -m pip install --quiet requests
    else
        echo "==> python3 + requests OK."
    fi
else
    echo "    WARNING: python3 not found — install it to use push_tool.py"
fi

# ---------------------------------------------------------------------------
# 6. Instructions
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo " Setup complete. Follow these steps to finish:"
echo "========================================================"
echo ""
echo " [1] Configure Web Search (Admin Panel > Settings > Web Search):"
echo "     - Enable Web Search:       ON"
echo "     - Web Search Engine:       searxng"
echo "     - SearXNG Query URL:       http://localhost:8080/search?q=<query>"
echo "     - Save"
echo ""
echo " [2] Install the Tool (Workspace > Tools > '+'):"
echo "     - Paste the contents of:  ${TOOL_DEST}"
echo "     - Or copy from:           ${TOOL_URL}"
echo "     - Save"
echo ""
echo " [3] Use in any chat:"
echo "     - Click the tools '+' icon in the chat bar"
echo "     - Toggle on 'Web Search & URL Fetch'"
echo "     - Works with Gemma4, Qwen, or any Ollama / vLLM model"
echo ""
echo " [4] Set up push_tool.py so you never paste via the browser again:"
echo "     Add to your shell profile (~/.bashrc or ~/.zshrc):"
echo ""
echo "     Option A — email + password (easiest, works with any version):"
echo "         export OWUI_URL=\"http://$(hostname -I | awk '{print $1}'):3000\""
echo "         export OWUI_EMAIL=\"you@example.com\""
echo "         export OWUI_PASSWORD=\"yourpassword\""
echo ""
echo "     Option B — API key (Admin Panel > Settings > Account > API Keys):"
echo "         export OWUI_URL=\"http://$(hostname -I | awk '{print $1}'):3000\""
echo "         export OWUI_API_KEY=\"sk-...\""
echo ""
echo "     Push once:         python3 ${PUSH_DEST}"
echo "     Auto-push on save: python3 ${PUSH_DEST} --watch"
echo ""
echo "========================================================"
echo " Done!"
echo "========================================================"
echo "Examples: "
echo "


🌤 Weather
curl -s "http://localhost:8080/search?q=weather+new+york&format=json" | python3 -m json.tool | head -30


📰 World News
curl -s "http://localhost:8080/search?q=world+news+today&format=json" | python3 -m json.tool | head -30


💰 Stock Market
# General market
curl -s "http://localhost:8080/search?q=stock+market+today&format=json" | python3 -m json.tool | head -30

# Specific stock
curl -s "http://localhost:8080/search?q=NVIDIA+stock+price&format=json" | python3 -m json.tool | head -30


🏈 Sports
# General scores
curl -s "http://localhost:8080/search?q=sports+scores+today&format=json" | python3 -m json.tool | head -30

# Specific sport
curl -s "http://localhost:8080/search?q=NBA+scores+today&format=json" | python3 -m json.tool | head -30


📖 Wikipedia
curl -s "http://localhost:8080/search?q=!wp+artificial+intelligence&format=json" | python3 -m json.tool | head -30


💻 Tech News
curl -s "http://localhost:8080/search?q=AI+news+today&format=json" | python3 -m json.tool | head -30


🤖 AI News
curl -s "http://localhost:8080/search?q=large+language+models+2026&format=json" | python3 -m json.tool | head -30


📡 RSS Feed Test (BBC specifically)
curl -s "http://localhost:8080/search?q=breaking+news&engines=bbc+world+news&format=json" | python3 -m json.tool | head -30


"



echo " Preforming a series of tests... (this will take a minute.). 

"
for query in "weather New York, New York" "world news today" "NVIDIA stock" "NFL scores" "artificial intelligence wikipedia" "tech news today"; do
  echo "==============================="
  echo "Testing: $query"
  echo "==============================="
  result=$(curl -s "http://localhost:8080/search?q=$(echo $query | sed 's/ /+/g')&format=json")
  echo $result | python3 -c "
import json,sys
data=json.load(sys.stdin)
results=data.get('results',[])
print(f'Found {len(results)} results')
for r in results[:2]:
    print(f'  - {r.get(\"title\",\"no title\")}')
    print(f'    {r.get(\"url\",\"no url\")}')
"
  echo ""
done
