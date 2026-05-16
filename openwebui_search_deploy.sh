#!/bin/bash
#
#  Christopher Gray
#    Version 0.0.2
#    Updated: 5/16/2026
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

# ---------------------------------------------------------------------------
# 1. SearXNG settings.yml
# ---------------------------------------------------------------------------
echo "==> Generating SearXNG secret key..."
SECRET=$(openssl rand -hex 32)

echo "==> Writing ${SEARXNG_CONFIG_DIR}/settings.yml..."
mkdir -p "$SEARXNG_CONFIG_DIR"

cat > "$SEARXNG_CONFIG_DIR/settings.yml" <<EOF
use_default_settings: true

server:
  secret_key: "${SECRET}"
  limiter: false
  image_proxy: true
  port: 8080
  bind_address: "0.0.0.0"

ui:
  static_use_hash: true
  default_locale: ""
  query_in_title: false
  infinite_scroll: false
  center_alignment: false

search:
  safe_search: 0
  autocomplete: ""
  default_lang: "en"
  formats:
    - html
    - json

engines:
  - name: google
    engine: google
    shortcut: g
    disabled: false

  - name: bing
    engine: bing
    shortcut: b
    disabled: false

  - name: duckduckgo
    engine: duckduckgo
    shortcut: ddg
    disabled: false

  - name: wikipedia
    engine: wikipedia
    shortcut: wp
    disabled: false

  - name: brave
    engine: brave
    shortcut: br
    disabled: false

outgoing:
  request_timeout: 15.0
  max_request_timeout: 30.0
  pool_connections: 100
  pool_maxsize: 20
  enable_http2: true
EOF

echo "==> settings.yml written."

# ---------------------------------------------------------------------------
# 2. Restart SearXNG
# ---------------------------------------------------------------------------
echo "==> Restarting SearXNG container..."
docker restart searxng

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
# 5. Instructions
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo " Setup complete. Follow these steps in Open WebUI:"
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
