#!/bin/bash
#  Christopher Gray
#    Version 0.0.8
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
# download it:
#   wget https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/searxng_settings.yml && cp searxng_settings.yml /opt/models/searxng/settings.yml

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

echo "==> Starting SearXNG container..."
docker run -d \
  --name searxng \
  --restart always \
  -p 8080:8080 \
  searxng/searxng


#echo "==> Restarting SearXNG container..."
#docker restart searxng

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
