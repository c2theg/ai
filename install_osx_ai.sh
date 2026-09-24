#!/usr/bin/env bash

echo "
 _____             _         _    _          _                                   
|     |___ ___ ___| |_ ___ _| |  | |_ _ _   |_|                                  
|   --|  _| -_| .'|  _| -_| . |  | . | | |   _                                   
|_____|_| |___|__,|_| |___|___|  |___|_  |  |_|                                  
                                     |___|                                       
                                                                                 
 _____ _       _     _           _              _____    __    _____             
|     | |_ ___|_|___| |_ ___ ___| |_ ___ ___   |     |__|  |  |   __|___ ___ _ _ 
|   --|   |  _| |_ -|  _| . | . |   | -_|  _|  | | | |  |  |  |  |  |  _| .'| | |
|_____|_|_|_| |_|___|_| |___|  _|_|_|___|_|    |_|_|_|_____|  |_____|_| |__,|_  |
                            |_|                                             |___|


Version:  0.0.5
Last Updated:  9/24/2026

"
# install_osx_ai.sh - local AI development environment for Apple Silicon Macs (M1+)
#
# Installs: Homebrew, dev basics, Python (via uv) with an ML venv, Node, local
# inference runtimes (Ollama, llama.cpp, LM Studio, MLX), AI coding agents
# (Claude Code, Codex, OpenCode, Gemini CLI), and editors (VS Code, Windsurf, Cursor).
#
# Idempotent: safe to re-run; it installs what is missing and upgrades nothing
# unless you pass --upgrade. A failed package is reported in the summary instead
# of aborting the whole run.
#
# Usage: ./install_osx_ai.sh [--upgrade] [--skip-editors] [--skip-python] [--pull-models]
#
set -uo pipefail

UPGRADE=0; SKIP_EDITORS=0; SKIP_PYTHON=0; PULL_MODELS=0
for arg in "$@"; do
  case "$arg" in
    --upgrade)      UPGRADE=1 ;;
    --skip-editors) SKIP_EDITORS=1 ;;
    --skip-python)  SKIP_PYTHON=1 ;;
    --pull-models)  PULL_MODELS=1 ;;
    -h|--help)      sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

AI_VENV="${AI_VENV:-$HOME/.venvs/ai}"
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"   # TF/torch wheels lag the newest CPython
FAILED=()

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }
try()  { "$@" || { FAILED+=("$*"); warn "failed: $*"; }; }

# ---------------------------------------------------------------- preflight
[[ "$(uname -s)" == "Darwin" ]] || { echo "This script is for macOS only." >&2; exit 1; }
[[ "$(uname -m)" == "arm64"  ]] || { echo "Apple Silicon (arm64) required; got $(uname -m)." >&2; exit 1; }
[[ $EUID -ne 0 ]] || { echo "Do not run as root/sudo; Homebrew refuses to." >&2; exit 1; }

echo "macOS $(sw_vers -productVersion), $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB RAM"

if ! xcode-select -p >/dev/null 2>&1; then
  log "Installing Xcode Command Line Tools (a GUI prompt will appear)"
  xcode-select --install
  until xcode-select -p >/dev/null 2>&1; do sleep 5; done
fi

# Ask for the admin password once up front and keep it alive, so later steps
# (Homebrew, casks) never stop to prompt.
sudo -v
while true; do sudo -n true; sleep 50; kill -0 "$$" 2>/dev/null || exit; done 2>/dev/null &
SUDO_KEEPALIVE_PID=$!
trap 'kill "$SUDO_KEEPALIVE_PID" 2>/dev/null' EXIT

# ---------------------------------------------------------------- Homebrew
if ! have brew && [[ ! -x /opt/homebrew/bin/brew ]]; then
  log "Installing Homebrew"
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" \
    || { echo "Homebrew install failed" >&2; exit 1; }
fi
eval "$(/opt/homebrew/bin/brew shellenv)"

# Persist brew on PATH once (the original appended a duplicate line on every run)
if ! grep -qs 'brew shellenv' "$HOME/.zprofile"; then
  echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
fi

export HOMEBREW_NO_ANALYTICS=1 HOMEBREW_NO_ENV_HINTS=1 NONINTERACTIVE=1
log "Updating Homebrew"
try brew update
brew --version | head -1

brew_formula() {
  local f="$1"
  if brew list --formula "$f" >/dev/null 2>&1; then
    [[ $UPGRADE -eq 1 ]] && try brew upgrade "$f"
  else
    echo "  + $f"; try brew install "$f"
  fi
}
brew_cask() {
  local c="$1"
  if brew list --cask "$c" >/dev/null 2>&1; then
    [[ $UPGRADE -eq 1 ]] && try brew upgrade --cask "$c"
  else
    echo "  + $c (cask)"; try brew install --cask "$c"
  fi
}

# ---------------------------------------------------------------- CLI basics
log "Developer basics"
for f in git git-lfs gh jq yq wget curl tmux htop btop ripgrep fd fzf tree cmake ninja pkg-config; do
  brew_formula "$f"
done
try git lfs install --skip-repo >/dev/null

# ---------------------------------------------------------------- Node / Python
log "Node.js"
brew_formula node
have node && echo "node $(node -v), npm $(npm -v)"

if [[ $SKIP_PYTHON -eq 0 ]]; then
  log "Python via uv"
  brew_formula uv
  try uv python install "$PYTHON_VERSION"

  # Homebrew/uv Pythons are PEP 668 "externally managed": a bare `pip3 install`
  # fails or pollutes the system. Use one shared venv instead.
  if [[ ! -x "$AI_VENV/bin/python" ]]; then
    log "Creating venv at $AI_VENV"
    mkdir -p "$(dirname "$AI_VENV")"
    try uv venv --python "$PYTHON_VERSION" "$AI_VENV"
  fi

  if [[ -x "$AI_VENV/bin/python" ]]; then
    log "Installing AI / ML packages into $AI_VENV"
    pip_install() { try uv pip install --python "$AI_VENV/bin/python" "$@"; }

    # Core ML. PyTorch's default macOS arm64 wheels include the MPS (Metal) backend.
    pip_install torch torchvision torchaudio
    # Apple-native inference/training
    pip_install mlx mlx-lm mlx-vlm coremltools
    # Hugging Face stack (provides the `hf` CLI)
    pip_install transformers accelerate datasets tokenizers safetensors sentence-transformers "huggingface_hub[cli]"
    # LLM app frameworks and vendor SDKs
    pip_install langchain langchain-community llama-index openai anthropic ollama
    # Vector stores / data / classic ML
    pip_install chromadb numpy pandas scipy scikit-learn matplotlib
    # Notebooks
    pip_install jupyterlab ipykernel
    # Optional extras (uncomment as needed):
    # pip_install gradio streamlit open-webui
    # tensorflow-macos is deprecated; for TF use `tensorflow` (+ `tensorflow-metal` is unmaintained).
    # pip_install tensorflow

    ZSHRC_MARK="# >>> ai venv >>>"
    if ! grep -qsF "$ZSHRC_MARK" "$HOME/.zshrc"; then
      cat >> "$HOME/.zshrc" <<EOF

$ZSHRC_MARK
alias ai-env='source $AI_VENV/bin/activate'
# <<< ai venv <<<
EOF
      echo "Added alias 'ai-env' to ~/.zshrc (activates $AI_VENV)"
    fi
  fi
fi

# ---------------------------------------------------------------- local inference
log "Local inference runtimes"
brew_formula ollama          # CLI + server; start with: brew services start ollama
brew_formula llama.cpp       # llama-cli, llama-server (Metal)
brew_cask    lm-studio       # always resolves to the current release (no hard-coded DMG version)

# ---------------------------------------------------------------- AI coding agents
log "AI coding agents"
brew_cask    claude-code     # Anthropic Claude Code
brew_cask    codex           # OpenAI Codex CLI
brew_formula opencode        # OpenCode (sst/opencode)
brew_formula gemini-cli      # Google Gemini CLI
# Alternatives if a brew package is missing:
#   Claude Code : curl -fsSL https://claude.ai/install.sh | bash
#   Codex       : npm install -g @openai/codex
#   OpenCode    : curl -fsSL https://opencode.ai/install | bash

# ---------------------------------------------------------------- containers
log "Containers"
brew_cask orbstack           # lighter than Docker Desktop on Apple Silicon; provides `docker`

# ---------------------------------------------------------------- editors
if [[ $SKIP_EDITORS -eq 0 ]]; then
  log "Editors / IDEs"
  brew_cask visual-studio-code
  brew_cask windsurf
  brew_cask cursor
fi

# ---------------------------------------------------------------- GGUF models (llama.cpp / LM Studio)
# "DD-Q4_K_XL" was taken as a typo for unsloth's Dynamic quant "UD-Q4_K_XL" (the same file
# download_llama_cpp_models.sh uses).
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
if [[ -x "$AI_VENV/bin/hf" ]]; then
  log "Downloading unsloth/Qwen3.5-9B-GGUF (UD-Q4_K_XL) to $MODEL_DIR"
  mkdir -p "$MODEL_DIR"
  try "$AI_VENV/bin/hf" download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q4_K_XL.gguf --local-dir "$MODEL_DIR"
  # Serve it:
  #   llama-server -m $MODEL_DIR/Qwen3.5-9B-UD-Q4_K_XL.gguf --alias qwen-9b --port 10022 -ngl 999 -c 32768 --jinja
  # or let llama.cpp fetch/cache it itself:
  #   llama-server -hf unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL --jinja
else
  warn "hf CLI not found in $AI_VENV; skipping Qwen3.5-9B download (re-run without --skip-python)"
fi

# ---------------------------------------------------------------- optional models
if [[ $PULL_MODELS -eq 1 ]]; then
  log "Pulling starter models"
  if have ollama; then
    brew services start ollama >/dev/null 2>&1 || true
    sleep 3
    try ollama pull llama3.2:3b
    try ollama pull qwen2.5-coder:7b
  fi
  # MLX example (downloads on first use):
  #   mlx_lm.generate --model mlx-community/Llama-3.2-3B-Instruct-4bit --prompt "Hello"
  #   mlx_lm.server   --model mlx-community/Llama-3.2-3B-Instruct-4bit
fi

# ---------------------------------------------------------------- verification
log "Installed versions"
show() { if have "$1"; then printf '  %-12s %s\n' "$1" "$("$@" 2>&1 | head -1)"; else printf '  %-12s MISSING\n' "$1"; fi; }
show brew --version
show node -v
show uv --version
show ollama --version
show llama-server --version
show claude --version
show codex --version
show opencode --version
show gemini --version
show docker --version
[[ -d "/Applications/LM Studio.app" ]] && echo "  LM Studio    installed" || echo "  LM Studio    MISSING"

if [[ -x "$AI_VENV/bin/python" ]]; then
  "$AI_VENV/bin/python" - <<'PY' 2>/dev/null || true
import torch, mlx.core as mx
print(f"  torch {torch.__version__}  MPS available: {torch.backends.mps.is_available()}")
print(f"  mlx   default device: {mx.default_device()}")
PY
fi

echo
if ((${#FAILED[@]})); then
  warn "Some steps failed:"; printf '   - %s\n' "${FAILED[@]}" >&2
else
  echo "All done."
fi

cat <<EOF

Next steps:
  exec zsh -l                      # reload shell (PATH + 'ai-env' alias)
  ai-env                           # activate $AI_VENV
  brew services start ollama       # run the Ollama server at login
  claude                           # sign in to Claude Code
  codex / opencode                 # sign in to Codex / OpenCode
  open -a "LM Studio"              # download models via the GUI
EOF
