#!/usr/bin/env bash
# =============================================================================
# Deep Research Agent - Setup Script
# Creates venv, installs dependencies, and copies config templates
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="venv"
PYTHON="python3"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()    { echo -e "${CYAN}[INFO]${NC}  $1"; }
success() { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail()    { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo ""
echo "========================================"
echo "  Deep Research Agent - Setup"
echo "========================================"
echo ""

# --- Check Python 3 ---
if ! command -v "$PYTHON" &>/dev/null; then
    fail "python3 not found. Please install Python 3.9+ first."
fi

PY_VERSION=$("$PYTHON" --version 2>&1 | awk '{print $2}')
info "Found $PYTHON ($PY_VERSION)"

# --- Create virtual environment ---
if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment already exists at ./$VENV_DIR"
    read -rp "  Recreate it? (y/N): " choice
    if [[ "$choice" =~ ^[Yy]$ ]]; then
        info "Removing old venv..."
        rm -rf "$VENV_DIR"
        info "Creating virtual environment..."
        "$PYTHON" -m venv "$VENV_DIR"
        success "Virtual environment created."
    else
        info "Keeping existing venv."
    fi
else
    info "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
    success "Virtual environment created."
fi

# --- Install dependencies ---
info "Upgrading pip..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet

info "Installing dependencies from requirements.txt..."
"$VENV_DIR/bin/pip" install -r requirements.txt --quiet
success "All dependencies installed."

# --- Copy .env template ---
if [ -f ".env" ]; then
    warn ".env already exists — skipping copy."
    info "  To reset: cp .env.example .env"
else
    if [ -f ".env.example" ]; then
        cp .env.example .env
        success "Created .env from .env.example"
        warn "Edit .env and add your API keys before running."
    else
        fail ".env.example not found — cannot create .env"
    fi
fi

# --- Copy config.yaml template ---
if [ -f "config.yaml" ]; then
    warn "config.yaml already exists — skipping copy."
    info "  To reset: cp config.yaml.example config.yaml"
else
    if [ -f "config.yaml.example" ]; then
        cp config.yaml.example config.yaml
        success "Created config.yaml from config.yaml.example"
    else
        fail "config.yaml.example not found — cannot create config.yaml"
    fi
fi

# --- Create logs directory ---
mkdir -p logs
success "logs/ directory ready."

# --- Summary ---
echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "  Next steps:"
echo "    1. Edit .env with your API keys:"
echo "       - OPENAI_API_KEY"
echo "       - TAVILY_API_KEY"
echo ""
echo "    2. (Optional) Edit config.yaml to tune settings"
echo ""
echo "    3. Start the server:"
echo "       ./start.sh"
echo ""
echo "    4. Or start fresh (clean all data first):"
echo "       ./start.sh --clean"
echo ""
