#!/usr/bin/env bash
# =============================================================================
# Deep Research Agent - Start Server
# Optional: --clean flag to wipe all data before starting
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="venv"
HOST="127.0.0.1"
PORT="8000"

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

# --- Parse arguments ---
CLEAN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean|-c)
            CLEAN=true
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port|-p)
            PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./start.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --clean, -c    Remove all data (database, reports, logs) before starting"
            echo "  --host HOST    Host to bind to (default: 127.0.0.1)"
            echo "  --port, -p N   Port to bind to (default: 8000)"
            echo "  --help, -h     Show this help message"
            exit 0
            ;;
        *)
            warn "Unknown option: $1"
            shift
            ;;
    esac
done

echo ""
echo "========================================"
echo "  Deep Research Agent"
echo "========================================"
echo ""

# --- Check venv exists ---
if [ ! -d "$VENV_DIR" ]; then
    fail "Virtual environment not found. Run ./setup.sh first."
fi

# --- Check .env exists ---
if [ ! -f ".env" ]; then
    fail ".env not found. Run ./setup.sh first."
fi

# --- Clean data if requested ---
if [ "$CLEAN" = true ]; then
    warn "Cleaning all data..."

    # Database files
    for f in research_state.db research_state.db-wal research_state.db-shm; do
        if [ -f "$f" ]; then
            rm -f "$f"
            info "  Removed $f"
        fi
    done

    # Report directory
    if [ -d "report" ]; then
        rm -rf "report"
        info "  Removed report/"
    fi

    # Logs
    if [ -d "logs" ]; then
        rm -rf "logs"
        info "  Removed logs/"
    fi
    mkdir -p logs

    success "All data cleaned. Starting fresh."
    echo ""
fi

# --- Start server ---
info "Starting web dashboard at http://${HOST}:${PORT}"
echo ""
exec "$VENV_DIR/bin/python" main.py serve --host "$HOST" --port "$PORT"
