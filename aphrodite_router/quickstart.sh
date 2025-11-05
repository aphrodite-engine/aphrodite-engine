#!/usr/bin/env bash
set -euo pipefail

# Aphrodite Router Quick Start Script
# This script helps you quickly set up and run the router

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUTER_DIR="${SCRIPT_DIR}/router"
CANDLE_DIR="${SCRIPT_DIR}/candle_binding"
CONFIG_FILE="${ROUTER_DIR}/config/config.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is not installed. Please install it first."
        return 1
    fi
    return 0
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing=0
    
    if ! check_command go; then
        missing=1
    fi
    
    if ! check_command cargo; then
        missing=1
    fi
    
    if ! check_command rustc; then
        missing=1
    fi
    
    if [ $missing -eq 1 ]; then
        log_error "Please install missing prerequisites. See GUIDE.md for details."
        exit 1
    fi
    
    log_info "Prerequisites check passed ✓"
}

build_rust_library() {
    log_info "Building Rust library..."
    
    cd "$CANDLE_DIR"
    
    if [ ! -f "target/release/libcandle_semantic_router.so" ]; then
        log_info "Rust library not found. Building (this may take 10-20 minutes)..."
        cargo build --release
    else
        log_info "Rust library already built. Skipping..."
    fi
    
    if [ ! -f "target/release/libcandle_semantic_router.so" ]; then
        log_error "Failed to build Rust library"
        exit 1
    fi
    
    log_info "Rust library built successfully ✓"
}

build_go_router() {
    log_info "Building Go router..."
    
    cd "$ROUTER_DIR"
    
    if [ ! -f "aphrodite-router" ]; then
        log_info "Building router binary..."
        go build -o aphrodite-router ./cmd/main.go
    else
        log_info "Router binary already exists. Skipping..."
    fi
    
    if [ ! -f "aphrodite-router" ]; then
        log_error "Failed to build router binary"
        exit 1
    fi
    
    log_info "Router binary built successfully ✓"
}

check_config() {
    log_info "Checking configuration..."
    
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Config file not found: $CONFIG_FILE"
        log_info "Please create a config file or update CONFIG_FILE path"
        exit 1
    fi
    
    log_info "Configuration file found ✓"
}

check_aphrodite_instances() {
    log_info "Checking Aphrodite API instances..."
    
    # Read endpoints from config (simple check)
    local endpoints=$(grep -A 5 "aphrodite_endpoints:" "$CONFIG_FILE" | grep "port:" | awk '{print $2}' || true)
    
    if [ -z "$endpoints" ]; then
        log_warn "No endpoints configured in config.yaml"
        log_warn "Please add at least one endpoint to aphrodite_endpoints section"
    else
        log_info "Found endpoints in config:"
        echo "$endpoints" | while read port; do
            log_info "  - Port $port"
        done
        
        log_warn "Make sure Aphrodite instances are running on these ports!"
        log_warn "You can start them with:"
        log_warn "  python -m aphrodite.entrypoints.api_server --model your-model --port 8000"
    fi
}

main() {
    echo "=========================================="
    echo "  Aphrodite Router Quick Start"
    echo "=========================================="
    echo ""
    
    check_prerequisites
    build_rust_library
    build_go_router
    check_config
    check_aphrodite_instances
    
    echo ""
    log_info "Build complete! ✓"
    echo ""
    echo "Next steps:"
    echo "  1. Make sure Aphrodite API instances are running"
    echo "  2. Start the router:"
    echo "     cd $ROUTER_DIR"
    echo "     ./aphrodite-router.sh --config config/config.yaml"
    echo ""
    echo "  3. Or run directly:"
    echo "     cd $ROUTER_DIR"
    echo "     export LD_LIBRARY_PATH=../candle_binding/target/release:\$LD_LIBRARY_PATH"
    echo "     ./aphrodite-router --config config/config.yaml"
    echo ""
    echo "  4. Test the router:"
    echo "     curl http://localhost:8080/healthz"
    echo ""
    echo "For more details, see GUIDE.md"
    echo ""
}

main "$@"

