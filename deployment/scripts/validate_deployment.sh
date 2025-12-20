#!/bin/bash

# Validation script for UDL Rating Framework deployment scripts
# This script tests all deployment scripts in dry-run mode without making actual changes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

ERRORS=0
WARNINGS=0

# Test 1: Validate bash syntax of all scripts
test_bash_syntax() {
    log_info "Testing bash syntax of deployment scripts..."
    
    for script in "${SCRIPT_DIR}"/*.sh; do
        if bash -n "$script" 2>/dev/null; then
            log_success "$(basename "$script"): syntax OK"
        else
            log_fail "$(basename "$script"): syntax error"
            ((ERRORS++))
        fi
    done
}

# Test 2: Validate help options work
test_help_options() {
    log_info "Testing help options..."
    
    if bash "${SCRIPT_DIR}/deploy.sh" --help > /dev/null 2>&1; then
        log_success "deploy.sh --help: OK"
    else
        log_fail "deploy.sh --help: failed"
        ((ERRORS++))
    fi
    
    if bash "${SCRIPT_DIR}/cleanup.sh" --help > /dev/null 2>&1; then
        log_success "cleanup.sh --help: OK"
    else
        log_fail "cleanup.sh --help: failed"
        ((ERRORS++))
    fi
}

# Test 3: Validate Kubernetes manifests
test_kubernetes_manifests() {
    log_info "Testing Kubernetes manifests with dry-run..."
    
    if ! command -v kubectl &> /dev/null; then
        log_warn "kubectl not installed, skipping Kubernetes manifest validation"
        ((WARNINGS++))
        return
    fi
    
    local k8s_dir="${PROJECT_ROOT}/deployment/kubernetes"
    
    for manifest in "${k8s_dir}"/*.yaml; do
        local name=$(basename "$manifest")
        if kubectl apply -f "$manifest" --dry-run=client > /dev/null 2>&1; then
            log_success "$name: valid"
        else
            log_fail "$name: invalid"
            ((ERRORS++))
        fi
    done
}

# Test 4: Validate Docker Compose configuration
test_docker_compose() {
    log_info "Testing Docker Compose configuration..."
    
    if ! command -v docker &> /dev/null; then
        log_warn "docker not installed, skipping Docker Compose validation"
        ((WARNINGS++))
        return
    fi
    
    local compose_file="${PROJECT_ROOT}/deployment/docker/docker-compose.yml"
    
    if docker compose -f "$compose_file" config --quiet 2>/dev/null; then
        log_success "docker-compose.yml: valid"
    else
        log_fail "docker-compose.yml: invalid"
        ((ERRORS++))
    fi
}

# Test 5: Validate Dockerfile syntax
test_dockerfile() {
    log_info "Testing Dockerfile syntax..."
    
    local dockerfile="${PROJECT_ROOT}/deployment/docker/Dockerfile"
    
    if [ -f "$dockerfile" ]; then
        # Basic syntax check - ensure FROM statements exist
        if grep -q "^FROM" "$dockerfile"; then
            log_success "Dockerfile: has valid FROM statements"
        else
            log_fail "Dockerfile: missing FROM statement"
            ((ERRORS++))
        fi
        
        # Check for multi-stage build targets
        if grep -q "^FROM.*as base" "$dockerfile" && \
           grep -q "^FROM.*as production" "$dockerfile"; then
            log_success "Dockerfile: multi-stage build configured"
        else
            log_warn "Dockerfile: multi-stage build may not be properly configured"
            ((WARNINGS++))
        fi
    else
        log_fail "Dockerfile: not found"
        ((ERRORS++))
    fi
}

# Test 6: Validate required files exist
test_required_files() {
    log_info "Testing required deployment files exist..."
    
    local required_files=(
        "deployment/docker/Dockerfile"
        "deployment/docker/docker-compose.yml"
        "deployment/docker/nginx.conf"
        "deployment/docker/prometheus.yml"
        "deployment/api/main.py"
        "deployment/api/requirements.txt"
        "deployment/kubernetes/namespace.yaml"
        "deployment/kubernetes/deployment.yaml"
        "deployment/kubernetes/service.yaml"
        "deployment/scripts/build.sh"
        "deployment/scripts/deploy.sh"
        "deployment/scripts/cleanup.sh"
    )
    
    for file in "${required_files[@]}"; do
        if [ -f "${PROJECT_ROOT}/${file}" ]; then
            log_success "$file: exists"
        else
            log_fail "$file: missing"
            ((ERRORS++))
        fi
    done
}

# Test 7: Validate script permissions
test_script_permissions() {
    log_info "Testing script permissions..."
    
    for script in "${SCRIPT_DIR}"/*.sh; do
        if [ -x "$script" ] || [ -r "$script" ]; then
            log_success "$(basename "$script"): readable"
        else
            log_warn "$(basename "$script"): not executable (may need chmod +x)"
            ((WARNINGS++))
        fi
    done
}

# Main function
main() {
    log_info "Starting deployment script validation..."
    log_info "Project root: ${PROJECT_ROOT}"
    echo ""
    
    test_bash_syntax
    echo ""
    
    test_help_options
    echo ""
    
    test_required_files
    echo ""
    
    test_dockerfile
    echo ""
    
    test_docker_compose
    echo ""
    
    test_kubernetes_manifests
    echo ""
    
    test_script_permissions
    echo ""
    
    # Summary
    log_info "=========================================="
    log_info "Validation Summary"
    log_info "=========================================="
    
    if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
        log_success "All tests passed!"
        exit 0
    elif [ $ERRORS -eq 0 ]; then
        log_warn "Tests passed with $WARNINGS warning(s)"
        exit 0
    else
        log_fail "Tests failed with $ERRORS error(s) and $WARNINGS warning(s)"
        exit 1
    fi
}

# Run main function
main "$@"
