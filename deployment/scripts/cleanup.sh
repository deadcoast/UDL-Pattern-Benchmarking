#!/bin/bash

# Cleanup script for UDL Rating Framework deployment
set -e

# Configuration
NAMESPACE="udl-rating"
KUBECTL_CONTEXT="${KUBECTL_CONTEXT:-}"

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

# Set kubectl context if specified
set_context() {
    if [ -n "${KUBECTL_CONTEXT}" ]; then
        log_info "Setting kubectl context to ${KUBECTL_CONTEXT}"
        kubectl config use-context "${KUBECTL_CONTEXT}"
    fi
}

# Cleanup Kubernetes deployment
cleanup_kubernetes() {
    log_info "Cleaning up UDL Rating Framework from Kubernetes..."
    
    # Check if namespace exists
    if ! kubectl get namespace ${NAMESPACE} &> /dev/null; then
        log_warn "Namespace ${NAMESPACE} does not exist"
        return 0
    fi
    
    # Delete ingress
    log_info "Removing ingress..."
    kubectl delete -f deployment/kubernetes/ingress.yaml --ignore-not-found=true
    
    # Delete horizontal pod autoscaler
    log_info "Removing autoscaling..."
    kubectl delete -f deployment/kubernetes/hpa.yaml --ignore-not-found=true
    
    # Delete services
    log_info "Removing services..."
    kubectl delete -f deployment/kubernetes/service.yaml --ignore-not-found=true
    
    # Delete deployments
    log_info "Removing deployments..."
    kubectl delete -f deployment/kubernetes/deployment.yaml --ignore-not-found=true
    
    # Delete monitoring stack
    log_info "Removing monitoring stack..."
    kubectl delete -f deployment/kubernetes/monitoring.yaml --ignore-not-found=true
    
    # Wait for pods to terminate
    log_info "Waiting for pods to terminate..."
    kubectl wait --for=delete pods --all -n ${NAMESPACE} --timeout=120s || true
    
    # Delete persistent volume claims (with confirmation)
    log_warn "This will delete all persistent data including models and logs!"
    read -p "Delete persistent volumes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing persistent volumes..."
        kubectl delete -f deployment/kubernetes/pvc.yaml --ignore-not-found=true
    else
        log_info "Keeping persistent volumes"
    fi
    
    # Delete configurations
    log_info "Removing configurations..."
    kubectl delete -f deployment/kubernetes/configmap.yaml --ignore-not-found=true
    
    # Delete secrets (with confirmation)
    read -p "Delete secrets? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing secrets..."
        kubectl delete -f deployment/kubernetes/secret.yaml --ignore-not-found=true
    else
        log_info "Keeping secrets"
    fi
    
    # Delete namespace (with confirmation)
    read -p "Delete namespace ${NAMESPACE}? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing namespace..."
        kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
    else
        log_info "Keeping namespace"
    fi
    
    log_info "Kubernetes cleanup completed!"
}

# Cleanup Docker Compose deployment
cleanup_docker_compose() {
    log_info "Cleaning up Docker Compose deployment..."
    
    cd deployment/docker
    
    # Stop and remove containers
    log_info "Stopping services..."
    docker-compose down
    
    # Remove volumes (with confirmation)
    read -p "Remove Docker volumes (this will delete all data)? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing volumes..."
        docker-compose down -v
    fi
    
    # Remove images (with confirmation)
    read -p "Remove Docker images? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing images..."
        docker-compose down --rmi all
    fi
    
    cd ../..
    
    log_info "Docker Compose cleanup completed!"
}

# Main cleanup logic
main() {
    local deployment_type="${1:-kubernetes}"
    
    log_warn "This will remove the UDL Rating Framework deployment!"
    log_warn "Deployment type: ${deployment_type}"
    
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleanup cancelled"
        exit 0
    fi
    
    case "${deployment_type}" in
        "kubernetes"|"k8s")
            set_context
            cleanup_kubernetes
            ;;
        "docker-compose"|"compose")
            cleanup_docker_compose
            ;;
        *)
            log_error "Unknown deployment type: ${deployment_type}"
            log_info "Supported types: kubernetes, docker-compose"
            exit 1
            ;;
    esac
    
    log_info "Cleanup process completed!"
}

# Show usage
show_usage() {
    echo "Usage: $0 [deployment_type]"
    echo ""
    echo "Deployment types:"
    echo "  kubernetes    Cleanup Kubernetes deployment (default)"
    echo "  docker-compose Cleanup Docker Compose deployment"
    echo ""
    echo "Environment variables:"
    echo "  KUBECTL_CONTEXT  Kubernetes context to use"
    echo ""
    echo "Examples:"
    echo "  $0 kubernetes"
    echo "  $0 docker-compose"
}

# Handle command line arguments
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

# Run main function
main "$@"