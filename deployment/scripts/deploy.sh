#!/bin/bash

# Deployment script for UDL Rating Framework
set -e

# Configuration
NAMESPACE="udl-rating"
KUBECTL_CONTEXT="${KUBECTL_CONTEXT:-}"
DRY_RUN="${DRY_RUN:-false}"

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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_info "Prerequisites check passed!"
}

# Set kubectl context if specified
set_context() {
    if [ -n "${KUBECTL_CONTEXT}" ]; then
        log_info "Setting kubectl context to ${KUBECTL_CONTEXT}"
        kubectl config use-context "${KUBECTL_CONTEXT}"
    fi
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying UDL Rating Framework to Kubernetes..."
    
    local dry_run_flag=""
    if [ "${DRY_RUN}" = "true" ]; then
        dry_run_flag="--dry-run=client"
        log_warn "Running in dry-run mode"
    fi
    
    # Create namespace
    log_info "Creating namespace..."
    kubectl apply -f deployment/kubernetes/namespace.yaml ${dry_run_flag}
    
    # Apply secrets (only if not dry run)
    if [ "${DRY_RUN}" != "true" ]; then
        log_warn "Please ensure secrets are properly configured in deployment/kubernetes/secret.yaml"
        read -p "Continue with secret deployment? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kubectl apply -f deployment/kubernetes/secret.yaml
        else
            log_warn "Skipping secret deployment. Please deploy manually."
        fi
    fi
    
    # Apply configurations
    log_info "Applying configurations..."
    kubectl apply -f deployment/kubernetes/configmap.yaml ${dry_run_flag}
    
    # Apply persistent volume claims
    log_info "Creating persistent volumes..."
    kubectl apply -f deployment/kubernetes/pvc.yaml ${dry_run_flag}
    
    # Deploy applications
    log_info "Deploying applications..."
    kubectl apply -f deployment/kubernetes/deployment.yaml ${dry_run_flag}
    
    # Create services
    log_info "Creating services..."
    kubectl apply -f deployment/kubernetes/service.yaml ${dry_run_flag}
    
    # Apply horizontal pod autoscaler
    log_info "Setting up autoscaling..."
    kubectl apply -f deployment/kubernetes/hpa.yaml ${dry_run_flag}
    
    # Apply ingress
    log_info "Setting up ingress..."
    kubectl apply -f deployment/kubernetes/ingress.yaml ${dry_run_flag}
    
    # Deploy monitoring (optional)
    read -p "Deploy monitoring stack (Prometheus/Grafana)? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Deploying monitoring stack..."
        kubectl apply -f deployment/kubernetes/monitoring.yaml ${dry_run_flag}
    fi
    
    if [ "${DRY_RUN}" != "true" ]; then
        # Wait for deployments to be ready
        log_info "Waiting for deployments to be ready..."
        kubectl wait --for=condition=available --timeout=300s deployment/udl-rating-api -n ${NAMESPACE}
        kubectl wait --for=condition=available --timeout=300s deployment/nginx-proxy -n ${NAMESPACE}
        
        # Show deployment status
        log_info "Deployment status:"
        kubectl get pods -n ${NAMESPACE}
        kubectl get services -n ${NAMESPACE}
        
        # Get external IP
        log_info "Getting external access information..."
        kubectl get ingress -n ${NAMESPACE}
        
        log_info "Deployment completed successfully!"
        log_info "API should be accessible via the ingress endpoint"
    else
        log_info "Dry-run completed successfully!"
    fi
}

# Deploy with Docker Compose (for local development)
deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    cd deployment/docker
    
    # Build images if needed
    if [ "${BUILD_IMAGES}" = "true" ]; then
        log_info "Building images..."
        docker-compose build
    fi
    
    # Start services
    log_info "Starting services..."
    docker-compose up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 10
    
    # Show status
    docker-compose ps
    
    log_info "Docker Compose deployment completed!"
    log_info "API available at: http://localhost:8000"
    log_info "Grafana available at: http://localhost:3000"
    log_info "Prometheus available at: http://localhost:9090"
    
    cd ../..
}

# Main deployment logic
main() {
    local deployment_type="${1:-kubernetes}"
    
    log_info "Starting deployment of UDL Rating Framework"
    log_info "Deployment type: ${deployment_type}"
    
    check_prerequisites
    
    case "${deployment_type}" in
        "kubernetes"|"k8s")
            set_context
            deploy_kubernetes
            ;;
        "docker-compose"|"compose")
            deploy_docker_compose
            ;;
        *)
            log_error "Unknown deployment type: ${deployment_type}"
            log_info "Supported types: kubernetes, docker-compose"
            exit 1
            ;;
    esac
    
    log_info "Deployment process completed!"
}

# Show usage
show_usage() {
    echo "Usage: $0 [deployment_type]"
    echo ""
    echo "Deployment types:"
    echo "  kubernetes    Deploy to Kubernetes cluster (default)"
    echo "  docker-compose Deploy locally with Docker Compose"
    echo ""
    echo "Environment variables:"
    echo "  KUBECTL_CONTEXT  Kubernetes context to use"
    echo "  DRY_RUN         Set to 'true' for dry-run mode"
    echo "  BUILD_IMAGES    Set to 'true' to build images"
    echo ""
    echo "Examples:"
    echo "  $0 kubernetes"
    echo "  DRY_RUN=true $0 kubernetes"
    echo "  BUILD_IMAGES=true $0 docker-compose"
}

# Handle command line arguments
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

# Run main function
main "$@"