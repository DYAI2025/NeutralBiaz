#!/bin/bash
# Deployment script for BiazNeutralize AI
# Usage: ./deploy.sh [environment] [options]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOYMENT_DIR="$PROJECT_ROOT/deployment"

# Default values
ENVIRONMENT="staging"
SKIP_TESTS=false
SKIP_BUILD=false
DRY_RUN=false
VERBOSE=false
FORCE_DEPLOY=false
ROLLBACK=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
BiazNeutralize AI Deployment Script

Usage: $0 [ENVIRONMENT] [OPTIONS]

Environments:
    staging     Deploy to staging environment (default)
    production  Deploy to production environment

Options:
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    -d, --dry-run          Show what would be done without executing
    -f, --force            Force deployment without confirmation
    -r, --rollback         Rollback to previous version
    --skip-tests           Skip running tests before deployment
    --skip-build           Skip building Docker images
    --tag TAG              Use specific image tag
    --config FILE          Use custom configuration file

Examples:
    $0 staging                          # Deploy to staging
    $0 production --force               # Force deploy to production
    $0 staging --dry-run --verbose      # Show staging deployment plan
    $0 --rollback production            # Rollback production deployment

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            staging|production)
                ENVIRONMENT="$1"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -f|--force)
                FORCE_DEPLOY=true
                shift
                ;;
            -r|--rollback)
                ROLLBACK=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
        log_error "Invalid environment: $ENVIRONMENT"
        show_help
        exit 1
    fi
    
    log_info "Deploying to environment: $ENVIRONMENT"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required commands
    local required_commands=("docker" "kubectl" "aws" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command '$cmd' not found"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        exit 1
    fi
    
    # Check kubectl context
    local current_context
    current_context=$(kubectl config current-context 2>/dev/null || echo "")
    if [[ -z "$current_context" ]]; then
        log_error "No kubectl context set"
        exit 1
    fi
    
    log_success "All prerequisites met"
}

# Load configuration
load_config() {
    local config_file="${CONFIG_FILE:-$DEPLOYMENT_DIR/configs/environments/$ENVIRONMENT.env}"
    
    if [[ -f "$config_file" ]]; then
        log_info "Loading configuration from $config_file"
        set -o allexport
        # shellcheck source=/dev/null
        source "$config_file"
        set +o allexport
    else
        log_warning "Configuration file not found: $config_file"
    fi
    
    # Set default image tag if not provided
    IMAGE_TAG="${IMAGE_TAG:-latest}"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping tests"
        return 0
    fi
    
    log_info "Running tests..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run: cd $PROJECT_ROOT && python -m pytest"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    if python -m pytest tests/ -v --tb=short; then
        log_success "All tests passed"
    else
        log_error "Tests failed"
        exit 1
    fi
}

# Build Docker images
build_images() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_warning "Skipping image build"
        return 0
    fi
    
    log_info "Building Docker images..."
    
    local backend_image="biazneutralize/backend:$IMAGE_TAG"
    local frontend_image="biazneutralize/frontend:$IMAGE_TAG"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would build: $backend_image"
        log_info "[DRY RUN] Would build: $frontend_image"
        return 0
    fi
    
    # Build backend image
    log_info "Building backend image: $backend_image"
    docker build \
        -f "$DEPLOYMENT_DIR/docker/Dockerfile.backend" \
        -t "$backend_image" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VERSION="$IMAGE_TAG" \
        --build-arg VCS_REF="$(git rev-parse HEAD)" \
        "$PROJECT_ROOT"
    
    # Build frontend image
    log_info "Building frontend image: $frontend_image"
    docker build \
        -f "$DEPLOYMENT_DIR/docker/Dockerfile.frontend" \
        -t "$frontend_image" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VERSION="$IMAGE_TAG" \
        --build-arg VCS_REF="$(git rev-parse HEAD)" \
        --build-arg REACT_APP_API_URL="$REACT_APP_API_URL" \
        --build-arg REACT_APP_ENVIRONMENT="$ENVIRONMENT" \
        "$PROJECT_ROOT"
    
    log_success "Images built successfully"
}

# Push images to registry
push_images() {
    log_info "Pushing images to registry..."
    
    local backend_image="biazneutralize/backend:$IMAGE_TAG"
    local frontend_image="biazneutralize/frontend:$IMAGE_TAG"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would push: $backend_image"
        log_info "[DRY RUN] Would push: $frontend_image"
        return 0
    fi
    
    docker push "$backend_image"
    docker push "$frontend_image"
    
    log_success "Images pushed successfully"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    local namespace
    if [[ "$ENVIRONMENT" == "production" ]]; then
        namespace="biazneutralize-prod"
    else
        namespace="biazneutralize"
    fi
    
    # Update image tags in deployment files
    local temp_dir
    temp_dir=$(mktemp -d)
    cp -r "$DEPLOYMENT_DIR/kubernetes/" "$temp_dir/"
    
    # Replace image tags
    sed -i "s|biazneutralize/backend:latest|biazneutralize/backend:$IMAGE_TAG|g" "$temp_dir/kubernetes/backend-deployment.yaml"
    sed -i "s|biazneutralize/frontend:latest|biazneutralize/frontend:$IMAGE_TAG|g" "$temp_dir/kubernetes/frontend-deployment.yaml"
    
    # Update namespace for production
    if [[ "$ENVIRONMENT" == "production" ]]; then
        sed -i "s/namespace: biazneutralize/namespace: $namespace/g" "$temp_dir/kubernetes/"*.yaml
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would apply Kubernetes manifests to namespace: $namespace"
        rm -rf "$temp_dir"
        return 0
    fi
    
    # Apply manifests
    kubectl apply -f "$temp_dir/kubernetes/namespace.yaml"
    kubectl apply -f "$temp_dir/kubernetes/configmap.yaml"
    kubectl apply -f "$temp_dir/kubernetes/secrets.yaml"
    kubectl apply -f "$temp_dir/kubernetes/postgres-deployment.yaml"
    kubectl apply -f "$temp_dir/kubernetes/redis-deployment.yaml"
    kubectl apply -f "$temp_dir/kubernetes/backend-deployment.yaml"
    kubectl apply -f "$temp_dir/kubernetes/frontend-deployment.yaml"
    
    # Wait for rollout to complete
    log_info "Waiting for deployment to complete..."
    kubectl rollout status deployment/backend-deployment -n "$namespace" --timeout=600s
    kubectl rollout status deployment/frontend-deployment -n "$namespace" --timeout=600s
    
    # Cleanup
    rm -rf "$temp_dir"
    
    log_success "Deployment completed successfully"
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    local base_url
    if [[ "$ENVIRONMENT" == "production" ]]; then
        base_url="https://biazneutralize.example.com"
    else
        base_url="https://staging.biazneutralize.example.com"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would test: $base_url/health"
        log_info "[DRY RUN] Would test: $base_url/api/health"
        return 0
    fi
    
    # Wait for services to be ready
    sleep 30
    
    # Test frontend health
    if curl -f "$base_url/health" &> /dev/null; then
        log_success "Frontend health check passed"
    else
        log_error "Frontend health check failed"
        exit 1
    fi
    
    # Test API health
    if curl -f "$base_url/api/health" &> /dev/null; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        exit 1
    fi
    
    log_success "All smoke tests passed"
}

# Rollback deployment
rollback_deployment() {
    log_info "Rolling back deployment..."
    
    local namespace
    if [[ "$ENVIRONMENT" == "production" ]]; then
        namespace="biazneutralize-prod"
    else
        namespace="biazneutralize"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would rollback deployments in namespace: $namespace"
        return 0
    fi
    
    kubectl rollout undo deployment/backend-deployment -n "$namespace"
    kubectl rollout undo deployment/frontend-deployment -n "$namespace"
    
    # Wait for rollback to complete
    kubectl rollout status deployment/backend-deployment -n "$namespace" --timeout=600s
    kubectl rollout status deployment/frontend-deployment -n "$namespace" --timeout=600s
    
    log_success "Rollback completed successfully"
}

# Confirm deployment
confirm_deployment() {
    if [[ "$FORCE_DEPLOY" == "true" || "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    
    echo
    log_warning "You are about to deploy to $ENVIRONMENT environment."
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Deployment cancelled"
        exit 0
    fi
}

# Main deployment function
main() {
    log_info "Starting BiazNeutralize AI deployment"
    log_info "Environment: $ENVIRONMENT"
    log_info "Image Tag: $IMAGE_TAG"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN MODE - No changes will be made"
    fi
    
    parse_args "$@"
    validate_environment
    check_prerequisites
    load_config
    
    if [[ "$ROLLBACK" == "true" ]]; then
        confirm_deployment
        rollback_deployment
        return 0
    fi
    
    confirm_deployment
    run_tests
    build_images
    push_images
    deploy_to_kubernetes
    run_smoke_tests
    
    log_success "Deployment completed successfully!"
    log_info "Application URL: https://${ENVIRONMENT}.biazneutralize.example.com"
}

# Trap errors and cleanup
trap 'log_error "Deployment failed at line $LINENO"' ERR

# Run main function
main "$@"