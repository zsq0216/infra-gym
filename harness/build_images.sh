#!/bin/bash
# =============================================================================
# build_images.sh — Build all infra-gym Docker images
# =============================================================================
#
# This script builds the base image and all version-group images for the
# vLLM infra-gym benchmark.
#
# Usage:
#   ./build_images.sh              # Build all images
#   ./build_images.sh base         # Build only the base image(s)
#   ./build_images.sh v0.5         # Build only the v0.5 group
#   ./build_images.sh v0.4 v0.6    # Build multiple groups
#   ./build_images.sh --no-cache   # Build without Docker cache
#   ./build_images.sh --dry-run    # Print commands without executing
#
# Environment variables:
#   DOCKER_REGISTRY   — Push images to this registry (default: local only)
#   DOCKER_TAG_PREFIX — Prefix for image tags (default: infra-gym)
#   BUILD_JOBS        — Number of parallel build jobs (default: 1)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
DOCKER_TAG_PREFIX="${DOCKER_TAG_PREFIX:-infra-gym}"
BUILD_JOBS="${BUILD_JOBS:-1}"
NO_CACHE=""
DRY_RUN=false

# ---------------------------------------------------------------------------
# Version group specifications
# Each entry: GROUP_NAME|CUDA_VERSION|PYTHON_VERSION|DOCKERFILE_PATH
# ---------------------------------------------------------------------------
declare -a VERSION_GROUPS=(
    "v0.1-v0.2|11.8.0|3.10|dockerfiles/Dockerfile.v0.1-v0.2"
    "v0.3|12.1.0|3.10|dockerfiles/Dockerfile.v0.3"
    "v0.4|12.1.0|3.10|dockerfiles/Dockerfile.v0.4"
    "v0.5|12.4.0|3.10|dockerfiles/Dockerfile.v0.5"
    "v0.6|12.4.0|3.12|dockerfiles/Dockerfile.v0.6"
)

# Base images needed (unique CUDA_VERSION x PYTHON_VERSION combos)
declare -a BASE_IMAGES=(
    "11.8.0|3.10"
    "12.1.0|3.10"
    "12.4.0|3.10"
    "12.4.0|3.12"
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
log_info() {
    echo -e "\033[1;34m[INFO]\033[0m $*"
}

log_success() {
    echo -e "\033[1;32m[OK]\033[0m $*"
}

log_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $*" >&2
}

log_warn() {
    echo -e "\033[1;33m[WARN]\033[0m $*"
}

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] $*"
    else
        "$@"
    fi
}

usage() {
    echo "Usage: $0 [OPTIONS] [VERSION_GROUP ...]"
    echo ""
    echo "Options:"
    echo "  --no-cache    Build without Docker cache"
    echo "  --dry-run     Print commands without executing"
    echo "  --push        Push images to registry after building"
    echo "  -h, --help    Show this help message"
    echo ""
    echo "Version groups:"
    echo "  base          Build only the base image(s)"
    echo "  v0.1-v0.2     vLLM 0.1.4, 0.1.7, 0.2.4 (CUDA 11.8, PyTorch 2.0)"
    echo "  v0.3          vLLM 0.3.3 (CUDA 12.1, PyTorch 2.1)"
    echo "  v0.4          vLLM 0.4.0-0.4.3 (CUDA 12.1, PyTorch 2.3)"
    echo "  v0.5          vLLM 0.5.0-0.5.5 (CUDA 12.4, PyTorch 2.4)"
    echo "  v0.6          vLLM 0.6.0-0.6.4 (CUDA 12.4, PyTorch 2.5)"
    echo ""
    echo "If no version group is specified, all images are built."
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
PUSH=false
TARGETS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --push)
            PUSH=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            TARGETS+=("$1")
            shift
            ;;
    esac
done

# If no targets specified, build everything
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=("all")
fi

# ---------------------------------------------------------------------------
# Build a base image
# ---------------------------------------------------------------------------
build_base_image() {
    local cuda_version="$1"
    local python_version="$2"
    local tag="${DOCKER_TAG_PREFIX}-base:cuda${cuda_version}-py${python_version}"

    log_info "Building base image: $tag"
    log_info "  CUDA=${cuda_version}, Python=${python_version}"

    run_cmd docker build \
        ${NO_CACHE} \
        --build-arg CUDA_VERSION="${cuda_version}" \
        --build-arg PYTHON_VERSION="${python_version}" \
        -t "${tag}" \
        -f "${SCRIPT_DIR}/Dockerfile.base" \
        "${SCRIPT_DIR}"

    if [ "$PUSH" = true ] && [ -n "$DOCKER_REGISTRY" ]; then
        local remote_tag="${DOCKER_REGISTRY}/${tag}"
        run_cmd docker tag "${tag}" "${remote_tag}"
        run_cmd docker push "${remote_tag}"
    fi

    log_success "Base image built: $tag"
}

# ---------------------------------------------------------------------------
# Build a version-group image
# ---------------------------------------------------------------------------
build_version_image() {
    local group_name="$1"
    local cuda_version="$2"
    local python_version="$3"
    local dockerfile="$4"
    local tag="${DOCKER_TAG_PREFIX}:${group_name}"
    local base_tag="${DOCKER_TAG_PREFIX}-base:cuda${cuda_version}-py${python_version}"

    log_info "Building version-group image: $tag"
    log_info "  Group=${group_name}, CUDA=${cuda_version}, Python=${python_version}"
    log_info "  Base image: ${base_tag}"
    log_info "  Dockerfile: ${dockerfile}"

    run_cmd docker build \
        ${NO_CACHE} \
        --build-arg CUDA_VERSION="${cuda_version}" \
        --build-arg PYTHON_VERSION="${python_version}" \
        -t "${tag}" \
        -f "${SCRIPT_DIR}/${dockerfile}" \
        "${SCRIPT_DIR}"

    if [ "$PUSH" = true ] && [ -n "$DOCKER_REGISTRY" ]; then
        local remote_tag="${DOCKER_REGISTRY}/${tag}"
        run_cmd docker tag "${tag}" "${remote_tag}"
        run_cmd docker push "${remote_tag}"
    fi

    log_success "Version-group image built: $tag"
}

# ---------------------------------------------------------------------------
# Determine which base images are needed for the selected targets
# ---------------------------------------------------------------------------
needs_base_image() {
    local cuda_version="$1"
    local python_version="$2"

    for target in "${TARGETS[@]}"; do
        if [ "$target" = "all" ] || [ "$target" = "base" ]; then
            return 0
        fi
    done

    # Check if any selected version group needs this base image
    for spec in "${VERSION_GROUPS[@]}"; do
        IFS='|' read -r group_name group_cuda group_python group_dockerfile <<< "$spec"
        if [ "$group_cuda" = "$cuda_version" ] && [ "$group_python" = "$python_version" ]; then
            for target in "${TARGETS[@]}"; do
                if [ "$target" = "$group_name" ]; then
                    return 0
                fi
            done
        fi
    done

    return 1
}

should_build_group() {
    local group_name="$1"
    for target in "${TARGETS[@]}"; do
        if [ "$target" = "all" ] || [ "$target" = "$group_name" ]; then
            return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# Main build sequence
# ---------------------------------------------------------------------------
log_info "=========================================="
log_info "  infra-gym Docker Image Builder"
log_info "=========================================="
log_info "Targets: ${TARGETS[*]}"
log_info "Script dir: ${SCRIPT_DIR}"
[ -n "$NO_CACHE" ] && log_info "Cache: disabled"
[ "$DRY_RUN" = true ] && log_warn "DRY RUN mode — no commands will be executed"
echo ""

# Step 1: Build required base images
log_info "Step 1: Building base images..."
for base_spec in "${BASE_IMAGES[@]}"; do
    IFS='|' read -r cuda_version python_version <<< "$base_spec"
    if needs_base_image "$cuda_version" "$python_version"; then
        build_base_image "$cuda_version" "$python_version"
    else
        log_info "  Skipping base cuda${cuda_version}-py${python_version} (not needed)"
    fi
done
echo ""

# Step 2: Build version-group images
if ! [[ " ${TARGETS[*]} " =~ " base " ]] || [[ " ${TARGETS[*]} " =~ " all " ]]; then
    log_info "Step 2: Building version-group images..."
    for spec in "${VERSION_GROUPS[@]}"; do
        IFS='|' read -r group_name cuda_version python_version dockerfile <<< "$spec"
        if should_build_group "$group_name"; then
            build_version_image "$group_name" "$cuda_version" "$python_version" "$dockerfile"
        else
            log_info "  Skipping ${group_name} (not selected)"
        fi
    done
fi

echo ""
log_success "=========================================="
log_success "  Build complete!"
log_success "=========================================="
echo ""

# List built images
log_info "Built images:"
if [ "$DRY_RUN" = false ]; then
    docker images --filter "reference=${DOCKER_TAG_PREFIX}*" --format "  {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" 2>/dev/null || true
fi
