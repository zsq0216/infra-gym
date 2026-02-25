#!/bin/bash
# =============================================================================
# setup_vllm_env.sh — Prepare a vLLM checkout for testing
# =============================================================================
#
# This script is invoked by the infra-gym harness inside the Docker container.
# It expects the vLLM source tree to be mounted at /workspace/vllm and performs:
#   1. Checkout to the specified base commit
#   2. Apply the test patch (if provided)
#   3. Install vLLM in editable/development mode
#   4. Install any additional test requirements from the repo
#
# Usage:
#   setup_vllm_env.sh --commit <sha> [--test-patch <path>] [--source-patch <path>] [--skip-build]
#
# Options:
#   --commit <sha>        Base commit to checkout (required)
#   --test-patch <path>   Path to the test patch file to apply
#   --source-patch <path> Path to the source (fix) patch file to apply
#   --skip-build          Skip building vLLM C extensions (for CPU-only tests)
#   --editable            Install vLLM in editable mode (pip install -e .)
# =============================================================================

set -euo pipefail

COMMIT=""
TEST_PATCH=""
SOURCE_PATCH=""
SKIP_BUILD=false
EDITABLE=false
VLLM_DIR="/workspace/vllm"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --commit)
            COMMIT="$2"
            shift 2
            ;;
        --test-patch)
            TEST_PATCH="$2"
            shift 2
            ;;
        --source-patch)
            SOURCE_PATCH="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --editable)
            EDITABLE=true
            shift
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$COMMIT" ]; then
    echo "ERROR: --commit is required"
    exit 1
fi

cd "$VLLM_DIR"

# ---------------------------------------------------------------------------
# Step 1: Checkout the base commit
# ---------------------------------------------------------------------------
echo "==> Checking out commit: $COMMIT"
git fetch --all --quiet 2>/dev/null || true
git checkout "$COMMIT" --force
git clean -fdx --quiet
git submodule update --init --recursive --quiet 2>/dev/null || true

# ---------------------------------------------------------------------------
# Step 2: Apply test patch (the test changes from the fix PR)
# ---------------------------------------------------------------------------
if [ -n "$TEST_PATCH" ] && [ -f "$TEST_PATCH" ]; then
    echo "==> Applying test patch: $TEST_PATCH"
    git apply --allow-empty "$TEST_PATCH" || {
        echo "WARNING: git apply failed, trying with --3way"
        git apply --allow-empty --3way "$TEST_PATCH" || {
            echo "ERROR: Could not apply test patch"
            exit 1
        }
    }
fi

# ---------------------------------------------------------------------------
# Step 3: Apply source patch (the actual fix — used for gold evaluation)
# ---------------------------------------------------------------------------
if [ -n "$SOURCE_PATCH" ] && [ -f "$SOURCE_PATCH" ]; then
    echo "==> Applying source patch: $SOURCE_PATCH"
    git apply --allow-empty "$SOURCE_PATCH" || {
        echo "WARNING: git apply failed, trying with --3way"
        git apply --allow-empty --3way "$SOURCE_PATCH" || {
            echo "ERROR: Could not apply source patch"
            exit 1
        }
    }
fi

# ---------------------------------------------------------------------------
# Step 4: Install vLLM and its dependencies
# ---------------------------------------------------------------------------
if [ "$SKIP_BUILD" = true ]; then
    echo "==> Skipping vLLM build (CPU-only mode)"
    # Install Python-only dependencies without compiling C extensions
    export VLLM_TARGET_DEVICE=empty
    if [ "$EDITABLE" = true ]; then
        pip install --no-build-isolation -e "." 2>/dev/null || \
            pip install --no-build-isolation -e ".[dev]" 2>/dev/null || \
            echo "WARNING: Could not install vLLM in editable mode, continuing anyway"
    else
        pip install --no-build-isolation "." 2>/dev/null || \
            echo "WARNING: Could not install vLLM, continuing anyway"
    fi
else
    echo "==> Installing vLLM from source (with CUDA extensions)"
    if [ "$EDITABLE" = true ]; then
        pip install -e "." 2>/dev/null || \
            pip install -e ".[dev]" 2>/dev/null || {
                echo "WARNING: Standard install failed, trying with MAX_JOBS=4"
                MAX_JOBS=4 pip install -e "." || true
            }
    else
        pip install "." 2>/dev/null || {
            echo "WARNING: Standard install failed, trying with MAX_JOBS=4"
            MAX_JOBS=4 pip install "." || true
        }
    fi
fi

# ---------------------------------------------------------------------------
# Step 5: Install test requirements if they exist in the repo
# ---------------------------------------------------------------------------
echo "==> Installing test requirements"
for req_file in \
    "requirements-test.txt" \
    "requirements/test.txt" \
    "requirements-dev.txt" \
    "requirements/dev.txt" \
    "tests/requirements.txt"; do
    if [ -f "$VLLM_DIR/$req_file" ]; then
        echo "    Installing from $req_file"
        pip install --no-cache-dir -r "$VLLM_DIR/$req_file" 2>/dev/null || \
            echo "    WARNING: Some packages from $req_file failed to install"
    fi
done

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo "==> Environment setup complete"
echo "    Python: $(python --version 2>&1)"
echo "    Commit: $(git rev-parse HEAD)"
echo "    vLLM location: $VLLM_DIR"
python -c "import vllm; print('    vLLM version:', vllm.__version__)" 2>/dev/null || \
    echo "    WARNING: vLLM not importable (may be expected for some test configurations)"
