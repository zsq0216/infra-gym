# vLLM Infra-Gym Harness

Test harness for the vLLM infra-gym benchmark. This harness builds Docker environments
for running reproducible tests across 91 instances spanning vLLM versions 0.1.4 through
0.6.4.post1.

## Architecture Overview

The harness uses a layered Docker image approach to minimize build times and disk usage:

```
nvidia/cuda:XX.X-devel-ubuntu22.04     (NVIDIA base)
  |
  +-- infra-gym-base:cudaXX.X-pyX.XX   (System packages, Python, test tools)
        |
        +-- infra-gym:v0.1-v0.2        (PyTorch 2.0, CUDA 11.8 deps)
        +-- infra-gym:v0.3             (PyTorch 2.1, CUDA 12.1 deps)
        +-- infra-gym:v0.4             (PyTorch 2.3, CUDA 12.1 deps)
        +-- infra-gym:v0.5             (PyTorch 2.4, CUDA 12.4 deps)
        +-- infra-gym:v0.6             (PyTorch 2.5, CUDA 12.4 deps)
```

The vLLM source code is **not** baked into the images. Instead, it is mounted at
runtime from a local git checkout, allowing the harness to switch between commits
without rebuilding.

### Version Groups

| Group | Versions | Instances | CUDA | PyTorch | Python |
|-------|----------|-----------|------|---------|--------|
| v0.1-v0.2 | 0.1.4, 0.1.7, 0.2.4 | 4 | 11.8 | 2.0.1 | 3.10 |
| v0.3 | 0.3.3 | 3 | 12.1 | 2.1.2 | 3.10 |
| v0.4 | 0.4.0.post1 -- 0.4.3 | 22 | 12.1 | 2.3.0 | 3.10 |
| v0.5 | 0.5.0 -- 0.5.5 | 44 | 12.4 | 2.4.0 | 3.10 |
| v0.6 | 0.6.0 -- 0.6.4.post1 | 18 | 12.4 | 2.5.0 | 3.12 |

## Prerequisites

- Docker (version 20.10 or later) with BuildKit enabled
- NVIDIA Container Toolkit (`nvidia-docker2`) for GPU tests
- At least 50 GB of disk space for all images
- A local clone of `vllm-project/vllm` with full git history

```bash
# Install NVIDIA Container Toolkit (if not already installed)
# See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Clone vLLM (if you do not already have it)
git clone https://github.com/vllm-project/vllm.git /path/to/vllm
```

## 1. Building Images

### Build all images

```bash
cd harness/
./build_images.sh
```

### Build a specific version group

```bash
./build_images.sh v0.5          # Build only the v0.5 image (and its base)
./build_images.sh v0.4 v0.6     # Build v0.4 and v0.6
./build_images.sh base          # Build only the base images
```

### Build options

```bash
./build_images.sh --no-cache    # Force a clean rebuild
./build_images.sh --dry-run     # Print commands without executing
./build_images.sh --push        # Push to registry (set DOCKER_REGISTRY first)
```

### Using docker compose

```bash
# Build all images via compose
docker compose build

# Build a specific service
docker compose build vllm-v0.5
```

## 2. Running the Harness for a Single Instance

Each instance in the benchmark is identified by an `instance_id` (e.g.,
`vllm-project__vllm-10324-10164`). To run the harness for a single instance:

### Step 1: Look up the instance

```bash
# Find the instance in the dataset
python3 -c "
import json
with open('../vllm_infra_gym.json') as f:
    data = json.load(f)
for d in data:
    if d['instance_id'] == 'vllm-project__vllm-10324-10164':
        print('Version:', d['version'])
        print('Commit:', d['base_commit'])
        print('Test files:', [t['filename'] for t in d['tests']['test_files']])
        break
"
```

### Step 2: Determine the version group

Use `version_specs.json` to find the Docker image:

```bash
python3 -c "
import json
with open('version_specs.json') as f:
    specs = json.load(f)
version = '0.6.3.post1'
group = specs['version_to_group'][version]
image = specs['version_groups'][group]['docker_image']
print('Group:', group)
print('Image:', image)
"
```

### Step 3: Run the test container

```bash
# Set the path to your local vLLM checkout
export VLLM_REPO_PATH=/path/to/vllm

# Run with GPU support
docker run --rm -it \
    --gpus all \
    --shm-size=16g \
    -v "${VLLM_REPO_PATH}:/workspace/vllm" \
    -v "$(pwd)/results:/workspace/results" \
    infra-gym:v0.6 \
    bash -c "
        setup_vllm_env.sh --commit 803f37eaaa11 --test-patch /workspace/patches/test.patch &&
        cd /workspace/vllm &&
        python -m pytest tests/entrypoints/test_chat_utils.py::test_resolve_content_format_hf_defined \
            -x --timeout=600 --tb=short \
            --junitxml=/workspace/results/result.xml
    "
```

### Step 4: Run without GPU (CPU-only tests)

Some tests do not require a GPU. Use the `--skip-build` flag to avoid compiling
CUDA extensions:

```bash
docker run --rm -it \
    -v "${VLLM_REPO_PATH}:/workspace/vllm" \
    -v "$(pwd)/results:/workspace/results" \
    -e CUDA_VISIBLE_DEVICES="" \
    infra-gym:v0.5 \
    bash -c "
        setup_vllm_env.sh --commit abc123 --skip-build --test-patch /workspace/patches/test.patch &&
        cd /workspace/vllm &&
        python -m pytest tests/test_utils.py -x --timeout=120
    "
```

### Using docker compose

```bash
export VLLM_REPO_PATH=/path/to/vllm

# Interactive shell in a version-group container
docker compose run --rm vllm-v0.5 bash

# Run a specific test
docker compose run --rm vllm-v0.5 bash -c "
    setup_vllm_env.sh --commit abc123 --test-patch /workspace/patches/test.patch &&
    python -m pytest tests/test_config.py -x
"

# CPU-only tests
docker compose run --rm vllm-cpu bash -c "
    setup_vllm_env.sh --commit abc123 --skip-build &&
    python -m pytest tests/test_utils.py -x
"
```

## 3. Running the Full Benchmark

To evaluate all 91 instances:

### Automated run with a script

```bash
#!/bin/bash
# run_benchmark.sh — Evaluate all instances in the benchmark

DATASET="../vllm_infra_gym.json"
VLLM_REPO="/path/to/vllm"
RESULTS_DIR="./results"

mkdir -p "$RESULTS_DIR"

python3 -c "
import json, subprocess, sys

with open('${DATASET}') as f:
    data = json.load(f)

with open('version_specs.json') as f:
    specs = json.load(f)

for instance in data:
    iid = instance['instance_id']
    version = instance['version']
    commit = instance['base_commit']
    group = specs['version_to_group'][version]
    image = specs['version_groups'][group]['docker_image']
    test_ids = instance['tests']['test_ids'].get('all_test_ids', [])

    print(f'Running {iid} (version={version}, image={image})')

    # Write patches to temp files
    # ... (patch extraction logic)

    # Run tests
    test_args = ' '.join(test_ids)
    result = subprocess.run([
        'docker', 'run', '--rm',
        '--gpus', 'all',
        '--shm-size=16g',
        '-v', f'${VLLM_REPO}:/workspace/vllm',
        '-v', f'${RESULTS_DIR}:/workspace/results',
        image,
        'bash', '-c',
        f'setup_vllm_env.sh --commit {commit} && '
        f'cd /workspace/vllm && '
        f'python -m pytest {test_args} -x --timeout=600 '
        f'--junitxml=/workspace/results/{iid}.xml'
    ], capture_output=True, text=True, timeout=1800)

    print(f'  Exit code: {result.returncode}')
"
```

### Parallel execution

For faster evaluation, run instances in parallel (one per GPU):

```bash
# Using GNU parallel (adjust -j for number of GPUs)
cat instance_ids.txt | parallel -j 4 \
    "docker run --rm --gpus '\"device={}\"' --shm-size=16g \
     -v /path/to/vllm:/workspace/vllm \
     infra-gym:v0.5 bash -c 'setup_vllm_env.sh --commit ... && pytest ...'"
```

## 4. Collecting Results

### JUnit XML results

Each test run produces a JUnit XML file in the results directory:

```bash
ls results/
# vllm-project__vllm-10324-10164.xml
# vllm-project__vllm-10693-10705.xml
# ...
```

### Parsing results

```python
import json
import xml.etree.ElementTree as ET
import os

results_dir = "results"
results = {}

for filename in os.listdir(results_dir):
    if not filename.endswith(".xml"):
        continue
    instance_id = filename.replace(".xml", "")
    tree = ET.parse(os.path.join(results_dir, filename))
    root = tree.getroot()

    # Extract test suite summary
    testsuite = root.find(".//testsuite")
    if testsuite is not None:
        results[instance_id] = {
            "tests": int(testsuite.get("tests", 0)),
            "failures": int(testsuite.get("failures", 0)),
            "errors": int(testsuite.get("errors", 0)),
            "skipped": int(testsuite.get("skipped", 0)),
            "time": float(testsuite.get("time", 0)),
            "passed": (
                int(testsuite.get("tests", 0))
                - int(testsuite.get("failures", 0))
                - int(testsuite.get("errors", 0))
                - int(testsuite.get("skipped", 0))
            ),
        }

# Summary statistics
total = len(results)
passed = sum(1 for r in results.values() if r["failures"] == 0 and r["errors"] == 0)
print(f"Total instances evaluated: {total}")
print(f"All tests passing: {passed}/{total} ({100*passed/total:.1f}%)")

# Write consolidated results
with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Aggregation by version group

```python
from collections import defaultdict

by_group = defaultdict(lambda: {"total": 0, "passed": 0})
# Load version_specs.json to map instance versions to groups
with open("version_specs.json") as f:
    specs = json.load(f)

with open("../vllm_infra_gym.json") as f:
    dataset = json.load(f)

for instance in dataset:
    iid = instance["instance_id"]
    version = instance["version"]
    group = specs["version_to_group"][version]
    by_group[group]["total"] += 1
    if iid in results and results[iid]["failures"] == 0 and results[iid]["errors"] == 0:
        by_group[group]["passed"] += 1

for group, stats in sorted(by_group.items()):
    pct = 100 * stats["passed"] / stats["total"] if stats["total"] > 0 else 0
    print(f"  {group}: {stats['passed']}/{stats['total']} ({pct:.1f}%)")
```

## File Structure

```
harness/
  Dockerfile.base              Base image template (system packages, Python, test tools)
  build_images.sh              Build script for all Docker images
  docker-compose.yml           Docker Compose orchestration with GPU support
  version_specs.json           Machine-readable version specifications
  README.md                    This file
  scripts/
    setup_vllm_env.sh          In-container setup script (checkout, patch, install)
  dockerfiles/
    Dockerfile.v0.1-v0.2       CUDA 11.8, PyTorch 2.0 (4 instances)
    Dockerfile.v0.3            CUDA 12.1, PyTorch 2.1 (3 instances)
    Dockerfile.v0.4            CUDA 12.1, PyTorch 2.3 (22 instances)
    Dockerfile.v0.5            CUDA 12.4, PyTorch 2.4 (44 instances)
    Dockerfile.v0.6            CUDA 12.4, PyTorch 2.5 (18 instances)
  results/                     Test results output directory (created at runtime)
  patches/                     Patch files directory (created at runtime)
```

## Troubleshooting

### FlashAttention / FlashInfer fails to install

These packages require a GPU-capable build environment. If building on a machine
without a GPU, the image will still work -- these packages are installed with
`|| echo "WARNING: ..."` fallbacks. Tests that require these packages will fail
at runtime, which is expected if the test needs GPU attention kernels.

### Out of shared memory

If you see `RuntimeError: DataLoader worker ... is killed by signal: Bus error`,
increase the shared memory size:

```bash
docker run --shm-size=32g ...
# or in docker-compose.yml, adjust shm_size
```

### CUDA version mismatch

The Docker images include specific CUDA toolkit versions. If your host NVIDIA driver
does not support the CUDA version in the image, you will see errors like
`CUDA driver version is insufficient`. Check compatibility at:
https://docs.nvidia.com/deploy/cuda-compatibility/

### Permission denied on mounted volume

The containers run as the `vllm` user. Ensure the mounted vLLM repo directory
is readable by UID 999 (the `vllm` user), or run with `--user root` for debugging:

```bash
docker run --user root ...
```
