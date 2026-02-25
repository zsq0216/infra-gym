#!/usr/bin/env python3
"""Add environment field to each instance classifying hardware requirements.

Classification logic combines:
1. Test file paths (directory-level semantics)
2. Test patch content analysis (GPU markers, model loading, distributed patterns)
3. Problem statement context

Categories:
- gpu_distributed: needs multiple GPUs (pipeline parallel, tensor parallel, distributed)
- gpu_model: needs single GPU (model inference, CUDA kernels, quantization)
- api_server: tests API server logic, may need GPU for backend
- unit_cpu: pure Python logic, no GPU needed
"""

import json
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(REPO_ROOT, "vllm_infra_gym.json")

# ---------------------------------------------------------------------------
# Path-based rules: map test file path patterns to categories
# ---------------------------------------------------------------------------
# Order matters: first match wins per file
PATH_RULES = [
    # Distributed: always multi-GPU
    (r"tests/distributed/test_pipeline_parallel", "gpu_distributed", 2),
    (r"tests/distributed/test_basic_distributed", "gpu_distributed", 2),
    (r"tests/distributed/test_chunked_prefill_distributed", "gpu_distributed", 2),
    (r"tests/distributed/test_comm_ops", "gpu_distributed", 2),
    (r"tests/distributed/test_multimodal_broadcast", "gpu_distributed", 2),
    (r"tests/distributed/", "gpu_distributed", 2),

    # GPU kernels: single GPU
    (r"tests/kernels/", "gpu_model", 1),

    # Model tests: need GPU for model loading + inference
    (r"tests/models/", "gpu_model", 1),
    (r"tests/lora/", "gpu_model", 1),
    (r"tests/quantization/", "gpu_model", 1),
    (r"tests/spec_decode/e2e/", "gpu_model", 1),
    (r"tests/spec_decode/", "gpu_model", 1),
    (r"tests/basic_correctness/", "gpu_model", 1),
    (r"tests/samplers/test_logprobs", "gpu_model", 1),
    (r"tests/samplers/test_beam_search", "gpu_model", 1),
    (r"tests/multi_step/", "gpu_model", 1),
    (r"tests/tensorizer_loader/", "gpu_model", 1),
    (r"tests/tpu/", "gpu_model", 1),
    (r"tests/worker/", "gpu_model", 1),
    (r"tests/mq_llm_engine/", "gpu_model", 1),
    (r"tests/test_regression", "gpu_model", 1),

    # API server tests: test HTTP endpoints, usually need GPU backend
    (r"tests/entrypoints/openai/", "api_server", 1),
    (r"tests/entrypoints/test_openai_server", "api_server", 1),
    (r"tests/async_engine/test_openapi_server", "api_server", 1),
    (r"tests/async_engine/test_api_server", "api_server", 1),
    (r"tests/tool_use/", "api_server", 1),
    (r"tests/tracing/", "api_server", 1),

    # CPU-only: pure Python logic
    (r"tests/entrypoints/test_chat_utils", "unit_cpu", 0),
    (r"tests/engine/test_stop_reason", "unit_cpu", 0),
    (r"tests/engine/test_stop_strings", "unit_cpu", 0),
    (r"tests/engine/output_processor/", "unit_cpu", 0),
    (r"tests/core/", "unit_cpu", 0),
    (r"tests/tokenization/", "unit_cpu", 0),
    (r"tests/multimodal/test_processor", "unit_cpu", 0),
    (r"tests/test_logger", "unit_cpu", 0),
    (r"tests/test_utils", "unit_cpu", 0),
    (r"tests/samplers/test_sampler", "gpu_model", 1),
    (r"tests/samplers/test_rejection_sampler", "unit_cpu", 0),
]

# ---------------------------------------------------------------------------
# Content-based rules: patterns in test_patch that indicate GPU needs
# ---------------------------------------------------------------------------
CONTENT_GPU_PATTERNS = [
    (r"torch\.cuda", "gpu_model"),
    (r"vllm\.LLM\(", "gpu_model"),
    (r"LLM\(", "gpu_model"),
    (r"num_gpus", "gpu_model"),
    (r"ServerRunner", "api_server"),
    (r"openai_client", "api_server"),
    (r"RemoteOpenAI", "api_server"),
]

CONTENT_DISTRIBUTED_PATTERNS = [
    (r"tensor_parallel_size\s*[>=]\s*[2-9]", 2),
    (r"tp_size\s*[>=]\s*[2-9]", 2),
    (r"pipeline_parallel_size\s*[>=]\s*[2-9]", 2),
    (r"pp_size\s*[>=]\s*[2-9]", 2),
    (r"ray\.init", 2),
    (r"world_size\s*[>=]\s*[2-9]", 2),
    (r"num_gpus\s*[>=]\s*[2-9]", 2),
]


def classify_instance(inst):
    """Return (category, min_gpus, gpu_required, reasoning)."""
    test_files = [tf["filename"] for tf in inst["tests"]["test_files"]]
    test_patch = inst["tests"].get("test_patch", "")
    full_diff = inst["fix"].get("full_diff", "")
    combined_text = test_patch + "\n" + full_diff

    # Track all signals
    signals = []  # (category, min_gpus, source)

    # 1. Path-based classification
    for fpath in test_files:
        # Skip non-test files (conftest.py, utils.py, fixtures, data files)
        basename = os.path.basename(fpath)
        if basename in ("conftest.py", "utils.py", "__init__.py"):
            continue
        if "/fixtures/" in fpath or "/prompts/" in fpath or "/data/" in fpath:
            continue
        if not fpath.endswith(".py"):
            continue

        for pattern, category, min_gpus in PATH_RULES:
            if re.search(pattern, fpath):
                signals.append((category, min_gpus, "path:" + fpath))
                break

    # 2. Content-based: check for distributed patterns (overrides single-GPU)
    for pattern, min_gpus in CONTENT_DISTRIBUTED_PATTERNS:
        if re.search(pattern, combined_text):
            signals.append(("gpu_distributed", min_gpus, "content:" + pattern))

    # 3. Content-based: check for GPU usage patterns
    for pattern, category in CONTENT_GPU_PATTERNS:
        if re.search(pattern, combined_text):
            signals.append((category, 1, "content:" + pattern))

    if not signals:
        # Default: if we have test files but couldn't classify, assume gpu_model
        # (most vLLM tests need a model loaded)
        if test_files:
            return "gpu_model", 1, True, "default: unclassified vLLM test"
        return "unit_cpu", 0, False, "no test files"

    # Priority: gpu_distributed > gpu_model > api_server > unit_cpu
    priority = {"gpu_distributed": 4, "gpu_model": 3, "api_server": 2, "unit_cpu": 1}
    best = max(signals, key=lambda s: priority.get(s[0], 0))
    category = best[0]
    min_gpus = max(s[1] for s in signals)

    # Determine gpu_required
    gpu_required = category in ("gpu_distributed", "gpu_model", "api_server")

    # Build reasoning from unique sources
    reasons = sorted(set(s[2] for s in signals if priority.get(s[0], 0) >= priority.get(category, 0)))

    return category, min_gpus, gpu_required, "; ".join(reasons[:3])


def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    print("Classifying {} instances...".format(len(data)))

    stats = {}
    for inst in data:
        category, min_gpus, gpu_required, reasoning = classify_instance(inst)

        inst["environment"] = {
            "category": category,
            "gpu_required": gpu_required,
            "min_gpus": min_gpus,
            "arch": "any" if not gpu_required else "nvidia",
            "reasoning": reasoning,
        }

        stats[category] = stats.get(category, 0) + 1

    # Write output
    with open(INPUT_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n=== Classification Results ===")
    for cat in ["gpu_distributed", "gpu_model", "api_server", "unit_cpu"]:
        count = stats.get(cat, 0)
        print("  {:<20s}: {:>3d} instances".format(cat, count))
    print("  {:<20s}: {:>3d}".format("TOTAL", len(data)))

    gpu_total = stats.get("gpu_distributed", 0) + stats.get("gpu_model", 0) + stats.get("api_server", 0)
    cpu_total = stats.get("unit_cpu", 0)
    print("\n  GPU required: {}  |  CPU-only: {}".format(gpu_total, cpu_total))

    # Multi-GPU breakdown
    multi = sum(1 for d in data if d["environment"]["min_gpus"] >= 2)
    print("  Multi-GPU (>=2): {}".format(multi))

    # Samples
    print("\n=== Samples ===")
    for cat in ["gpu_distributed", "gpu_model", "api_server", "unit_cpu"]:
        examples = [d for d in data if d["environment"]["category"] == cat][:2]
        for inst in examples:
            env = inst["environment"]
            files = [tf["filename"] for tf in inst["tests"]["test_files"]]
            print("\n  {} [{}]".format(inst["instance_id"], cat))
            print("    min_gpus={}, gpu_required={}, arch={}".format(
                env["min_gpus"], env["gpu_required"], env["arch"]))
            print("    reasoning: {}".format(env["reasoning"]))
            print("    test_files: {}".format(files[:3]))


if __name__ == "__main__":
    main()
