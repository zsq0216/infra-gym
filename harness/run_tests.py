#!/usr/bin/env python3
"""
Test harness for the vLLM infra-gym benchmark.

Automates FAIL_TO_PASS and PASS_TO_PASS test collection for each instance in
the dataset. For every instance it:

  Phase 1 (pre-fix): checkout base_commit, apply test_patch only, run pytest
  Phase 2 (post-fix): additionally apply the source fix patch, re-run pytest

Then it cross-references results:
  FAIL_TO_PASS  = tests that FAILED in phase 1 AND PASSED in phase 2
  PASS_TO_PASS  = tests that PASSED in phase 1 AND PASSED in phase 2

Usage examples
--------------
Run a single instance locally:
    python run_tests.py --instance-id vllm-project__vllm-10324-10164

Run all instances using Docker images:
    python run_tests.py --instance-id all --docker

Run with custom paths:
    python run_tests.py \\
        --instance-id all \\
        --dataset ../vllm_infra_gym.json \\
        --workdir /data/workdir \\
        --output-dir ./results \\
        --timeout 600

Requirements
------------
- Python >= 3.8 (stdlib only -- no third-party packages)
- git >= 2.20  (for git worktree support)
- Docker (optional, for --docker mode)
- pytest must be installed in the target environment (local or container)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VLLM_REPO_URL = "https://github.com/vllm-project/vllm.git"
BARE_CLONE_DIR_NAME = "vllm.git"          # inside workdir
JUNIT_PHASE1 = "phase1.xml"
JUNIT_PHASE2 = "phase2.xml"
RESULT_FILE_SUFFIX = ".json"
VERSION_SPECS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "version_specs.json")

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("infra-gym-harness")


# ---------------------------------------------------------------------------
# Helpers: version-to-Docker-image mapping
# ---------------------------------------------------------------------------

_version_to_group = None  # type: Optional[Dict[str, str]]


def get_version_group(version):
    # type: (str) -> str
    """Map a vLLM version string (e.g. '0.5.3.post1') to its version group
    (e.g. 'v0.5') using version_specs.json. Falls back to a heuristic."""
    global _version_to_group
    if _version_to_group is None:
        try:
            with open(VERSION_SPECS_FILE, "r") as fh:
                specs = json.load(fh)
            _version_to_group = specs.get("version_to_group", {})
        except (IOError, json.JSONDecodeError) as exc:
            logger.warning("Could not load version_specs.json: %s", exc)
            _version_to_group = {}

    group = _version_to_group.get(version)
    if group:
        return group

    # Heuristic fallback: "0.5.3.post1" -> "v0.5"
    parts = version.split(".")
    if len(parts) >= 2:
        return "v{}.{}".format(parts[0], parts[1])
    return "v{}".format(version)


def get_docker_image_name(image_prefix, version):
    # type: (str, str) -> str
    """Construct Docker image name from prefix and version.

    Images are built per version GROUP (e.g. infra-gym:v0.5),
    not per individual version.
    """
    group = get_version_group(version)
    return "{}:{}".format(image_prefix, group)


# ---------------------------------------------------------------------------
# Helpers: subprocess
# ---------------------------------------------------------------------------

def run_cmd(
    cmd,                       # type: List[str]
    cwd=None,                  # type: Optional[str]
    timeout=300,               # type: int
    env=None,                  # type: Optional[Dict[str, str]]
    capture=True,              # type: bool
    check=False,               # type: bool
):
    # type: (...) -> subprocess.CompletedProcess
    """Run a command and return its CompletedProcess result.

    Wraps subprocess.run with sensible defaults for this harness:
    - stdout/stderr captured by default
    - timeout support
    - optional check for non-zero return codes
    """
    logger.debug("Running: %s (cwd=%s)", " ".join(shlex.quote(c) for c in cmd), cwd)
    kwargs = dict(
        cwd=cwd,
        timeout=timeout,
        env=env,
    )  # type: Dict[str, Any]
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    try:
        result = subprocess.run(cmd, **kwargs)
    except subprocess.TimeoutExpired:
        logger.warning("Command timed out after %ds: %s", timeout, cmd)
        raise
    if check and result.returncode != 0:
        stderr_text = ""
        if capture and result.stderr:
            stderr_text = result.stderr.decode("utf-8", errors="replace")
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )
    return result


# ---------------------------------------------------------------------------
# Helpers: JUnit XML parsing
# ---------------------------------------------------------------------------

def parse_junit_xml(xml_path):
    # type: (str) -> Dict[str, List[str]]
    """Parse a JUnit XML file produced by pytest --junit-xml.

    Returns a dict with keys:
        passed  - list of fully-qualified test node IDs
        failed  - list of fully-qualified test node IDs
        errors  - list of fully-qualified test node IDs
        skipped - list of fully-qualified test node IDs
    """
    result = {
        "passed": [],
        "failed": [],
        "errors": [],
        "skipped": [],
    }  # type: Dict[str, List[str]]

    if not os.path.isfile(xml_path):
        logger.warning("JUnit XML not found: %s", xml_path)
        return result

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        logger.warning("Failed to parse JUnit XML %s: %s", xml_path, exc)
        return result

    root = tree.getroot()

    # pytest JUnit XML structure:
    #   <testsuites> or <testsuite> at root
    #   <testcase classname="..." name="..." ...>
    #       <failure .../>  or  <error .../>  or  <skipped .../>
    #   </testcase>
    #
    # A testcase with no failure/error/skipped child is considered passed.
    for testcase in root.iter("testcase"):
        classname = testcase.get("classname", "")
        name = testcase.get("name", "")

        # Reconstruct a pytest-style node ID: file::class::method or file::func
        # pytest sets classname = "tests.foo.test_bar" and name = "test_func[param]"
        # We convert dots in classname back to path separators for the file part.
        node_id = _make_node_id(classname, name)

        failure = testcase.find("failure")
        error = testcase.find("error")
        skipped = testcase.find("skipped")

        if failure is not None:
            result["failed"].append(node_id)
        elif error is not None:
            result["errors"].append(node_id)
        elif skipped is not None:
            result["skipped"].append(node_id)
        else:
            result["passed"].append(node_id)

    return result


def _make_node_id(classname, name):
    # type: (str, str) -> str
    """Convert JUnit classname + name into a pytest node ID.

    Heuristic: pytest generates classname like
        "tests.entrypoints.test_chat_utils"
    which we turn into
        "tests/entrypoints/test_chat_utils::name"
    If the classname contains a class portion (e.g. TestFoo), we keep it after
    the last file-level segment.
    """
    if not classname:
        # Collection errors may store a dotted module path in *name*,
        # e.g. name="tests.test_logger".  Normalise it to a file path
        # (e.g. "tests/test_logger.py") so that prefix matching with
        # per-function node IDs works in the classification step.
        if "." in name and "::" not in name and not name.endswith(".py"):
            return "/".join(name.split(".")) + ".py"
        return name

    parts = classname.split(".")

    # Try to find where the module path ends and where a test class begins.
    # Convention: test classes start with uppercase or "Test".
    file_parts = []
    class_parts = []
    found_class = False
    for p in parts:
        if not found_class and (p[0:1].islower() or p.startswith("test_")):
            file_parts.append(p)
        else:
            found_class = True
            class_parts.append(p)

    file_path = "/".join(file_parts) + ".py" if file_parts else classname.replace(".", "/") + ".py"
    # Remove double .py if already present
    if file_path.endswith(".py.py"):
        file_path = file_path[:-3]

    if class_parts:
        return "{}::{}::{}".format(file_path, ".".join(class_parts), name)
    else:
        return "{}::{}".format(file_path, name)


# ---------------------------------------------------------------------------
# Helpers: pytest output parsing (fallback when JUnit fails)
# ---------------------------------------------------------------------------

def parse_pytest_log(log_path):
    # type: (str) -> Dict[str, List[str]]
    """Fallback parser: read pytest verbose output and classify tests.

    Looks for lines like:
        tests/foo.py::test_bar PASSED
        tests/foo.py::test_baz FAILED
    """
    result = {
        "passed": [],
        "failed": [],
        "errors": [],
        "skipped": [],
    }  # type: Dict[str, List[str]]

    if not os.path.isfile(log_path):
        return result

    with open(log_path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if " PASSED" in line:
                node_id = line.split(" PASSED")[0].strip()
                if "::" in node_id:
                    result["passed"].append(node_id)
            elif " FAILED" in line:
                node_id = line.split(" FAILED")[0].strip()
                if "::" in node_id:
                    result["failed"].append(node_id)
            elif " ERROR" in line:
                node_id = line.split(" ERROR")[0].strip()
                if "::" in node_id:
                    result["errors"].append(node_id)
            elif " SKIPPED" in line:
                node_id = line.split(" SKIPPED")[0].strip()
                if "::" in node_id:
                    result["skipped"].append(node_id)

    return result


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------

def ensure_bare_clone(workdir, repo_url=VLLM_REPO_URL):
    # type: (str, str) -> str
    """Create (or reuse) a bare clone of the vLLM repo as an object store.

    Returns the path to the bare clone directory.
    """
    bare_path = os.path.join(workdir, BARE_CLONE_DIR_NAME)

    if os.path.isdir(bare_path):
        logger.info("Reusing existing bare clone at %s", bare_path)
        # Fetch latest objects (in case we need newer commits)
        try:
            run_cmd(["git", "fetch", "--all"], cwd=bare_path, timeout=600)
        except Exception as exc:
            logger.warning("git fetch failed on bare clone: %s", exc)
        return bare_path

    logger.info("Creating bare clone of %s ...", repo_url)
    os.makedirs(workdir, exist_ok=True)
    run_cmd(
        ["git", "clone", "--bare", repo_url, bare_path],
        timeout=1200,
        check=True,
    )
    return bare_path


def _force_remove_dir(path):
    # type: (str) -> None
    """Remove a directory tree, handling root-owned files left by Docker.

    Docker containers run as root by default, so files created inside a
    bind-mounted volume are owned by root and ``shutil.rmtree`` will fail
    with PermissionError.  We first try a plain ``shutil.rmtree``; if that
    fails we fall back to ``chmod -R`` + retry, and finally ``sudo rm -rf``.
    """
    if not os.path.isdir(path):
        return
    try:
        shutil.rmtree(path)
        return
    except Exception:
        pass
    # Try fixing permissions then retrying
    try:
        subprocess.run(["chmod", "-R", "u+rwX", path], timeout=60,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.rmtree(path)
        return
    except Exception:
        pass
    # Last resort: sudo
    try:
        subprocess.run(["sudo", "rm", "-rf", path], timeout=60,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        logger.warning("Could not remove directory %s", path)


def setup_worktree(bare_path, worktree_path, commit_sha):
    # type: (str, str, str) -> None
    """Create a git worktree at worktree_path checked out at commit_sha.

    If the worktree already exists, remove and recreate it to ensure a clean
    state.
    """
    # Prune stale worktree references before attempting to create a new one
    try:
        run_cmd(["git", "worktree", "prune"], cwd=bare_path, timeout=60)
    except Exception:
        pass

    if os.path.isdir(worktree_path):
        logger.info("Removing stale worktree %s", worktree_path)
        # Try git worktree remove first
        try:
            run_cmd(
                ["git", "worktree", "remove", "--force", worktree_path],
                cwd=bare_path,
                timeout=60,
            )
        except Exception:
            pass
        # git worktree remove may succeed at unlinking the worktree
        # reference but leave root-owned files (created by Docker) on
        # disk.  Force-delete whatever remains.
        _force_remove_dir(worktree_path)
        try:
            run_cmd(["git", "worktree", "prune"], cwd=bare_path, timeout=60)
        except Exception:
            pass

    # Use a temporary detached branch name to avoid collisions
    branch_name = "harness-" + os.path.basename(worktree_path)

    # Remove branch if it already exists from a prior run
    try:
        run_cmd(
            ["git", "branch", "-D", branch_name],
            cwd=bare_path,
            timeout=30,
        )
    except Exception:
        pass

    logger.info("Creating worktree at %s (commit %s)", worktree_path, commit_sha[:12])
    run_cmd(
        [
            "git", "worktree", "add",
            "--detach",
            worktree_path,
            commit_sha,
        ],
        cwd=bare_path,
        timeout=300,
        check=True,
    )


def cleanup_worktree(bare_path, worktree_path):
    # type: (str, str) -> None
    """Remove a git worktree and its directory."""
    try:
        run_cmd(
            ["git", "worktree", "remove", "--force", worktree_path],
            cwd=bare_path,
            timeout=60,
        )
    except Exception:
        _force_remove_dir(worktree_path)
    try:
        run_cmd(["git", "worktree", "prune"], cwd=bare_path, timeout=60)
    except Exception:
        pass


def apply_patch(repo_path, patch_text, label="patch"):
    # type: (str, str, str) -> bool
    """Apply a git-format patch via git apply. Returns True on success."""
    if not patch_text or not patch_text.strip():
        logger.info("No %s to apply (empty).", label)
        return True

    # Write patch to a temp file
    patch_file = os.path.join(repo_path, ".tmp_{}.patch".format(label))
    try:
        with open(patch_file, "w") as fh:
            fh.write(patch_text)

        result = run_cmd(
            ["git", "apply", "--verbose", patch_file],
            cwd=repo_path,
            timeout=60,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
            logger.warning("git apply (%s) returned %d: %s", label, result.returncode, stderr)

            # Try with --3way as a fallback
            logger.info("Retrying %s with --3way ...", label)
            result2 = run_cmd(
                ["git", "apply", "--3way", patch_file],
                cwd=repo_path,
                timeout=60,
            )
            if result2.returncode != 0:
                stderr2 = result2.stderr.decode("utf-8", errors="replace") if result2.stderr else ""
                logger.error("git apply --3way (%s) also failed: %s", label, stderr2)
                return False
        return True
    finally:
        if os.path.exists(patch_file):
            os.remove(patch_file)


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------

def determine_test_targets(instance):
    # type: (Dict[str, Any]) -> List[str]
    """Determine which pytest targets to run for an instance.

    Priority:
      1. tests.test_ids.all_test_ids  (explicit pytest node IDs)
      2. tests.test_ids.affected_test_files  (file-level targets)
      3. tests.test_files[*].filename  (from PR file list)
    """
    test_ids_info = instance.get("tests", {}).get("test_ids", {})

    # Prefer explicit test IDs
    all_test_ids = test_ids_info.get("all_test_ids", [])
    if all_test_ids:
        return list(all_test_ids)

    # Fall back to affected test files
    affected_files = test_ids_info.get("affected_test_files", [])
    if affected_files:
        return list(affected_files)

    # Last resort: files from the test_files list
    test_files_list = instance.get("tests", {}).get("test_files", [])
    if test_files_list:
        return [tf["filename"] for tf in test_files_list if "filename" in tf]

    return []


def run_pytest_local(
    repo_path,        # type: str
    test_targets,     # type: List[str]
    junit_xml_path,   # type: str
    log_path,         # type: str
    timeout=300,      # type: int
    env_extra=None,   # type: Optional[Dict[str, str]]
):
    # type: (...) -> Dict[str, List[str]]
    """Run pytest locally and return parsed results.

    Uses --junit-xml for structured output and also captures verbose log
    as a fallback.
    """
    if not test_targets:
        logger.warning("No test targets to run.")
        return {"passed": [], "failed": [], "errors": [], "skipped": []}

    cmd = [
        sys.executable, "-m", "pytest",
        "--tb=short",
        "--no-header",
        "-rN",
        "-v",
        "--junit-xml={}".format(junit_xml_path),
        "--timeout={}".format(timeout),
    ] + test_targets

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    # Ensure the repo itself is on PYTHONPATH so imports work
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_path + (os.pathsep + existing_pp if existing_pp else "")

    logger.info("Running pytest with %d target(s) ...", len(test_targets))
    try:
        result = run_cmd(
            cmd,
            cwd=repo_path,
            timeout=timeout + 60,  # buffer beyond per-test timeout
            env=env,
            capture=True,
        )
        # Write verbose output to log file
        stdout_text = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        stderr_text = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        with open(log_path, "w") as fh:
            fh.write(stdout_text)
            fh.write("\n--- STDERR ---\n")
            fh.write(stderr_text)
    except subprocess.TimeoutExpired:
        logger.warning("pytest timed out after %ds", timeout + 60)
        with open(log_path, "w") as fh:
            fh.write("TIMEOUT after {}s\n".format(timeout + 60))
        return {"passed": [], "failed": [], "errors": ["TIMEOUT"], "skipped": []}

    # Parse JUnit XML first; fall back to log parsing
    results = parse_junit_xml(junit_xml_path)
    if not any(results[k] for k in results):
        logger.info("JUnit XML empty or missing, falling back to log parsing.")
        results = parse_pytest_log(log_path)

    return results


def _build_docker_setup_commands(category, setup_timeout=300):
    # type: (str, int) -> List[str]
    """Return shell commands to install vLLM and its dependencies inside Docker.

    These run before pytest so that ``import vllm`` (and transitive deps like
    ``psutil``, ``huggingface_hub``, etc.) succeed even though the container
    image only has Python + pytest.
    """
    commands = []  # type: List[str]

    # Use HuggingFace mirror to avoid connectivity issues in China.
    # Also forward HF_TOKEN from host if set (needed for gated models).
    commands.append('export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"')
    commands.append(
        'if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN="$HF_TOKEN"; fi'
    )

    # Install project requirements (psutil, transformers, etc.)
    req_install = (
        'for f in requirements-common.txt requirements.txt requirements-cpu.txt; '
        'do [ -f "$f" ] && timeout {t} pip install -r "$f"; done || true'
    ).format(t=setup_timeout)
    # Install test-specific requirements
    test_req_install = (
        'for f in requirements-test.txt requirements-dev.txt; '
        'do [ -f "$f" ] && timeout {t} pip install -r "$f"; done || true'
    ).format(t=setup_timeout)
    # Install vLLM itself in editable mode.
    # For unit_cpu we set VLLM_TARGET_DEVICE=empty to skip CUDA extension builds.
    # If the full install fails (old setup.py with hard CUDA deps), fall back to
    # --no-deps so at least the package is importable.
    if "cpu" in category:
        vllm_install = (
            'timeout {t} bash -c \'VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e "."\' '
            '|| timeout {t} bash -c \'pip install --no-build-isolation --no-deps -e "."\' || true'
        ).format(t=setup_timeout)
    else:
        vllm_install = (
            'timeout {t} pip install --no-build-isolation -e "." '
            '|| timeout {t} pip install --no-build-isolation --no-deps -e "." || true'
        ).format(t=setup_timeout)
    # PYTHONPATH fallback for old versions where editable install fails entirely
    pythonpath = 'export PYTHONPATH=/workspace:${PYTHONPATH:-}'

    # Some tests reference data files via relative paths like ./data/...
    # which resolve against CWD (/workspace), but the actual files live
    # under /workspace/tests/data/.  Create a symlink so both paths work.
    link_test_data = (
        '[ -d tests/data ] && [ ! -e data ] && ln -s tests/data data || true'
    )

    return commands + [req_install, test_req_install, vllm_install, pythonpath, link_test_data]


def run_pytest_docker(
    repo_path,          # type: str
    test_targets,       # type: List[str]
    junit_xml_path,     # type: str
    log_path,           # type: str
    image_name,         # type: str
    timeout=120,        # type: int
    setup_commands=None, # type: Optional[List[str]]
    setup_timeout=300,  # type: int
):
    # type: (...) -> Dict[str, List[str]]
    """Run pytest inside a Docker container and return parsed results.

    The repo directory is mounted into the container at /workspace.
    The container image is expected to have pytest installed.

    If *setup_commands* is provided, the commands are prepended before pytest
    (joined with ``; ``) and executed via ``bash -c``.  Network access is
    enabled so that ``pip install`` can fetch packages, and the timeout
    budget is increased to account for installation time.
    """
    if not test_targets:
        logger.warning("No test targets to run.")
        return {"passed": [], "failed": [], "errors": [], "skipped": []}

    container_workspace = "/workspace"
    container_junit = os.path.join(container_workspace, os.path.basename(junit_xml_path))

    pytest_args = [
        "python", "-m", "pytest",
        "--tb=short",
        "--no-header",
        "-rN",
        "-v",
        "--junit-xml={}".format(container_junit),
        "--timeout={}".format(timeout),
    ] + test_targets

    has_setup = setup_commands and len(setup_commands) > 0
    timeout_buffer = setup_timeout if has_setup else 120

    docker_cmd = [
        "docker", "run",
        "--rm",
        "-v", "{}:{}".format(os.path.abspath(repo_path), container_workspace),
        "-w", container_workspace,
        "--memory=16g",          # memory limit
    ]

    # Forward HuggingFace env vars from host into the container so that
    # the setup commands can use them (mirror endpoint, gated model token).
    for env_key in ("HF_ENDPOINT", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env_key)
        if val:
            docker_cmd += ["-e", "{}={}".format(env_key, val)]

    if not has_setup:
        docker_cmd.append("--network=none")

    docker_cmd.append(image_name)

    if has_setup:
        # Build a single shell command: setup1; setup2; ...; pytest ...
        pytest_str = " ".join(shlex.quote(a) for a in pytest_args)
        full_script = "; ".join(setup_commands) + "; " + pytest_str
        docker_cmd += ["bash", "-c", full_script]
    else:
        docker_cmd += pytest_args

    logger.info("Running pytest in Docker (%s) with %d target(s) ...",
                image_name, len(test_targets))
    try:
        result = run_cmd(
            docker_cmd,
            timeout=timeout + timeout_buffer,
            capture=True,
        )
        stdout_text = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        stderr_text = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        with open(log_path, "w") as fh:
            fh.write(stdout_text)
            fh.write("\n--- STDERR ---\n")
            fh.write(stderr_text)
    except subprocess.TimeoutExpired:
        logger.warning("Docker pytest timed out after %ds", timeout + timeout_buffer)
        with open(log_path, "w") as fh:
            fh.write("TIMEOUT after {}s\n".format(timeout + timeout_buffer))
        return {"passed": [], "failed": [], "errors": ["TIMEOUT"], "skipped": []}

    # pytest writes JUnit XML inside the mounted volume at
    # /workspace/<basename>.xml which maps to {repo_path}/<basename>.xml on
    # the host.  However the caller expects it at junit_xml_path (usually
    # inside the output dir).  Copy it there if needed.
    worktree_junit = os.path.join(repo_path, os.path.basename(junit_xml_path))
    if os.path.isfile(worktree_junit) and os.path.abspath(worktree_junit) != os.path.abspath(junit_xml_path):
        os.makedirs(os.path.dirname(junit_xml_path), exist_ok=True)
        shutil.move(worktree_junit, junit_xml_path)

    # The JUnit XML should have been written inside the mounted volume
    results = parse_junit_xml(junit_xml_path)
    if not any(results[k] for k in results):
        logger.info("JUnit XML empty or missing, falling back to log parsing.")
        results = parse_pytest_log(log_path)

    return results


# ---------------------------------------------------------------------------
# Per-instance pipeline
# ---------------------------------------------------------------------------

def process_instance(
    instance,          # type: Dict[str, Any]
    workdir,           # type: str
    output_dir,        # type: str
    timeout,           # type: int
    use_docker,        # type: bool
    image_prefix,      # type: str
    keep_worktrees=False,  # type: bool
    setup_timeout=300, # type: int
):
    # type: (...) -> Dict[str, Any]
    """Run the full FAIL_TO_PASS / PASS_TO_PASS pipeline for one instance.

    Returns a result dict ready to be serialized as JSON.
    """
    instance_id = instance["instance_id"]
    base_commit = instance["base_commit"]
    version = instance.get("version", "unknown")
    test_patch = instance.get("tests", {}).get("test_patch", "")
    source_patch = instance.get("fix", {}).get("patch", "")

    logger.info("=" * 72)
    logger.info("Processing instance: %s  (version %s)", instance_id, version)
    logger.info("=" * 72)

    result = {
        "instance_id": instance_id,
        "version": version,
        "base_commit": base_commit,
        "phase1": {"passed": [], "failed": [], "errors": [], "skipped": []},
        "phase2": {"passed": [], "failed": [], "errors": [], "skipped": []},
        "FAIL_TO_PASS": [],
        "PASS_TO_PASS": [],
        "status": "error",
        "error_message": "",
        "timestamps": {
            "start": time.time(),
            "phase1_start": 0.0,
            "phase1_end": 0.0,
            "phase2_start": 0.0,
            "phase2_end": 0.0,
            "end": 0.0,
        },
    }  # type: Dict[str, Any]

    # Paths
    instance_workdir = os.path.join(workdir, instance_id)
    worktree_path = os.path.join(instance_workdir, "repo")
    inst_output_dir = os.path.join(output_dir, instance_id)
    os.makedirs(inst_output_dir, exist_ok=True)

    bare_path = os.path.join(workdir, BARE_CLONE_DIR_NAME)

    # ------------------------------------------------------------------
    # Step 0: Ensure bare clone exists
    # ------------------------------------------------------------------
    try:
        bare_path = ensure_bare_clone(workdir)
    except Exception as exc:
        result["error_message"] = "Failed to create/update bare clone: {}".format(exc)
        result["timestamps"]["end"] = time.time()
        logger.error(result["error_message"])
        return result

    # ------------------------------------------------------------------
    # Step 1: Setup worktree at base_commit
    # ------------------------------------------------------------------
    try:
        setup_worktree(bare_path, worktree_path, base_commit)
    except Exception as exc:
        result["error_message"] = "Failed to setup worktree: {}".format(exc)
        result["timestamps"]["end"] = time.time()
        logger.error(result["error_message"])
        return result

    # ------------------------------------------------------------------
    # Step 2: Apply test_patch only
    # ------------------------------------------------------------------
    if not apply_patch(worktree_path, test_patch, label="test_patch"):
        result["error_message"] = "Failed to apply test_patch"
        result["timestamps"]["end"] = time.time()
        logger.error(result["error_message"])
        if not keep_worktrees:
            cleanup_worktree(bare_path, worktree_path)
        return result

    # ------------------------------------------------------------------
    # Step 3: Determine test targets
    # ------------------------------------------------------------------
    test_targets = determine_test_targets(instance)
    if not test_targets:
        result["error_message"] = "No test targets found for this instance"
        result["timestamps"]["end"] = time.time()
        logger.warning(result["error_message"])
        if not keep_worktrees:
            cleanup_worktree(bare_path, worktree_path)
        return result

    logger.info("Test targets (%d): %s", len(test_targets),
                test_targets[:5] if len(test_targets) > 5 else test_targets)

    # ------------------------------------------------------------------
    # Phase 1: Run tests WITHOUT source fix (only test_patch applied)
    # ------------------------------------------------------------------
    phase1_junit = os.path.join(inst_output_dir, JUNIT_PHASE1)
    phase1_log = os.path.join(inst_output_dir, "phase1.log")

    result["timestamps"]["phase1_start"] = time.time()

    image_name = get_docker_image_name(image_prefix, version) if use_docker else ""

    setup_commands = None  # type: Optional[List[str]]
    if use_docker:
        category = instance.get("environment", {}).get("category", "")
        setup_commands = _build_docker_setup_commands(category, setup_timeout=setup_timeout)

    if use_docker:
        phase1_results = run_pytest_docker(
            worktree_path, test_targets, phase1_junit, phase1_log,
            image_name, timeout=timeout, setup_commands=setup_commands,
            setup_timeout=setup_timeout,
        )
    else:
        phase1_results = run_pytest_local(
            worktree_path, test_targets, phase1_junit, phase1_log,
            timeout=timeout,
        )

    result["phase1"] = phase1_results
    result["timestamps"]["phase1_end"] = time.time()

    logger.info(
        "Phase 1 results: %d passed, %d failed, %d errors, %d skipped",
        len(phase1_results["passed"]),
        len(phase1_results["failed"]),
        len(phase1_results["errors"]),
        len(phase1_results["skipped"]),
    )

    # ------------------------------------------------------------------
    # Step 4: Apply source fix patch on top of the test patch
    # ------------------------------------------------------------------
    if not apply_patch(worktree_path, source_patch, label="source_patch"):
        result["error_message"] = "Failed to apply source patch"
        result["status"] = "partial"  # Phase 1 ran OK
        result["timestamps"]["end"] = time.time()
        logger.error(result["error_message"])
        if not keep_worktrees:
            cleanup_worktree(bare_path, worktree_path)
        return result

    # ------------------------------------------------------------------
    # Phase 2: Run tests WITH source fix (test_patch + source_patch)
    # ------------------------------------------------------------------
    phase2_junit = os.path.join(inst_output_dir, JUNIT_PHASE2)
    phase2_log = os.path.join(inst_output_dir, "phase2.log")

    result["timestamps"]["phase2_start"] = time.time()

    if use_docker:
        phase2_results = run_pytest_docker(
            worktree_path, test_targets, phase2_junit, phase2_log,
            image_name, timeout=timeout, setup_commands=setup_commands,
            setup_timeout=setup_timeout,
        )
    else:
        phase2_results = run_pytest_local(
            worktree_path, test_targets, phase2_junit, phase2_log,
            timeout=timeout,
        )

    result["phase2"] = phase2_results
    result["timestamps"]["phase2_end"] = time.time()

    logger.info(
        "Phase 2 results: %d passed, %d failed, %d errors, %d skipped",
        len(phase2_results["passed"]),
        len(phase2_results["failed"]),
        len(phase2_results["errors"]),
        len(phase2_results["skipped"]),
    )

    # ------------------------------------------------------------------
    # Step 5: Classify tests
    # ------------------------------------------------------------------
    phase1_failed_set = set(phase1_results["failed"] + phase1_results["errors"])
    phase1_passed_set = set(phase1_results["passed"])
    phase2_passed_set = set(phase2_results["passed"])

    # FAIL_TO_PASS: failed before fix, pass after fix
    fail_to_pass = sorted(phase1_failed_set & phase2_passed_set)

    # Handle module-level collection errors: a Phase 1 error like
    # "tests/test_logger.py" (no "::") should match any Phase 2 pass
    # within that module, e.g. "tests/test_logger.py::test_func".
    f2p_set = set(fail_to_pass)
    module_errors_resolved = set()  # module-level errors that found matches
    for err_id in phase1_results["errors"]:
        if "::" not in err_id and err_id not in phase2_passed_set:
            prefix = err_id + "::" if err_id.endswith(".py") else err_id + ".py::"
            for p2 in phase2_passed_set:
                if p2.startswith(prefix) and p2 not in f2p_set:
                    f2p_set.add(p2)
                    module_errors_resolved.add(err_id)
    fail_to_pass = sorted(f2p_set)

    # PASS_TO_PASS: passed before fix, still pass after fix
    pass_to_pass = sorted(phase1_passed_set & phase2_passed_set)

    result["FAIL_TO_PASS"] = fail_to_pass
    result["PASS_TO_PASS"] = pass_to_pass
    result["status"] = "success"
    result["timestamps"]["end"] = time.time()

    logger.info("FAIL_TO_PASS: %d tests", len(fail_to_pass))
    logger.info("PASS_TO_PASS: %d tests", len(pass_to_pass))

    # Also log problematic tests (failed in both phases)
    # Exclude module-level errors that were resolved by prefix matching above.
    both_failed = sorted(
        (phase1_failed_set - phase2_passed_set) - module_errors_resolved
    )
    if both_failed:
        logger.warning("Tests that FAILED in both phases (%d): %s",
                        len(both_failed), both_failed[:5])

    # Regressions: passed before, now fail (should not happen with a fix)
    regressions = sorted(phase1_passed_set - phase2_passed_set)
    if regressions:
        logger.warning("Regressions (passed->failed): %d: %s",
                        len(regressions), regressions[:5])
    result["regressions"] = regressions
    result["both_failed"] = both_failed

    # ------------------------------------------------------------------
    # Cleanup worktree
    # ------------------------------------------------------------------
    if not keep_worktrees:
        cleanup_worktree(bare_path, worktree_path)
    else:
        logger.info("Keeping worktree at %s (--keep-worktrees)", worktree_path)

    return result


def save_result(result, output_dir):
    # type: (Dict[str, Any], str) -> str
    """Save per-instance result to a JSON file. Returns the file path."""
    instance_id = result["instance_id"]
    # Save in the instance sub-directory
    inst_output_dir = os.path.join(output_dir, instance_id)
    os.makedirs(inst_output_dir, exist_ok=True)
    result_path = os.path.join(inst_output_dir, "result.json")
    with open(result_path, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=False)
    logger.info("Saved result to %s", result_path)

    # Also save a flat copy at output_dir/{instance_id}.json for easy collection
    flat_path = os.path.join(output_dir, instance_id + RESULT_FILE_SUFFIX)
    with open(flat_path, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=False)
    return flat_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_dataset(dataset_path):
    # type: (str) -> List[Dict[str, Any]]
    """Load the infra-gym dataset JSON."""
    with open(dataset_path, "r") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array at top level, got {}".format(type(data).__name__))
    return data


def filter_instances(dataset, instance_id_filter, category_filter=None):
    # type: (List[Dict[str, Any]], str, Optional[str]) -> List[Dict[str, Any]]
    """Filter instances by ID and/or category.

    instance_id_filter: 'all', a single ID, or comma-separated IDs.
    category_filter: None (no filter), or comma-separated categories
        (gpu_distributed, gpu_model, api_server, unit_cpu).
    """
    VALID_CATEGORIES = {"gpu_distributed", "gpu_model", "api_server", "unit_cpu"}

    # Step 1: filter by instance ID
    if instance_id_filter.lower() == "all":
        filtered = list(dataset)
    else:
        requested_ids = set(iid.strip() for iid in instance_id_filter.split(","))
        filtered = [inst for inst in dataset if inst["instance_id"] in requested_ids]

        if not filtered:
            available = [inst["instance_id"] for inst in dataset[:10]]
            raise ValueError(
                "No instances matched '{}'. Available IDs include: {}".format(
                    instance_id_filter, available
                )
            )

        missing = requested_ids - set(inst["instance_id"] for inst in filtered)
        if missing:
            logger.warning("Requested instance IDs not found in dataset: %s", missing)

    # Step 2: filter by category
    if category_filter:
        requested_cats = set(c.strip() for c in category_filter.split(","))
        unknown = requested_cats - VALID_CATEGORIES
        if unknown:
            raise ValueError(
                "Unknown category: {}. Valid categories: {}".format(
                    unknown, sorted(VALID_CATEGORIES)
                )
            )

        before = len(filtered)
        filtered = [
            inst for inst in filtered
            if inst.get("environment", {}).get("category") in requested_cats
        ]
        logger.info(
            "Category filter %s: %d -> %d instances",
            requested_cats, before, len(filtered),
        )

        if not filtered:
            raise ValueError(
                "No instances matched category '{}' (from {} candidates).".format(
                    category_filter, before
                )
            )

    return filtered


def print_summary(results):
    # type: (List[Dict[str, Any]]) -> None
    """Print a tabular summary of all processed instances."""
    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print("{:<50s} {:>6s} {:>6s} {:>8s}".format(
        "Instance", "F2P", "P2P", "Status"))
    print("-" * 88)

    total_f2p = 0
    total_p2p = 0
    statuses = {"success": 0, "error": 0, "partial": 0, "timeout": 0}

    for r in sorted(results, key=lambda x: x["instance_id"]):
        f2p = len(r.get("FAIL_TO_PASS", []))
        p2p = len(r.get("PASS_TO_PASS", []))
        status = r.get("status", "unknown")
        print("{:<50s} {:>6d} {:>6d} {:>8s}".format(
            r["instance_id"][:50], f2p, p2p, status))
        total_f2p += f2p
        total_p2p += p2p
        statuses[status] = statuses.get(status, 0) + 1

    print("-" * 88)
    print("{:<50s} {:>6d} {:>6d}".format("TOTAL", total_f2p, total_p2p))
    print()
    print("Status breakdown: {}".format(
        ", ".join("{}={}".format(k, v) for k, v in sorted(statuses.items()) if v > 0)))
    print("=" * 88)


def build_parser():
    # type: () -> argparse.ArgumentParser
    parser = argparse.ArgumentParser(
        description="Test harness for the vLLM infra-gym benchmark. "
                    "Collects FAIL_TO_PASS and PASS_TO_PASS test sets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single instance locally
  %(prog)s --instance-id vllm-project__vllm-10324-10164

  # Run all instances with Docker
  %(prog)s --instance-id all --docker

  # Run only CPU-only instances (no GPU needed)
  %(prog)s --category unit_cpu

  # Run GPU single-card and API server instances
  %(prog)s --category gpu_model,api_server --docker

  # Custom paths and timeout
  %(prog)s --instance-id all --dataset ./data.json --workdir /tmp/work --timeout 600
""",
    )
    parser.add_argument(
        "--instance-id",
        default="all",
        help='Instance ID to run, or "all" for every instance. '
             "Comma-separated list is also accepted. (default: all)",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Filter instances by environment category. "
             "Comma-separated list accepted. "
             "Valid values: gpu_distributed, gpu_model, api_server, unit_cpu",
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm_infra_gym.json"),
        help="Path to the infra-gym dataset JSON file. "
             "(default: ../vllm_infra_gym.json relative to this script)",
    )
    parser.add_argument(
        "--workdir",
        default="/tmp/infra-gym-workdir",
        help="Working directory for git clones and worktrees. (default: /tmp/infra-gym-workdir)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        help="Directory to save per-instance result files. (default: ./results/)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-test timeout in seconds. (default: 120)",
    )
    parser.add_argument(
        "--setup-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for Docker setup (pip install) steps. "
             "Also used as the buffer for overall Docker subprocess timeout. "
             "(default: 300)",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        default=False,
        help="Run pytest inside Docker containers instead of locally.",
    )
    parser.add_argument(
        "--image-prefix",
        default="infra-gym",
        help='Docker image prefix. Image name = {prefix}:{version}. '
             '(default: "infra-gym")',
    )
    parser.add_argument(
        "--keep-worktrees",
        action="store_true",
        default=False,
        help="Do not clean up git worktrees after processing (useful for debugging).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG) logging.",
    )
    return parser


def main():
    # type: () -> None
    parser = build_parser()
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=LOG_FORMAT)

    # Resolve paths
    dataset_path = os.path.abspath(args.dataset)
    workdir = os.path.abspath(args.workdir)
    output_dir = os.path.abspath(args.output_dir)

    logger.info("Dataset:    %s", dataset_path)
    logger.info("Workdir:    %s", workdir)
    logger.info("Output dir: %s", output_dir)
    logger.info("Timeout:    %ds (setup: %ds)", args.timeout, args.setup_timeout)
    logger.info("Docker:     %s", args.docker)

    # Load dataset
    if not os.path.isfile(dataset_path):
        logger.error("Dataset file not found: %s", dataset_path)
        sys.exit(1)

    dataset = load_dataset(dataset_path)
    logger.info("Loaded %d instances from dataset.", len(dataset))

    # Filter instances
    try:
        instances = filter_instances(dataset, args.instance_id, args.category)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info("Will process %d instance(s).", len(instances))
    os.makedirs(output_dir, exist_ok=True)

    # Process each instance
    all_results = []  # type: List[Dict[str, Any]]

    for idx, instance in enumerate(instances, 1):
        iid = instance["instance_id"]
        logger.info("[%d/%d] Starting instance: %s", idx, len(instances), iid)

        try:
            result = process_instance(
                instance=instance,
                workdir=workdir,
                output_dir=output_dir,
                timeout=args.timeout,
                use_docker=args.docker,
                image_prefix=args.image_prefix,
                keep_worktrees=args.keep_worktrees,
                setup_timeout=args.setup_timeout,
            )
        except subprocess.TimeoutExpired:
            result = {
                "instance_id": iid,
                "version": instance.get("version", "unknown"),
                "base_commit": instance.get("base_commit", ""),
                "phase1": {"passed": [], "failed": [], "errors": [], "skipped": []},
                "phase2": {"passed": [], "failed": [], "errors": [], "skipped": []},
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
                "status": "timeout",
                "error_message": "Overall instance processing timed out",
            }
        except Exception as exc:
            logger.exception("Unexpected error processing %s", iid)
            result = {
                "instance_id": iid,
                "version": instance.get("version", "unknown"),
                "base_commit": instance.get("base_commit", ""),
                "phase1": {"passed": [], "failed": [], "errors": [], "skipped": []},
                "phase2": {"passed": [], "failed": [], "errors": [], "skipped": []},
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
                "status": "error",
                "error_message": str(exc),
            }

        save_result(result, output_dir)
        all_results.append(result)

    # Print summary
    print_summary(all_results)

    # Exit with non-zero if any instance errored
    error_count = sum(1 for r in all_results if r["status"] != "success")
    if error_count:
        logger.warning("%d instance(s) did not complete successfully.", error_count)
        sys.exit(1)


if __name__ == "__main__":
    main()
