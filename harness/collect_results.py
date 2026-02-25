#!/usr/bin/env python3
"""
Collect per-instance results and merge them back into the main dataset JSON.

Reads all {output_dir}/{instance_id}.json files produced by run_tests.py
and writes the FAIL_TO_PASS and PASS_TO_PASS lists as JSON-encoded strings
into the dataset, matching the SWE-bench format.

Usage:
    python collect_results.py
    python collect_results.py --results-dir ./results --dataset ../vllm_infra_gym.json --output ../vllm_infra_gym_evaluated.json

SWE-bench output format per instance:
    {
      ...existing fields...,
      "FAIL_TO_PASS": "[\"tests/foo.py::test_bar\", ...]",   # JSON-encoded string
      "PASS_TO_PASS": "[\"tests/foo.py::test_baz\", ...]",   # JSON-encoded string
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from typing import Any, Dict, List, Optional

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger("collect-results")


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def load_result_files(results_dir):
    # type: (str) -> Dict[str, Dict[str, Any]]
    """Load all per-instance result JSON files from the results directory.

    Returns a dict mapping instance_id -> result dict.
    Looks for both flat files ({results_dir}/{instance_id}.json) and
    nested files ({results_dir}/{instance_id}/result.json).
    """
    results = {}  # type: Dict[str, Dict[str, Any]]

    if not os.path.isdir(results_dir):
        logger.error("Results directory does not exist: %s", results_dir)
        return results

    for entry in os.listdir(results_dir):
        entry_path = os.path.join(results_dir, entry)

        # Case 1: flat file like "vllm-project__vllm-10324-10164.json"
        if entry.endswith(".json") and os.path.isfile(entry_path):
            try:
                with open(entry_path, "r") as fh:
                    data = json.load(fh)
                iid = data.get("instance_id", entry.replace(".json", ""))
                results[iid] = data
            except (json.JSONDecodeError, IOError) as exc:
                logger.warning("Failed to load %s: %s", entry_path, exc)

        # Case 2: directory with result.json inside
        elif os.path.isdir(entry_path):
            nested_path = os.path.join(entry_path, "result.json")
            if os.path.isfile(nested_path):
                try:
                    with open(nested_path, "r") as fh:
                        data = json.load(fh)
                    iid = data.get("instance_id", entry)
                    # Don't overwrite flat file result (they should be identical)
                    if iid not in results:
                        results[iid] = data
                except (json.JSONDecodeError, IOError) as exc:
                    logger.warning("Failed to load %s: %s", nested_path, exc)

    logger.info("Loaded %d result file(s) from %s", len(results), results_dir)
    return results


# ---------------------------------------------------------------------------
# Merge results into dataset
# ---------------------------------------------------------------------------

def merge_results_into_dataset(
    dataset,      # type: List[Dict[str, Any]]
    results,      # type: Dict[str, Dict[str, Any]]
    swebench_format=True,  # type: bool
):
    # type: (...) -> List[Dict[str, Any]]
    """Merge FAIL_TO_PASS and PASS_TO_PASS from results into the dataset.

    When swebench_format is True, the lists are stored as JSON-encoded strings
    (matching the SWE-bench dataset format). Otherwise they are stored as
    plain Python lists.

    Returns the updated dataset (mutated in place).
    """
    matched = 0
    for instance in dataset:
        iid = instance["instance_id"]
        if iid not in results:
            continue

        r = results[iid]
        f2p = r.get("FAIL_TO_PASS", [])
        p2p = r.get("PASS_TO_PASS", [])

        if swebench_format:
            instance["FAIL_TO_PASS"] = json.dumps(f2p)
            instance["PASS_TO_PASS"] = json.dumps(p2p)
        else:
            instance["FAIL_TO_PASS"] = f2p
            instance["PASS_TO_PASS"] = p2p

        # Also store metadata about the evaluation
        instance["_eval_status"] = r.get("status", "unknown")
        instance["_eval_error"] = r.get("error_message", "")

        matched += 1

    logger.info("Merged results for %d / %d instances.", matched, len(dataset))
    return dataset


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(results):
    # type: (Dict[str, Dict[str, Any]]) -> None
    """Print summary statistics for all collected results."""

    total = len(results)
    statuses = {}  # type: Dict[str, int]
    total_f2p = 0
    total_p2p = 0
    instances_with_f2p = 0
    instances_with_p2p = 0
    regressions_count = 0
    both_failed_count = 0

    for iid, r in sorted(results.items()):
        status = r.get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1

        f2p = r.get("FAIL_TO_PASS", [])
        p2p = r.get("PASS_TO_PASS", [])
        total_f2p += len(f2p)
        total_p2p += len(p2p)
        if f2p:
            instances_with_f2p += 1
        if p2p:
            instances_with_p2p += 1

        regressions = r.get("regressions", [])
        regressions_count += len(regressions)
        both_failed = r.get("both_failed", [])
        both_failed_count += len(both_failed)

    print()
    print("=" * 72)
    print("COLLECTION SUMMARY")
    print("=" * 72)
    print()
    print("Total instances with results:     {}".format(total))
    print()
    print("Status breakdown:")
    for status, count in sorted(statuses.items()):
        pct = 100.0 * count / total if total else 0
        print("  {:<15s} {:>5d}  ({:.1f}%)".format(status, count, pct))
    print()
    print("FAIL_TO_PASS:")
    print("  Total test transitions:         {}".format(total_f2p))
    print("  Instances with >= 1 F2P:        {} / {}  ({:.1f}%)".format(
        instances_with_f2p, total,
        100.0 * instances_with_f2p / total if total else 0))
    print()
    print("PASS_TO_PASS:")
    print("  Total test transitions:         {}".format(total_p2p))
    print("  Instances with >= 1 P2P:        {} / {}  ({:.1f}%)".format(
        instances_with_p2p, total,
        100.0 * instances_with_p2p / total if total else 0))
    print()
    if regressions_count:
        print("WARNING: {} regressions detected (tests that passed before fix but failed after)".format(
            regressions_count))
    if both_failed_count:
        print("NOTE: {} tests failed in BOTH phases (potential environment issues)".format(
            both_failed_count))
    print()

    # Per-instance detail table
    print("{:<50s} {:>6s} {:>6s} {:>6s} {:>6s} {:>8s}".format(
        "Instance", "F2P", "P2P", "Regr", "BothF", "Status"))
    print("-" * 88)
    for iid, r in sorted(results.items()):
        f2p = len(r.get("FAIL_TO_PASS", []))
        p2p = len(r.get("PASS_TO_PASS", []))
        regr = len(r.get("regressions", []))
        bf = len(r.get("both_failed", []))
        status = r.get("status", "unknown")
        print("{:<50s} {:>6d} {:>6d} {:>6d} {:>6d} {:>8s}".format(
            iid[:50], f2p, p2p, regr, bf, status))
    print("-" * 88)
    print("{:<50s} {:>6d} {:>6d} {:>6d} {:>6d}".format(
        "TOTAL", total_f2p, total_p2p, regressions_count, both_failed_count))
    print("=" * 72)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_results(dataset, results):
    # type: (List[Dict[str, Any]], Dict[str, Dict[str, Any]]) -> None
    """Print warnings for potential issues in the collected results."""

    dataset_ids = set(inst["instance_id"] for inst in dataset)
    result_ids = set(results.keys())

    missing = dataset_ids - result_ids
    extra = result_ids - dataset_ids

    if missing:
        print("\nWARNING: {} instance(s) in dataset have no results:".format(len(missing)))
        for iid in sorted(missing)[:20]:
            print("  - {}".format(iid))
        if len(missing) > 20:
            print("  ... and {} more".format(len(missing) - 20))

    if extra:
        print("\nWARNING: {} result(s) not found in dataset:".format(len(extra)))
        for iid in sorted(extra)[:10]:
            print("  - {}".format(iid))

    # Check for instances with no FAIL_TO_PASS (might indicate broken test setup)
    no_f2p = []
    for iid, r in results.items():
        if r.get("status") == "success" and not r.get("FAIL_TO_PASS"):
            no_f2p.append(iid)

    if no_f2p:
        print("\nNOTE: {} successful instance(s) have ZERO FAIL_TO_PASS tests:".format(len(no_f2p)))
        for iid in sorted(no_f2p)[:10]:
            print("  - {}".format(iid))
        if len(no_f2p) > 10:
            print("  ... and {} more".format(len(no_f2p) - 10))
        print("  These may have tests that do not actually exercise the bug,")
        print("  or the test framework setup failed silently.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    # type: () -> argparse.ArgumentParser
    parser = argparse.ArgumentParser(
        description="Collect per-instance test results and merge them into the dataset JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect results and write an updated dataset
  %(prog)s --results-dir ./results --dataset ../vllm_infra_gym.json

  # Write to a separate output file (don't modify original)
  %(prog)s --output ../vllm_infra_gym_evaluated.json

  # Summary only, no merge
  %(prog)s --results-dir ./results --summary-only
""",
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        help="Directory containing per-instance result JSON files. (default: ./results/)",
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm_infra_gym.json"),
        help="Path to the original infra-gym dataset JSON. (default: ../vllm_infra_gym.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the merged dataset JSON. "
             "Defaults to the same path as --dataset (overwrites in place). "
             "Use a different path to avoid modifying the original.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        default=False,
        help="Only print summary statistics; do not merge into dataset.",
    )
    parser.add_argument(
        "--no-swebench-format",
        action="store_true",
        default=False,
        help="Store FAIL_TO_PASS and PASS_TO_PASS as plain lists instead of "
             "JSON-encoded strings (non-SWE-bench format).",
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

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=LOG_FORMAT)

    results_dir = os.path.abspath(args.results_dir)
    dataset_path = os.path.abspath(args.dataset)
    output_path = os.path.abspath(args.output) if args.output else dataset_path

    # Load results
    results = load_result_files(results_dir)
    if not results:
        logger.error("No results found in %s", results_dir)
        sys.exit(1)

    # Print summary
    print_summary(results)

    # Load dataset for validation and optional merge
    if not os.path.isfile(dataset_path):
        logger.warning("Dataset file not found: %s -- skipping merge and validation.", dataset_path)
        sys.exit(0)

    with open(dataset_path, "r") as fh:
        dataset = json.load(fh)

    validate_results(dataset, results)

    if args.summary_only:
        logger.info("Summary-only mode; not writing output.")
        sys.exit(0)

    # Merge
    swebench_format = not args.no_swebench_format
    merge_results_into_dataset(dataset, results, swebench_format=swebench_format)

    # Write output
    logger.info("Writing merged dataset to %s", output_path)
    with open(output_path, "w") as fh:
        json.dump(dataset, fh, indent=2, ensure_ascii=False)

    logger.info("Done. %d instances in output file.", len(dataset))

    # Final tally
    merged_count = sum(
        1 for inst in dataset
        if "FAIL_TO_PASS" in inst and inst["FAIL_TO_PASS"]
    )
    print("\nWrote {} instances to {}".format(len(dataset), output_path))
    print("{} instances have FAIL_TO_PASS data.".format(merged_count))


if __name__ == "__main__":
    main()
