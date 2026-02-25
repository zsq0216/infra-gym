#!/usr/bin/env python3
"""
Build the structured infra-gym dataset for vllm MVP.
Combines issue details, PR details, original CSV metadata into a single JSON.
"""

import csv
import json
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")


def load_csv_metadata():
    """Load original CSV data for vllm test entries."""
    meta = {}
    with open(os.path.join(DATA_DIR, "dataset_with_tests.csv"), encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["repo"] == "vllm" and row["Has_Tests"] == "Yes":
                meta[row["URL"]] = {
                    "phase": row["Phase"],
                    "symptoms": row["Symptom(s)"],
                    "component": row["Component"],
                    "root_causes": row["Root Cause(s)"],
                }
    return meta


def classify_test_files(changed_files):
    """Separate test files from source files."""
    test_files = []
    source_files = []
    for f in changed_files:
        if f["is_test"]:
            test_files.append(f)
        else:
            source_files.append(f)
    return source_files, test_files


def compute_difficulty(source_files, diff):
    """Estimate difficulty based on patch size and file count."""
    total_additions = sum(f["additions"] for f in source_files)
    total_deletions = sum(f["deletions"] for f in source_files)
    total_changes = total_additions + total_deletions
    num_files = len(source_files)

    if num_files <= 1 and total_changes <= 10:
        return "easy"
    elif num_files <= 3 and total_changes <= 50:
        return "medium"
    else:
        return "hard"


def build_dataset():
    # Load all data
    with open(os.path.join(DATA_DIR, "vllm_test_entries.json")) as f:
        entries = json.load(f)

    with open(os.path.join(DATA_DIR, "vllm_issue_details.json")) as f:
        issue_details = json.load(f)

    with open(os.path.join(DATA_DIR, "vllm_pr_details.json")) as f:
        pr_details = json.load(f)

    csv_meta = load_csv_metadata()

    dataset = []
    stats = {"easy": 0, "medium": 0, "hard": 0}

    for entry in entries:
        issue_url = entry["issue_url"]
        issue = issue_details.get(issue_url)
        prs = pr_details.get(issue_url, [])
        meta = csv_meta.get(issue_url, {})

        if not issue or not prs:
            continue

        # Use the first (most relevant) PR as the primary fix
        # But keep all PRs for reference
        for pr in prs:
            source_files, test_files = classify_test_files(pr["changed_files"])

            if not test_files:
                continue  # Skip PRs without tests in this dataset

            # Build source-only diff (exclude test file patches)
            source_diff_parts = []
            for sf in source_files:
                if sf.get("patch"):
                    source_diff_parts.append(f"--- a/{sf['filename']}\n+++ b/{sf['filename']}\n{sf['patch']}")
            source_diff = "\n".join(source_diff_parts)

            # Build test-only diff
            test_diff_parts = []
            for tf in test_files:
                if tf.get("patch"):
                    test_diff_parts.append(f"--- a/{tf['filename']}\n+++ b/{tf['filename']}\n{tf['patch']}")
            test_diff = "\n".join(test_diff_parts)

            difficulty = compute_difficulty(source_files, source_diff)
            stats[difficulty] += 1

            task = {
                # Identifiers
                "instance_id": f"vllm__{issue['issue_number']}__{pr['pr_number']}",
                "repo": "vllm-project/vllm",

                # Issue info (this is what the model sees as input)
                "issue": {
                    "number": issue["issue_number"],
                    "url": issue["issue_url"],
                    "title": issue["title"],
                    "body": issue["body"],
                    "labels": issue["labels"],
                    "author": issue["author"],
                    "created_at": issue["created_at"],
                    "comments": issue["comments"],
                },

                # Bug classification (from original dataset)
                "classification": {
                    "phase": meta.get("phase", ""),
                    "symptoms": meta.get("symptoms", ""),
                    "component": meta.get("component", ""),
                    "root_causes": meta.get("root_causes", ""),
                },

                # Ground truth fix (PR info)
                "fix": {
                    "pr_number": pr["pr_number"],
                    "pr_url": pr["pr_url"],
                    "pr_title": pr["pr_title"],
                    "pr_body": pr["pr_body"],
                    "merged": pr["merged"],
                    "base_sha": pr["base_sha"],
                    "base_ref": pr["base_ref"],
                    "head_sha": pr["head_sha"],
                    "merge_commit_sha": pr["merge_commit_sha"],
                    "source_diff": source_diff,
                    "full_diff": pr["diff"],
                    "source_files": [
                        {
                            "filename": f["filename"],
                            "status": f["status"],
                            "additions": f["additions"],
                            "deletions": f["deletions"],
                        }
                        for f in source_files
                    ],
                },

                # Test info (for evaluation)
                "tests": {
                    "test_diff": test_diff,
                    "test_files": [
                        {
                            "filename": f["filename"],
                            "status": f["status"],
                            "additions": f["additions"],
                            "deletions": f["deletions"],
                            "blob_url": f["blob_url"],
                        }
                        for f in test_files
                    ],
                },

                # Metadata
                "difficulty": difficulty,
                "num_source_files_changed": len(source_files),
                "num_test_files_changed": len(test_files),
                "total_source_additions": sum(f["additions"] for f in source_files),
                "total_source_deletions": sum(f["deletions"] for f in source_files),
            }

            dataset.append(task)

    # Sort by instance_id
    dataset.sort(key=lambda x: x["instance_id"])

    # Remove duplicates (same issue+PR)
    seen = set()
    deduped = []
    for task in dataset:
        if task["instance_id"] not in seen:
            seen.add(task["instance_id"])
            deduped.append(task)
    dataset = deduped

    # Save
    output_path = os.path.join(REPO_ROOT, "vllm_infra_gym.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    # Also save a summary CSV
    summary_path = os.path.join(DATA_DIR, "vllm_infra_gym_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "instance_id", "issue_url", "pr_url", "issue_title",
            "difficulty", "phase", "symptoms", "component", "root_causes",
            "num_source_files", "source_additions", "source_deletions",
            "num_test_files", "base_sha",
        ])
        writer.writeheader()
        for task in dataset:
            writer.writerow({
                "instance_id": task["instance_id"],
                "issue_url": task["issue"]["url"],
                "pr_url": task["fix"]["pr_url"],
                "issue_title": task["issue"]["title"],
                "difficulty": task["difficulty"],
                "phase": task["classification"]["phase"],
                "symptoms": task["classification"]["symptoms"],
                "component": task["classification"]["component"],
                "root_causes": task["classification"]["root_causes"],
                "num_source_files": task["num_source_files_changed"],
                "source_additions": task["total_source_additions"],
                "source_deletions": task["total_source_deletions"],
                "num_test_files": task["num_test_files_changed"],
                "base_sha": task["fix"]["base_sha"],
            })

    print(f"=== vllm Infra-Gym MVP Dataset ===", flush=True)
    print(f"Total tasks: {len(dataset)}", flush=True)
    print(f"Difficulty distribution: {stats}", flush=True)
    print(f"", flush=True)

    # More stats
    phases = {}
    components = {}
    for t in dataset:
        p = t["classification"]["phase"]
        c = t["classification"]["component"]
        phases[p] = phases.get(p, 0) + 1
        components[c] = components.get(c, 0) + 1

    print("By phase:", flush=True)
    for k, v in sorted(phases.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}", flush=True)

    print("\nBy component:", flush=True)
    for k, v in sorted(components.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {v}", flush=True)

    print(f"\nSaved to:", flush=True)
    print(f"  {output_path}", flush=True)
    print(f"  {summary_path}", flush=True)


if __name__ == "__main__":
    build_dataset()
