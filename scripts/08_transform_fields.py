#!/usr/bin/env python3
"""Transform infra-gym fields to align with SWE-bench format.

1. instance_id: vllm__10324__10164 -> vllm-project__vllm-10324-10164
2. problem_statement: issue.title + issue.body
3. hints_text: concatenated issue.comments (all comments)
"""

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(REPO_ROOT, "vllm_infra_gym.json")


def transform_instance_id(inst):
    """Convert instance_id to SWE-bench style.

    SWE-bench format: {owner}__{repo}-{pr}
    Since we have multiple issues per PR, use: {owner}__{repo}-{issue}-{pr}
    e.g. vllm-project__vllm-10324-10164
    """
    repo = inst["repo"]  # "vllm-project/vllm"
    owner, repo_name = repo.split("/")
    issue_num = inst["issue"]["number"]
    pr_num = inst["fix"]["pr_number"]
    return f"{owner}__{repo_name}-{issue_num}-{pr_num}"


def build_problem_statement(inst):
    """Concatenate issue title and body, matching SWE-bench format."""
    title = inst["issue"]["title"]
    body = inst["issue"]["body"]
    return f"{title}\n\n{body}"


def build_hints_text(inst):
    """Concatenate issue comments into hints_text.

    SWE-bench uses comments posted before the PR's first commit.
    We don't have the PR creation timestamp, so we include all comments.
    """
    comments = inst["issue"].get("comments", [])
    if not comments:
        return ""
    parts = []
    for c in comments:
        parts.append(f"{c['author']} ({c['created_at']}):\n{c['body']}")
    return "\n\n".join(parts)


def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    print(f"Processing {len(data)} instances...")

    for inst in data:
        old_id = inst["instance_id"]
        new_id = transform_instance_id(inst)
        inst["instance_id"] = new_id
        inst["problem_statement"] = build_problem_statement(inst)
        inst["hints_text"] = build_hints_text(inst)

    # Write output
    with open(INPUT_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Verification
    print("\n--- Verification ---")
    ids = [d["instance_id"] for d in data]
    print(f"All IDs unique: {len(ids) == len(set(ids))}")
    print(f"Sample IDs:")
    for d in data[:5]:
        print(f"  {d['instance_id']}")

    all_have_ps = all(d.get("problem_statement") for d in data)
    print(f"\nAll have non-empty problem_statement: {all_have_ps}")
    print(f"Sample problem_statement lengths: {[len(d['problem_statement']) for d in data[:5]]}")

    have_hints = sum(1 for d in data if d.get("hints_text"))
    print(f"\nInstances with hints_text: {have_hints}/{len(data)}")
    print(f"Sample hints_text (first 100 chars):")
    for d in data[:3]:
        ht = d["hints_text"]
        print(f"  {d['instance_id']}: {ht[:100]!r}{'...' if len(ht) > 100 else ''}")


if __name__ == "__main__":
    main()
