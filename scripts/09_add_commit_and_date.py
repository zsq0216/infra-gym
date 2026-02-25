#!/usr/bin/env python3
"""Add base_commit and created_at (PR creation time) to each instance."""

import json
import os
import time
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(REPO_ROOT, "vllm_infra_gym.json")
REPO = "vllm-project/vllm"


def fetch_pr_created_at(pr_number):
    """Fetch PR creation timestamp from GitHub API."""
    url = f"https://api.github.com/repos/{REPO}/pulls/{pr_number}"
    req = urllib.request.Request(url, headers={"User-Agent": "Python"})
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read())
    remaining = int(resp.headers.get("X-RateLimit-Remaining", 0))
    return data["created_at"], remaining


def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    print(f"Processing {len(data)} instances...")

    # 1. base_commit: simple mapping from fix.base_sha
    for inst in data:
        inst["base_commit"] = inst["fix"]["base_sha"]

    # 2. created_at: fetch PR creation time from GitHub API
    # Cache by PR number since multiple issues can share a PR
    pr_cache = {}
    unique_prs = sorted(set(d["fix"]["pr_number"] for d in data))
    print(f"Fetching created_at for {len(unique_prs)} unique PRs...")

    for i, pr_num in enumerate(unique_prs):
        try:
            created_at, remaining = fetch_pr_created_at(pr_num)
            pr_cache[pr_num] = created_at
            print(f"  [{i+1}/{len(unique_prs)}] PR #{pr_num}: {created_at} (rate limit: {remaining})")
            # Be polite with rate limiting
            if remaining < 10:
                print("  Rate limit low, sleeping 60s...")
                time.sleep(60)
            elif remaining < 30:
                time.sleep(2)
            else:
                time.sleep(0.5)
        except Exception as e:
            print(f"  ERROR fetching PR #{pr_num}: {e}")
            pr_cache[pr_num] = None

    # Apply to instances
    for inst in data:
        pr_num = inst["fix"]["pr_number"]
        inst["created_at"] = pr_cache.get(pr_num, "")

    # Write output
    with open(INPUT_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Verification
    print("\n--- Verification ---")
    all_base = all(d.get("base_commit") for d in data)
    all_created = all(d.get("created_at") for d in data)
    print(f"All have base_commit: {all_base}")
    print(f"All have created_at: {all_created}")

    print("\nSamples:")
    for d in data[:5]:
        print(f"  {d['instance_id']}")
        print(f"    base_commit: {d['base_commit']}")
        print(f"    created_at:  {d['created_at']}")


if __name__ == "__main__":
    main()
