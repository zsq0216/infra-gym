#!/usr/bin/env python3
"""
Fetch full PR metadata for vllm test entries:
- base_sha, merge_sha, diff, changed files, test files, PR description
"""

import json
import os
import re
import time
import urllib.request
import urllib.error

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

TOKEN = os.environ["GITHUB_TOKEN"]


def api_get(url, accept="application/vnd.github+json"):
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    for attempt in range(3):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req) as resp:
                remaining = int(resp.headers.get("X-RateLimit-Remaining", 999))
                if remaining < 100:
                    print(f"  [Rate limit: {remaining}]", flush=True)
                if remaining < 50:
                    time.sleep(60)
                content = resp.read()
                if accept == "application/vnd.github.v3.diff":
                    return content.decode("utf-8", errors="replace")
                return json.loads(content.decode())
        except urllib.error.HTTPError as e:
            if e.code in (404, 410, 422):
                return None
            elif e.code == 403:
                time.sleep(60)
                continue
            else:
                if attempt < 2:
                    time.sleep(5)
                    continue
                return None
        except Exception:
            if attempt < 2:
                time.sleep(5)
                continue
            return None
    return None


def api_get_all_pages(url):
    all_data = []
    while url:
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
                if isinstance(data, list):
                    all_data.extend(data)
                link = resp.headers.get("Link", "")
                url = None
                for part in link.split(","):
                    if 'rel="next"' in part:
                        url = part.split("<")[1].split(">")[0]
        except Exception:
            break
    return all_data


def fetch_pr_details(pr_url):
    """Fetch comprehensive PR details."""
    match = re.match(r'https://github\.com/([^/]+)/([^/]+)/pull/(\d+)', pr_url)
    if not match:
        return None
    owner, repo, pr_num = match.group(1), match.group(2), match.group(3)
    api_base = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}"

    # 1. PR metadata
    pr_data = api_get(api_base)
    if not pr_data:
        return None

    # 2. PR diff
    diff = api_get(api_base, accept="application/vnd.github.v3.diff")

    # 3. Changed files
    files = api_get_all_pages(f"{api_base}/files?per_page=100")

    result = {
        "pr_number": pr_data.get("number"),
        "pr_url": pr_data.get("html_url"),
        "pr_title": pr_data.get("title"),
        "pr_body": pr_data.get("body"),
        "pr_state": pr_data.get("state"),
        "merged": pr_data.get("merged", False),
        "base_sha": pr_data.get("base", {}).get("sha"),
        "base_ref": pr_data.get("base", {}).get("ref"),
        "head_sha": pr_data.get("head", {}).get("sha"),
        "merge_commit_sha": pr_data.get("merge_commit_sha"),
        "diff": diff,
        "changed_files": [],
    }

    test_patterns = [
        r'test[s]?/', r'test_', r'_test\.', r'_test_',
        r'\.test\.', r'\.spec\.', r'__tests__/', r'testing/',
    ]

    for f in files:
        if not isinstance(f, dict):
            continue
        fname = f.get("filename", "")
        is_test = any(re.search(p, fname.lower()) for p in test_patterns)
        result["changed_files"].append({
            "filename": fname,
            "status": f.get("status"),
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
            "patch": f.get("patch", ""),
            "is_test": is_test,
            "blob_url": f.get("blob_url", ""),
        })

    return result


def main():
    with open(os.path.join(DATA_DIR, "vllm_test_entries.json")) as f:
        entries = json.load(f)

    print(f"Fetching PR details for {len(entries)} entries", flush=True)

    results = {}
    for i, entry in enumerate(entries):
        pr_urls_str = entry["pr_urls"]
        pr_urls = [u.strip() for u in pr_urls_str.split("|") if u.strip()]

        print(f"[{i+1}/{len(entries)}] {entry['issue_url']} ({len(pr_urls)} PRs)", flush=True)

        pr_details = []
        for pr_url in pr_urls:
            detail = fetch_pr_details(pr_url)
            if detail:
                pr_details.append(detail)
                print(f"  PR #{detail['pr_number']}: {detail['pr_title'][:60]}", flush=True)
                print(f"    base_sha: {detail['base_sha'][:12] if detail['base_sha'] else 'N/A'}, "
                      f"files: {len(detail['changed_files'])}, "
                      f"tests: {sum(1 for f in detail['changed_files'] if f['is_test'])}", flush=True)

        results[entry["issue_url"]] = pr_details

    with open(os.path.join(DATA_DIR, "vllm_pr_details.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Saved PR details to vllm_pr_details.json", flush=True)
    print(f"Total PRs fetched: {sum(len(v) for v in results.values())}", flush=True)


if __name__ == "__main__":
    main()
