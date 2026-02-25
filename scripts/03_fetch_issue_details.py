#!/usr/bin/env python3
"""
Fetch issue details for vllm test entries:
- title, body, labels, state, comments
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


def api_get(url):
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    for attempt in range(3):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req) as resp:
                remaining = int(resp.headers.get("X-RateLimit-Remaining", 999))
                if remaining < 50:
                    time.sleep(60)
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
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


def fetch_issue_details(issue_url):
    """Fetch issue details including comments."""
    match = re.match(r'https://github\.com/([^/]+)/([^/]+)/issues/(\d+)', issue_url)
    if not match:
        return None
    owner, repo, issue_num = match.group(1), match.group(2), match.group(3)
    api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_num}"

    issue = api_get(api_url)
    if not issue:
        return None

    # Fetch comments (useful for debugging context)
    comments_data = api_get(f"{api_url}/comments?per_page=50")
    comments = []
    if comments_data:
        for c in comments_data:
            if isinstance(c, dict):
                comments.append({
                    "author": c.get("user", {}).get("login", ""),
                    "body": c.get("body", ""),
                    "created_at": c.get("created_at", ""),
                })

    return {
        "issue_number": issue.get("number"),
        "issue_url": issue.get("html_url"),
        "title": issue.get("title"),
        "body": issue.get("body"),
        "state": issue.get("state"),
        "labels": [l.get("name", "") for l in issue.get("labels", [])],
        "created_at": issue.get("created_at"),
        "closed_at": issue.get("closed_at"),
        "author": issue.get("user", {}).get("login", ""),
        "comments_count": issue.get("comments", 0),
        "comments": comments,
    }


def main():
    with open(os.path.join(DATA_DIR, "vllm_test_entries.json")) as f:
        entries = json.load(f)

    print(f"Fetching issue details for {len(entries)} entries", flush=True)

    results = {}
    for i, entry in enumerate(entries):
        issue_url = entry["issue_url"]
        print(f"[{i+1}/{len(entries)}] {issue_url}", flush=True)

        detail = fetch_issue_details(issue_url)
        if detail:
            results[issue_url] = detail
            print(f"  {detail['title'][:70]}", flush=True)
        else:
            print(f"  FAILED to fetch", flush=True)

    with open(os.path.join(DATA_DIR, "vllm_issue_details.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Saved {len(results)} issue details to vllm_issue_details.json", flush=True)


if __name__ == "__main__":
    main()
