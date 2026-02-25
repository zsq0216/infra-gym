#!/usr/bin/env python3
"""
Supplement dataset_with_tests.csv by searching for PRs for issues marked "No PR Found".
Uses GitHub GraphQL search API with post-filtering for exact #N references.
"""

import csv
import json
import os
import re
import time
import urllib.request
import urllib.error

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

TOKEN = os.environ["GITHUB_TOKEN"]

TEST_PATTERNS = [
    r'test[s]?/',
    r'test_',
    r'_test\.',
    r'_test_',
    r'\.test\.',
    r'\.spec\.',
    r'__tests__/',
    r'testing/',
    r'spec[s]?/',
]


def graphql_query(query):
    for attempt in range(3):
        data = json.dumps({"query": query}).encode()
        req = urllib.request.Request("https://api.github.com/graphql", data=data, headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        })
        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode())
                if "errors" in result and not result.get("data"):
                    print(f"  GraphQL errors: {result['errors'][:1]}", flush=True)
                return result.get("data", {})
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            if e.code == 403:
                print("  Rate limited, waiting 60s...", flush=True)
                time.sleep(60)
                continue
            elif e.code in (502, 503):
                time.sleep(5)
                continue
            else:
                print(f"  HTTP {e.code}: {body[:200]}", flush=True)
                if attempt < 2:
                    time.sleep(5)
                    continue
                return {}
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
                continue
            return {}
    return {}


def rest_api_get(url):
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
                data = json.loads(resp.read().decode())
                link_header = resp.headers.get("Link", "")
                return data, link_header
        except urllib.error.HTTPError as e:
            if e.code in (404, 410, 422, 409):
                return None, ""
            elif e.code == 403:
                time.sleep(60)
                continue
            else:
                if attempt < 2:
                    time.sleep(5)
                    continue
                return None, ""
        except Exception:
            if attempt < 2:
                time.sleep(5)
                continue
            return None, ""
    return None, ""


def batch_search_prs(issues_batch):
    """
    Search for PRs referencing each issue using GraphQL batched search.
    issues_batch: list of (owner, repo, issue_number, original_url)
    Returns: dict of original_url -> list of PR dicts
    """
    if not issues_batch:
        return {}

    fragments = []
    for idx, (owner, repo, issue_num, _) in enumerate(issues_batch):
        alias = f"s{idx}"
        # Search for PRs that mention the issue number
        search_q = f"repo:{owner}/{repo} is:pr {issue_num}"
        fragments.append(f'''
    {alias}: search(query: "{search_q}", type: ISSUE, first: 15) {{
      nodes {{
        ... on PullRequest {{
          number
          title
          url
          body
        }}
      }}
    }}''')

    query = "query {\n" + "\n".join(fragments) + "\n}"
    data = graphql_query(query)

    results = {}
    for idx, (owner, repo, issue_num, orig_url) in enumerate(issues_batch):
        alias = f"s{idx}"
        search_data = data.get(alias)
        if not search_data:
            results[orig_url] = []
            continue

        nodes = search_data.get("nodes", [])
        # Filter: PR must reference #N in title or body
        issue_ref = f"#{issue_num}"
        issue_url_ref = f"issues/{issue_num}"
        matched_prs = []

        for node in nodes:
            if not node or not node.get("number"):
                continue
            title = node.get("title", "") or ""
            body = node.get("body", "") or ""
            text = title + " " + body

            # Check for exact issue reference
            if issue_ref in text or issue_url_ref in text:
                matched_prs.append({
                    "number": node["number"],
                    "url": node["url"],
                    "title": node.get("title", ""),
                })

        results[orig_url] = matched_prs

    return results


def is_test_file(filename):
    fn_lower = filename.lower()
    for pattern in TEST_PATTERNS:
        if re.search(pattern, fn_lower):
            return True
    return False


def check_pr_for_tests(pr_url):
    match = re.match(r'https://github\.com/([^/]+)/([^/]+)/pull/(\d+)', pr_url)
    if not match:
        return []
    owner, repo, pr_num = match.group(1), match.group(2), match.group(3)

    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}/files?per_page=100"
    all_files = []
    max_pages = 5
    page = 0
    while url and page < max_pages:
        data, link_header = rest_api_get(url)
        if not data:
            break
        all_files.extend(data)
        url = None
        if link_header:
            for part in link_header.split(","):
                if 'rel="next"' in part:
                    url = part.split("<")[1].split(">")[0]
        page += 1

    test_files = []
    for f in all_files:
        if not isinstance(f, dict):
            continue
        filename = f.get("filename", "")
        if is_test_file(filename):
            blob_url = f.get("blob_url", "")
            test_files.append(blob_url if blob_url else filename)
    return test_files


def parse_issue_url(url):
    match = re.match(r'https://github\.com/([^/]+)/([^/]+)/issues/(\d+)', url)
    if match:
        return match.group(1), match.group(2), int(match.group(3))
    return None, None, None


def main():
    csv_file = os.path.join(DATA_DIR, "dataset_with_tests.csv")

    # Read current results
    rows = []
    with open(csv_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            rows.append(row)

    # Find "No PR Found" entries
    no_pr_indices = []
    for i, row in enumerate(rows):
        if row["Has_Tests"] == "No PR Found":
            owner, repo, issue_num = parse_issue_url(row["URL"])
            if owner:
                no_pr_indices.append((i, owner, repo, issue_num, row["URL"]))

    print(f"Total 'No PR Found' issues to search: {len(no_pr_indices)}", flush=True)

    # Phase 1: Batch search for PRs
    BATCH_SIZE = 10  # Smaller batches to avoid GraphQL complexity limits
    url_to_prs = {}

    batch_items = [(owner, repo, issue_num, url) for (_, owner, repo, issue_num, url) in no_pr_indices]

    for batch_start in range(0, len(batch_items), BATCH_SIZE):
        batch = batch_items[batch_start:batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, len(batch_items))
        print(f"Search batch {batch_start+1}-{batch_end}/{len(batch_items)}", flush=True)

        result = batch_search_prs(batch)
        url_to_prs.update(result)
        time.sleep(1)  # Be gentle with search API

    # Count how many found PRs
    found_count = sum(1 for prs in url_to_prs.values() if prs)
    print(f"\nFound PRs for {found_count}/{len(no_pr_indices)} previously-empty issues", flush=True)

    # Phase 2: Check new PRs for tests
    pr_test_cache = {}
    prs_to_check = set()
    for prs in url_to_prs.values():
        for pr in prs:
            prs_to_check.add(pr["url"])

    print(f"Unique new PRs to check for tests: {len(prs_to_check)}", flush=True)

    for idx, pr_url in enumerate(sorted(prs_to_check)):
        if (idx + 1) % 20 == 0:
            print(f"  Checking PR files {idx+1}/{len(prs_to_check)}", flush=True)
        pr_test_cache[pr_url] = check_pr_for_tests(pr_url)

    # Phase 3: Update rows
    updated_count = 0
    for i, owner, repo, issue_num, url in no_pr_indices:
        prs = url_to_prs.get(url, [])
        if not prs:
            continue

        pr_urls = [pr["url"] for pr in prs]
        all_test_urls = []
        for pr in prs:
            test_urls = pr_test_cache.get(pr["url"], [])
            all_test_urls.extend(test_urls)

        rows[i]["PR_URLs"] = " | ".join(pr_urls)
        rows[i]["Has_Tests"] = "Yes" if all_test_urls else "No"
        rows[i]["Test_URLs"] = " | ".join(all_test_urls)
        updated_count += 1

    print(f"\nUpdated {updated_count} rows", flush=True)

    # Write back
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Print summary
    from collections import defaultdict
    stats = defaultdict(lambda: defaultdict(int))
    for row in rows:
        stats[row["repo"]][row["Has_Tests"]] += 1
        stats[row["repo"]]["total"] += 1

    print(f"\n=== Updated Summary ===", flush=True)
    print(f"{'Repo':<25} {'Total':>6} {'Has PR':>8} {'With Tests':>12} {'No Tests':>10} {'No PR':>8}", flush=True)
    print("-" * 75, flush=True)
    for repo in sorted(stats.keys()):
        s = stats[repo]
        has_pr = s.get("Yes", 0) + s.get("No", 0)
        print(f"{repo:<25} {s['total']:>6} {has_pr:>8} {s.get('Yes',0):>12} {s.get('No',0):>10} {s.get('No PR Found',0):>8}", flush=True)
    print("-" * 75, flush=True)
    total = sum(s["total"] for s in stats.values())
    yes = sum(s.get("Yes", 0) for s in stats.values())
    no = sum(s.get("No", 0) for s in stats.values())
    no_pr = sum(s.get("No PR Found", 0) for s in stats.values())
    print(f"{'TOTAL':<25} {total:>6} {yes+no:>8} {yes:>12} {no:>10} {no_pr:>8}", flush=True)
    print(f"\nResults saved to {csv_file}", flush=True)


if __name__ == "__main__":
    main()
