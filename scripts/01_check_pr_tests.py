#!/usr/bin/env python3
"""
Read dataset.csv, find linked PRs for each issue via GitHub GraphQL API,
check if PRs contain tests, and save results to dataset_with_tests.csv.

Uses GraphQL batching for efficiency.
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
GRAPHQL_URL = "https://api.github.com/graphql"
REST_HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

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
    """Execute a GraphQL query."""
    for attempt in range(3):
        data = json.dumps({"query": query}).encode()
        req = urllib.request.Request(GRAPHQL_URL, data=data, headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        })
        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode())
                if "errors" in result:
                    print(f"  GraphQL errors: {result['errors'][:2]}", flush=True)
                return result.get("data", {})
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            if e.code == 403 and "rate limit" in body.lower():
                print("  GraphQL rate limited, waiting 60s...", flush=True)
                time.sleep(60)
                continue
            elif e.code == 502:
                print(f"  502 error, retrying...", flush=True)
                time.sleep(5)
                continue
            else:
                print(f"  GraphQL HTTP {e.code}: {body[:200]}", flush=True)
                if attempt < 2:
                    time.sleep(5)
                    continue
                return {}
        except Exception as e:
            print(f"  GraphQL error: {e}", flush=True)
            if attempt < 2:
                time.sleep(5)
                continue
            return {}
    return {}


def batch_find_prs(issues_batch):
    """
    Find PRs for a batch of issues using a single GraphQL query.
    issues_batch: list of (owner, repo, issue_number, original_url)
    Returns: dict of original_url -> list of PR dicts {number, url, title}
    """
    if not issues_batch:
        return {}

    # Build batched query using aliases
    fragments = []
    for idx, (owner, repo, issue_num, _) in enumerate(issues_batch):
        alias = f"issue_{idx}"
        fragments.append(f'''
    {alias}: repository(owner: "{owner}", name: "{repo}") {{
      issue(number: {issue_num}) {{
        number
        state
        timelineItems(itemTypes: [CROSS_REFERENCED_EVENT, CONNECTED_EVENT, CLOSED_EVENT], first: 30) {{
          nodes {{
            __typename
            ... on CrossReferencedEvent {{
              source {{
                __typename
                ... on PullRequest {{
                  number
                  title
                  url
                }}
              }}
            }}
            ... on ConnectedEvent {{
              subject {{
                __typename
                ... on PullRequest {{
                  number
                  title
                  url
                }}
              }}
            }}
            ... on ClosedEvent {{
              closer {{
                __typename
                ... on PullRequest {{
                  number
                  title
                  url
                }}
                ... on Commit {{
                  oid
                  url
                  associatedPullRequests(first: 5) {{
                    nodes {{
                      number
                      title
                      url
                    }}
                  }}
                }}
              }}
            }}
          }}
        }}
      }}
    }}''')

    query = "query {\n" + "\n".join(fragments) + "\n}"
    data = graphql_query(query)

    results = {}
    for idx, (owner, repo, issue_num, orig_url) in enumerate(issues_batch):
        alias = f"issue_{idx}"
        repo_data = data.get(alias)
        if not repo_data or not repo_data.get("issue"):
            results[orig_url] = []
            continue

        issue_data = repo_data["issue"]
        pr_map = {}  # number -> {number, url, title}

        nodes = issue_data.get("timelineItems", {}).get("nodes", [])
        for node in nodes:
            typename = node.get("__typename", "")

            if typename == "ClosedEvent":
                closer = node.get("closer")
                if not closer:
                    continue
                ct = closer.get("__typename", "")
                if ct == "PullRequest":
                    pr_map[closer["number"]] = {
                        "number": closer["number"],
                        "url": closer["url"],
                        "title": closer.get("title", ""),
                    }
                elif ct == "Commit":
                    # Get associated PRs from the commit
                    assoc = closer.get("associatedPullRequests", {}).get("nodes", [])
                    for pr in assoc:
                        if pr and pr.get("number"):
                            pr_map[pr["number"]] = {
                                "number": pr["number"],
                                "url": pr["url"],
                                "title": pr.get("title", ""),
                            }

            elif typename == "CrossReferencedEvent":
                source = node.get("source")
                if source and source.get("__typename") == "PullRequest" and source.get("number"):
                    pr_map[source["number"]] = {
                        "number": source["number"],
                        "url": source["url"],
                        "title": source.get("title", ""),
                    }

            elif typename == "ConnectedEvent":
                subject = node.get("subject")
                if subject and subject.get("__typename") == "PullRequest" and subject.get("number"):
                    pr_map[subject["number"]] = {
                        "number": subject["number"],
                        "url": subject["url"],
                        "title": subject.get("title", ""),
                    }

        results[orig_url] = list(pr_map.values())

    return results


def rest_api_get(url):
    """Make a REST API GET request."""
    for attempt in range(3):
        req = urllib.request.Request(url, headers=REST_HEADERS)
        try:
            with urllib.request.urlopen(req) as resp:
                remaining = int(resp.headers.get("X-RateLimit-Remaining", 999))
                if remaining < 100:
                    print(f"  [REST rate limit: {remaining} remaining]", flush=True)
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


def is_test_file(filename):
    """Check if a filename looks like a test file."""
    fn_lower = filename.lower()
    for pattern in TEST_PATTERNS:
        if re.search(pattern, fn_lower):
            return True
    return False


def check_pr_for_tests(pr_url):
    """Check if a PR contains test file changes. Returns list of test file blob URLs."""
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
    """Extract owner, repo, issue_number from a GitHub issue URL."""
    match = re.match(r'https://github\.com/([^/]+)/([^/]+)/issues/(\d+)', url)
    if match:
        return match.group(1), match.group(2), int(match.group(3))
    return None, None, None


def main():
    input_file = os.path.join(DATA_DIR, "dataset.csv")
    output_file = os.path.join(DATA_DIR, "dataset_with_tests.csv")

    # Read input
    rows = []
    with open(input_file, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            rows.append(row)

    print(f"Total issues: {len(rows)}", flush=True)

    out_fieldnames = fieldnames + ["PR_URLs", "Has_Tests", "Test_URLs"]

    # Load existing progress for resumption
    processed = {}
    try:
        with open(output_file, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed[row["URL"]] = row
        print(f"Resuming: {len(processed)} already processed", flush=True)
    except FileNotFoundError:
        pass

    # Prepare batches of issues that need processing
    to_process = []
    for row in rows:
        url = row["URL"]
        if url not in processed:
            owner, repo, issue_num = parse_issue_url(url)
            if owner:
                to_process.append((owner, repo, issue_num, url))

    print(f"Need to process: {len(to_process)} issues", flush=True)

    # Phase 1: Batch find PRs using GraphQL (batches of 25 to avoid query size limits)
    BATCH_SIZE = 25
    url_to_prs = {}  # url -> list of PR dicts

    for batch_start in range(0, len(to_process), BATCH_SIZE):
        batch = to_process[batch_start:batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, len(to_process))
        print(f"GraphQL batch {batch_start+1}-{batch_end}/{len(to_process)}", flush=True)

        result = batch_find_prs(batch)
        url_to_prs.update(result)

        # Small delay between batches
        time.sleep(0.5)

    # Phase 2: Check PRs for test files (REST API)
    pr_test_cache = {}  # pr_url -> list of test file URLs
    prs_to_check = set()
    for prs in url_to_prs.values():
        for pr in prs:
            prs_to_check.add(pr["url"])

    print(f"\nTotal unique PRs to check for tests: {len(prs_to_check)}", flush=True)

    for idx, pr_url in enumerate(sorted(prs_to_check)):
        if (idx + 1) % 20 == 0:
            print(f"  Checking PR files {idx+1}/{len(prs_to_check)}", flush=True)
        test_files = check_pr_for_tests(pr_url)
        pr_test_cache[pr_url] = test_files

    # Phase 3: Write output
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()

        stats = {"Yes": 0, "No": 0, "No PR Found": 0}

        for row in rows:
            issue_url = row["URL"]

            if issue_url in processed:
                writer.writerow(processed[issue_url])
                v = processed[issue_url].get("Has_Tests", "No PR Found")
                stats[v] = stats.get(v, 0) + 1
                continue

            prs = url_to_prs.get(issue_url, [])
            pr_urls = [pr["url"] for pr in prs]

            all_test_urls = []
            for pr in prs:
                test_urls = pr_test_cache.get(pr["url"], [])
                all_test_urls.extend(test_urls)

            has_tests = "Yes" if all_test_urls else ("No" if prs else "No PR Found")
            stats[has_tests] = stats.get(has_tests, 0) + 1

            row["PR_URLs"] = " | ".join(pr_urls)
            row["Has_Tests"] = has_tests
            row["Test_URLs"] = " | ".join(all_test_urls)
            writer.writerow(row)

    print(f"\n=== Summary ===", flush=True)
    print(f"Total: {len(rows)}", flush=True)
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}", flush=True)
    print(f"\nResults saved to {output_file}", flush=True)


if __name__ == "__main__":
    main()
