#!/usr/bin/env python3
"""
Build version mapping for vLLM instances.

For each instance in vllm_infra_gym.json, determine the vLLM version
based on the instance's created_at date and the vLLM release timeline.

Strategy:
- Fetch all vLLM releases from GitHub API (using curl to avoid Python SSL issues)
- For each instance, find the most recent release tag whose publication date
  is BEFORE the instance's created_at (PR creation date)
- Write the version field into each instance
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RELEASES_CACHE = os.path.join(SCRIPT_DIR, "releases_page1.json")


def fetch_github_releases_via_curl(repo, cache_path):
    # type: (str, str) -> List[Dict]
    """Fetch all releases from GitHub using curl (avoids Python SSL issues)."""
    all_releases = []  # type: List[Dict]

    for page in range(1, 5):
        url = "https://api.github.com/repos/{}/releases?per_page=100&page={}".format(repo, page)
        print("  Fetching releases page {} via curl...".format(page))

        result = subprocess.run(
            [
                "curl", "--connect-timeout", "10", "--max-time", "120",
                "-s", "-H", "Accept: application/vnd.github.v3+json",
                url
            ],
            capture_output=True, text=True, timeout=180
        )

        if result.returncode != 0:
            print("    curl failed with return code {}".format(result.returncode))
            break

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            print("    Failed to parse JSON response")
            break

        if isinstance(data, dict) and "message" in data:
            print("    GitHub API error: {}".format(data["message"]))
            break

        if not data:
            break

        all_releases.extend(data)
        print("    Got {} releases on this page".format(len(data)))

        if len(data) < 100:
            break

    # Cache the results
    if all_releases:
        with open(cache_path, "w") as f:
            json.dump(all_releases, f)
        print("  Cached {} releases to {}".format(len(all_releases), cache_path))

    return all_releases


def load_releases(repo):
    # type: (str) -> List[Dict]
    """Load releases, using cache if available, otherwise fetching."""
    if os.path.exists(RELEASES_CACHE):
        print("  Loading cached releases from {}".format(RELEASES_CACHE))
        with open(RELEASES_CACHE, "r") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            print("  Loaded {} cached releases".format(len(data)))
            return data

    return fetch_github_releases_via_curl(repo, RELEASES_CACHE)


def parse_version_from_tag(tag_name):
    # type: (str) -> Optional[str]
    """Parse a version string from a tag name like 'v0.3.0' -> '0.3.0'."""
    tag = tag_name.strip()
    if tag.startswith("v"):
        tag = tag[1:]
    # Filter out non-version tags
    parts = tag.split(".")
    if len(parts) < 2:
        return None
    try:
        int(parts[0])
        int(parts[1])
    except ValueError:
        return None
    return tag


def build_release_timeline(releases):
    # type: (List[Dict]) -> List[Tuple[str, str, str, bool]]
    """
    Build a sorted timeline of releases.

    Returns list of (published_date_str, version, tag_name, is_prerelease)
    sorted by date ascending.
    """
    timeline = []
    for rel in releases:
        tag_name = rel.get("tag_name", "")
        version = parse_version_from_tag(tag_name)
        if version is None:
            continue

        # Use published_at date
        published_at = rel.get("published_at") or rel.get("created_at")
        if not published_at:
            continue

        is_prerelease = rel.get("prerelease", False)
        is_draft = rel.get("draft", False)

        if is_draft:
            continue

        # Skip RC/prerelease tags for version assignment
        # (we only want stable releases to define the version)
        if is_prerelease:
            continue

        # Also skip versions with "rc" in them
        if "rc" in version.lower():
            continue

        timeline.append((published_at, version, tag_name, is_prerelease))

    # Sort by date ascending
    timeline.sort(key=lambda x: x[0])
    return timeline


def find_version_for_date(created_at, timeline):
    # type: (str, List[Tuple[str, str, str, bool]]) -> str
    """
    Find the most recent release version that was published BEFORE or ON created_at.

    Args:
        created_at: ISO date string of the PR creation
        timeline: sorted list of (published_at, version, tag_name, is_prerelease)

    Returns:
        version string
    """
    # Parse the created_at date
    if "T" in created_at:
        created_dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
    else:
        created_dt = datetime.strptime(created_at, "%Y-%m-%d")

    best_version = None  # type: Optional[str]

    for published_at, version, tag_name, is_prerelease in timeline:
        if "T" in published_at:
            pub_dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
        else:
            pub_dt = datetime.strptime(published_at, "%Y-%m-%d")

        if pub_dt <= created_dt:
            best_version = version
        else:
            break

    if best_version is None:
        best_version = "0.1"

    return best_version


def version_sort_key(v):
    # type: (str) -> List
    """Generate a sort key for version strings like '0.3.0', '0.4.1.post1'."""
    parts = []
    for segment in v.split("."):
        # Split "4post1" into numeric and non-numeric parts
        num = ""
        rest = ""
        for ch in segment:
            if ch.isdigit() and not rest:
                num += ch
            else:
                rest += ch
        if num:
            parts.append(int(num))
        if rest:
            parts.append(rest)
    return parts


def main():
    # type: () -> None
    json_path = os.path.join(REPO_ROOT, "vllm_infra_gym.json")

    # Load the JSON
    print("Loading {}...".format(json_path))
    with open(json_path, "r") as f:
        instances = json.load(f)
    print("Loaded {} instances".format(len(instances)))

    # Collect unique base_commits and created_at dates
    base_commits = set()
    dates = set()
    for inst in instances:
        bc = inst.get("base_commit", "")
        ca = inst.get("created_at", "")
        if bc:
            base_commits.add(bc)
        if ca:
            dates.add(ca)

    print("Found {} unique base_commits".format(len(base_commits)))
    print("Date range: {} to {}".format(min(dates), max(dates)))

    # Load releases
    print("\nLoading vLLM releases...")
    releases = load_releases("vllm-project/vllm")
    print("Got {} releases".format(len(releases)))

    # Build release timeline (stable releases only)
    timeline = build_release_timeline(releases)
    print("\nStable release timeline ({} entries):".format(len(timeline)))
    for pub_at, version, tag, is_pre in timeline:
        print("  {} -> {}".format(pub_at[:10], version))

    # Map each instance to a version
    print("\nMapping instances to versions...")
    version_counts = {}  # type: Dict[str, int]

    for inst in instances:
        created_at = inst.get("created_at", "")
        if not created_at:
            print("  WARNING: No created_at for {}".format(inst.get("instance_id", "?")))
            inst["version"] = "unknown"
            continue

        version = find_version_for_date(created_at, timeline)
        inst["version"] = version

        version_counts[version] = version_counts.get(version, 0) + 1

    # Save the updated JSON
    print("\nSaving updated JSON...")
    with open(json_path, "w") as f:
        json.dump(instances, f, indent=2)
    print("Saved to {}".format(json_path))

    # Print verification
    print("\n=== Version Distribution ===")
    sorted_versions = sorted(version_counts.keys(), key=version_sort_key)
    for version in sorted_versions:
        count = version_counts[version]
        bar = "#" * count
        print("  v{:20s} {:3d} instances  {}".format(version, count, bar))

    print("\nTotal: {} instances".format(sum(version_counts.values())))

    # Show a few examples
    print("\n=== Sample Mappings ===")
    for inst in instances[:10]:
        print("  {} -> v{} (created: {})".format(
            inst["instance_id"],
            inst["version"],
            inst.get("created_at", "?")[:10]
        ))


if __name__ == "__main__":
    main()
