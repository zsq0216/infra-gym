#!/usr/bin/env python3
"""Extract git-format test_patch and patch from full_diff for each instance."""

import json
import os
import re
import sys
from typing import List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(REPO_ROOT, "vllm_infra_gym.json")


def split_diff_by_file(full_diff):
    """Split a full git diff into per-file diff blocks.

    Each block starts with 'diff --git a/... b/...' and includes
    all content until the next 'diff --git' or end of string.
    """
    # Split on 'diff --git' boundaries, keeping the delimiter
    parts = re.split(r'(?=^diff --git )', full_diff, flags=re.MULTILINE)
    # Filter out empty parts
    return [p for p in parts if p.strip()]


def extract_filename_from_diff_block(block):
    """Extract the b/ filename from a 'diff --git a/X b/Y' header.

    Handles filenames with spaces by looking for the '\\n' line boundary
    and extracting from the 'b/' portion.
    """
    # First try: find the --- a/ and +++ b/ lines which are unambiguous
    m = re.search(r'^\+\+\+ b/(.+)$', block, flags=re.MULTILINE)
    if m:
        return m.group(1).rstrip()
    # Fallback: parse the diff --git header (works for paths without spaces)
    m = re.match(r'diff --git a/\S+ b/(\S+)', block)
    return m.group(1) if m else None


def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    print(f"Processing {len(data)} instances...")

    for i, inst in enumerate(data):
        instance_id = inst["instance_id"]
        full_diff = inst["fix"]["full_diff"]
        test_filenames = {tf["filename"] for tf in inst["tests"]["test_files"]}

        # Split full_diff into per-file blocks
        blocks = split_diff_by_file(full_diff)

        test_blocks = []
        source_blocks = []

        for block in blocks:
            fname = extract_filename_from_diff_block(block)
            if fname is None:
                print(f"  WARNING: could not parse filename in block for {instance_id}")
                continue
            if fname in test_filenames:
                test_blocks.append(block)
            else:
                source_blocks.append(block)

        test_patch = "".join(test_blocks)
        patch = "".join(source_blocks)

        # Write into the instance
        inst["tests"]["test_patch"] = test_patch
        inst["fix"]["patch"] = patch

        # Sanity check: all blocks accounted for (no data lost)
        all_classified = set(test_blocks) | set(source_blocks)
        if len(all_classified) != len(blocks):
            print(f"  WARNING [{instance_id}]: {len(blocks)} blocks but only {len(all_classified)} classified")

    # Write output
    with open(INPUT_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Verification
    print("\n--- Verification ---")
    all_have_test_patch = all("test_patch" in inst["tests"] for inst in data)
    all_have_patch = all("patch" in inst["fix"] for inst in data)
    print(f"All {len(data)} instances have test_patch: {all_have_test_patch}")
    print(f"All {len(data)} instances have patch: {all_have_patch}")

    # Check test_patch starts with 'diff --git'
    empty_test_patches = 0
    bad_test_patches = 0
    for inst in data:
        tp = inst["tests"]["test_patch"]
        if not tp:
            empty_test_patches += 1
        elif not tp.startswith("diff --git"):
            bad_test_patches += 1

    print(f"Empty test_patch count: {empty_test_patches}")
    print(f"test_patch NOT starting with 'diff --git': {bad_test_patches}")

    # Sample a few
    print("\n--- Samples ---")
    for inst in data[:3]:
        tp = inst["tests"]["test_patch"]
        p = inst["fix"]["patch"]
        print(f"\n{inst['instance_id']}:")
        print(f"  test_patch length: {len(tp)}, starts: {tp[:60]!r}...")
        print(f"  patch length: {len(p)}, starts: {p[:60]!r}...")


if __name__ == "__main__":
    main()
