#!/usr/bin/env python3
"""
Extract pytest test case IDs from test_patch diffs in vllm_infra_gym.json.

For each instance, parses the `tests.test_patch` git diff and identifies
which specific test cases (in pytest node ID format) are affected by the patch.

Writes a `tests.test_ids` field into each instance and updates the JSON in-place.
"""

import json
import re
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Diff parsing utilities
# ---------------------------------------------------------------------------

# Matches the file header in a unified diff:
#   diff --git a/foo/bar.py b/foo/bar.py
FILE_HEADER_RE = re.compile(r'^diff --git a/(.+?) b/(.+?)$')

# Matches a hunk header:
#   @@ -766,8 +766,8 @@ def test_resolve_content_format_hf_defined(model, expected_format):
# The optional trailing context is in group(3).
HUNK_HEADER_RE = re.compile(
    r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@(?: (.*))?$'
)

# Matches a top-level function definition (no leading whitespace beyond the +)
# Also handles async def
TOPLEVEL_FUNC_RE = re.compile(r'^(?:async )?def (test_\w+)\s*\(')

# Matches a method definition inside a class (indented)
# Also handles async def
METHOD_FUNC_RE = re.compile(r'^    (?:async )?def (test_\w+)\s*\(')

# Matches a class definition
CLASS_DEF_RE = re.compile(r'^class (\w+)\s*[\(:]')

# Matches a parametrize decorator
PARAMETRIZE_RE = re.compile(r'@pytest\.mark\.parametrize')


def parse_diff_files(patch_text: str) -> List[Dict]:
    """
    Parse a unified diff string into a list of file diffs.
    Each file diff contains:
      - filepath: the b-side path
      - is_new_file: whether the file was newly created
      - hunks: list of hunk dicts with:
          - start_line: new-side start line
          - num_lines: new-side line count
          - context_func: the function/class name from the @@ header context
          - lines: list of (type, content) where type is '+', '-', or ' '
    """
    files = []
    current_file = None
    current_hunk = None
    is_new_file = False

    for raw_line in patch_text.split('\n'):
        # Check for file header
        m = FILE_HEADER_RE.match(raw_line)
        if m:
            if current_file is not None:
                files.append(current_file)
            is_new_file = False
            current_file = {
                'filepath': m.group(2),
                'is_new_file': False,
                'hunks': [],
            }
            current_hunk = None
            continue

        # Detect new file mode
        if raw_line.startswith('new file mode'):
            if current_file is not None:
                current_file['is_new_file'] = True
            continue

        # Check for hunk header
        m = HUNK_HEADER_RE.match(raw_line)
        if m and current_file is not None:
            start_line = int(m.group(1))
            num_lines = int(m.group(2)) if m.group(2) else 1
            context_func = (m.group(3) or '').strip()
            current_hunk = {
                'start_line': start_line,
                'num_lines': num_lines,
                'context_func': context_func,
                'lines': [],
            }
            current_file['hunks'].append(current_hunk)
            continue

        # Parse hunk body lines
        if current_hunk is not None:
            if raw_line.startswith('+'):
                current_hunk['lines'].append(('+', raw_line[1:]))
            elif raw_line.startswith('-'):
                current_hunk['lines'].append(('-', raw_line[1:]))
            elif raw_line.startswith(' '):
                current_hunk['lines'].append((' ', raw_line[1:]))
            # Lines starting with '\' (no newline at end) are metadata; skip.

    if current_file is not None:
        files.append(current_file)

    return files


def extract_context_name(context_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    From a hunk context string (e.g. 'def test_foo(bar):' or 'class TestFoo:'),
    extract a (class_name, func_name) tuple.

    Returns (None, 'test_foo') for function context,
            ('TestFoo', None) for class context, or
            (None, None) if unrecognizable.
    """
    if not context_str:
        return (None, None)

    # Check for function definition
    func_m = re.match(r'(?:async\s+)?def (\w+)\s*\(', context_str.strip())
    if func_m:
        return (None, func_m.group(1))

    # Check for class definition
    class_m = re.match(r'class (\w+)\s*[\(:]', context_str.strip())
    if class_m:
        return (class_m.group(1), None)

    return (None, None)


def is_test_file(filepath: str) -> bool:
    """Check if a file is a Python test file (or conftest)."""
    basename = os.path.basename(filepath)
    return (
        filepath.endswith('.py') and
        (basename.startswith('test_') or basename == 'conftest.py')
    )


def extract_test_ids_from_patch(patch_text: str) -> Dict:
    """
    Parse a test_patch diff and extract affected pytest test IDs.

    Returns a dict with:
      - added_tests: list of test IDs for entirely new test functions
      - modified_tests: list of test IDs for modified existing tests
      - modified_fixtures: list of filepaths where fixtures/conftest were changed
      - affected_test_files: list of test filepaths that were changed but
            where no specific test function could be identified (e.g., module-level
            data changes, helper function changes)
      - all_test_ids: combined deduplicated list of all test IDs
    """
    if not patch_text or not patch_text.strip():
        return {
            'added_tests': [],
            'modified_tests': [],
            'modified_fixtures': [],
            'affected_test_files': [],
            'all_test_ids': [],
        }

    file_diffs = parse_diff_files(patch_text)

    added_tests = []  # type: List[str]
    modified_tests = []  # type: List[str]
    modified_fixtures = []  # type: List[str]
    affected_test_files = []  # type: List[str]
    seen_ids = set()  # type: Set[str]

    # Track which files had at least one test ID identified
    files_with_test_ids = set()  # type: Set[str]

    for fdiff in file_diffs:
        filepath = fdiff['filepath']

        # Skip non-Python test files (e.g., .txt prompt files)
        if not filepath.endswith('.py'):
            continue

        is_conftest = os.path.basename(filepath) == 'conftest.py'
        is_new = fdiff['is_new_file']
        basename = os.path.basename(filepath)

        # Check if this is a utility/non-test Python file (e.g., tests/quantization/utils.py)
        is_utility = not basename.startswith('test_') and basename != 'conftest.py'

        if is_conftest:
            # Conftest changes affect fixtures, note them specially.
            modified_fixtures.append(filepath)
            # Also try to extract specific fixture/test names from conftest.
            _extract_conftest_tests(fdiff, filepath, added_tests, modified_tests, seen_ids)
            continue

        if is_utility:
            # Utility file changes are noted but don't contain test functions.
            affected_test_files.append(filepath)
            continue

        ids_before = len(seen_ids)

        # Process each hunk to find test functions
        for hunk in fdiff['hunks']:
            _process_hunk(
                hunk, filepath, is_new,
                added_tests, modified_tests, seen_ids
            )

        ids_after = len(seen_ids)

        if ids_after > ids_before:
            files_with_test_ids.add(filepath)

    # For test files that had changes but no specific test IDs extracted,
    # record the whole file as affected. This happens when:
    # - Module-level data changes (model lists, constants, imports)
    # - Helper function body changes (e.g., vllm_to_hf_output)
    # - Changes to parametrize data blocks at module level
    for fdiff in file_diffs:
        filepath = fdiff['filepath']
        if not filepath.endswith('.py'):
            continue
        basename = os.path.basename(filepath)
        is_conftest = basename == 'conftest.py'
        is_utility = not basename.startswith('test_') and not is_conftest
        if is_conftest or is_utility:
            continue
        if filepath not in files_with_test_ids:
            if filepath not in affected_test_files:
                affected_test_files.append(filepath)

    all_ids = list(dict.fromkeys(added_tests + modified_tests))  # dedupe, preserve order

    return {
        'added_tests': added_tests,
        'modified_tests': modified_tests,
        'modified_fixtures': modified_fixtures,
        'affected_test_files': affected_test_files,
        'all_test_ids': all_ids,
    }


def _extract_conftest_tests(
    fdiff: Dict, filepath: str,
    added_tests: List[str], modified_tests: List[str],
    seen_ids: Set[str]
) -> None:
    """Extract any test functions defined in conftest (rare but possible)."""
    for hunk in fdiff['hunks']:
        _process_hunk(
            hunk, filepath, fdiff['is_new_file'],
            added_tests, modified_tests, seen_ids
        )


def _build_test_id(filepath: str, class_name: Optional[str], func_name: str) -> str:
    """Build a pytest node ID: filepath::ClassName::test_name or filepath::test_name."""
    if class_name:
        return "{}::{}::{}".format(filepath, class_name, func_name)
    return "{}::{}".format(filepath, func_name)


def _process_hunk(
    hunk: Dict, filepath: str, is_new_file: bool,
    added_tests: List[str], modified_tests: List[str],
    seen_ids: Set[str]
) -> None:
    """
    Process a single hunk to find affected test functions.

    Strategy:
    1. Scan added lines for new `def test_` definitions.
    2. Use the hunk context header to identify the enclosing function/class
       when modifications are made to an existing test body.
    3. Detect modified @pytest.mark.parametrize decorators and associate
       them with the next test function.
    """
    context_str = hunk['context_func']
    context_class, context_func = extract_context_name(context_str)

    added_lines = []
    removed_lines = []
    all_new_content_lines = []  # ordered content as it appears on the new side

    has_modifications = False  # True if there are actual +/- changes (not just context)
    has_parametrize_change = False

    # Track current class scope from the hunk content itself
    current_class = context_class  # inherited from hunk header

    # Collect new test function defs and track parametrize changes
    new_test_funcs = []  # (func_name, class_name, is_added)
    parametrize_on_next = False

    for line_type, content in hunk['lines']:
        if line_type == '+':
            has_modifications = True
            added_lines.append(content)

            # Check for parametrize decorator on added line
            if PARAMETRIZE_RE.search(content):
                parametrize_on_next = True

            # Check for class definition
            class_m = CLASS_DEF_RE.match(content)
            if class_m:
                current_class = class_m.group(1)

            # Check for top-level test function definition (new)
            func_m = TOPLEVEL_FUNC_RE.match(content)
            if func_m:
                test_id = _build_test_id(filepath, None, func_m.group(1))
                if test_id not in seen_ids:
                    seen_ids.add(test_id)
                    added_tests.append(test_id)
                parametrize_on_next = False
                continue

            # Check for class method test function definition (new)
            method_m = METHOD_FUNC_RE.match(content)
            if method_m:
                # Method inside a class - use current_class
                cls = current_class
                test_id = _build_test_id(filepath, cls, method_m.group(1))
                if test_id not in seen_ids:
                    seen_ids.add(test_id)
                    added_tests.append(test_id)
                parametrize_on_next = False
                continue

        elif line_type == '-':
            has_modifications = True
            removed_lines.append(content)

            # Check for parametrize decorator on removed line
            if PARAMETRIZE_RE.search(content):
                has_parametrize_change = True

        elif line_type == ' ':
            # Context lines - track class scope changes
            class_m = CLASS_DEF_RE.match(content)
            if class_m:
                current_class = class_m.group(1)

            # If we had a parametrize modification and we see a test func in context,
            # that test is affected.
            if has_parametrize_change or parametrize_on_next:
                func_m = TOPLEVEL_FUNC_RE.match(content)
                if func_m:
                    test_id = _build_test_id(filepath, None, func_m.group(1))
                    if test_id not in seen_ids:
                        seen_ids.add(test_id)
                        modified_tests.append(test_id)
                    has_parametrize_change = False
                    parametrize_on_next = False

                method_m = METHOD_FUNC_RE.match(content)
                if method_m:
                    test_id = _build_test_id(filepath, current_class, method_m.group(1))
                    if test_id not in seen_ids:
                        seen_ids.add(test_id)
                        modified_tests.append(test_id)
                    has_parametrize_change = False
                    parametrize_on_next = False

    # If there were modifications but no new test defs found in the hunk body,
    # the hunk is modifying an existing function. Use the context_func from the header.
    if has_modifications and context_func:
        # Only record if it looks like a test function
        if context_func.startswith('test_'):
            test_id = _build_test_id(filepath, context_class, context_func)
            if test_id not in seen_ids:
                seen_ids.add(test_id)
                modified_tests.append(test_id)

    # Handle parametrize change affecting the context function
    if has_parametrize_change and context_func and context_func.startswith('test_'):
        test_id = _build_test_id(filepath, context_class, context_func)
        if test_id not in seen_ids:
            seen_ids.add(test_id)
            modified_tests.append(test_id)

    # Scan for test functions on added lines that follow parametrize decorators
    # within the same hunk (walk through added lines looking for patterns)
    _scan_added_lines_for_parametrize_targets(
        hunk, filepath, current_class, added_tests, modified_tests, seen_ids
    )


def _scan_added_lines_for_parametrize_targets(
    hunk: Dict, filepath: str, default_class: Optional[str],
    added_tests: List[str], modified_tests: List[str],
    seen_ids: Set[str]
) -> None:
    """
    Walk through the hunk lines looking for a pattern where a modified
    parametrize decorator is followed by a test function on a context line.
    This handles the case where the parametrize values changed but the
    def line itself is unchanged.
    """
    saw_parametrize_change = False

    for line_type, content in hunk['lines']:
        if line_type in ('+', '-'):
            if PARAMETRIZE_RE.search(content):
                saw_parametrize_change = True
        elif line_type == ' ' and saw_parametrize_change:
            # Look for a test function def on the context line
            func_m = TOPLEVEL_FUNC_RE.match(content)
            if func_m:
                test_id = _build_test_id(filepath, None, func_m.group(1))
                if test_id not in seen_ids:
                    seen_ids.add(test_id)
                    modified_tests.append(test_id)
                saw_parametrize_change = False
            method_m = METHOD_FUNC_RE.match(content)
            if method_m:
                test_id = _build_test_id(filepath, default_class, method_m.group(1))
                if test_id not in seen_ids:
                    seen_ids.add(test_id)
                    modified_tests.append(test_id)
                saw_parametrize_change = False
            # If it's another decorator line (non-parametrize), keep looking
            if content.strip().startswith('@'):
                continue
            # If it's a blank line or non-def context, stop looking
            if not content.strip().startswith('def ') and not content.strip().startswith('@'):
                saw_parametrize_change = False


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def main():
    json_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'vllm_infra_gym.json'
    )

    print("Loading JSON from: {}".format(json_path))
    with open(json_path, 'r') as f:
        data = json.load(f)

    print("Processing {} instances...".format(len(data)))
    print("=" * 70)

    total_added = 0
    total_modified = 0
    total_fixtures = 0
    total_affected_files = 0
    instances_with_tests = 0
    instances_no_tests = 0

    for i, instance in enumerate(data):
        instance_id = instance.get('instance_id', 'unknown-{}'.format(i))
        tests_info = instance.get('tests', {})
        test_patch = tests_info.get('test_patch', '')

        result = extract_test_ids_from_patch(test_patch)

        # Write the test_ids into the instance
        tests_info['test_ids'] = result

        n_added = len(result['added_tests'])
        n_modified = len(result['modified_tests'])
        n_fixtures = len(result['modified_fixtures'])
        n_affected = len(result['affected_test_files'])
        n_total = len(result['all_test_ids'])

        total_added += n_added
        total_modified += n_modified
        total_fixtures += n_fixtures
        total_affected_files += n_affected

        if n_total > 0 or n_fixtures > 0 or n_affected > 0:
            instances_with_tests += 1
        else:
            instances_no_tests += 1

    # Update JSON in-place
    print("Writing updated JSON back to: {}".format(json_path))
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Done writing.")

    # ---------------------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print("Total instances:             {}".format(len(data)))
    print("Instances with test IDs:     {}".format(instances_with_tests))
    print("Instances with NO test IDs:  {}".format(instances_no_tests))
    print("Total added test IDs:        {}".format(total_added))
    print("Total modified test IDs:     {}".format(total_modified))
    print("Total fixture file changes:  {}".format(total_fixtures))
    print("Total affected test files:   {}".format(total_affected_files))
    print("  (files with changes but no specific test ID identified)")
    print()

    # ---------------------------------------------------------------------------
    # Sample outputs for verification
    # ---------------------------------------------------------------------------
    print("=" * 70)
    print("SAMPLE OUTPUTS (first 10 instances)")
    print("=" * 70)
    for i, instance in enumerate(data[:10]):
        instance_id = instance.get('instance_id', 'unknown')
        test_ids = instance.get('tests', {}).get('test_ids', {})
        all_ids = test_ids.get('all_test_ids', [])
        added = test_ids.get('added_tests', [])
        modified = test_ids.get('modified_tests', [])
        fixtures = test_ids.get('modified_fixtures', [])
        affected = test_ids.get('affected_test_files', [])

        print("\n--- Instance {}: {} ---".format(i + 1, instance_id))
        if added:
            print("  ADDED tests:")
            for tid in added:
                print("    + {}".format(tid))
        if modified:
            print("  MODIFIED tests:")
            for tid in modified:
                print("    ~ {}".format(tid))
        if fixtures:
            print("  MODIFIED fixtures/conftest:")
            for fp in fixtures:
                print("    * {}".format(fp))
        if affected:
            print("  AFFECTED test files (no specific test ID):")
            for fp in affected:
                print("    ? {}".format(fp))
        if not all_ids and not fixtures and not affected:
            print("  (no test IDs extracted)")

    # ---------------------------------------------------------------------------
    # Distribution of test counts
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DISTRIBUTION OF TEST ID COUNTS PER INSTANCE")
    print("=" * 70)
    counts = []
    for instance in data:
        test_ids = instance.get('tests', {}).get('test_ids', {})
        n = len(test_ids.get('all_test_ids', []))
        counts.append(n)

    from collections import Counter
    dist = Counter(counts)
    for count in sorted(dist.keys()):
        print("  {} test IDs: {} instances".format(count, dist[count]))

    # Print some instances with the most test IDs for verification
    print("\n" + "=" * 70)
    print("INSTANCES WITH MOST TEST IDS (top 5)")
    print("=" * 70)
    ranked = sorted(
        enumerate(data),
        key=lambda x: len(x[1].get('tests', {}).get('test_ids', {}).get('all_test_ids', [])),
        reverse=True
    )
    for idx, instance in ranked[:5]:
        iid = instance.get('instance_id', 'unknown')
        test_ids = instance.get('tests', {}).get('test_ids', {})
        all_ids = test_ids.get('all_test_ids', [])
        print("\n  {} ({} test IDs):".format(iid, len(all_ids)))
        for tid in all_ids:
            print("    {}".format(tid))

    print("\nAll done.")


if __name__ == '__main__':
    main()
