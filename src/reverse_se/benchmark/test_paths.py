"""Path-based test-file detection, shared by the Commit0 benchmark
(commit0.py: which files are tests vs implementation) and the structural
analysis scripts (structural_ambiguity.py, doc_groundedness.py)."""

from __future__ import annotations

import re
from pathlib import Path

TEST_DIR_NAMES = {"test", "tests", "__tests__", "spec", "specs", "testing"}

# Matches stems like test_foo, testFoo, foo_test, FooTest, foo_tests,
# conftest (ends in "test"), CategoryTesterTest -- deliberately loose since
# a false "is test" keeps a file out of the implementation (safe); a false
# negative would treat a test file as implementation (unsafe).
_TEST_STEM_RE = re.compile(r"(^test[_a-z0-9]*$|[_a-z0-9]*tests?$)", re.IGNORECASE)


def is_test_path(rel_path: Path) -> bool:
    if any(part.lower() in TEST_DIR_NAMES for part in rel_path.parts[:-1]):
        return True
    return bool(_TEST_STEM_RE.match(rel_path.stem))
