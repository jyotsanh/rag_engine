"""Golden (snapshot) files: compare output with a checked-in file, or rewrite it.

`assert_golden(path, actual, update=...)` fails with a unified diff when the output
changed. Run `uv run pytest --update-golden <tests>` to accept a change, then review the
diff of the golden files like any other code change.
"""

import difflib
from pathlib import Path

GOLDEN_ROOT = Path(__file__).parent.parent / "fixtures" / "golden"


def assert_golden(path: Path, actual: str, *, update: bool) -> None:
    if update:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
        return
    if not path.exists():
        raise AssertionError(
            f"golden file {path} is missing; run with --update-golden to create it"
        )
    expected = path.read_text(encoding="utf-8")
    if actual != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile=f"{path.name} (golden)",
                tofile=f"{path.name} (actual)",
                n=2,
            )
        )
        raise AssertionError(
            f"output differs from {path.name} (accept with --update-golden):\n{diff}"
        )
