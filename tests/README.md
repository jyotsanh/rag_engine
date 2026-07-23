# Test conventions

How we write tests in this repo. Keep new tests consistent with this so the
suite stays scannable and every test traces back to the issue that introduced it.

## One class per sub-issue

Group a sub-issue's tests into a single class named:

```
Test<PascalCaseFeature>_<issueNumber>
```

- `<PascalCaseFeature>` — the thing under test, from the issue title.
- `<issueNumber>` — the GitHub sub-issue number, suffixed after an underscore.

Examples:

```python
class TestBaseClassSkeleton_70:   # issue #70 — base class skeleton
class TestModelSerialization_71:  # issue #71 — to_mongo / from_mongo
```

The number ties test output straight back to the issue (and its PR), and keeps
successive sub-issues of the same epic side by side in one file.

## Class shape

- **Plain pytest classes NOT `unittest.TestCase`.** No `self.assert*`; use bare
  `assert`. No `__init__`.
- **Class docstring**: one line naming the sub-issue (`#NN — ...`) and what's under
  test. Add a second line for anything non-obvious (what a probe exercises, why a
  fixture is used).
- **Methods take fixtures as parameters** (`client: TestClient`, `auth_headers`, …)
  — never module globals for per-test state.
- **Class-level constants** for shared literals (e.g. `URL = f"{API_V1_STR}/file/..."`).

## Method shape

- Name: `test_<subject>_<expected_behavior>` — e.g.
  `test_from_mongo_rejects_empty_data`, `test_create_index_returns_201_on_success`.
- **Arrange / Act / Assert**, separated by blank lines. Add `# Arrange` / `# Act` /
  `# Assert` (or a short intent comment) when the flow isn't obvious; a trivial
  one-liner can skip them.
- One behavior per test. Prefer several small tests over one that asserts many
  unrelated things.

## Section banner

Separate the test classes from module-level helpers/fixtures with a banner:

```python
# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
```

## Helpers & fixtures

- Shared model subclasses / builders used only for testing: module-level and
  **underscore-prefixed** (`class _Doc(...)`) so pytest does not collect them as
  test classes.
- Reusable setup shared across files → a pytest fixture in `conftest.py`.

## Full example

```python
import uuid

import pytest
from pydantic import Field

from rag_engine.domain.base import NoSQLBaseDocument


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _Doc(NoSQLBaseDocument):
    name: str = "example"


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestModelSerialization_71:
    """#71 — Mongo <-> model serialization via to_mongo / from_mongo."""

    def test_from_mongo_round_trip_reproduces_the_model(self):
        # Arrange
        doc = _Doc(name="hello")

        # Act
        restored = _Doc.from_mongo(doc.to_mongo())

        # Assert
        assert restored == doc
        assert restored.name == doc.name

    def test_from_mongo_rejects_empty_data(self):
        with pytest.raises(ValueError):
            _Doc.from_mongo({})
```

## Running

```bash
uv run pytest -q                      # whole suite
uv run pytest tests/domain -q         # a directory
uv run pytest -k TestModelSerialization_71 -v   # one sub-issue's class
```
