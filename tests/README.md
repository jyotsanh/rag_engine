# Test conventions

## Layout
- `tests/unit/` — no network, no containers. HTTP mocked with `respx`, infra replaced by fakes.
- `tests/integration/` — real MongoDB / Redis / Qdrant via testcontainers, real Celery worker, local fixture website.
- `tests/fixtures/site/` — the static fixture website served by a local server fixture.

## Markers
| Marker | Meaning |
|---|---|
| `unit` | Fast, isolated (default path) |
| `integration` | Needs containers / Celery worker |
| `browser` | Needs Playwright Chromium |
| `live` | Hits real external services (OpenAI, public websites). Never run by default |
| `eval` | Retrieval quality evaluation |

Fast path: `uv run pytest -q -m "not integration and not browser and not live and not eval"`

## Naming
- One plain pytest class per plan step: `Test<PascalCaseFeature>_S<step>` (e.g. `TestUrlNormalization_S23`).
- Class docstring names the step: `"""S23 — URL normalization: <what is under test>."""`
- **Not** `unittest.TestCase`: bare `assert`, methods take fixtures as parameters.
- Method names: `test_<subject>_<expected_behavior>`.
- A section banner sits above the test classes:
  ```python
  # ---- #
  # Tests
  # ---- #
  ```
- Test-only helper subclasses/models live at module level and are underscore-prefixed (`_FakeEmbedder`) so pytest does not collect them.
- Arrange / Act / Assert separated by blank lines; add `# Arrange` / `# Act` / `# Assert` comments when the flow is not obvious.

## Checks before handing back a step
```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q
```
