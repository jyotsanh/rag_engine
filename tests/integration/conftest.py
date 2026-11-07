"""Integration tests: the container fixtures live in `tests/support/containers.py`.

They are registered for the whole test tree in `tests/conftest.py` (`pytest_plugins`), so
browser tests that need MongoDB/Redis (Step 54) can use them too.
"""
