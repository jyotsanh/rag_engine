"""Browser tests: real headless Chromium via Playwright.

Every test here is marked `browser` (from the directory). If Chromium cannot be launched
(not installed: `uv run playwright install chromium`), the tests are skipped, not failed.
"""

import asyncio

import pytest
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright


async def _can_launch() -> str | None:
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            await browser.close()
    except PlaywrightError as exc:
        return str(exc).splitlines()[0]
    return None


@pytest.fixture(scope="session")
def chromium_available() -> None:
    problem = asyncio.run(_can_launch())
    if problem is not None:
        pytest.skip(f"Chromium unavailable: {problem}")


@pytest.fixture(autouse=True)
def _require_chromium(chromium_available: None) -> None:
    """Applied to every browser test."""
