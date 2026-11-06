"""The crawl worker image: Chromium installed, renders the fixture SPA, SSRF still enforced."""

import json
import subprocess

import pytest

from tests.support.fixture_site import FixtureSite

# dummy secrets: the image never contains a .env, configuration is the environment's job
_ENV = {
    "ENVIRONMENT": "dev",
    "ADMIN_API_KEY": "container-admin-key-" + "x" * 24,
    "API_KEY_PEPPER": "container-pepper-" + "y" * 24,
    "OPENAI_API_KEY": "sk-container-test",
    "LOG_LEVEL": "WARNING",
}
_RUN_TIMEOUT_SECONDS = 180


def _docker_run(
    image: str, *command: str, env: dict[str, str] | None = None, entrypoint: str | None = None
) -> subprocess.CompletedProcess[str]:
    args = ["docker", "run", "--rm", "--network", "host"]  # host network: reach the fixture site
    for name, value in {**_ENV, **(env or {})}.items():
        args += ["-e", f"{name}={value}"]
    if entrypoint is not None:
        args += ["--entrypoint", entrypoint]
    return subprocess.run(
        [*args, image, *command],
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_SECONDS,
        check=False,
    )


def _smoke(image: str, url: str, *extra: str, **env: str) -> tuple[int, dict]:
    proc = _docker_run(
        image, "python", "-m", "rag_engine.crawling.fetchers.smoke", url, *extra, env=env
    )
    lines = proc.stdout.strip().splitlines()
    try:
        report = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        pytest.fail(f"no JSON report (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return proc.returncode, report


# ---- #
# Tests
# ---- #


@pytest.mark.timeout(1800)  # the first test pays for the image build
class TestWorkerImage_S37:
    """S37 — Chromium in the worker image: render smoke test, SSRF, non-root, no secrets."""

    def test_renders_the_fixture_spa(self, worker_image: str, fixture_site: FixtureSite) -> None:
        code, report = _smoke(
            worker_image,
            fixture_site.url("/app/"),
            "--expect",
            "Widget configurator",
            SSRF_ALLOW_PRIVATE_NETWORKS="true",
        )

        assert code == 0, report
        assert report["rendered_with"] == "browser"
        assert report["expect_found"] is True
        assert "empty_mount" in report["js_signals"]
        assert report["chromium"]  # e.g. "156.0.8078.4"
        # the browser's requests were made by our SSRF-safe client inside the container
        assert fixture_site.hits("/app/") == 2

    def test_server_rendered_page_does_not_need_the_browser(
        self, worker_image: str, fixture_site: FixtureSite
    ) -> None:
        code, report = _smoke(
            worker_image,
            fixture_site.url("/about.html"),
            "--allow-http",
            SSRF_ALLOW_PRIVATE_NETWORKS="true",
        )

        assert code == 0, report
        assert report["rendered_with"] == "http"

    def test_ssrf_guard_is_enforced_in_the_image(
        self, worker_image: str, fixture_site: FixtureSite
    ) -> None:
        # default settings: loopback (where the fixture site lives) is not reachable
        code, report = _smoke(worker_image, fixture_site.url("/app/"))

        assert code == 1
        assert report["ok"] is False
        assert report["error"] == "ssrf_blocked"
        assert fixture_site.hits("/app/") == 0

    def test_runs_as_non_root_without_secrets_or_tests(self, worker_image: str) -> None:
        proc = _docker_run(
            worker_image,
            "-c",
            "id -u; test -e /app/.env && echo has-env; test -d /app/tests && echo has-tests; "
            'ls "$PLAYWRIGHT_BROWSERS_PATH"',
            entrypoint="sh",
        )

        lines = proc.stdout.split()
        assert proc.returncode == 0, proc.stderr
        assert lines[0] == "10001"
        assert "has-env" not in lines and "has-tests" not in lines
        assert any(name.startswith("chromium_headless_shell-") for name in lines)
