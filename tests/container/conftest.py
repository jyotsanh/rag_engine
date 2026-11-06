"""Container tests: build the Docker images and run them (opt-in: `--run-container`).

The image is built once per session from the repo root. A missing Docker daemon skips
the tests rather than failing them.
"""

import subprocess
from collections.abc import Iterator
from pathlib import Path

import docker
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_IMAGE = "rag-engine-worker:test"
BUILD_TIMEOUT_SECONDS = 1500


@pytest.fixture(scope="session")
def docker_daemon() -> None:
    try:
        docker.from_env().ping()
    except Exception as exc:  # any failure means "no usable docker daemon"
        pytest.skip(f"docker unavailable: {exc}")


@pytest.fixture(scope="session")
def worker_image(docker_daemon: None) -> Iterator[str]:
    build = subprocess.run(
        ["docker", "build", "--target", "worker", "-t", WORKER_IMAGE, str(REPO_ROOT)],
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT_SECONDS,
        check=False,
    )
    if build.returncode != 0:
        pytest.fail(f"worker image build failed:\n{build.stderr[-4000:]}")
    yield WORKER_IMAGE
