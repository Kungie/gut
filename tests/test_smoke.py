"""Smoke tests: the package imports, cheaply."""

from __future__ import annotations

import subprocess
import sys

import gut


def test_version_is_exposed() -> None:
    assert gut.__version__


def test_importing_gut_pulls_in_no_backend_dependencies() -> None:
    """`import gut` must stay instant: no HTTP client, no vendor SDK, no PyTorch until asked."""
    probe = (
        "import sys, gut; "
        "print(sorted(m for m in ('httpx', 'torch', 'transformers', 'typesafe_sdk') "
        "if m in sys.modules))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert loaded == "[]"


def test_every_exported_name_resolves() -> None:
    for name in gut.__all__:
        if name in {"JevBackend", "TransformersBackend", "ZeroShotBackend"}:
            continue  # optional extras; resolved lazily, tested where they are installed
        assert getattr(gut, name) is not None, name


def test_the_version_is_the_same_in_both_places() -> None:
    """The release workflow publishes what pyproject.toml says and refuses a tag that differs."""
    import re
    from pathlib import Path

    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    assert match is not None
    assert gut.__version__ == match.group(1)
