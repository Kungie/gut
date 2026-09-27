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
