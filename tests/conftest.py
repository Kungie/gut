"""Shared fixtures.

`gut.configure()` sets process-wide state, so every test starts from the unconfigured defaults
whether or not it touched configuration itself.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from gut._config import reset_configuration

pytest_plugins = ["pytester"]


@pytest.fixture
def anyio_backend() -> str:
    """Run the async tests on asyncio. anyio's plugin requires this to exist."""
    return "asyncio"


@pytest.fixture(autouse=True)
def _reset_gut_configuration() -> Iterator[None]:
    reset_configuration()
    yield
    reset_configuration()
