"""Smoke tests: the package imports and the pytest plugin is registered."""

from __future__ import annotations

import pytest

import gut


def test_version_is_exposed() -> None:
    assert gut.__version__


def test_gut_evals_flag_is_registered(pytestconfig: pytest.Config) -> None:
    # The pytest11 entry point should make --gut-evals a known option.
    assert pytestconfig.getoption("--gut-evals") is False
