"""pytest plugin entry point for `gut` example-file evaluations.

Registered via the ``pytest11`` entry point in ``pyproject.toml`` so that installing
``gut`` makes ``--gut-evals`` available. Collection logic lands in step 9.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--gut-evals`` flag."""
    group = parser.getgroup("gut")
    group.addoption(
        "--gut-evals",
        action="store_true",
        default=False,
        help="Collect gut predicate example files and run them as tests.",
    )
