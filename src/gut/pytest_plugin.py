"""pytest integration: run predicate example files as tests.

```bash
pytest --gut-evals predicates/
```

Each YAML file becomes one test, because the unit that passes or fails is the file's accuracy, not
an individual example. A failure prints every case that went the wrong way, with what the model
actually said, so the next edit is to the examples or the question rather than to a guess.

Collection is gated behind the flag and behind the file's own shape: without `--gut-evals` nothing
here runs, and with it only YAML documents containing `examples` are collected. Someone else's
`docker-compose.yml` is not a predicate file.

Registered through the `pytest11` entry point, so this module is imported in every pytest run of
every project that installs `gut`. `pytest` itself is free to import here -- it is the thing doing
the importing -- but `gut`'s own machinery is pulled in only when a file is actually collected.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from gut._evals import EvalResult

EVALS_FLAG = "--gut-evals"

_results: list[EvalResult] = []


class EvalFailedError(Exception):
    """A predicate file scored below its own `min_accuracy`."""


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the `--gut-evals` flag."""
    group = parser.getgroup("gut")
    group.addoption(
        EVALS_FLAG,
        action="store_true",
        default=False,
        help="Collect gut predicate example files and run them as tests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Start each session with an empty report."""
    _results.clear()


class EvalItem(pytest.Item):
    """The accuracy of one predicate example file."""

    def runtest(self) -> None:
        """Run every example in the file and check it against `min_accuracy`."""
        from gut._config import current_backend
        from gut._evals import format_failures, load_suite, run_suite

        suite = load_suite(self.path)
        result = run_suite(suite, current_backend())
        _results.append(result)
        if not result.passed:
            raise EvalFailedError(format_failures(result))

    def repr_failure(self, excinfo: Any, style: Any = None) -> Any:
        """Show the mismatched examples rather than a Python traceback."""
        if isinstance(excinfo.value, EvalFailedError):
            return str(excinfo.value)
        return super().repr_failure(excinfo, style)

    def reportinfo(self) -> tuple[Path, int, str]:
        """Name the file in test output."""
        return self.path, 0, f"gut eval: {self.name}"


class EvalFile(pytest.File):
    """One predicate example file."""

    def collect(self) -> Any:
        """Yield the single item that scores this file."""
        yield EvalItem.from_parent(self, name=self.path.stem)


def pytest_collect_file(file_path: Path, parent: pytest.Collector) -> pytest.Collector | None:
    """Collect YAML files that are predicate example files, when the flag is on."""
    if file_path.suffix not in (".yaml", ".yml"):
        return None
    if not parent.config.getoption(EVALS_FLAG, default=False):
        return None

    import yaml

    from gut._evals import looks_like_suite

    try:
        document = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not looks_like_suite(document):
        return None
    return EvalFile.from_parent(parent, path=file_path)


def pytest_terminal_summary(terminalreporter: Any) -> None:
    """Report accuracy per file, including the ones that passed."""
    if not _results:
        return
    terminalreporter.write_sep("-", "gut evals")
    for result in sorted(_results, key=lambda item: item.suite.name):
        mark = "PASS" if result.passed else "FAIL"
        terminalreporter.write_line(
            f"{mark}  {result.suite.name:<32} "
            f"{result.accuracy:>6.0%}  ({result.correct}/{len(result.results)}) "
            f"min {result.suite.min_accuracy:.0%}"
        )
