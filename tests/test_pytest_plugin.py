"""Tests for the pytest integration, run through pytest's own pytester fixture."""

from __future__ import annotations

import pytest

CONFTEST = """
import gut

gut.configure(
    backend=gut.FakeBackend(default={default!r}),
    cache=gut.NullCache(),
)
"""

PASSING = """
question: "the customer threatens to cancel"
min_accuracy: 0.5
examples:
  - text: "I'm cancelling."
    expected: yes
  - text: "How do I cancel?"
    expected: no
"""

FAILING = """
question: "the customer threatens to cancel"
min_accuracy: 1.0
examples:
  - text: "I'm cancelling."
    expected: yes
  - text: "How do I cancel?"
    expected: no
"""


def test_nothing_is_collected_without_the_flag(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(CONFTEST.format(default=0.9))
    pytester.makefile(".yaml", cancel=PASSING)

    result = pytester.runpytest()
    result.assert_outcomes()  # nothing collected at all
    assert "gut evals" not in result.stdout.str()


def test_the_flag_collects_predicate_files(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(CONFTEST.format(default=0.9))
    pytester.makefile(".yaml", cancel=PASSING)

    result = pytester.runpytest("--gut-evals")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*PASS  cancel*50%*(1/2)*min 50%*"])


def test_a_file_below_its_bar_fails_with_the_mismatches(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(CONFTEST.format(default=0.9))
    pytester.makefile(".yaml", cancel=FAILING)

    result = pytester.runpytest("--gut-evals")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        [
            "*accuracy 50%*below min_accuracy 100%*",
            "*expected False, got yes (p=0.900)*",
            "*How do I cancel?*",
        ]
    )
    # A failure shows the mismatches, not a Python traceback.
    assert "Traceback" not in result.stdout.str()


def test_a_perfect_file_passes(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(CONFTEST.format(default=0.1))
    pytester.makefile(
        ".yaml",
        calm='question: "is angry"\nexamples:\n  - text: "thanks!"\n    expected: no\n',
    )

    result = pytester.runpytest("--gut-evals")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*PASS  calm*100%*(1/1)*"])


def test_other_yaml_is_left_alone(pytester: pytest.Pytester) -> None:
    """Someone else's docker-compose.yml is not a predicate file."""
    pytester.makeconftest(CONFTEST.format(default=0.9))
    pytester.makefile(".yml", **{"docker-compose": "services:\n  web:\n    image: nginx\n"})
    pytester.makefile(".yaml", settings="name: something\nvalues: [1, 2]\n")

    result = pytester.runpytest("--gut-evals")
    result.assert_outcomes()


def test_unparseable_yaml_is_skipped_not_crashed(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(CONFTEST.format(default=0.9))
    pytester.makefile(".yaml", broken="question: [unclosed\n")

    result = pytester.runpytest("--gut-evals")
    result.assert_outcomes()
    assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED


def test_several_files_are_each_reported(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(CONFTEST.format(default=0.9))
    pytester.makefile(".yaml", alpha=PASSING, beta=FAILING)

    result = pytester.runpytest("--gut-evals")
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*PASS  alpha*", "*FAIL  beta*"])


def test_a_malformed_predicate_file_fails_the_test(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(CONFTEST.format(default=0.9))
    pytester.makefile(".yaml", broken="examples:\n  - text: t\n    expected: yes\n")

    result = pytester.runpytest("--gut-evals")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*needs a 'question'*"])


def test_the_test_is_named_after_the_file(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(CONFTEST.format(default=0.9))
    pytester.makefile(".yaml", cancel_threat=PASSING)

    result = pytester.runpytest("--gut-evals", "-v")
    result.stdout.fnmatch_lines(["*cancel_threat.yaml::cancel_threat PASSED*"])


def test_a_specific_file_can_be_selected(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(CONFTEST.format(default=0.9))
    pytester.makefile(".yaml", alpha=PASSING, beta=PASSING)

    result = pytester.runpytest("--gut-evals", "alpha.yaml")
    result.assert_outcomes(passed=1)
    assert "beta" not in result.stdout.str()


def test_the_flag_is_documented_in_help(pytester: pytest.Pytester) -> None:
    result = pytester.runpytest("--help")
    result.stdout.fnmatch_lines(["*--gut-evals*predicate example files*"])
