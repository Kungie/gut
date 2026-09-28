"""Tests for the `gut` command: judgments over lines of text, from a shell."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import gut
from gut import FakeBackend
from gut._cli import CHUNK, MAX_FILE_CHARS, Record, chunks, main

LINES = "WIN a FREE iPhone\n\nlunch at 1?\nWIN big, click here\nmaybe later\n"


def by_text(state: gut.State, name: str, spec: gut.QuestionSpec) -> float | str | int | None:
    text = str(state)
    match spec:
        case gut.NoulSpec():
            return 0.97 if "WIN" in text else 0.5 if "maybe" in text else 0.02
        case gut.ChoiceSpec():
            return "billing" if "invoice" in text else "other"
        case _:
            return 2 if "down" in text else 0


Run = Callable[..., tuple[int, str, str]]


@pytest.fixture
def run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Callable[..., tuple[int, str, str]]:
    """`gut ARGS` with `stdin`, answered by a fixture backend: (status, stdout, stderr)."""

    def invoke(*argv: str, stdin: str = LINES, backend: Any = "default") -> tuple[int, str, str]:
        chosen = FakeBackend(rule=by_text) if backend == "default" else backend
        monkeypatch.setattr("gut._cli.backend_from_env", lambda env: chosen)
        status = main(list(argv), stdin=io.StringIO(stdin))
        out, err = capsys.readouterr()
        return status, out, err

    return invoke


# --------------------------------------------------------------------------- filter


def test_filter_prints_the_lines_a_claim_is_true_of(run: Run) -> None:
    status, out, err = run("filter", "is spam")
    assert status == 0
    assert out.splitlines() == ["WIN a FREE iPhone", "WIN big, click here"]
    assert err.startswith("gut: 4 records · 2 yes, 2 no · 4 calls")
    assert "fake-1.0" in err


def test_empty_input_is_nothing_to_do(run: Run) -> None:
    status, out, err = run("filter", "is spam", stdin="\n\n")
    assert (status, out) == (1, "")
    assert err.strip() == "gut: 0 records · 0 yes, 0 no · 0 calls, $0 · 0.0s"


def test_filter_finds_nothing_like_grep(run: Run) -> None:
    status, out, _ = run("filter", "is spam", stdin="hello\nlunch?\n")
    assert (status, out) == (1, "")


def test_filter_can_show_the_other_answers(run: Run) -> None:
    _, out, _ = run("filter", "is spam", "--show", "no")
    assert out.splitlines() == ["lunch at 1?", "maybe later"]
    _, out, err = run("filter", "is spam", "--ask-human", "--show", "unsure", "--stakes", "high")
    assert out.splitlines() == ["maybe later"]
    assert "1 unsure" in err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["filter", "is spam", "--show", "unsure"], "needs --ask-human"),
        (["filter", "is spam", "--stakes", "high"], "only matters with --ask-human"),
    ],
)
def test_filter_says_which_options_go_together(run: Run, argv: list[str], message: str) -> None:
    status, _, err = run(*argv)
    assert status == 2
    assert message in err


def test_options_may_come_after_the_files(run: Run, tmp_path: Path) -> None:
    source = tmp_path / "in.txt"
    source.write_text("WIN now\nhello\n", encoding="utf-8")
    status, out, _ = run("filter", "is spam", str(source), "--quiet", "--concurrency", "2")
    assert (status, out) == (0, "WIN now\n")


def test_files_are_read_in_order_and_dash_is_standard_input(run: Run, tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    first.write_text("WIN one\n", encoding="utf-8")
    _, out, _ = run("filter", "is spam", str(first), "-", "-q", stdin="WIN two\n")
    assert out.splitlines() == ["WIN one", "WIN two"]


def test_a_missing_file_is_an_error(run: Run, tmp_path: Path) -> None:
    status, _, err = run("filter", "is spam", str(tmp_path / "nope.txt"))
    assert status == 2
    assert "cannot read" in err


def test_read_files_judges_the_file_a_line_names(run: Run, tmp_path: Path) -> None:
    spam = tmp_path / "spam.txt"
    spam.write_text("WIN a prize", encoding="utf-8")
    fine = tmp_path / "fine.txt"
    fine.write_text("minutes of the meeting", encoding="utf-8")
    binary = tmp_path / "image.bin"
    binary.write_bytes(b"\x89PNG\0\0\0")
    huge = tmp_path / "huge.txt"
    huge.write_text("WIN " * (MAX_FILE_CHARS // 4 + 1), encoding="utf-8")
    paths = [spam, fine, binary, huge, tmp_path / "gone.txt"]
    status, out, err = run(
        "filter", "is spam", "--read-files", stdin="\n".join(map(str, paths)) + "\n"
    )
    assert status == 0
    assert out.splitlines() == [str(spam)]
    assert "skipped" in err
    assert f"{binary}: binary" in err
    assert f"{huge}: longer than" in err
    assert "gone.txt" in err
    assert "3 skipped" in err


# --------------------------------------------------------------------------- map


def test_map_prints_every_answer_as_json(run: Run) -> None:
    status, out, err = run(
        "map",
        "--likely",
        "spam=is spam",
        "--classify",
        "team=billing,other",
        "--rate",
        "urgency=can wait,today,now",
        stdin="WIN now\nmy invoice is wrong, the site is down\n",
    )
    assert status == 0
    first, second = (json.loads(line) for line in out.splitlines())
    assert first == {
        "line": 1,
        "text": "WIN now",
        "spam": {"outcome": "yes", "p": 0.97},
        "team": {"outcome": "yes", "value": "other", "confidence": 0.9},
        "urgency": {"outcome": "yes", "score": 0.0, "level": "can wait", "confidence": 1.0},
    }
    assert second["team"]["value"] == "billing"
    assert second["urgency"]["level"] == "now"
    assert "2 records · 2 calls" in err


def test_map_takes_options_with_descriptions_as_json(run: Run) -> None:
    options = '{"billing": "invoices and refunds", "other": "anything else"}'
    _, out, _ = run("map", "--classify", f"team={options}", stdin="an invoice\n")
    assert json.loads(out)["team"]["value"] == "billing"


def test_map_asks_every_question_about_a_line_in_one_call(run: Run) -> None:
    backend = FakeBackend(rule=by_text)
    run("map", "--likely", "a=is spam", "--likely", "b=is urgent", backend=backend, stdin="x\ny\n")
    assert backend.call_count == 2


def test_map_can_say_unsure(run: Run) -> None:
    _, out, _ = run(
        "map", "--likely", "spam=is spam", "--ask-human", "--stakes", "high", stdin="maybe\n"
    )
    assert json.loads(out)["spam"]["outcome"] == "unsure"
    _, out, _ = run("map", "--classify", "team=billing,other", "--ask-human", stdin="x\n")
    assert json.loads(out)["team"]["value"] in ("billing", "other", None)


def test_map_names_paths_when_reading_files(run: Run, tmp_path: Path) -> None:
    page = tmp_path / "page.txt"
    page.write_text("WIN", encoding="utf-8")
    _, out, _ = run("map", "--likely", "spam=is spam", "--read-files", stdin=f"{page}\n")
    assert json.loads(out)["path"] == str(page)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["map"], "at least one of"),
        (["map", "--likely", "spam"], "NAME=..."),
        (["map", "--likely", "a=x", "--likely", "a=y"], "its own NAME"),
        (["map", "--classify", "team={nope"], "not valid JSON"),
        (["map", "--classify", 'team={"a": 1}'], "at least two options"),
        (["map", "--rate", "u=a,b", "--lean", "yes"], "--lean only applies"),
        (["map", "--classify", "team=only"], "at least two options"),
    ],
)
def test_map_explains_a_bad_question(run: Run, argv: list[str], message: str) -> None:
    status, _, err = run(*argv, stdin="x\n")
    assert status == 2
    assert message in err


# --------------------------------------------------------------------------- budgets


def test_a_budget_stops_the_run_and_says_where_to_resume(run: Run) -> None:
    backend = FakeBackend(rule=by_text, cost=0.01)
    text = "".join(f"line {n}\n" for n in range(1, 201))
    status, out, err = run(
        "filter", "is spam", "--show", "no", "--max-cost", "0.5", backend=backend, stdin=text
    )
    assert status == 3
    printed = out.splitlines()
    assert 1 <= len(printed) < 200
    resume = len(printed) + 1
    assert f"resume from line {resume}: tail -n +{resume} FILE | gut ..." in err
    assert "spent of a $0.5 budget" in err


def test_a_budget_needs_a_backend_that_reports_cost(run: Run) -> None:
    backend = FakeBackend(rule=by_text, cost=None)
    status, out, err = run("filter", "is spam", "--show", "no", "--max-cost", "1", backend=backend)
    assert status == 3
    assert out.splitlines() == []  # only the first record was asked, and it was spam
    assert backend.call_count == 1
    assert "does not report" in err
    assert "resume" not in err


def test_a_zero_budget_allows_a_free_model(run: Run) -> None:
    backend = FakeBackend(rule=by_text, cost=0.0)
    status, _, err = run("filter", "is spam", "--max-cost", "$0", backend=backend)
    assert status == 0
    assert "4 calls, $0" in err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["filter", "x", "--max-cost", "lots"], "not an amount"),
        (["filter", "x", "--max-cost", "-1"], "zero or more"),
        (["filter", "x", "--concurrency", "0"], "at least 1"),
        (["filter", "x", "--concurrency", "many"], "not a number"),
        (["filter", "x", "--show", "maybe"], "choose from"),
    ],
)
def test_bad_values_are_refused_by_the_parser(
    run: Run, argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exited:
        run(*argv)
    assert exited.value.code == 2
    assert message in capsys.readouterr().err


# --------------------------------------------------------------------------- the model


def test_without_a_model_the_command_says_how_to_choose_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("GUT_BACKEND", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    status = main(["filter", "is spam"], stdin=io.StringIO("x\n"))
    assert status == 2
    err = capsys.readouterr().err
    assert "TYPESAFE_API_KEY" in err
    assert "GUT_BACKEND" in err


def test_backend_and_model_flags_override_the_environment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GUT_BACKEND", "fake")
    status = main(["filter", "is spam", "--backend", "openai"], stdin=io.StringIO("x\n"))
    assert status == 2
    assert "needs GUT_MODEL" in capsys.readouterr().err
    status = main(
        ["filter", "is spam", "--backend", "fake", "--model", "m", "-q"], stdin=io.StringIO("x\n")
    )
    assert status in (0, 1)


def test_version_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.strip() == f"gut {gut.__version__}"
    with pytest.raises(SystemExit):
        main([])
    with pytest.raises(SystemExit):
        main(["filter", "--help"])
    assert "claim" in capsys.readouterr().out


def test_records_come_in_chunks_the_first_of_its_own_size() -> None:
    records = iter([Record(n, str(n), str(n)) for n in range(1, 6)])
    assert [len(c) for c in chunks(records, 1, 2)] == [1, 2, 2]
    assert CHUNK > 1


def test_the_command_runs_as_a_program() -> None:
    env = {key: value for key, value in os.environ.items() if key != "TYPESAFE_API_KEY"}
    env["GUT_BACKEND"] = "fake"
    done = subprocess.run(
        [sys.executable, "-m", "gut._cli", "map", "--likely", "spam=is spam"],
        input="one\ntwo\n",
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert [json.loads(line)["line"] for line in done.stdout.splitlines()] == [1, 2]
    assert done.stderr.startswith("gut: 2 records")
