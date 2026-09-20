"""Tests for @semantic: what the analysis collects, what it refuses to, and the call count."""

from __future__ import annotations

import asyncio
import enum
import inspect
import logging
import threading
from collections.abc import Callable
from typing import Any

import pytest

import gut
from gut import FakeBackend, classify, likely, rate, semantic
from gut._backends.base import BackendResponse
from gut._backends.fake import FakeValue
from gut._errors import BackendError
from gut._questions import ChoiceSpec, QuestionSpec, ScoreSpec, State
from gut._semantic import _function_node, build_plan

URGENCY = ["can wait", "this week", "today"]
CANCEL_QUESTION = "the customer threatens to cancel"


class Team(enum.Enum):
    BILLING = "invoices and charges"
    PLATFORM = "outages and errors"
    OTHER = "anything else"


def _by_question_type(state: State, name: str, spec: QuestionSpec) -> FakeValue | None:
    match spec:
        case ChoiceSpec():
            return next(iter(spec.criteria))
        case ScoreSpec():
            return 1
        case _:
            return None


@pytest.fixture
def backend() -> FakeBackend:
    fake = FakeBackend(default=0.9, rule=_by_question_type)
    # Caching off, so every assertion about call counts is about batching alone.
    gut.configure(backend=fake, cache=gut.NullCache())
    return fake


# --------------------------------------------------------------------------- the headline


@semantic
def handle(ticket: str) -> str:
    if likely(ticket, "is a bug report"):
        if likely(ticket, "has reproduction steps"):
            return "triage"
        return "ask for steps"
    if likely(ticket, "asks for a refund"):
        return f"refund {classify(ticket, Team).value}"
    return f"urgency {rate(ticket, URGENCY).score}"


def test_five_questions_become_one_backend_call(backend: FakeBackend) -> None:
    assert len(handle.gut_plan.questions) == 5  # type: ignore[attr-defined]

    handle("a ticket")

    assert backend.call_count == 1
    assert len(backend.calls[0].names) == 5


def test_answers_report_that_they_were_prefetched(backend: FakeBackend) -> None:
    @semantic
    def inspect_source(ticket: str) -> gut.Decision:
        return likely(ticket, "is a bug report")

    decision = inspect_source("a ticket")
    assert decision.source == "prefetch"
    assert decision.cached is True
    assert decision.latency_ms is None
    assert backend.call_count == 1


def test_the_undecorated_version_makes_a_call_per_question(backend: FakeBackend) -> None:
    def plain(ticket: str) -> None:
        likely(ticket, "is a bug report")
        likely(ticket, "has reproduction steps")
        likely(ticket, "asks for a refund")

    plain("a ticket")
    assert backend.call_count == 3


# --------------------------------------------------------------------------- what is collected


def test_all_three_primitives_are_collected() -> None:
    @semantic
    def every(ticket: str) -> None:
        likely(ticket, "is a bug report")
        classify(ticket, Team)
        rate(ticket, URGENCY)

    kinds = {q.spec.canonical()["type"] for q in every.gut_plan.questions}  # type: ignore[attr-defined]
    assert kinds == {"noul", "choice", "score"}


def test_the_module_qualified_spelling_is_collected() -> None:
    @semantic
    def qualified(ticket: str) -> None:
        gut.likely(ticket, "is a bug report")

    assert len(qualified.gut_plan.questions) == 1  # type: ignore[attr-defined]


def test_a_module_level_name_for_the_question_is_collected() -> None:
    @semantic
    def named(ticket: str) -> None:
        likely(ticket, CANCEL_QUESTION)

    (question,) = named.gut_plan.questions  # type: ignore[attr-defined]
    assert question.spec.instructions == CANCEL_QUESTION


def test_described_outcomes_are_part_of_the_planned_question() -> None:
    @semantic
    def described(ticket: str) -> None:
        likely(ticket, "is this spam", yes_means="unsolicited advertising")

    (question,) = described.gut_plan.questions  # type: ignore[attr-defined]
    assert question.spec.canonical()["criteria"] == {"true": "unsolicited advertising"}


def test_the_same_question_twice_is_planned_once(backend: FakeBackend) -> None:
    @semantic
    def twice(ticket: str) -> None:
        likely(ticket, "is a bug report")
        likely(ticket, "is a bug report")

    assert len(twice.gut_plan.questions) == 1  # type: ignore[attr-defined]
    twice("a ticket")
    assert backend.call_count == 1


def test_each_subject_gets_its_own_call(backend: FakeBackend) -> None:
    @semantic
    def two_subjects(first: str, second: str) -> None:
        likely(first, "is a bug report")
        likely(first, "asks for a refund")
        likely(second, "is a bug report")

    two_subjects("one", "two")
    assert backend.call_count == 2
    assert sorted(len(call.names) for call in backend.calls) == [1, 2]


# --------------------------------------------------------------------------- what is refused


def test_a_computed_question_is_left_alone(backend: FakeBackend) -> None:
    @semantic
    def computed(ticket: str, topic: str) -> None:
        likely(ticket, f"is about {topic}")

    assert computed.gut_plan.questions == ()  # type: ignore[attr-defined]
    computed("a ticket", "billing")
    assert backend.call_count == 1  # asked normally, not batched


def test_a_subject_that_is_not_a_parameter_is_left_alone() -> None:
    @semantic
    def local_subject(ticket: str) -> None:
        summary = ticket[:10]
        likely(summary, "is a bug report")

    assert local_subject.gut_plan.questions == ()  # type: ignore[attr-defined]


def test_a_reassigned_parameter_is_excluded(backend: FakeBackend) -> None:
    """Prefetching against the original value would answer about the wrong state."""

    @semantic
    def rebinds(ticket: str) -> None:
        ticket = ticket.strip()
        likely(ticket, "is a bug report")

    assert rebinds.gut_plan.questions == ()  # type: ignore[attr-defined]
    assert "no statically resolvable" in str(rebinds.gut_plan.skipped)  # type: ignore[attr-defined]


def test_a_parameter_rebound_by_a_loop_is_excluded() -> None:
    @semantic
    def loops(ticket: str, others: list[str]) -> None:
        likely(ticket, "is a bug report")
        for ticket in others:
            likely(ticket, "asks for a refund")

    assert loops.gut_plan.questions == ()  # type: ignore[attr-defined]


def test_calls_inside_a_nested_function_are_left_alone() -> None:
    @semantic
    def nested(ticket: str) -> None:
        def inner() -> None:
            likely(ticket, "is a bug report")

        inner()

    assert nested.gut_plan.questions == ()  # type: ignore[attr-defined]


def test_a_per_call_backend_is_left_alone() -> None:
    other = FakeBackend(default=0.1)

    @semantic
    def explicit(ticket: str) -> None:
        likely(ticket, "is a bug report", backend=other)

    assert explicit.gut_plan.questions == ()  # type: ignore[attr-defined]


def test_starred_arguments_are_left_alone() -> None:
    extra: dict[str, Any] = {"cost_false_yes": 1, "cost_false_no": 1}

    @semantic
    def splatted(ticket: str) -> None:
        likely(ticket, "is a bug report", **extra)

    assert splatted.gut_plan.questions == ()  # type: ignore[attr-defined]


def test_a_shadowed_name_is_not_mistaken_for_ours() -> None:
    """Someone else's `likely` must not be collected just because of its name."""
    source = (
        "from gut import semantic\n"
        "def likely(subject, question):\n"
        "    return True\n"
        "@semantic\n"
        "def handler(ticket):\n"
        "    return likely(ticket, 'is a bug report')\n"
    )
    namespace: dict[str, Any] = {}
    exec(compile(source, "fake_module.py", "exec"), namespace)
    assert namespace["handler"].gut_plan.questions == ()


def test_an_enum_that_is_not_resolvable_is_left_alone() -> None:
    @semantic
    def local_enum(ticket: str) -> None:
        class Local(enum.Enum):
            A = "first"
            B = "second"

        classify(ticket, Local)

    assert local_enum.gut_plan.questions == ()  # type: ignore[attr-defined]


def test_an_enum_with_unusable_values_is_left_alone() -> None:
    @semantic
    def numbered(ticket: str) -> None:
        classify(ticket, Numbered)

    assert numbered.gut_plan.questions == ()  # type: ignore[attr-defined]


class Numbered(enum.Enum):
    A = 1
    B = 2


def test_a_rubric_with_the_wrong_number_of_levels_is_left_alone() -> None:
    @semantic
    def one_level(ticket: str) -> None:
        rate(ticket, ["only one"])

    assert one_level.gut_plan.questions == ()  # type: ignore[attr-defined]


def test_a_rubric_of_non_strings_is_left_alone() -> None:
    @semantic
    def numbers(ticket: str) -> None:
        rate(ticket, [1, 2, 3])  # type: ignore[list-item]

    assert numbers.gut_plan.questions == ()  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- analysis corners

NOT_A_QUESTION = 42
NOT_A_RUBRIC = 7


def test_a_keyword_question_is_collected() -> None:
    @semantic
    def keyworded(ticket: str) -> None:
        likely(ticket, question="is a bug report")

    (question,) = keyworded.gut_plan.questions  # type: ignore[attr-defined]
    assert question.spec.instructions == "is a bug report"


def test_variadic_parameters_are_still_parameters() -> None:
    @semantic
    def variadic(ticket: str, *extra: str, **options: str) -> None:
        likely(ticket, "is a bug report")

    assert len(variadic.gut_plan.questions) == 1  # type: ignore[attr-defined]


def test_a_parameter_a_nested_function_can_rebind_is_excluded() -> None:
    """`nonlocal` reaches the parameter from inside, so its value at call time is not the one
    that was passed in."""

    @semantic
    def outer(ticket: str) -> None:
        def inner() -> None:
            nonlocal ticket
            ticket = ticket.strip()

        inner()
        likely(ticket, "is a bug report")

    assert outer.gut_plan.questions == ()  # type: ignore[attr-defined]


def test_a_literal_subject_is_not_a_parameter() -> None:
    @semantic
    def literal_subject(ticket: str) -> None:
        likely("a fixed string", "is a bug report")

    assert literal_subject.gut_plan.questions == ()  # type: ignore[attr-defined]


def test_a_starred_positional_argument_is_left_alone() -> None:
    things = ("a ticket", "is a bug report")

    @semantic
    def splatted_positional(ticket: str) -> None:
        likely(*things)

    assert splatted_positional.gut_plan.questions == ()  # type: ignore[attr-defined]


def test_a_name_bound_to_a_non_string_is_not_a_question() -> None:
    @semantic
    def numeric_question(ticket: str) -> None:
        likely(ticket, NOT_A_QUESTION)  # type: ignore[arg-type]

    assert numeric_question.gut_plan.questions == ()  # type: ignore[attr-defined]


def test_a_rubric_with_a_computed_element_is_left_alone() -> None:
    @semantic
    def mixed(ticket: str, worst: str) -> None:
        rate(ticket, ["can wait", worst])

    assert mixed.gut_plan.questions == ()  # type: ignore[attr-defined]


def test_a_name_bound_to_a_non_sequence_is_not_a_rubric() -> None:
    @semantic
    def numeric_rubric(ticket: str) -> None:
        rate(ticket, NOT_A_RUBRIC)  # type: ignore[arg-type]

    assert numeric_rubric.gut_plan.questions == ()  # type: ignore[attr-defined]


def test_a_computed_outcome_description_is_left_alone() -> None:
    @semantic
    def computed_criteria(ticket: str, meaning: str) -> None:
        likely(ticket, "is this spam", yes_means=meaning)

    assert computed_criteria.gut_plan.questions == ()  # type: ignore[attr-defined]


def test_a_computed_no_description_is_left_alone() -> None:
    @semantic
    def computed_no(ticket: str, meaning: str) -> None:
        likely(ticket, "is this spam", no_means=meaning)

    assert computed_no.gut_plan.questions == ()  # type: ignore[attr-defined]


def test_a_computed_classify_question_is_left_alone() -> None:
    @semantic
    def computed_question(ticket: str, prompt: str) -> None:
        classify(ticket, Team, question=prompt)

    assert computed_question.gut_plan.questions == ()  # type: ignore[attr-defined]


def test_a_nested_lambda_deep_in_the_body_is_skipped() -> None:
    @semantic
    def deep(ticket: str, flag: bool) -> None:
        if flag:
            later: Callable[[], object] = lambda: likely(ticket, "is a bug report")  # noqa: E731
            later()

    assert deep.gut_plan.questions == ()  # type: ignore[attr-defined]


def test_a_malformed_call_does_not_break_decoration() -> None:
    """A call missing a required argument is the caller's bug to hit at runtime.

    Decoration must not raise first, and must not half-collect it either.
    """

    @semantic
    def malformed(ticket: str) -> None:
        classify(ticket)  # type: ignore[call-arg]
        rate(ticket)  # type: ignore[call-arg]

    assert malformed.gut_plan.questions == ()  # type: ignore[attr-defined]


def test_function_node_handles_source_without_a_definition() -> None:
    assert _function_node("x = 1\n") is None
    assert _function_node("x = 1\ndef later():\n    pass\n") is not None


# --------------------------------------------------------------------------- degrading safely


def test_a_function_without_retrievable_source_runs_unchanged(
    backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    source = (
        "from gut import likely\n"
        "def handler(ticket):\n"
        "    return likely(ticket, 'is a bug report')\n"
    )
    namespace: dict[str, Any] = {}
    exec(compile(source, "<not on disk>", "exec"), namespace)

    with caplog.at_level(logging.DEBUG, logger="gut"):
        decorated = semantic(namespace["handler"])

    assert decorated.gut_plan.questions == ()  # type: ignore[attr-defined]
    assert "source unavailable" in str(decorated.gut_plan.skipped)  # type: ignore[attr-defined]
    assert any("collected nothing" in record.message for record in caplog.records)

    # And it still works.
    assert decorated("a ticket").p == 0.9
    assert backend.call_count == 1


# --------------------------------------------------------------------------- coroutines


@pytest.mark.anyio
async def test_a_coroutine_batches_too(backend: FakeBackend) -> None:
    @semantic
    async def handler(ticket: str) -> tuple[float, float, float]:
        return (
            likely(ticket, "is a bug report").p,
            likely(ticket, "asks for a refund").p,
            likely(ticket, "has reproduction steps").p,
        )

    assert len(handler.gut_plan.questions) == 3  # type: ignore[attr-defined]
    assert await handler("a ticket") == (0.9, 0.9, 0.9)
    assert backend.call_count == 1


@pytest.mark.anyio
async def test_the_event_loop_keeps_running_during_the_prefetch() -> None:
    """Deterministic, not timed: if the loop were blocked this deadlocks and the wait fails.

    The backend parks inside `ask` until a coroutine on the loop releases it. That coroutine can
    only run if the prefetch is off the event loop, which is the whole claim.
    """
    entered = threading.Event()
    release = threading.Event()

    class Parking(FakeBackend):
        def ask(self, state: State, questions: Any) -> Any:
            entered.set()
            assert release.wait(timeout=5), "the event loop never got a turn"
            return super().ask(state, questions)

    gut.configure(backend=Parking(default=0.5), cache=gut.NullCache())

    @semantic
    async def handler(ticket: str) -> float:
        return likely(ticket, "is a bug report").p

    async def releaser() -> None:
        while not entered.is_set():
            await asyncio.sleep(0.005)
        release.set()

    result, _ = await asyncio.gather(handler("a ticket"), releaser())
    assert result == 0.5


@pytest.mark.anyio
async def test_concurrent_handlers_do_not_see_each_others_answers(
    backend: FakeBackend,
) -> None:
    """Each task enters the scope in its own context, so nothing leaks sideways."""
    gut.configure(
        backend=FakeBackend(rule=lambda state, name, spec: 0.1 if state == "first" else 0.9),
        cache=gut.NullCache(),
    )

    @semantic
    async def handler(ticket: str) -> float:
        await asyncio.sleep(0)  # hand the loop to the other task, mid-body
        return likely(ticket, "is a bug report").p

    first, second = await asyncio.gather(handler("first"), handler("second"))
    assert (first, second) == (0.1, 0.9)


@pytest.mark.anyio
async def test_a_coroutine_without_questions_is_untouched(backend: FakeBackend) -> None:
    @semantic
    async def handler(ticket: str) -> str:
        return ticket.upper()

    assert await handler("hi") == "HI"
    assert backend.call_count == 0


@pytest.mark.anyio
async def test_wrong_arguments_still_raise_from_the_coroutine(backend: FakeBackend) -> None:
    @semantic
    async def needs_two(ticket: str, other: str) -> None:
        likely(ticket, "is a bug report")

    with pytest.raises(TypeError):
        await needs_two("only one")  # type: ignore[call-arg]
    assert backend.call_count == 0


@pytest.mark.anyio
async def test_a_failing_prefetch_still_falls_back(caplog: pytest.LogCaptureFixture) -> None:
    class Broken(FakeBackend):
        def ask(self, state: State, questions: Any) -> Any:
            if len(questions) > 1:
                raise BackendError("batches not supported here")
            return super().ask(state, questions)

    gut.configure(backend=Broken(default=0.5), cache=gut.NullCache())

    @semantic
    async def handler(ticket: str) -> tuple[float, float]:
        return (
            likely(ticket, "is a bug report").p,
            likely(ticket, "asks for a refund").p,
        )

    with caplog.at_level(logging.DEBUG, logger="gut"):
        assert await handler("a ticket") == (0.5, 0.5)
    assert any("falling back" in record.message for record in caplog.records)


def test_a_decorated_coroutine_is_still_a_coroutine_function() -> None:
    @semantic
    async def handler(ticket: str) -> None:
        likely(ticket, "is a bug report")

    assert inspect.iscoroutinefunction(handler)
    assert handler.__name__ == "handler"


def test_a_failing_prefetch_falls_back_to_single_calls(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Flaky:
        model_id = "flaky-1.0"

        def __init__(self) -> None:
            self.calls = 0

        def ask(self, state: State, questions: dict[str, QuestionSpec]) -> BackendResponse:
            self.calls += 1
            if len(questions) > 1:
                raise BackendError("batches not supported here")
            return FakeBackend(default=0.5).ask(state, questions)

    flaky = Flaky()
    gut.configure(backend=flaky, cache=gut.NullCache())  # type: ignore[arg-type]

    @semantic
    def two(ticket: str) -> tuple[float, float]:
        return (
            likely(ticket, "is a bug report").p,
            likely(ticket, "asks for a refund").p,
        )

    with caplog.at_level(logging.DEBUG, logger="gut"):
        assert two("a ticket") == (0.5, 0.5)

    assert any("falling back" in record.message for record in caplog.records)
    assert flaky.calls == 3  # one failed batch, then one call per question


def test_wrong_arguments_still_raise_from_the_function(backend: FakeBackend) -> None:
    @semantic
    def needs_two(ticket: str, other: str) -> None:
        likely(ticket, "is a bug report")

    with pytest.raises(TypeError):
        needs_two("only one")  # type: ignore[call-arg]
    assert backend.call_count == 0


def test_a_subject_that_is_not_state_is_skipped(backend: FakeBackend) -> None:
    @semantic
    def numeric(ticket: int) -> None:
        likely(ticket, "is a bug report")  # type: ignore[arg-type]

    assert len(numeric.gut_plan.questions) == 1  # type: ignore[attr-defined]
    numeric(42)
    # Nothing was prefetched, so the body asked normally.
    assert backend.call_count == 1


def test_a_decorated_function_with_no_questions_is_untouched(backend: FakeBackend) -> None:
    @semantic
    def plain(ticket: str) -> str:
        return ticket.upper()

    assert plain("hi") == "HI"
    assert backend.call_count == 0


def test_build_plan_never_raises_on_a_builtin() -> None:
    plan = build_plan(len)
    assert plan.questions == ()
    assert "source unavailable" in str(plan.skipped)


def test_metadata_survives_decoration() -> None:
    @semantic
    def documented(ticket: str) -> None:
        """A docstring."""
        likely(ticket, "is a bug report")

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "A docstring."
