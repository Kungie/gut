"""Tests for the awaitable API: the same decisions, without blocking the event loop."""

from __future__ import annotations

import asyncio
import enum
import logging
import threading
from collections.abc import Mapping, Sequence

import pytest

import gut
from gut import BackendError, FakeBackend
from gut._backends._many import Item, aask_all, aask_one
from gut._batching import afetch

pytestmark = pytest.mark.anyio


class Team(enum.Enum):
    BILLING = "invoices"
    PLATFORM = "outages"
    OTHER = "anything else"


def by_text(state: gut.State, name: str, spec: gut.QuestionSpec) -> float | str | int | None:
    match spec:
        case gut.NoulSpec():
            return 0.97 if "WIN" in str(state) else 0.5 if "maybe" in str(state) else 0.02
        case gut.ChoiceSpec():
            return "BILLING" if "invoice" in str(state) else "OTHER"
        case _:
            return 2 if "down" in str(state) else 0


class Native(FakeBackend):
    """A backend that speaks `async` and refuses to be asked synchronously."""

    in_flight: int
    most_in_flight: int

    def ask(self, state, questions):  # type: ignore[no-untyped-def]
        raise AssertionError("an async backend must be awaited, not asked synchronously")

    async def aask(
        self, state: gut.State, questions: Mapping[str, gut.QuestionSpec]
    ) -> gut.BackendResponse:
        self.in_flight = getattr(self, "in_flight", 0) + 1
        self.most_in_flight = max(getattr(self, "most_in_flight", 0), self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        return FakeBackend.ask(self, state, questions)


@pytest.fixture
def backend() -> FakeBackend:
    fake = FakeBackend(rule=by_text)
    gut.configure(backend=fake, cache=gut.NullCache())
    return fake


# --------------------------------------------------------------------------- one subject


async def test_the_awaitable_forms_decide_exactly_as_the_blocking_ones(
    backend: FakeBackend,
) -> None:
    for text in ("WIN A PRIZE", "lunch?", "maybe a deal"):
        awaited = await gut.alikely(text, "is spam", ask_human=True)
        blocking = gut.likely(text, "is spam", ask_human=True)
        assert (awaited.outcome, awaited.p) == (blocking.outcome, blocking.p)
    team = await gut.aclassify("invoice is wrong", Team, question="which team")
    assert team.value is Team.BILLING
    assert team.question == "which team"
    urgency = await gut.arate(
        "site is down", ["later", "soon", "now"], stakes="low", ask_human=True
    )
    assert urgency.nearest_level == 2
    assert urgency.min_confidence == 0.5


async def test_a_blocking_backend_runs_in_a_worker_thread(backend: FakeBackend) -> None:
    threads: list[str] = []
    original = backend.ask

    def ask(state, questions):  # type: ignore[no-untyped-def]
        threads.append(threading.current_thread().name)
        return original(state, questions)

    backend.ask = ask  # type: ignore[method-assign]
    await gut.alikely("x", "is spam")
    assert threads and threads[0] != threading.main_thread().name


async def test_a_native_backend_is_awaited_and_never_blocks() -> None:
    native = Native(rule=by_text)
    gut.configure(backend=native, cache=gut.NullCache())
    decisions = await asyncio.gather(*(gut.alikely(f"WIN {i}", "is spam") for i in range(5)))
    assert all(d == gut.YES for d in decisions)
    assert native.most_in_flight == 5  # all five were waiting at once: nothing blocked the loop


async def test_the_cache_serves_awaited_questions_too() -> None:
    fake = FakeBackend(rule=by_text)
    gut.configure(backend=fake)
    first = await gut.alikely("WIN", "is spam")
    second = await gut.alikely("WIN", "is spam")
    assert (first.source, second.source) == ("backend", "cache")
    assert first.latency_ms is not None
    assert fake.call_count == 1


async def test_a_backend_that_forgets_the_answer_is_an_error() -> None:
    class Empty(Native):
        async def aask(self, state, questions):  # type: ignore[no-untyped-def]
            return gut.BackendResponse(answers={}, model="empty")

    with pytest.raises(BackendError, match="no answer for the question asked"):
        await gut.alikely("x", "q", backend=Empty())


async def test_posture_mistakes_are_caught_before_anything_is_awaited(backend: FakeBackend) -> None:
    with pytest.raises(gut.PolicyError, match="not both"):
        await gut.alikely("x", "q", ask_human=True, cost_false_yes=1, cost_false_no=1)
    with pytest.warns(UserWarning, match="no catch-all member"):

        class Colour(enum.Enum):
            RED = "red"
            BLUE = "blue"

        await gut.aclassify("x", Colour, backend=FakeBackend(default="RED"))


# --------------------------------------------------------------------------- many subjects


async def test_each_awaits_the_same_decisions(backend: FakeBackend) -> None:
    texts = ["WIN A PRIZE", "lunch?", "WIN A PRIZE"]
    awaited = await gut.each(texts).alikely("is spam")
    assert [d.outcome for d in awaited] == [gut.YES, gut.NO, gut.YES]
    assert [d.source for d in awaited] == ["batch"] * 3
    assert backend.call_count == 2
    teams = await gut.each(["invoice", "hello"]).aclassify(Team, stakes="high", ask_human=True)
    assert [d.value for d in teams] == [Team.BILLING, Team.OTHER]
    levels = await gut.each(["down", "fine"]).arate(["later", "soon", "now"])
    assert [d.nearest_level for d in levels] == [2, 0]
    with pytest.raises(gut.PolicyError, match="not both"):
        await gut.each(["x"]).alikely("q", ask_human=True, cost_false_yes=1, cost_false_no=1)


async def test_each_keeps_a_native_backend_to_its_concurrency() -> None:
    native = Native(rule=by_text)
    gut.configure(cache=gut.NullCache())
    texts = [f"WIN {i}" for i in range(10)]
    decisions = await gut.each(texts, backend=native, concurrency=3).alikely("is spam")
    assert len(decisions) == 10
    assert native.most_in_flight == 3


async def test_each_hands_a_native_batch_everything_at_once() -> None:
    class Batched(Native):
        batches: list[int]

        async def aask_many(self, items):  # type: ignore[no-untyped-def]
            self.batches.append(len(items))
            return [FakeBackend.ask(self, state, questions) for state, questions in items]

    batched = Batched(default=0.9)
    batched.batches = []
    gut.configure(cache=gut.NullCache())
    await gut.each(["a", "b", "c"], backend=batched).alikely("q")
    assert batched.batches == [3]


async def test_a_blocking_batch_backend_still_batches_from_a_thread() -> None:
    class Batched(FakeBackend):
        batches: list[int]

        def ask_many(self, items):  # type: ignore[no-untyped-def]
            self.batches.append(len(items))
            return [self.ask(state, questions) for state, questions in items]

    batched = Batched(default=0.9)
    batched.batches = []
    gut.configure(cache=gut.NullCache())
    await gut.each(["a", "b", "c"], backend=batched).alikely("q")
    assert batched.batches == [3]


async def test_aask_all_checks_its_arguments_and_its_backend() -> None:
    with pytest.raises(BackendError, match="concurrency must be at least 1"):
        await aask_all(FakeBackend(default=0.9), [], concurrency=0)
    assert await aask_all(FakeBackend(default=0.9), []) == []

    class Short(Native):
        async def aask_many(self, items):  # type: ignore[no-untyped-def]
            return []

    item: list[Item] = [("a", {"q": gut.NoulSpec("q")})]
    with pytest.raises(BackendError, match="aask_many returned 0 responses for 1 subjects"):
        await aask_all(Short(default=0.9), item)


async def test_aask_one_prefers_the_native_path() -> None:
    response = await aask_one(Native(default=0.9), "a", {"q": gut.NoulSpec("q")})
    assert response.answers["q"] == gut.NoulAnswer(p=0.9)


# --------------------------------------------------------------------------- judge and @semantic


async def test_a_judge_resolves_on_the_event_loop() -> None:
    native = Native(rule=by_text)
    gut.configure(backend=native, cache=gut.NullCache())
    with gut.judge("WIN an invoice") as j:
        spam = j.likely("is spam", ask_human=True)
        team = j.classify(Team)
    await j.aresolve()
    await j.aresolve()  # nothing left pending: no second request
    assert j.requests == 1
    assert spam == gut.YES  # reading no longer asks anything: the backend would refuse
    assert team.value is Team.BILLING


async def test_a_split_batch_goes_out_concurrently() -> None:
    class Tight(Native):
        context_limit_tokens = 40  # small enough that every question needs its own request

    tight = Tight(default=0.9)
    gut.configure(cache=gut.NullCache())
    specs: Sequence[gut.QuestionSpec] = [
        gut.NoulSpec(f"question number {i} " * 3) for i in range(4)
    ]
    answers = await afetch("a subject", specs, tight)
    assert set(answers) == set(specs)
    assert tight.call_count == 4
    assert tight.most_in_flight == 4


@gut.semantic
async def handle(ticket: str, note: str = "") -> tuple[gut.Decision, gut.Decision, gut.Decision]:
    spam = await gut.alikely(ticket, "is spam")
    urgent = gut.likely(ticket, "is urgent")
    noted = await gut.alikely(note, "is spam")
    return spam, urgent, noted


async def test_semantic_collects_awaited_judgments_and_prefetches_natively() -> None:
    native = Native(rule=by_text)
    gut.configure(backend=native, cache=gut.NullCache())
    spam, urgent, noted = await handle("WIN A PRIZE", note="lunch?")
    assert [d.source for d in (spam, urgent, noted)] == ["prefetch"] * 3
    assert spam == gut.YES
    assert noted == gut.NO
    assert native.call_count == 2  # one request per parameter, both in flight together
    assert native.most_in_flight == 2
    assert {q.spec.instructions for q in handle.gut_plan.questions} == {"is spam", "is urgent"}  # type: ignore[attr-defined]


async def test_a_failed_prefetch_leaves_the_body_to_ask(caplog: pytest.LogCaptureFixture) -> None:
    class Flaky(FakeBackend):
        """Fails any batch; answers single questions, awaited or not."""

        async def aask(self, state, questions):  # type: ignore[no-untyped-def]
            if len(questions) > 1:
                raise BackendError("the batch endpoint is down")
            return self.ask(state, questions)

    gut.configure(backend=Flaky(rule=by_text), cache=gut.NullCache())
    with caplog.at_level(logging.DEBUG, logger="gut"):
        spam, urgent, _ = await handle("WIN A PRIZE")
    assert (spam.source, urgent.source) == ("backend", "backend")
    assert "prefetch for 'ticket' failed" in caplog.text


async def test_a_cancelled_prefetch_is_not_swallowed() -> None:
    class Cancelled(Native):
        async def aask(self, state, questions):  # type: ignore[no-untyped-def]
            raise asyncio.CancelledError

    gut.configure(backend=Cancelled(default=0.9), cache=gut.NullCache())
    with pytest.raises(asyncio.CancelledError):
        await handle("x")


# --------------------------------------------------------------------------- cascades


async def test_a_cascade_awaits_each_stage_and_escalates_only_the_unsure() -> None:
    cheap = Native(rule=by_text, model="cheap")
    costly = Native(default=0.95, model="costly")
    cascade = gut.Cascade(cheap, costly)
    gut.configure(backend=cascade, cache=gut.NullCache())
    decisions = await gut.each(["WIN", "lunch?", "maybe a deal"]).alikely("is spam")
    assert [d.model for d in decisions] == ["cheap", "cheap", "costly"]
    single = await cascade.aask("maybe", {"q": gut.NoulSpec("is spam")})
    assert single.model_for("q") == "costly"


async def test_a_cascade_mixes_blocking_and_native_stages(caplog: pytest.LogCaptureFixture) -> None:
    broken = FakeBackend(model="broken")  # blocking, and every question fails
    cascade = gut.Cascade(broken, Native(default=0.9, model="spare"))
    with caplog.at_level(logging.INFO, logger="gut"):
        response = await cascade.aask("x", {"q": gut.NoulSpec("q")})
    assert response.model_for("q") == "spare"
    assert "broken failed, escalating" in caplog.text


async def test_a_cascade_whose_last_stage_fails_raises() -> None:
    cascade = gut.Cascade(Native(default=0.5), FakeBackend(model="last"))
    with pytest.raises(BackendError, match="no fixture"):
        await cascade.aask("x", {"q": gut.NoulSpec("q")})


async def test_a_cascade_stops_early_when_everything_is_settled() -> None:
    second = Native(default=0.5, model="second")
    cascade = gut.Cascade(Native(default=0.99), second)
    await cascade.aask_many([("a", {"q": gut.NoulSpec("q")})])
    assert second.call_count == 0
