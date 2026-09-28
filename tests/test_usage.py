"""Tests for `gut.usage()`: counting what calls cost, and stopping at a budget."""

from __future__ import annotations

import asyncio
import enum
from collections.abc import Mapping

import pytest

import gut
from gut import BudgetExceeded, ConfigurationError, FakeBackend
from gut._backends.base import BackendResponse, NoulAnswer, add_cost, reported_cost


class Team(enum.Enum):
    BILLING = "invoices"
    OTHER = "anything else"


def priced(cost: float | None = 0.001) -> FakeBackend:
    return FakeBackend(rule=gut.deterministic_rule, cost=cost)


def test_calls_and_their_cost_are_counted() -> None:
    gut.configure(backend=priced(0.001))
    with gut.usage() as spent:
        gut.likely("a", "is spam")
        gut.classify("b", Team)
        gut.rate("c", ["low", "high"])
    assert spent.calls == 3
    assert spent.cost == pytest.approx(0.003)
    assert spent.cost_known
    assert str(spent) == "3 calls, $0.003"
    assert spent.to_dict() == {
        "calls": 3,
        "input_tokens": 0,
        "cost": 0.003,
        "unpriced_calls": 0,
        "max_cost": None,
    }


def test_an_answer_from_the_cache_costs_nothing() -> None:
    gut.configure(backend=priced())
    with gut.usage() as spent:
        gut.likely("a", "is spam")
        gut.likely("a", "is spam")
    assert spent.calls == 1


def test_calls_outside_the_block_are_not_counted() -> None:
    gut.configure(backend=priced())
    gut.likely("before", "is spam")
    with gut.usage() as spent:
        pass
    gut.likely("after", "is spam")
    assert spent.calls == 0
    assert str(spent) == "0 calls, $0"


def test_each_counts_one_call_per_subject() -> None:
    gut.configure(backend=priced(0.002))
    with gut.usage() as spent:
        gut.each(["a", "b", "c", "a"]).likely("is spam")
    assert spent.calls == 3
    assert spent.cost == pytest.approx(0.006)


def test_a_prefetched_batch_is_counted_once() -> None:
    gut.configure(backend=priced(0.004))
    with gut.usage() as spent:
        with gut.judge("ticket") as j:
            bug = j.likely("is a bug report")
            team = j.classify(Team)
        assert bug.outcome in (gut.YES, gut.NO)
        assert team.value in Team
    assert spent.calls == 1
    assert spent.cost == pytest.approx(0.004)


def test_tokens_are_summed_when_reported() -> None:
    class Tokens(FakeBackend):
        def ask(
            self, state: gut.State, questions: Mapping[str, gut.QuestionSpec]
        ) -> BackendResponse:
            response = super().ask(state, questions)
            return BackendResponse(answers=response.answers, model=response.model, input_tokens=50)

    gut.configure(backend=Tokens(default=0.9))
    with gut.usage() as spent:
        gut.likely("a", "is spam")
        gut.likely("b", "is spam")
    assert spent.input_tokens == 100
    assert str(spent) == "2 calls, 100 input tokens, cost not reported"


def test_a_backend_that_does_not_report_cost_is_counted_as_unpriced() -> None:
    gut.configure(backend=priced(None))
    with gut.usage() as spent:
        gut.likely("a", "is spam")
    assert not spent.cost_known
    assert spent.unpriced_calls == 1
    assert spent.to_dict()["cost"] is None
    assert str(spent) == "1 call, cost not reported"


def test_blocks_nest_and_each_counts_what_ran_inside_it() -> None:
    gut.configure(backend=priced(0.001))
    with gut.usage() as outer:
        gut.likely("a", "is spam")
        with gut.usage() as inner:
            gut.likely("b", "is spam")
    assert (outer.calls, inner.calls) == (2, 1)


# --------------------------------------------------------------------------- budgets


def test_a_budget_stops_the_call_that_would_follow_it() -> None:
    gut.configure(backend=priced(0.001))
    with gut.usage(max_cost=0.0025) as spent:
        for text in ("a", "b", "c"):
            gut.likely(text, "is spam")
        with pytest.raises(BudgetExceeded, match=r"\$0.003 spent of a \$0.0025 budget") as caught:
            gut.likely("d", "is spam")
    assert caught.value.usage is spent
    assert spent.calls == 3


def test_a_budget_still_allows_the_cache() -> None:
    gut.configure(backend=priced(0.01))
    with gut.usage(max_cost=0.01):
        gut.likely("a", "is spam")
        gut.likely("a", "is spam")  # answered from the cache: no call, so nothing to stop


def test_a_budget_is_checked_before_each_batch() -> None:
    gut.configure(backend=priced(0.01))
    with gut.usage(max_cost=0.015):
        gut.each(["a", "b"]).likely("is spam")  # starts under budget, so the batch runs
        with pytest.raises(BudgetExceeded):
            gut.each(["c", "d"]).likely("is spam")


def test_a_budget_cannot_be_kept_without_a_reported_cost() -> None:
    gut.configure(backend=priced(None))
    with gut.usage(max_cost=1.0):
        gut.likely("a", "is spam")
        with pytest.raises(BudgetExceeded, match="does not report"):
            gut.likely("b", "is spam")


def test_a_budget_of_zero_allows_what_is_free() -> None:
    gut.configure(backend=priced(0.0))
    with gut.usage(max_cost=0) as spent:
        gut.each(["a", "b", "c"]).likely("is spam")
    assert spent.calls == 3


def test_a_negative_budget_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="zero or more"), gut.usage(max_cost=-1):
        pass  # pragma: no cover


# --------------------------------------------------------------------------- async


@pytest.mark.anyio
async def test_awaited_calls_are_counted_and_budgeted() -> None:
    gut.configure(backend=priced(0.001))
    with gut.usage(max_cost=0.0035) as spent:
        await gut.alikely("a", "is spam")
        await gut.each(["b", "c"]).alikely("is spam")
        both = await asyncio.gather(gut.aclassify("d", Team), gut.arate("e", ["low", "high"]))
        assert len(both) == 2
        with pytest.raises(BudgetExceeded):
            await gut.alikely("f", "is spam")
    assert spent.calls == 5


@pytest.mark.anyio
async def test_an_awaited_prefetch_from_the_cache_needs_no_budget() -> None:
    gut.configure(backend=priced(0.01))
    gut.likely("ticket", "is a bug report")
    with gut.usage(max_cost=0.001) as spent:
        gut.likely("paid", "is spam")  # spends the whole budget
        with gut.judge("ticket") as j:
            bug = j.likely("is a bug report")
        await j.aresolve()  # already cached: nothing is sent, so nothing is refused
        assert bug.outcome in (gut.YES, gut.NO)
    assert spent.calls == 1


@pytest.mark.anyio
async def test_an_awaited_prefetch_is_counted() -> None:
    gut.configure(backend=priced(0.002))
    with gut.usage() as spent:
        with gut.judge("ticket") as j:
            j.likely("is a bug report")
            j.classify(Team)
        await j.aresolve()
    assert spent.calls == 1


# --------------------------------------------------------------------------- the pieces


def test_a_cascade_reports_what_all_its_stages_cost() -> None:
    first = FakeBackend(default=0.5, cost=0.0, model="small")  # unsure: escalates
    second = FakeBackend(default=0.95, cost=0.01, model="big")
    gut.configure(backend=gut.Cascade(first, second))
    with gut.usage() as spent:
        gut.likely("a", "is spam")
    assert spent.cost == pytest.approx(0.01)
    third = FakeBackend(default=0.95, cost=None)
    response = gut.Cascade(first, third).ask("b", {"q": gut.NoulSpec("is spam")})
    assert response.cost is None


def test_costs_add_until_one_is_unknown() -> None:
    assert add_cost(0.0, 0.5) == 0.5
    assert add_cost(0.5, None) is None
    assert add_cost(None, 0.5) is None


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"cost": 0.25}, 0.25),
        ({"cost": 0}, 0.0),
        ({"cost": -1}, None),
        ({"cost": "0.1"}, None),
        ({"cost": True}, None),
        ({}, None),
        (None, None),
    ],
)
def test_a_reported_cost_is_only_ever_a_price(usage: object, expected: float | None) -> None:
    assert reported_cost(usage) == expected


def test_a_free_answer_still_counts_as_a_call() -> None:
    response = BackendResponse(answers={"q": NoulAnswer(p=0.5)}, model="m", cost=0.0)
    with gut.usage() as spent:
        from gut._usage import after_call

        after_call([response])
    assert (spent.calls, spent.cost, spent.cost_known) == (1, 0.0, True)
