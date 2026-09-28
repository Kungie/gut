"""`OpenAICompatibleBackend` against a real server.

Skipped unless you point it at one, so it never runs in CI:

    ollama pull qwen2.5:1.5b
    GUT_LIVE_BASE_URL=http://localhost:11434/v1 GUT_LIVE_MODEL=qwen2.5:1.5b pytest -m live

It checks plumbing -- that a real server's logprobs are read correctly, both label orders are asked,
and a batch comes back in order -- with cases obvious enough that any competent model agrees.
"""

from __future__ import annotations

import enum
import os

import pytest

import gut

BASE_URL = os.environ.get("GUT_LIVE_BASE_URL", "")
MODEL = os.environ.get("GUT_LIVE_MODEL", "")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not (BASE_URL and MODEL), reason="set GUT_LIVE_BASE_URL and GUT_LIVE_MODEL"),
]

SPAM = "WIN a FREE iPhone!!! Click the link below and enter your card details to claim your prize"
MEETING = "Hi Sam, can we move tomorrow's design review to 3pm? The room is booked until then."


class Team(enum.Enum):
    BILLING = "payments, invoices, refunds and money"
    ENGINEERING = "bugs, crashes, errors and broken software"
    OTHER = "anything else"


@pytest.fixture(autouse=True)
def server() -> None:
    backend = gut.OpenAICompatibleBackend(
        MODEL, base_url=BASE_URL, api_key=os.environ.get("GUT_LIVE_API_KEY")
    )
    gut.configure(backend=backend, cache=gut.NullCache())


def test_obvious_yes_and_no() -> None:
    assert gut.likely(SPAM, "is spam") == gut.YES
    assert gut.likely(MEETING, "is spam") == gut.NO


def test_obvious_classification() -> None:
    decision = gut.classify("The charger you sold me melted. I want my money back.", Team)
    assert decision.value is Team.BILLING


def test_a_rating_lands_between_the_levels() -> None:
    decision = gut.rate(
        "Production is down and the whole team is blocked", ["can wait", "this week", "right now"]
    )
    assert 0.0 <= decision.score <= 2.0
    assert decision.nearest_level >= 1


def test_each_keeps_the_order() -> None:
    decisions = gut.each([SPAM, MEETING, SPAM]).likely("is spam")
    assert [d.outcome for d in decisions] == [gut.YES, gut.NO, gut.YES]
    assert all(d.model for d in decisions)


@pytest.mark.anyio
async def test_awaited_answers_match_blocking_ones() -> None:
    awaited = await gut.alikely(SPAM, "is spam")
    blocking = gut.likely(SPAM, "is spam")
    assert awaited.outcome == blocking.outcome == gut.YES
    assert awaited.p == pytest.approx(blocking.p, abs=0.05)
    decisions = await gut.each([SPAM, MEETING]).alikely("is spam")
    assert [d.outcome for d in decisions] == [gut.YES, gut.NO]
