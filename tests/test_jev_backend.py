"""Tests for the Jev backend.

The translation layer is exercised against the SDK's real question and answer types rather than
hand-rolled doubles, so a change in their shape fails here instead of in production. Only the
transport is stubbed.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import pytest
import typesafe_sdk as ts

import gut
from gut._backends.base import ChoiceAnswer, NoulAnswer, ScoreAnswer
from gut._backends.jev import JevBackend
from gut._errors import BackendError
from gut._questions import ChoiceSpec, NoulSpec, ScoreSpec

BUG = NoulSpec("is a bug report")
DESCRIBED = NoulSpec("is this spam", yes_means="unsolicited advertising", no_means="a real message")
TEAM = ChoiceSpec("which team owns this", {"BILLING": "invoices", "PLATFORM": "outages"})
UNNAMED = ChoiceSpec(None, {"A": None, "B": None})
URGENCY = ScoreSpec(None, ["can wait", "this week", "today"])


def response(answers: Mapping[str, Any], model: str = "jev-1.13.0") -> ts.SystemOneResponse:
    return ts.SystemOneResponse(
        model=model,
        usage=ts.Usage(input_tokens=120, output_tokens=12),
        answers=dict(answers),
    )


class StubClient:
    """Stands in for `TypeSafeClient`, recording calls instead of making them."""

    def __init__(self, reply: object = None, error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def system_one(self, *, state: Any, questions: Any, model: str) -> Any:
        self.calls.append({"state": state, "questions": questions, "model": model})
        if self.error is not None:
            raise self.error
        return self.reply

    def close(self) -> None:
        self.closed = True


def backend_with(
    reply: object = None, error: Exception | None = None, **kwargs: Any
) -> tuple[JevBackend, StubClient]:
    client = StubClient(reply=reply, error=error)
    return JevBackend(client=client, **kwargs), client  # type: ignore[arg-type]


# --------------------------------------------------------------------------- configuration


def test_the_default_model_comes_from_the_sdk() -> None:
    backend, _ = backend_with()
    assert backend.model_id == ts.constants.DEFAULT_MODEL


def test_a_pinned_version_is_what_gets_asked_for() -> None:
    backend, client = backend_with(
        reply=response({"q": ts.NoulAnswer(noul=0.5)}), model="jev-1.13.0"
    )
    assert backend.model_id == "jev-1.13.0"
    backend.ask("a ticket", {"q": BUG})
    assert client.calls[0]["model"] == "jev-1.13.0"


def test_a_supplied_client_is_used_as_configured() -> None:
    with pytest.raises(BackendError, match="used as configured"):
        backend_with(timeout=5.0)
    with pytest.raises(BackendError, match="used as configured"):
        backend_with(api_key="k")


def test_retry_shorthand_and_full_policy_are_mutually_exclusive() -> None:
    with pytest.raises(BackendError, match="max_retries or retry, not both"):
        JevBackend(api_key="k", max_retries=3, retry=ts.RetryPolicy(max_retries=1))


def test_both_retry_forms_build_a_client() -> None:
    """No network happens at construction, so this only proves the arguments are accepted."""
    JevBackend(api_key="k", max_retries=5).close()
    JevBackend(api_key="k", retry=ts.RetryPolicy(max_retries=1, backoff_initial=0.0)).close()


def test_the_underlying_client_is_reachable() -> None:
    backend, client = backend_with()
    # The stub is deliberately not a TypeSafeClient, which is the whole point of substituting it.
    assert backend.client is client  # type: ignore[comparison-overlap]


def test_an_api_key_in_the_environment_infers_the_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Building a billable client implicitly needs an explicit statement of intent (D13)."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-not-used")
    from gut._config import current_backend

    inferred = current_backend()
    assert isinstance(inferred, JevBackend)
    assert inferred.model_id == ts.constants.DEFAULT_MODEL
    inferred.close()


def test_a_blank_api_key_does_not_count_as_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "   ")
    from gut._config import current_backend
    from gut._errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="No backend is configured"):
        current_backend()


def test_gut_exposes_the_backend_lazily() -> None:
    """`import gut` must not drag in the vendor SDK; the attribute resolves on first use."""
    import gut._backends as backends

    assert gut.JevBackend is JevBackend
    assert backends.JevBackend is JevBackend
    with pytest.raises(AttributeError, match="no attribute 'NoSuchBackend'"):
        gut.NoSuchBackend  # noqa: B018
    with pytest.raises(AttributeError, match="no attribute 'NoSuchBackend'"):
        backends.NoSuchBackend  # noqa: B018


# --------------------------------------------------------------------------- questions out


def test_a_plain_yes_no_question_carries_no_criteria() -> None:
    backend, client = backend_with(reply=response({"q": ts.NoulAnswer(noul=0.5)}))
    backend.ask("a ticket", {"q": BUG})
    sent = client.calls[0]["questions"]["q"]
    assert sent.model_dump() == {"type": "noul", "instructions": "is a bug report"}


def test_described_outcomes_become_noul_criteria() -> None:
    backend, client = backend_with(reply=response({"q": ts.NoulAnswer(noul=0.5)}))
    backend.ask("a ticket", {"q": DESCRIBED})
    assert client.calls[0]["questions"]["q"].model_dump() == {
        "type": "noul",
        "instructions": "is this spam",
        "criteria": {"true": "unsolicited advertising", "false": "a real message"},
    }


def test_choice_and_score_questions_translate() -> None:
    answers = {
        "team": ts.ChoiceAnswer(
            choice="BILLING", confidence=0.8, probabilities={"BILLING": 0.8, "PLATFORM": 0.2}
        ),
        "urgency": ts.ScoreAnswer(
            score=1.0, confidence=0.9, probabilities={1: 0.9, 0: 0.1}, legend={0: "a", 1: "b"}
        ),
    }
    backend, client = backend_with(reply=response(answers))
    backend.ask({"subject": "down"}, {"team": TEAM, "urgency": URGENCY})

    sent = client.calls[0]["questions"]
    assert sent["team"].model_dump() == {
        "type": "choice",
        "instructions": "which team owns this",
        "criteria": {"BILLING": "invoices", "PLATFORM": "outages"},
    }
    # An unnamed question sends no instructions at all, which the API allows.
    assert sent["urgency"].model_dump() == {
        "type": "score",
        "criteria": ["can wait", "this week", "today"],
    }
    assert client.calls[0]["state"] == {"subject": "down"}


def test_an_unnamed_choice_sends_only_its_options() -> None:
    backend, client = backend_with(
        reply=response(
            {"q": ts.ChoiceAnswer(choice="A", confidence=0.7, probabilities={"A": 0.7, "B": 0.3})}
        )
    )
    backend.ask("s", {"q": UNNAMED})
    assert client.calls[0]["questions"]["q"].model_dump() == {
        "type": "choice",
        "criteria": {"A": None, "B": None},
    }


def test_every_question_rides_in_one_request() -> None:
    answers = {
        "a": ts.NoulAnswer(noul=0.1),
        "b": ts.NoulAnswer(noul=0.2),
        "c": ts.NoulAnswer(noul=0.3),
    }
    backend, client = backend_with(reply=response(answers))
    result = backend.ask("a ticket", {"a": BUG, "b": DESCRIBED, "c": NoulSpec("asks for a refund")})
    assert len(client.calls) == 1
    assert len(result.answers) == 3


# --------------------------------------------------------------------------- answers back


def test_answers_translate_into_our_shapes() -> None:
    answers = {
        "bug": ts.NoulAnswer(noul=0.91),
        "team": ts.ChoiceAnswer(
            choice="BILLING", confidence=0.8, probabilities={"BILLING": 0.8, "PLATFORM": 0.2}
        ),
        "urgency": ts.ScoreAnswer(
            score=1.3,
            confidence=0.7,
            probabilities={1: 0.7, 2: 0.3},
            legend={0: "a", 1: "b", 2: "c"},
        ),
    }
    backend, _ = backend_with(reply=response(answers))
    result = backend.ask("s", {"bug": BUG, "team": TEAM, "urgency": URGENCY})

    assert result.answers["bug"] == NoulAnswer(p=0.91)
    assert result.answers["team"] == ChoiceAnswer(
        choice="BILLING", confidence=0.8, probabilities={"BILLING": 0.8, "PLATFORM": 0.2}
    )
    assert result.answers["urgency"] == ScoreAnswer(
        score=1.3, confidence=0.7, probabilities={1: 0.7, 2: 0.3}, legend={0: "a", 1: "b", 2: "c"}
    )


def test_the_resolved_version_is_reported_not_the_alias() -> None:
    backend, _ = backend_with(
        reply=response({"q": ts.NoulAnswer(noul=0.5)}, model="jev-1.13.0"), model="jev-latest"
    )
    result = backend.ask("s", {"q": BUG})
    assert backend.model_id == "jev-latest"
    assert result.model == "jev-1.13.0"
    assert result.input_tokens == 120


def test_a_structured_legend_entry_is_rendered_as_text() -> None:
    """We only ever send strings, but the schema allows objects; never crash on one."""
    answer = ts.ScoreAnswer(
        score=0.0, confidence=1.0, probabilities={0: 1.0}, legend={0: {"label": "x"}}
    )
    backend, _ = backend_with(reply=response({"q": answer}))
    result = backend.ask("s", {"q": URGENCY})
    urgency = result.answers["q"]
    assert isinstance(urgency, ScoreAnswer)
    assert urgency.legend[0] == '{"label":"x"}'


# --------------------------------------------------------------------------- failures


def test_an_empty_batch_is_refused_before_the_network() -> None:
    backend, client = backend_with()
    with pytest.raises(BackendError, match="at least one question"):
        backend.ask("s", {})
    assert client.calls == []


def test_a_missing_answer_names_what_is_missing() -> None:
    backend, _ = backend_with(reply=response({"a": ts.NoulAnswer(noul=0.5)}))
    with pytest.raises(BackendError, match="answered without 'b'"):
        backend.ask("s", {"a": BUG, "b": DESCRIBED})


def test_an_unrecognised_answer_type_is_reported() -> None:
    backend, _ = backend_with(reply=type("R", (), {"answers": {"q": object()}, "model": "m"})())
    with pytest.raises(BackendError, match="unrecognised type"):
        backend.ask("s", {"q": BUG})


def test_sdk_errors_are_wrapped_and_the_cause_is_kept() -> None:
    original = ts.TypeSafeError("upstream exploded")
    backend, _ = backend_with(error=original)
    with pytest.raises(BackendError, match="Jev request failed: upstream exploded") as caught:
        backend.ask("s", {"q": BUG})
    assert caught.value.__cause__ is original


def test_we_do_not_stack_a_second_retry_loop() -> None:
    """The SDK already retries with backoff and honours retry-after (D4).

    A failing call must reach the client exactly once from our side; retrying here would double
    the delay on a 429.
    """
    backend, client = backend_with(error=ts.TypeSafeError("boom"))
    with pytest.raises(BackendError):
        backend.ask("s", {"q": BUG})
    assert len(client.calls) == 1


def test_closing_closes_the_client() -> None:
    backend, client = backend_with()
    with backend:
        pass
    assert client.closed is True


# --------------------------------------------------------------------------- live


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("TYPESAFE_API_KEY", "").strip(),
    reason="needs TYPESAFE_API_KEY",
)
def test_live_jev_answers_a_batch() -> None:
    """The one test that touches the real API. Everything else runs offline."""
    with JevBackend() as backend:
        result = backend.ask(
            "If this happens again I am cancelling my subscription.",
            {
                "cancel": NoulSpec("the customer threatens to cancel"),
                "tone": ChoiceSpec(
                    "what is the tone", {"CALM": "neutral or polite", "ANGRY": "upset or hostile"}
                ),
                "urgency": ScoreSpec(None, ["can wait", "this week", "today"]),
            },
        )

    cancel = result.answers["cancel"]
    assert isinstance(cancel, NoulAnswer)
    assert 0.0 <= cancel.p <= 1.0

    tone = result.answers["tone"]
    assert isinstance(tone, ChoiceAnswer)
    assert tone.choice in {"CALM", "ANGRY"}
    assert sum(tone.probabilities.values()) == pytest.approx(1.0, abs=0.01)

    urgency = result.answers["urgency"]
    assert isinstance(urgency, ScoreAnswer)
    assert 0.0 <= urgency.score <= 2.0

    # The response names the version that answered, never the alias we asked for.
    assert result.model != "jev-latest"
    assert result.model.startswith("jev-")
