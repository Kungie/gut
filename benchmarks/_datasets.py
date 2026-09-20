"""The three datasets: where they come from, and what we ask about them.

Every source is pinned to a commit where the host has commits, and verified by SHA-256 either way.
Licences differ and so does what may be committed -- see D31 in DECISIONS.md.

**Question wording was written against the dev sample and then frozen.** Nothing below was changed
after a test set was scored. That is the entire reason the test numbers mean anything, so the
wording lives here as constants rather than being tuned at the call site.
"""

from __future__ import annotations

import csv
import enum
import io
import json
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

from benchmarks._core import Example, Source, fetch
from gut._api import _criteria_from_enum
from gut._questions import ChoiceSpec, NoulSpec, QuestionSpec

Kind = Literal["binary", "choice"]


@dataclass(frozen=True, slots=True)
class Benchmark:
    """One dataset and one question about it."""

    name: str
    kind: Kind
    load: Callable[[], list[Example]]
    spec: QuestionSpec
    what: str
    """One line on what this measures."""
    licence: str
    citation: str
    redistributable: bool
    """Whether the cassette can be committed, which follows from the licence."""
    positive: str | None = None
    """Binary tasks: the label that counts as yes."""
    abstain_label: str | None = None
    """Choice tasks: the catch-all member, if any."""
    oos_label: str | None = None
    keyword: Callable[[str], bool] | None = None
    keyword_name: str = "keyword rule"
    published: str | None = None
    """A trained-model result from the dataset's own paper or competition, for context."""
    quota: dict[str, tuple[int, int]] | None = None
    """Labels sampled to a fixed `(dev, test)` count instead of their natural share."""


# --------------------------------------------------------------------------- CLINC150

CLINC = Source(
    url="https://raw.githubusercontent.com/clinc/oos-eval/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_full.json",
    sha256="36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0",
    revision="828f8093932c8fe6ca7936c3d2e52903b1c523de",
)

OUT_OF_SCOPE: Final = "oos"
ABSTAIN: Final = "OTHER"


def load_clinc() -> list[Example]:
    """Every in-scope and out-of-scope query from the `full` split."""
    document = json.loads(fetch(CLINC))
    examples: list[Example] = []
    for split in ("train", "val", "test"):
        examples.extend(Example(text=text, label=intent) for text, intent in document[split])
    for split in ("oos_train", "oos_val", "oos_test"):
        examples.extend(Example(text=text, label=OUT_OF_SCOPE) for text, _ in document[split])
    return examples


def clinc_intents() -> type[enum.Enum]:
    """The 150 intents plus a catch-all, as the Enum `classify` would be handed.

    Descriptions are the label names with underscores removed -- mechanically, not hand-written.
    Writing 150 descriptions by hand would be tuning the prompt against the labels, which is
    exactly the thing the dev/test split exists to prevent.
    """
    document = json.loads(fetch(CLINC))
    labels = sorted({intent for _, intent in document["train"]})
    members = {label.upper(): label.replace("_", " ") for label in labels}
    members[ABSTAIN] = "none of the above: the request is outside what this assistant handles"
    return enum.Enum("Intent", members)  # type: ignore[return-value]


CLINC_QUESTION: Final = "which of these does the user want"


# --------------------------------------------------------------------------- NLBSE issues

NLBSE_REVISION: Final = "2927bc67eb42db8affd16eaf3e5a6d74f3063961"
NLBSE_TEST = Source(
    url=f"https://raw.githubusercontent.com/nlbse2024/issue-report-classification/{NLBSE_REVISION}/data/issues_test.csv",
    sha256="4f7d8619d4e5adbea126e548fd8c214449288f3a93bb3bc130c54cd307af7e85",
    revision=NLBSE_REVISION,
)
NLBSE_TRAIN = Source(
    url=f"https://raw.githubusercontent.com/nlbse2024/issue-report-classification/{NLBSE_REVISION}/data/issues_train.csv",
    sha256="18dc42a30aa33dccadb723ad3baeb164d38bff521496f985ca2791c26b8939f5",
    revision=NLBSE_REVISION,
)


def load_nlbse() -> list[Example]:
    """Real GitHub issues from five projects, labelled bug / feature / question."""
    # Some issue bodies are far past csv's default 128 KB field cap.
    csv.field_size_limit(10_000_000)
    examples: list[Example] = []
    for source in (NLBSE_TRAIN, NLBSE_TEST):
        # Some rows carry stray NUL bytes, which csv refuses outright.
        text = fetch(source).decode("utf-8", errors="replace").replace("\x00", "")
        rows = csv.DictReader(io.StringIO(text))
        examples.extend(
            Example(
                text=f"{row['title']}\n\n{row['body']}".strip(),
                label=row["label"],
                meta={"repo": row["repo"]},
            )
            for row in rows
        )
    return examples


class IssueKind(enum.Enum):
    """What a GitHub issue is asking for. The labels the dataset uses."""

    BUG = "reports that something is broken or behaving incorrectly"
    FEATURE = "asks for something new, or for an existing behaviour to change"
    QUESTION = "asks how to do something, or for an explanation"


NLBSE_BUG_QUESTION: Final = "this issue reports that something is broken or behaving incorrectly"
NLBSE_KIND_QUESTION: Final = "what kind of issue is this"

BUG_WORDS: Final = re.compile(
    r"\b(bug|error|crash|crashes|crashed|exception|traceback|stack ?trace|fails?|failing|"
    r"failure|broken|breaks?|regression|not work\w*|doesn'?t work|unexpected)\b",
    re.IGNORECASE,
)


def looks_like_a_bug(text: str) -> bool:
    """A regex a developer might reasonably write for this, in good faith."""
    return BUG_WORDS.search(text) is not None


# --------------------------------------------------------------------------- SMS spam

SMS = Source(
    url="https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    sha256="1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3",
)

SPAM: Final = "spam"


def load_sms() -> list[Example]:
    """5,574 real SMS messages, labelled ham or spam."""
    with zipfile.ZipFile(io.BytesIO(fetch(SMS))) as archive:
        raw = archive.read("SMSSpamCollection").decode("utf-8", errors="replace")
    examples: list[Example] = []
    for line in raw.splitlines():
        label, _, text = line.partition("\t")
        if text:
            examples.append(Example(text=text, label=label))
    return examples


SMS_QUESTION: Final = (
    "this message is spam: unsolicited marketing, a prize or competition claim, or an attempt to "
    "get the reader to call a premium number or send money"
)

SPAM_WORDS: Final = re.compile(
    r"\b(free|win|winner|won|prize|cash|claim|urgent|congratulations|txt|text stop|"
    r"guaranteed|award\w*|selected|ringtone|subscri\w+|unsubscribe|call now|click)\b|"
    r"\b(0800|0871|0906|09\d{8})\b|£\d",
    re.IGNORECASE,
)


def looks_like_spam(text: str) -> bool:
    """A keyword filter of the kind the README's pitch says is brittle. Written in good faith."""
    return len(SPAM_WORDS.findall(text)) >= 2


# --------------------------------------------------------------------------- the registry


def benchmarks() -> dict[str, Benchmark]:
    """Every benchmark, by name."""
    intents = clinc_intents()
    return {
        "clinc": Benchmark(
            name="clinc",
            kind="choice",
            load=load_clinc,
            spec=ChoiceSpec(instructions=CLINC_QUESTION, criteria=_criteria_from_enum(intents)),
            what="151-way intent routing, and whether an out-of-scope request is declined",
            licence="CC BY 3.0 per the dataset card; the upstream repository declares none",
            citation="Larson et al. 2019, An Evaluation Dataset for Intent Classification and "
            "Out-of-Scope Prediction (EMNLP-IJCNLP)",
            redistributable=True,
            abstain_label=ABSTAIN,
            oos_label=OUT_OF_SCOPE,
            quota={OUT_OF_SCOPE: (40, 100)},
            published="Larson et al. report 96.9% in-scope accuracy for a fine-tuned BERT on the "
            "full split, with out-of-scope recall far lower",
        ),
        "nlbse-bug": Benchmark(
            name="nlbse-bug",
            kind="binary",
            load=load_nlbse,
            spec=NoulSpec(NLBSE_BUG_QUESTION),
            what="is this GitHub issue a bug report",
            licence="none declared; not redistributable",
            citation="NLBSE'24 issue report classification tool competition",
            redistributable=False,
            positive="bug",
            keyword=looks_like_a_bug,
            keyword_name="regex: bug/error/crash/...",
            published="NLBSE'24 baselines report ~0.87 F1 for a fine-tuned RoBERTa across repos",
        ),
        "nlbse-kind": Benchmark(
            name="nlbse-kind",
            kind="choice",
            load=load_nlbse,
            spec=ChoiceSpec(
                instructions=NLBSE_KIND_QUESTION, criteria=_criteria_from_enum(IssueKind)
            ),
            what="routing a GitHub issue to bug / feature / question",
            licence="none declared; not redistributable",
            citation="NLBSE'24 issue report classification tool competition",
            redistributable=False,
            published="NLBSE'24 baselines report ~0.87 F1 for a fine-tuned RoBERTa across repos",
        ),
        "sms": Benchmark(
            name="sms",
            kind="binary",
            load=load_sms,
            spec=NoulSpec(SMS_QUESTION),
            what="a single yes/no judgment, the cleanest calibration check available",
            licence="CC BY 4.0",
            citation="Almeida, Gomez Hidalgo & Yamakami 2011, SMS Spam Collection v.1 "
            "(UCI, DOI 10.24432/C5CC84)",
            redistributable=True,
            positive=SPAM,
            keyword=looks_like_spam,
            keyword_name="keyword filter: free/win/prize/...",
            published="Almeida et al. report ~97.6% accuracy for an SVM trained on this corpus",
        ),
    }
