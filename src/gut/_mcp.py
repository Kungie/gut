"""gut as an MCP server, so an agent can hand its judgment calls to a small model.

    uvx gutfeel-mcp

Four tools, `likely`, `classify`, `rate` and `each`, over stdio. They take the same arguments as the
Python functions and answer with `Decision.to_dict()`, so an agent sees the outcome, the model's
probability and which model gave it. The model comes from the environment, Jev first:

| variable | meaning |
|---|---|
| `TYPESAFE_API_KEY` | use TypeSafe's Jev -- the default whenever it is set |
| `GUT_BACKEND` | one of `gut._env.BACKENDS`: `jev`, `openrouter`, `ollaya`, `ollama`, ... |
| `GUT_MODEL` | the model to ask, for backends that take one |
| `GUT_BASE_URL` | an OpenAI-compatible server's URL, for `openai` and `ollama` |

The backend is built on the first call, not at startup, so the handshake is instant even for a
local model that takes a few seconds to load.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Annotated, Any

import anyio.to_thread
from pydantic import Field

from gut._api import aclassify, alikely, arate
from gut._backends import Backend
from gut._decision import BaseDecision
from gut._each import each as each_subject
from gut._env import NO_MODEL, FromEnvironment
from gut._errors import ConfigurationError, GutError
from gut._options import catch_all_note as _note
from gut._options import options_enum
from gut._posture import Lean, Stakes
from gut._usage import usage

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

MAX_SUBJECTS = 1000

INSTRUCTIONS = """\
gut answers judgment calls about text with a small, fast, cheap model, and says when it does not \
know.

Use it for questions whose answer is a judgment rather than a fact you can compute: is this \
comment spam, which team owns this ticket, how urgent is this email. It is at its best on many \
texts at once: use `each` to triage a whole inbox or a page of search results in one call, where \
reading every item yourself would be slow and expensive.

Every answer carries an outcome and a probability (or a confidence). `unsure` only happens when \
you pass ask_human=true, and means the model could not tell: read the text yourself or ask the \
user. Confidence is how peaked the model's answer is, not the probability that it is right.

Questions work best as short, concrete claims about the text ("asks for a refund", "is written \
in German"), not as open questions or judgments of quality.
"""

Subject = Annotated[str, Field(description="The text to judge: an email, a comment, a diff.")]
Question = Annotated[
    str,
    Field(
        description='A short claim about the text, e.g. "asks for a refund". Not an open question.'
    ),
]
Options = Annotated[
    dict[str, str] | list[str],
    Field(
        description="The categories: {label: description}, or a list of labels. Include a "
        'catch-all such as "other", or every text is forced into one of them.'
    ),
]
Levels = Annotated[
    list[str],
    Field(description="Between 2 and 10 level descriptions, in order; the first is level 0."),
]
AskHuman = Annotated[
    bool,
    Field(description="Allow the outcome `unsure` when the model cannot tell. Off by default."),
]
StakesArg = Annotated[
    Stakes | None,
    Field(description="How sure the model must be before an answer counts. Needs ask_human."),
]
LeanArg = Annotated[Lean | None, Field(description="Which way to err when unsure is not allowed.")]


def report(decision: BaseDecision) -> dict[str, Any]:
    """A decision as an agent needs it.

    Without the call-site id and the source, which describe this server's code rather than the
    answer.
    """
    record = decision.to_dict()
    record.pop("id", None)
    record.pop("source", None)
    return record


def _check_stakes(stakes: Stakes | None, ask_human: bool) -> None:
    # A program gets a warning for this; an agent would never see one, so it is an error here.
    if stakes is not None and not ask_human:
        raise ConfigurationError(
            "stakes sets how sure the model must be before an answer counts, so it only matters "
            "with ask_human=true, which lets the answer be unsure. Pass both, or neither."
        )


class Tools:
    """What the server exposes, over one lazily built backend."""

    def __init__(self, backend: Backend | None = None) -> None:
        self._configured = FromEnvironment()
        self._fixed = backend

    async def backend(self) -> Backend | None:
        """The backend to ask; building one may load a model, so that happens off the loop."""
        if self._fixed is not None:
            return self._fixed
        return await anyio.to_thread.run_sync(self._configured.get)

    async def likely(
        self,
        subject: Subject,
        question: Question,
        ask_human: AskHuman = False,
        stakes: StakesArg = None,
        lean: LeanArg = None,
    ) -> dict[str, Any]:
        """Judge whether a claim is true of a text. Answers yes or no with the model's
        probability that the claim is true -- or unsure, when ask_human is set."""
        _check_stakes(stakes, ask_human)
        decision = await alikely(
            subject,
            question,
            ask_human=ask_human,
            stakes=stakes,
            lean=lean,
            backend=await self.backend(),
        )
        return report(decision)

    async def classify(
        self,
        subject: Subject,
        options: Options,
        question: Annotated[
            str | None, Field(description="What to decide, if the options do not say it.")
        ] = None,
        ask_human: AskHuman = False,
        stakes: StakesArg = None,
    ) -> dict[str, Any]:
        """Pick the option that fits a text best: a team, a topic, a language, an intent. Answers
        with the chosen label, every option's probability and a confidence."""
        _check_stakes(stakes, ask_human)
        enum_class = options_enum(options)
        decision = await aclassify(
            subject,
            enum_class,
            question=question,
            ask_human=ask_human,
            stakes=stakes,
            backend=await self.backend(),
        )
        return report(decision) | _note(enum_class)

    async def rate(
        self,
        subject: Subject,
        levels: Levels,
        question: Annotated[
            str | None, Field(description="What to rate, if the levels do not say it.")
        ] = None,
        ask_human: AskHuman = False,
        stakes: StakesArg = None,
    ) -> dict[str, Any]:
        """Rate a text on an ordered scale such as urgency or severity. Answers with a score,
        which can fall between levels, the nearest level and a confidence."""
        _check_stakes(stakes, ask_human)
        decision = await arate(
            subject,
            levels,
            question=question,
            ask_human=ask_human,
            stakes=stakes,
            backend=await self.backend(),
        )
        return report(decision)

    async def each(
        self,
        subjects: Annotated[
            list[str], Field(description="The texts to judge, all with the same question.")
        ],
        question: Annotated[
            str | None,
            Field(
                description="A claim, as for likely. Alone, it asks likely; with options or "
                "levels, it says what to decide."
            ),
        ] = None,
        options: Annotated[
            dict[str, str] | list[str] | None,
            Field(description="Categories, as for classify."),
        ] = None,
        levels: Annotated[list[str] | None, Field(description="A scale, as for rate.")] = None,
        ask_human: AskHuman = False,
        stakes: StakesArg = None,
    ) -> dict[str, Any]:
        """Ask the same question about many texts at once: likely with just a question, classify
        with options, rate with levels. Far faster than one call per text."""
        _check_stakes(stakes, ask_human)
        if options is not None and levels is not None:
            raise ConfigurationError("Pass options or levels, not both.")
        if len(subjects) > MAX_SUBJECTS:
            raise ConfigurationError(f"At most {MAX_SUBJECTS} subjects per call.")
        batch = each_subject(subjects, backend=await self.backend())
        decisions: Sequence[BaseDecision]
        if options is not None:
            enum_class = options_enum(options)
            decisions = await batch.aclassify(
                enum_class, question=question, ask_human=ask_human, stakes=stakes
            )
            return {"results": [report(d) for d in decisions]} | _note(enum_class)
        if levels is not None:
            decisions = await batch.arate(
                levels, question=question, ask_human=ask_human, stakes=stakes
            )
        elif question:
            decisions = await batch.alikely(question, ask_human=ask_human, stakes=stakes)
        else:
            raise ConfigurationError("Pass a question, options or levels.")
        return {"results": [report(d) for d in decisions]}


Handler = Callable[..., Awaitable[dict[str, Any]]]


def _as_tool_errors(method: Handler, tool_error: type[Exception]) -> Handler:
    """Hand gut's own errors to the agent as tool errors, whose text it gets to read.

    Anything else is a crash, which the SDK reports without the details.
    """

    @functools.wraps(method)
    async def call(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            with usage() as spent:
                result = await method(*args, **kwargs)
            if spent.calls:
                # What this answer cost, when the backend says: an agent can keep its own tally.
                result["usage"] = {"calls": spent.calls, "cost": spent.to_dict()["cost"]}
            return result
        except GutError as error:
            text = NO_MODEL if "No backend is configured" in str(error) else str(error)
            raise tool_error(text) from error

    return call


def build(backend: Backend | None = None) -> MCPServer:
    """The MCP server, with `backend` if given and otherwise the one the environment names."""
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp.types import ToolAnnotations
    except ImportError as error:  # pragma: no cover - the extra is part of the dev environment
        raise ConfigurationError("The MCP server needs its SDK: pip install gutfeel-mcp") from error

    from gut import __version__

    tools = Tools(backend)
    server: MCPServer = MCPServer(
        "gut",
        title="gut",
        description="Judgment calls on small, fast, cheap models.",
        instructions=INSTRUCTIONS,
        website_url="https://gutpy.dev/",
        version=__version__,
    )
    # Asking a model changes nothing, and the same question gets the same answer.
    annotations = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)
    for method in (tools.likely, tools.classify, tools.rate, tools.each):
        server.tool(annotations=annotations, structured_output=True)(
            _as_tool_errors(method, ToolError)
        )
    return server


def main() -> None:  # pragma: no cover - run over stdio by the end-to-end test, in a subprocess
    """Serve over stdio: the entry point behind `gutfeel-mcp`."""
    build().run("stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
