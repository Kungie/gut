"""`@semantic`: collapsing a function's judgments into one request.

The decorator reads the function's source once, at decoration time, and works out which questions it
will ask about which of its own parameters. At call time it asks them all up front -- one request
per subject -- and leaves the answers in a prefetch scope. The function body then runs completely
unchanged; its `likely(...)` calls find their answers waiting.

Two properties matter more than coverage here:

**It never changes what your code does.** Anything the analysis cannot prove statically is simply
not collected, and that call goes to the backend on its own exactly as it would without the
decorator. A mismatch between what was planned and what is actually asked cannot produce a wrong
answer either, because the scope is keyed by a fingerprint of the real question and the real
subject: a plan that guessed wrong just fails to match and the normal path takes over.

**It is speculative.** Questions behind branches that never run are still asked. That is the trade
the decorator exists to make: billing is on input tokens, the state is paid for once per request,
and so five questions in one call cost barely more than one. If a question is expensive for reasons
other than tokens, keep it out of a decorated function.

Coroutine functions work the same way. The prefetch is the only call that touches the network, so
it is run in a worker thread and awaited; everything after it is answered from memory and never
blocks the loop.
"""

from __future__ import annotations

import ast
import asyncio
import functools
import inspect
import logging
import textwrap
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ParamSpec, TypeVar, cast

from gut._api import _criteria_from_enum, classify, likely, rate
from gut._backends.base import Backend
from gut._batching import fetch
from gut._config import current_backend
from gut._errors import GutError
from gut._questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec
from gut._scope import Prefetch, prefetch_scope

logger = logging.getLogger("gut")

P = ParamSpec("P")
R = TypeVar("R")

_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


@dataclass(frozen=True, slots=True)
class PlannedQuestion:
    """One question the decorator expects the body to ask."""

    parameter: str
    """The function parameter holding the subject."""
    spec: QuestionSpec
    """The question, fully determined at decoration time."""


@dataclass(frozen=True, slots=True)
class Plan:
    """What `@semantic` worked out about a function."""

    questions: tuple[PlannedQuestion, ...] = ()
    skipped: str | None = None
    """Why nothing was planned, when nothing was. Informational; an empty plan is not an error."""

    def by_parameter(self) -> dict[str, list[QuestionSpec]]:
        """Planned questions grouped by the parameter they ask about."""
        grouped: dict[str, list[QuestionSpec]] = {}
        for question in self.questions:
            grouped.setdefault(question.parameter, []).append(question.spec)
        return grouped


# --------------------------------------------------------------------------- static analysis


def _function_node(source: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    module = ast.parse(textwrap.dedent(source))
    for node in module.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return node
    return None


def _parameter_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = node.args
    every = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    if args.vararg is not None:
        every.append(args.vararg)
    if args.kwarg is not None:
        every.append(args.kwarg)
    return {argument.arg for argument in every}


def _rebound_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Names the body assigns to, deletes, or declares global or nonlocal.

    A parameter that gets reassigned no longer refers to the value the caller passed, so prefetching
    against the original would answer about the wrong state. Such parameters are excluded entirely
    rather than tracked through the flow -- being conservative here costs a batch, being clever and
    wrong costs a correct answer.
    """
    rebound: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store | ast.Del):
            rebound.add(child.id)
        elif isinstance(child, ast.Global | ast.Nonlocal):
            rebound.update(child.names)
    return rebound


def _body_calls(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.Call]:
    """Every call in the function's own body, not in anything nested inside it.

    A call inside a nested function or comprehension may run with different bindings, or not at all,
    so it is left to resolve itself.
    """
    found: list[ast.Call] = []

    def walk(current: ast.AST) -> None:
        for child in ast.iter_child_nodes(current):
            if isinstance(child, _NESTED_SCOPES):
                continue
            if isinstance(child, ast.Call):
                found.append(child)
            walk(child)

    for statement in node.body:
        if isinstance(statement, _NESTED_SCOPES):
            continue
        if isinstance(statement, ast.Call):  # pragma: no cover - a bare call is an Expr
            found.append(statement)
        walk(statement)
    return found


def _resolve(node: ast.expr, namespace: Mapping[str, Any]) -> Any:
    """Resolve `likely`, `gut.likely` and similar to the object they name, or `None`."""
    if isinstance(node, ast.Name):
        return namespace.get(node.id)
    if isinstance(node, ast.Attribute):
        parent = _resolve(node.value, namespace)
        return None if parent is None else getattr(parent, node.attr, None)
    return None


def _as_text(node: ast.expr | None, namespace: Mapping[str, Any]) -> str | None:
    """A string literal, or a module-level name bound to a string."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    resolved = _resolve(node, namespace)
    return resolved if isinstance(resolved, str) else None


def _as_levels(node: ast.expr | None, namespace: Mapping[str, Any]) -> list[str] | None:
    """A list or tuple of string literals, or a module-level name bound to one."""
    if node is None:
        return None
    if isinstance(node, ast.List | ast.Tuple):
        levels = [
            element.value
            for element in node.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        ]
        return levels if len(levels) == len(node.elts) else None
    resolved = _resolve(node, namespace)
    if isinstance(resolved, Sequence) and not isinstance(resolved, str):
        return list(resolved) if all(isinstance(item, str) for item in resolved) else None
    return None


def _as_enum(node: ast.expr | None, namespace: Mapping[str, Any]) -> type[Enum] | None:
    """A name bound to an `Enum` subclass at decoration time."""
    if node is None:
        return None
    resolved = _resolve(node, namespace)
    if isinstance(resolved, type) and issubclass(resolved, Enum):
        return resolved
    return None


@dataclass
class _CallArguments:
    """Positional and keyword arguments of a call, in a form the builders can read."""

    positional: list[ast.expr] = field(default_factory=list)
    keyword: dict[str, ast.expr] = field(default_factory=dict)
    starred: bool = False

    def at(self, index: int, name: str) -> ast.expr | None:
        if len(self.positional) > index:
            return self.positional[index]
        return self.keyword.get(name)


def _arguments(call: ast.Call) -> _CallArguments:
    arguments = _CallArguments()
    for argument in call.args:
        if isinstance(argument, ast.Starred):
            arguments.starred = True
        else:
            arguments.positional.append(argument)
    for keyword in call.keywords:
        if keyword.arg is None:
            arguments.starred = True
        else:
            arguments.keyword[keyword.arg] = keyword.value
    return arguments


def _subject_parameter(
    arguments: _CallArguments, parameters: set[str], rebound: set[str]
) -> str | None:
    subject = arguments.at(0, "subject")
    if not isinstance(subject, ast.Name):
        return None
    if subject.id not in parameters or subject.id in rebound:
        return None
    return subject.id


def _spec_for(call: ast.Call, target: Any, namespace: Mapping[str, Any]) -> QuestionSpec | None:
    """Build the question a call will ask, when every part of it is knowable now."""
    arguments = _arguments(call)
    # A per-call backend may not be the one the prefetch would use, and **kwargs could carry one.
    if arguments.starred or "backend" in arguments.keyword:
        return None

    question = _as_text(arguments.keyword.get("question"), namespace)
    if target is likely:
        instructions = _as_text(arguments.at(1, "question"), namespace)
        if instructions is None:
            return None
        yes_means = _as_text(arguments.keyword.get("yes_means"), namespace)
        no_means = _as_text(arguments.keyword.get("no_means"), namespace)
        if ("yes_means" in arguments.keyword) != (yes_means is not None):
            return None
        if ("no_means" in arguments.keyword) != (no_means is not None):
            return None
        return NoulSpec(instructions=instructions, yes_means=yes_means, no_means=no_means)

    if ("question" in arguments.keyword) != (question is not None):
        return None

    if target is classify:
        enum_class = _as_enum(arguments.at(1, "enum_class"), namespace)
        if enum_class is None:
            return None
        try:
            return ChoiceSpec(instructions=question, criteria=_criteria_from_enum(enum_class))
        except GutError:
            return None

    if target is rate:
        levels = _as_levels(arguments.at(1, "levels"), namespace)
        if levels is None:
            return None
        try:
            return ScoreSpec(instructions=question, criteria=levels)
        except GutError:
            return None

    # Unreachable: callers only pass a target already known to be one of the three.
    return None  # pragma: no cover


def build_plan(func: Callable[..., Any]) -> Plan:
    """Work out which questions `func` will ask about its own parameters.

    Never raises for an unanalysable function: the result is simply an empty plan with a reason.
    """
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError) as error:
        return Plan(skipped=f"source unavailable ({error})")

    try:
        node = _function_node(source)
    except SyntaxError as error:  # pragma: no cover - source that does not parse standalone
        return Plan(skipped=f"source does not parse ({error})")
    if node is None:  # pragma: no cover - getsource on a function always yields a def
        return Plan(skipped="no function definition found in source")

    namespace = getattr(func, "__globals__", {})
    parameters = _parameter_names(node)
    rebound = _rebound_names(node)

    planned: list[PlannedQuestion] = []
    seen: set[tuple[str, str]] = set()
    for call in _body_calls(node):
        target = _resolve(call.func, namespace)
        if target not in (likely, classify, rate):
            continue
        arguments = _arguments(call)
        parameter = _subject_parameter(arguments, parameters, rebound)
        if parameter is None:
            continue
        spec = _spec_for(call, target, namespace)
        if spec is None:
            continue
        key = (parameter, spec.fingerprint)
        if key in seen:
            continue
        seen.add(key)
        planned.append(PlannedQuestion(parameter=parameter, spec=spec))

    if not planned:
        return Plan(skipped="no statically resolvable questions about parameters")
    return Plan(questions=tuple(planned))


# --------------------------------------------------------------------------- the decorator


def _prefetch_for(plan: Plan, arguments: Mapping[str, Any], backend: Backend) -> Prefetch:
    """Answer everything the plan expects, in as few calls as it can manage."""
    prefetch = Prefetch()
    for parameter, specs in plan.by_parameter().items():
        if parameter not in arguments:  # pragma: no cover - apply_defaults() fills every parameter
            continue
        subject = arguments[parameter]
        if not isinstance(subject, str | Mapping | Sequence):
            continue
        try:
            resolved = fetch(subject, specs, backend)
        except Exception as error:
            # A failed prefetch must never break the call; the body asks normally.
            logger.debug(
                "gut: prefetch for %r failed, falling back to single calls: %s", parameter, error
            )
            continue
        for spec, entry in resolved.items():
            prefetch.add(subject, spec, entry)
    return prefetch


def _async_wrapper(
    func: Callable[P, Coroutine[Any, Any, R]], plan: Plan, signature: inspect.Signature
) -> Callable[P, Coroutine[Any, Any, R]]:
    """The same batching for a coroutine function, with the one blocking call moved off the loop.

    The only I/O `@semantic` does is the prefetch itself; once it has run, every `likely` in the
    body is answered from the scope without touching the network. So running that single call in a
    worker thread is enough to make a decorated coroutine batch properly without blocking the event
    loop -- and it is a strict improvement on today's behaviour, where the body issues one blocking
    request per judgment.

    A backend that speaks `async` natively would avoid the thread entirely. That is a larger change
    and is noted as a next step rather than done here. See D27.
    """

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if not plan.questions:
            return await func(*args, **kwargs)
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:
            # Let the function raise the argument error, with its own traceback.
            return await func(*args, **kwargs)
        bound.apply_defaults()

        backend = current_backend()
        arguments = dict(bound.arguments)
        prefetch = await asyncio.to_thread(_prefetch_for, plan, arguments, backend)
        # The scope is entered in the coroutine's own context, so it follows this task and no
        # other: a sibling task awaiting concurrently has its own, and sees none of this.
        with prefetch_scope(prefetch):
            return await func(*args, **kwargs)

    wrapper.gut_plan = plan  # type: ignore[attr-defined]
    return wrapper


def semantic(func: Callable[P, R]) -> Callable[P, R]:
    """Batch every question a function asks about its own parameters into one request each.

    Example:
        ```python
        @semantic
        def handle(ticket):
            if likely(ticket, "is a bug report"):
                if likely(ticket, "has reproduction steps"):   # already answered
                    ...
            elif likely(ticket, "asks for a refund"):          # already answered
                ...
        ```

    What gets collected: calls to `likely`, `classify` and `rate` in the function's own body, whose
    subject is one of its parameters and whose question is knowable at decoration time. Everything
    else runs exactly as it would undecorated.

    Coroutine functions are supported: the prefetch runs in a worker thread so the event loop keeps
    turning, and the body's judgments are then answered from memory.

    The plan is available as `handle.gut_plan` for inspection.
    """
    plan = build_plan(func)
    if plan.skipped is not None:
        logger.debug("gut: @semantic collected nothing from %r: %s", func, plan.skipped)

    signature = inspect.signature(func)

    if inspect.iscoroutinefunction(func):
        return cast("Callable[P, R]", _async_wrapper(func, plan, signature))

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if not plan.questions:
            return func(*args, **kwargs)
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:
            # Let the function itself raise the argument error, with its own traceback.
            return func(*args, **kwargs)
        bound.apply_defaults()
        prefetch = _prefetch_for(plan, bound.arguments, current_backend())
        with prefetch_scope(prefetch):
            return func(*args, **kwargs)

    wrapper.gut_plan = plan  # type: ignore[attr-defined]
    return wrapper
