"""Decision-site identity.

Every decision carries an id that answers "which decision in the code is this?", so that outcomes
recorded today can be matched against outcomes recorded next month and calibrated against reality.

The id is derived from the question itself plus *where it is asked from* -- not from the state,
which changes on every call. See D11 in DECISIONS.md for why the location is the module and function
name rather than the file and line the handoff suggested.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from types import FrameType

_PACKAGE = __name__.split(".")[0]


@dataclass(frozen=True, slots=True)
class CallSite:
    """Where in the caller's code a decision was made."""

    module: str
    """Dotted module name, or `"<unknown>"` when it cannot be determined."""
    function: str
    """Function or method name, `"<module>"` at import time."""
    filename: str
    """Source file, recorded for debugging only -- it is not part of the id."""
    lineno: int
    """Source line, recorded for debugging only -- it is not part of the id."""

    @property
    def location(self) -> str:
        """The part of the site that the id is built from."""
        return f"{self.module}:{self.function}"

    def __str__(self) -> str:
        return f"{self.filename}:{self.lineno} in {self.location}"


UNKNOWN_SITE = CallSite(module="<unknown>", function="<unknown>", filename="<unknown>", lineno=0)
"""Used when the stack cannot be walked, so an id is always available."""


def _is_internal(frame: FrameType) -> bool:
    name = str(frame.f_globals.get("__name__", ""))
    return name == _PACKAGE or name.startswith(f"{_PACKAGE}.")


def caller_site() -> CallSite:
    """The nearest frame outside `gut` itself.

    Walks out through however many internal frames sit between here and user code, so the site does
    not shift when `gut`'s own call depth changes between releases.
    """
    frame: FrameType | None = sys._getframe(1)
    while frame is not None and _is_internal(frame):
        frame = frame.f_back
    if frame is None:
        return UNKNOWN_SITE
    return CallSite(
        module=str(frame.f_globals.get("__name__", "<unknown>")),
        function=frame.f_code.co_name,
        filename=frame.f_code.co_filename,
        lineno=frame.f_lineno,
    )


def decision_id(question_fingerprint: str, site: CallSite) -> str:
    """A stable identifier for "this question, asked from this place"."""
    payload = f"{site.location}|{question_fingerprint}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]
