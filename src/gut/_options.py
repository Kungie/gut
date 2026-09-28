"""Categories given as plain data -- a list of labels, or labels with descriptions -- turned into
the `Enum` that `classify` takes. Shared by the command line and the MCP server, where nobody writes
a class.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum

from gut._api import _warned_enums, has_catch_all
from gut._errors import ConfigurationError


def options_enum(options: Mapping[str, str] | Sequence[str]) -> type[Enum]:
    """Categories sent as data, as the `Enum` that `classify` takes."""
    if isinstance(options, Mapping):
        pairs = [(str(k).strip(), str(v).strip() or str(k).strip()) for k, v in options.items()]
    else:
        pairs = [(str(o).strip(), str(o).strip()) for o in options]
    labels = [label for label, _ in pairs]
    if len(pairs) < 2:
        raise ConfigurationError("classify needs at least two options.")
    if any(not label or label.startswith("_") for label in labels):
        raise ConfigurationError("Option labels must be non-empty and must not start with '_'.")
    if len(set(labels)) < len(labels):
        raise ConfigurationError("Option labels must be different from each other.")
    if len({description for _, description in pairs}) < len(pairs):
        # An Enum folds members with equal values into one, which would silently drop an option.
        raise ConfigurationError("Two options have the same description; give each its own.")
    enum_class: type[Enum] = Enum("Options", pairs)  # type: ignore[misc]
    # The answer carries a note instead of the warning a program would get.
    _warned_enums.add(enum_class)
    return enum_class


def catch_all_note(enum_class: type[Enum]) -> dict[str, str]:
    """A note for an answer that had no "none of these" to fall back on."""
    if has_catch_all(enum_class):
        return {}
    return {
        "note": "There was no catch-all option such as 'other', so the text was put in one of "
        "the options even if none fits."
    }
