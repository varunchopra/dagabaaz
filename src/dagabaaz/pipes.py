"""Built-in pipe functions for task-parameter expressions.

Each pipe receives the current value followed by optional string arguments,
as in ``{edge.key | replace(old,new)}``. Registration requires matching
entries in ``BUILTIN_PIPES`` and ``PIPE_ARITY``.
Routing arrays enter this module as tuples, and pipes that create arrays return
tuples.
Text conversion restores dictionaries and lists before calling ``str``.
"""

import json
import math
import types
from collections.abc import Callable, Mapping
from pathlib import PurePosixPath
from typing import Final, cast
from urllib.parse import quote, unquote

try:
    import re2 as _re_engine
except ImportError:
    import re as _re_engine  # type: ignore[assignment]
    import warnings

    warnings.warn(
        "google-re2 is not installed; regular expressions will use Python's "
        "re module without ReDoS protection. Install Dagabaaz with the re2 "
        "extra to enable protected matching.",
        stacklevel=2,
    )

from dagabaaz.json import JsonValue, freeze_json, thaw_json
from dagabaaz.models import ExpressionError


def _text(value: object) -> str:
    return str(thaw_json(cast(JsonValue, value)))


def _upper(value: object) -> object:
    return _text(value).upper()


def _lower(value: object) -> object:
    return _text(value).lower()


def _trim(value: object) -> object:
    return _text(value).strip()


def _title(value: object) -> object:
    return _text(value).title()


def _replace(value: object, old: str, new: str = "") -> object:
    return _text(value).replace(old, new)


def _strip(value: object, chars: str = "") -> object:
    return _text(value).strip(chars) if chars else _text(value).strip()


def _lstrip(value: object, chars: str = "") -> object:
    return _text(value).lstrip(chars) if chars else _text(value).lstrip()


def _rstrip(value: object, chars: str = "") -> object:
    return _text(value).rstrip(chars) if chars else _text(value).rstrip()


def _default(value: object, fallback: str = "") -> object:
    return value if value is not None and value != "" else fallback


def _required(value: object) -> object:
    if value is None or value == "":
        msg = "Required value is missing"
        raise ExpressionError(msg)
    return value


def _first(value: object) -> object:
    if isinstance(value, tuple):
        return value[0] if value else None
    return value


def _last(value: object) -> object:
    if isinstance(value, tuple):
        return value[-1] if value else None
    return value


def _nth(value: object, index: str = "0") -> object:
    if isinstance(value, tuple):
        try:
            idx = int(index)
        except (ValueError, TypeError):
            return None
        return value[idx] if 0 <= idx < len(value) else None
    return value


def _join(value: object, sep: str = ", ") -> object:
    if isinstance(value, tuple):
        return sep.join(_text(item) for item in value if item is not None)
    return _text(value)


def _basename(value: object) -> object:
    return PurePosixPath(_text(value)).name


def _dirname(value: object) -> object:
    return str(PurePosixPath(_text(value)).parent)


def _stem(value: object) -> object:
    return PurePosixPath(_text(value)).stem


def _ext(value: object) -> object:
    return PurePosixPath(_text(value)).suffix


def _urlencode(value: object) -> object:
    return quote(_text(value), safe="")


def _urldecode(value: object) -> object:
    return unquote(_text(value))


def _int(value: object) -> object:
    try:
        float_val = float(_text(value))
        if math.isnan(float_val) or math.isinf(float_val):
            return 0
        return int(float_val)
    except (ValueError, TypeError, OverflowError):
        return 0


def _string(value: object) -> object:
    if isinstance(value, tuple):
        return ", ".join(_text(item) for item in value)
    return _text(value)


def _truncate(value: object, length: str = "100") -> object:
    """Truncation retains at most ``length`` characters.

    An ``append`` pipe may add an ellipsis after truncation.
    """
    text = _text(value)
    try:
        max_len = int(length)
    except (ValueError, TypeError):
        return text
    return text[:max_len] if len(text) > max_len else text


def _prepend(value: object, prefix: str = "") -> object:
    return prefix + _text(value)


def _append(value: object, suffix: str = "") -> object:
    return _text(value) + suffix


def _match(value: object, pattern: str = "") -> object:
    if not pattern:
        return ""
    try:
        match_result = _re_engine.search(pattern, _text(value))
    except _re_engine.error:
        return ""
    return match_result.group(0) if match_result else ""


def _json_get(value: object, key: str = "") -> object:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, Mapping):
            return freeze_json(parsed.get(key))
        return None
    if isinstance(value, Mapping):
        return freeze_json(value.get(key))
    return None


def _flatten(value: object) -> object:
    """One level of a routing array is flattened.

    Nested arrays are expanded once; other values are returned unchanged.
    """
    if not isinstance(value, tuple):
        return value
    result: list[object] = []
    for item in value:
        if isinstance(item, tuple):
            result.extend(item)
        else:
            result.append(item)
    return tuple(result)


def _compact(value: object) -> object:
    """Array compaction removes ``None`` and retains other values."""
    if isinstance(value, tuple):
        return tuple(item for item in value if item is not None)
    return value


def _pad(value: object, width: str = "2") -> object:
    """Numeric values are zero-padded to the requested width.

    Conversion to an integer truncates a fractional value, so ``3.7`` becomes
    ``"03"`` at width 2. Non-numeric values are returned unchanged.
    """
    try:
        return str(int(float(_text(value)))).zfill(int(width))
    except (ValueError, OverflowError):
        return _text(value)


def _not(value: object) -> object:
    """Negation follows Python truth-value testing."""

    return not bool(value)


def _eq(value: object, expected: str) -> object:
    return _text(value) == expected


def _neq(value: object, expected: str) -> object:
    return _text(value) != expected


def _gt(value: object, threshold: str) -> object:
    """Non-numeric input returns ``False``."""
    try:
        return float(value) > float(threshold)
    except (ValueError, TypeError):
        return False


def _lt(value: object, threshold: str) -> object:
    """Non-numeric input returns ``False``."""
    try:
        return float(value) < float(threshold)
    except (ValueError, TypeError):
        return False


def _in(value: object, *options: str) -> object:
    """Expression syntax is ``{edge.kind | in(first, second)}``."""
    return _text(value) in options


BUILTIN_PIPES: Final[types.MappingProxyType[str, Callable[..., object]]] = (
    types.MappingProxyType(
        {
            "upper": _upper,
            "lower": _lower,
            "trim": _trim,
            "title": _title,
            "replace": _replace,
            "strip": _strip,
            "lstrip": _lstrip,
            "rstrip": _rstrip,
            "default": _default,
            "required": _required,
            "first": _first,
            "last": _last,
            "nth": _nth,
            "join": _join,
            "basename": _basename,
            "dirname": _dirname,
            "stem": _stem,
            "ext": _ext,
            "urlencode": _urlencode,
            "urldecode": _urldecode,
            "int": _int,
            "string": _string,
            "truncate": _truncate,
            "prepend": _prepend,
            "append": _append,
            "match": _match,
            "json_get": _json_get,
            "flatten": _flatten,
            "compact": _compact,
            "pad": _pad,
            "not": _not,
            "eq": _eq,
            "neq": _neq,
            "gt": _gt,
            "lt": _lt,
            "in": _in,
        }
    )
)


# Each pair gives the minimum and maximum number of explicit arguments.
PIPE_ARITY: Final[dict[str, tuple[int, int]]] = {
    "upper": (0, 0),
    "lower": (0, 0),
    "trim": (0, 0),
    "title": (0, 0),
    "replace": (1, 2),
    "strip": (0, 1),
    "lstrip": (0, 1),
    "rstrip": (0, 1),
    "default": (0, 1),
    "required": (0, 0),
    "first": (0, 0),
    "last": (0, 0),
    "nth": (0, 1),
    "join": (0, 1),
    "basename": (0, 0),
    "dirname": (0, 0),
    "stem": (0, 0),
    "ext": (0, 0),
    "urlencode": (0, 0),
    "urldecode": (0, 0),
    "int": (0, 0),
    "string": (0, 0),
    "truncate": (0, 1),
    "prepend": (0, 1),
    "append": (0, 1),
    "match": (0, 1),
    "json_get": (0, 1),
    "flatten": (0, 0),
    "compact": (0, 0),
    "pad": (0, 1),
    "not": (0, 0),
    "eq": (1, 1),
    "neq": (1, 1),
    "gt": (1, 1),
    "lt": (1, 1),
    "in": (1, 32),
}
