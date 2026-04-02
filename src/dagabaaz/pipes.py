"""Built-in pipe functions for the expression engine.

Pure transforms that operate on scalar or list values. Each pipe receives
the current value as its first argument, plus optional string arguments
from the expression syntax: ``{slug.key | pipe(arg1, arg2)}``.

Growth area — new pipes are added here with one function + one dict entry
+ one arity entry. No registration decorator, no mutable state.
"""

import json
import math
import types
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Final
from urllib.parse import quote, unquote

try:
    import re2 as _re_engine
except ImportError:
    import re as _re_engine  # type: ignore[assignment]
    import warnings

    warnings.warn(
        "google-re2 not installed, falling back to stdlib re. "
        "User-supplied regex patterns will not have ReDoS protection. "
        "Install with: pip install google-re2",
        stacklevel=2,
    )

from dagabaaz.models import ExpressionError


def _upper(value: object) -> object:
    return str(value).upper()


def _lower(value: object) -> object:
    return str(value).lower()


def _trim(value: object) -> object:
    return str(value).strip()


def _title(value: object) -> object:
    return str(value).title()


def _replace(value: object, old: str, new: str = "") -> object:
    return str(value).replace(old, new)


def _strip(value: object, chars: str = "") -> object:
    return str(value).strip(chars) if chars else str(value).strip()


def _lstrip(value: object, chars: str = "") -> object:
    return str(value).lstrip(chars) if chars else str(value).lstrip()


def _rstrip(value: object, chars: str = "") -> object:
    return str(value).rstrip(chars) if chars else str(value).rstrip()


def _default(value: object, fallback: str = "") -> object:
    return value if value is not None and value != "" else fallback


def _required(value: object) -> object:
    if value is None or value == "":
        msg = "Required value is missing"
        raise ExpressionError(msg)
    return value


def _first(value: object) -> object:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _last(value: object) -> object:
    if isinstance(value, list):
        return value[-1] if value else None
    return value


def _nth(value: object, index: str = "0") -> object:
    if isinstance(value, list):
        try:
            idx = int(index)
        except (ValueError, TypeError):
            return None
        return value[idx] if 0 <= idx < len(value) else None
    return value


def _join(value: object, sep: str = ", ") -> object:
    if isinstance(value, list):
        return sep.join(str(item) for item in value if item is not None)
    return str(value)


def _basename(value: object) -> object:
    return PurePosixPath(str(value)).name


def _dirname(value: object) -> object:
    return str(PurePosixPath(str(value)).parent)


def _stem(value: object) -> object:
    return PurePosixPath(str(value)).stem


def _ext(value: object) -> object:
    return PurePosixPath(str(value)).suffix


def _urlencode(value: object) -> object:
    return quote(str(value), safe="")


def _urldecode(value: object) -> object:
    return unquote(str(value))


def _int(value: object) -> object:
    try:
        float_val = float(str(value))
        if math.isnan(float_val) or math.isinf(float_val):
            return 0
        return int(float_val)
    except (ValueError, TypeError, OverflowError):
        return 0


def _string(value: object) -> object:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _truncate(value: object, length: str = "100") -> object:
    """Truncate to exact length — no ellipsis appended.

    For ellipsis, use ``{x | truncate:97 | append:...}``.
    """
    text = str(value)
    try:
        max_len = int(length)
    except (ValueError, TypeError):
        return text
    return text[:max_len] if len(text) > max_len else text


def _prepend(value: object, prefix: str = "") -> object:
    return prefix + str(value)


def _append(value: object, suffix: str = "") -> object:
    return str(value) + suffix


def _match(value: object, pattern: str = "") -> object:
    if not pattern:
        return ""
    try:
        match_result = _re_engine.search(pattern, str(value))
    except _re_engine.error:
        return ""
    return match_result.group(0) if match_result else ""


def _json_get(value: object, key: str = "") -> object:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed.get(key)
        return None
    if isinstance(value, dict):
        return value.get(key)
    return None


def _flatten(value: object) -> object:
    """Flatten one level of nesting (non-recursive, matches Terraform model).

    ``[[1, 2], 3, [4]]`` → ``[1, 2, 3, 4]``. Apply repeatedly for deeper.
    Non-list input is returned as-is.
    """
    if not isinstance(value, list):
        return value
    result: list[object] = []
    for item in value:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result


def _compact(value: object) -> object:
    """Remove None values from a list (not empty strings, not 0, not False).

    Matches Ruby/Lodash compact semantics: only None is stripped.
    Preserves positional correspondence for non-None values.
    """
    if isinstance(value, list):
        return [v for v in value if v is not None]
    return value


def _pad(value: object, width: str = "2") -> object:
    """Zero-pad a numeric value to the given width.

    Converts to int first (truncates floats: 3.7 → "03" with width=2).
    Non-numeric values are returned as-is.
    """
    try:
        return str(int(float(str(value)))).zfill(int(width))
    except (ValueError, OverflowError):
        return str(value)


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
        }
    )
)


# (min_args, max_args) beyond the implicit first `value` parameter.
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
}
