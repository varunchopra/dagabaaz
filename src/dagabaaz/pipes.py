"""Built-in pipe functions for task-parameter expressions.

Each pipe receives the current value followed by optional string arguments,
as in ``{edge.key | replace(old,new)}``. Registration requires matching
entries in ``BUILTIN_PIPES`` and ``PIPE_ARITY``.
Pipe inputs and results use ordinary Python dictionaries and lists.
"""

import json
import types
from collections.abc import Callable, Iterable, Mapping
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from typing import Final, cast
from urllib.parse import quote, unquote

from dagabaaz.constants import MAX_EXPRESSION_RESULT_BYTES, MAX_PIPE_INTEGER_DIGITS

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

from dagabaaz.json import (
    JsonValue,
    JsonWireValue,
    json_string_payload_size,
    routing_text,
    thaw_json,
)
from dagabaaz.models import ExpressionError


def _text(value: object) -> str:
    return routing_text(cast(JsonValue | JsonWireValue, value))


def _json_string_payload_size(value: str, *, limit: int | None = None) -> int:
    """Return the UTF-8 JSON byte count excluding the surrounding quotes."""

    try:
        return json_string_payload_size(value, limit=limit)
    except ValueError as exc:
        raise ExpressionError("expression text is not valid UTF-8") from exc


def _check_text_result_size(payload_size: int) -> None:
    size = payload_size + 2
    if size > MAX_EXPRESSION_RESULT_BYTES:
        raise ExpressionError(
            f"expression result would be {size} bytes; maximum is {MAX_EXPRESSION_RESULT_BYTES}"
        )


def _materialise_join(separator: str, parts: list[str]) -> str:
    return separator.join(parts)


def _materialise_replace(text: str, old: str, new: str) -> str:
    return text.replace(old, new)


def _bounded_text_join(parts: Iterable[str], separator: str = "") -> str:
    """Check the encoded result before joining its parts."""

    collected: list[str] = []
    payload_size = 0
    separator_size = _json_string_payload_size(separator)
    for part in parts:
        candidate = payload_size + _json_string_payload_size(part)
        if collected:
            candidate += separator_size
        _check_text_result_size(candidate)
        collected.append(part)
        payload_size = candidate
    return _materialise_join(separator, collected)


def _decimal(value: object) -> Decimal | None:
    if isinstance(value, bool):
        return None
    try:
        number = Decimal(_text(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return number if number.is_finite() else None


def _integer(number: Decimal) -> int | None:
    if number.adjusted() >= MAX_PIPE_INTEGER_DIGITS:
        return None
    return int(number)


def _upper(value: object) -> object:
    return _text(value).upper()


def _lower(value: object) -> object:
    return _text(value).lower()


def _trim(value: object) -> object:
    return _text(value).strip()


def _title(value: object) -> object:
    return _text(value).title()


def _replace(value: object, old: str, new: str = "") -> object:
    text = _text(value)
    replacements = len(text) + 1 if old == "" else text.count(old)
    # The encoded size is additive, so the guard does not build the replacement.
    output_size = _json_string_payload_size(text)
    if replacements:
        output_size += replacements * (
            _json_string_payload_size(new) - _json_string_payload_size(old)
        )
    _check_text_result_size(output_size)
    return _materialise_replace(text, old, new)


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
        return _bounded_text_join(
            (_text(item) for item in value if item is not None),
            sep,
        )
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
    text = _text(value)
    output_size = 0
    unreserved = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-~"
    for char in text:
        try:
            encoded = char.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ExpressionError("expression text is not valid UTF-8") from exc
        output_size += sum(1 if byte in unreserved else 3 for byte in encoded)
        _check_text_result_size(output_size)
    return quote(text, safe="")


def _urldecode(value: object) -> object:
    return unquote(_text(value))


def _int(value: object) -> object:
    number = _decimal(value)
    integer = None if number is None else _integer(number)
    return 0 if integer is None else integer


def _string(value: object) -> object:
    if isinstance(value, list):
        return _bounded_text_join((_text(item) for item in value), ", ")
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
    return _bounded_text_join((prefix, _text(value)))


def _append(value: object, suffix: str = "") -> object:
    return _bounded_text_join((_text(value), suffix))


def _match(value: object, pattern: str = "") -> object:
    if not pattern:
        return ""
    try:
        match_result = _re_engine.search(pattern, _text(value))
    except (_re_engine.error, RecursionError):
        return ""
    return match_result.group(0) if match_result else ""


def _json_get(value: object, key: str = "") -> object:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, RecursionError):
            return None
        if isinstance(parsed, Mapping):
            return parsed.get(key)
        return None
    if isinstance(value, Mapping):
        return thaw_json(cast(JsonValue | JsonWireValue, value.get(key)))
    return None


def _flatten(value: object) -> object:
    """One level of a routing array is flattened.

    Nested arrays are expanded once; other values are returned unchanged.
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
    """Array compaction removes ``None`` and retains other values."""
    if isinstance(value, list):
        return [item for item in value if item is not None]
    return value


def _pad(value: object, width: str = "2") -> object:
    """Numeric values are zero-padded to the requested width.

    Conversion to an integer truncates a fractional value, so ``3.7`` becomes
    ``"03"`` at width 2. Non-numeric values are returned unchanged.
    """
    number = _decimal(value)
    if number is None:
        return _text(value)
    try:
        integer = _integer(number)
        output_width = int(width)
        if integer is None or output_width > MAX_PIPE_INTEGER_DIGITS:
            return _text(value)
        return str(integer).zfill(output_width)
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
    left = _decimal(value)
    right = _decimal(threshold)
    return left is not None and right is not None and left > right


def _lt(value: object, threshold: str) -> object:
    """Non-numeric input returns ``False``."""
    left = _decimal(value)
    right = _decimal(threshold)
    return left is not None and right is not None and left < right


def _in(value: object, *options: str) -> object:
    """Expression syntax is ``{edge.kind | in(first, second)}``."""
    return _text(value) in options


BUILTIN_PIPES: Final[types.MappingProxyType[str, Callable[..., object]]] = types.MappingProxyType(
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
