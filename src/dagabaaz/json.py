"""Validation, comparison and container boundaries for routing JSON.

Persisted objects and arrays use ``FrozenDict`` and ``tuple``. Expressions and
serialisation use ``dict`` and ``list``. Object keys must be strings and
numbers must be finite.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from types import MappingProxyType

from dagabaaz.constants import MAX_JSON_DEPTH, MAX_JSON_KEYS, MAX_JSON_VALUES

type JsonScalar = None | bool | int | float | str
type JsonInput = JsonScalar | list[JsonInput] | tuple[JsonInput, ...] | Mapping[str, JsonInput]
type JsonValue = JsonScalar | tuple[JsonValue, ...] | FrozenDict
type JsonWireValue = JsonScalar | list[JsonWireValue] | dict[str, JsonWireValue]

# JsonValue is the persisted representation. JsonWireValue is used by public
# expression APIs and serialisation. freeze_json and thaw_json cross the boundary.


def json_string_payload_size(value: str, *, limit: int | None = None) -> int:
    """Return the UTF-8 JSON byte count without materialising an encoded string."""

    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("JSON string size limit must be a non-negative integer")
    size = 0
    chunk_size = 4_096
    for start in range(0, len(value), chunk_size):
        try:
            encoded = json.dumps(
                value[start : start + chunk_size],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("JSON text is not valid UTF-8") from exc
        size += len(encoded) - 2
        if limit is not None and size > limit:
            return limit + 1
    return size


def bounded_json_size(value: JsonValue | JsonWireValue, limit: int) -> int:
    """Return the exact encoded size, or ``limit + 1`` once the limit is crossed."""

    if type(limit) is not int or limit < 0:
        raise ValueError("JSON size limit must be a non-negative integer")

    def measure(item: object, budget: int) -> int:
        if item is None:
            return 4 if budget >= 4 else budget + 1
        if isinstance(item, bool):
            size = 4 if item else 5
            return size if budget >= size else budget + 1
        if isinstance(item, str):
            if budget < 2:
                return budget + 1
            payload_size = json_string_payload_size(item, limit=budget - 2)
            return payload_size + 2 if payload_size <= budget - 2 else budget + 1
        if isinstance(item, (int, float)):
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("JSON numbers must be finite")
            try:
                size = len(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
            except (TypeError, ValueError, UnicodeEncodeError) as exc:
                raise ValueError("JSON number cannot be serialised") from exc
            return size if size <= budget else budget + 1
        if isinstance(item, (list, tuple)):
            total = 2
            if total > budget:
                return budget + 1
            for index, member in enumerate(item):
                separator_size = 1 if index else 0
                remaining = budget - total - separator_size
                if remaining < 0:
                    return budget + 1
                member_size = measure(member, remaining)
                if member_size > remaining:
                    return budget + 1
                total += separator_size + member_size
            return total
        if isinstance(item, Mapping):
            total = 2
            if total > budget:
                return budget + 1
            for index, (key, member) in enumerate(item.items()):
                if not isinstance(key, str):
                    raise ValueError("JSON object keys must be strings")
                separator_size = 1 if index else 0
                remaining = budget - total - separator_size
                if remaining < 0:
                    return budget + 1
                key_size = measure(key, remaining)
                if key_size > remaining:
                    return budget + 1
                remaining -= key_size + 1
                if remaining < 0:
                    return budget + 1
                member_size = measure(member, remaining)
                if member_size > remaining:
                    return budget + 1
                total += separator_size + key_size + 1 + member_size
            return total
        raise ValueError(f"value of type {type(item).__name__} is not JSON-compatible")

    return measure(value, limit)


class FrozenDict(Mapping[str, JsonValue]):
    """A copy of a JSON object that cannot be changed."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, object] | None = None) -> None:
        frozen = freeze_json({} if values is None else values)
        assert isinstance(frozen, FrozenDict)
        object.__setattr__(self, "_values", frozen._values)

    @classmethod
    def _from_frozen(cls, values: Mapping[str, JsonValue]) -> FrozenDict:
        instance = object.__new__(cls)
        object.__setattr__(instance, "_values", MappingProxyType(dict(values)))
        return instance

    def __getitem__(self, key: str) -> JsonValue:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"FrozenDict({dict(self._values)!r})"

    def __setattr__(self, _name: str, _value: object) -> None:
        raise TypeError("routing JSON is immutable")

    def __setitem__(self, _key: str, _value: JsonValue) -> None:
        raise TypeError("routing JSON is immutable")

    def __delitem__(self, _key: str) -> None:
        raise TypeError("routing JSON is immutable")

    def __ior__(self, _other: object) -> FrozenDict:
        raise TypeError("routing JSON is immutable")

    def __copy__(self) -> FrozenDict:
        return self

    def __deepcopy__(self, _memo: dict[int, object]) -> FrozenDict:
        return self

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: object,
        _handler: object,
    ) -> dict[str, object]:
        return {"type": "object", "additionalProperties": True}


def json_values_equal(left: JsonValue, right: JsonValue) -> bool:
    """Frozen routing values are compared recursively.

    JSON has one number type, so integers and finite floats compare
    numerically. Booleans remain distinct from numbers. Array order matters;
    object key order does not. Both arguments must have passed through
    ``freeze_json``.
    """

    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        return isinstance(left, (int, float)) and isinstance(right, (int, float)) and left == right
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    if isinstance(left, tuple) or isinstance(right, tuple):
        return (
            isinstance(left, tuple)
            and isinstance(right, tuple)
            and len(left) == len(right)
            and all(json_values_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    if isinstance(left, FrozenDict) or isinstance(right, FrozenDict):
        return (
            isinstance(left, FrozenDict)
            and isinstance(right, FrozenDict)
            and len(left) == len(right)
            and all(
                key in right and json_values_equal(value, right[key]) for key, value in left.items()
            )
        )
    return False


def json_equality_key(value: JsonValue) -> object:
    """Return a hashable key with the same semantics as ``json_values_equal``."""

    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, (int, float)):
        return ("number", value)
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, tuple):
        return ("array", tuple(json_equality_key(item) for item in value))
    if isinstance(value, FrozenDict):
        return (
            "object",
            frozenset((key, json_equality_key(item)) for key, item in value.items()),
        )
    raise ValueError(f"value of type {type(value).__name__} is not frozen JSON")


def json_representations_equal(left: JsonValue, right: JsonValue) -> bool:
    """Persisted values match only when their observable representations match."""

    if type(left) is not type(right):
        return False
    if isinstance(left, tuple):
        return len(left) == len(right) and all(
            json_representations_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, FrozenDict):
        if tuple(left) != tuple(right):
            return False
        return all(json_representations_equal(left[key], right[key]) for key in left)
    if isinstance(left, float):
        return left.hex() == right.hex()
    return left == right


def freeze_json(
    value: object,
    *,
    _depth: int = 0,
    _key_counter: list[int] | None = None,
    _value_counter: list[int] | None = None,
) -> JsonValue:
    """Routing JSON is copied into immutable containers and checked against limits."""

    if _depth > MAX_JSON_DEPTH:
        raise ValueError(f"routing JSON exceeds maximum depth {MAX_JSON_DEPTH}")
    key_counter = _key_counter if _key_counter is not None else [0]
    value_counter = _value_counter if _value_counter is not None else [0]

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("routing JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("routing JSON object keys must be strings")
            key_counter[0] += 1
            if key_counter[0] > MAX_JSON_KEYS:
                raise ValueError(f"routing JSON exceeds maximum key count {MAX_JSON_KEYS}")
            value_counter[0] += 1
            if value_counter[0] > MAX_JSON_VALUES:
                raise ValueError(f"routing JSON exceeds maximum value count {MAX_JSON_VALUES}")
            frozen[key] = freeze_json(
                item,
                _depth=_depth + 1,
                _key_counter=key_counter,
                _value_counter=value_counter,
            )
        return FrozenDict._from_frozen(frozen)
    if isinstance(value, (list, tuple)):
        frozen_items: list[JsonValue] = []
        for item in value:
            value_counter[0] += 1
            if value_counter[0] > MAX_JSON_VALUES:
                raise ValueError(f"routing JSON exceeds maximum value count {MAX_JSON_VALUES}")
            frozen_items.append(
                freeze_json(
                    item,
                    _depth=_depth + 1,
                    _key_counter=key_counter,
                    _value_counter=value_counter,
                )
            )
        return tuple(frozen_items)
    raise ValueError(f"value of type {type(value).__name__} is not JSON-compatible")


def freeze_object(value: Mapping[str, object] | None) -> FrozenDict:
    """A routing object is copied into a ``FrozenDict``."""

    if value is not None and not isinstance(value, Mapping):
        raise ValueError("routing JSON object must be a mapping")
    frozen = freeze_json({} if value is None else value)
    assert isinstance(frozen, FrozenDict)
    return frozen


def thaw_json(value: JsonValue | JsonWireValue) -> JsonWireValue:
    """Return a copy using ``dict`` and ``list`` containers."""

    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(item) for item in value]
    return value


def canonical_json_bytes(value: JsonValue | JsonWireValue) -> bytes:
    """Canonical serialisation accepts frozen routing JSON or JSON wire data."""

    serialisable = thaw_json(value) if isinstance(value, (FrozenDict, tuple)) else value

    return json.dumps(
        serialisable,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def routing_text(value: JsonValue | JsonWireValue) -> str:
    """String conversion uses public containers rather than persisted ones."""

    return str(thaw_json(value))
