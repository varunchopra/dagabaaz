"""Validation, comparison and serialisation for routing JSON.

Incoming objects and arrays are copied to ``FrozenDict`` and ``tuple``.
``thaw_json`` restores ``dict`` and ``list`` at the serialisation boundary.
Object keys must be strings and numbers must be finite.
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

# JsonInput is accepted at model and API boundaries. freeze_json converts it to
# JsonValue; thaw_json restores dict and list containers for serialisation.


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
        return (
            isinstance(left, (int, float))
            and isinstance(right, (int, float))
            and left == right
        )
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
                key in right and json_values_equal(value, right[key])
                for key, value in left.items()
            )
        )
    return False


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
                raise ValueError(
                    f"routing JSON exceeds maximum value count {MAX_JSON_VALUES}"
                )
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
                raise ValueError(
                    f"routing JSON exceeds maximum value count {MAX_JSON_VALUES}"
                )
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


def thaw_json(value: JsonValue) -> JsonWireValue:
    """JSON serialisation uses ``dict`` and ``list`` containers."""

    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
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
