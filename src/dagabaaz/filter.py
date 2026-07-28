"""Filtering and selection for ``OutputRef`` routing fields.

Every rule in an ``EdgeFilter`` must match. Filtering retains source order,
and comparisons preserve the distinction between JSON booleans, numbers and
strings.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, cast

from dagabaaz.constants import FilterOperator, SortOrder
from dagabaaz.json import JsonValue, json_values_equal
from dagabaaz.models import EdgeFilter, FilterRule, OutputRef, PlanningError, Selection

_MISSING: Final = object()


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _ordered_pair(actual: JsonValue, expected: JsonValue) -> tuple[object, object] | None:
    if _number(actual) and _number(expected):
        return actual, expected
    if isinstance(actual, str) and isinstance(expected, str):
        return actual, expected
    return None


def _contains(actual: JsonValue, expected: JsonValue) -> bool:
    if isinstance(actual, str) and isinstance(expected, str):
        return expected in actual
    if isinstance(actual, tuple):
        return any(json_values_equal(item, expected) for item in actual)
    return False


def _matches(rule: FilterRule, output: OutputRef) -> bool:
    actual = output.fields.get(rule.field, _MISSING)
    expected = rule.value
    operator = rule.operator

    if operator == FilterOperator.EXISTS:
        return actual is not _MISSING and actual is not None
    if operator == FilterOperator.NOT_EXISTS:
        return actual is _MISSING or actual is None
    if actual is _MISSING:
        return False
    actual = cast(JsonValue, actual)
    if operator == FilterOperator.EQ:
        return json_values_equal(actual, expected)
    if operator == FilterOperator.NEQ:
        return not json_values_equal(actual, expected)
    if operator in (FilterOperator.IN, FilterOperator.NOT_IN):
        candidates = expected if isinstance(expected, tuple) else (expected,)
        found = any(json_values_equal(actual, candidate) for candidate in candidates)
        return found if operator == FilterOperator.IN else not found
    if operator == FilterOperator.CONTAINS:
        return _contains(actual, expected)
    if operator == FilterOperator.NOT_CONTAINS:
        return not _contains(actual, expected)
    if operator == FilterOperator.STARTS_WITH:
        return isinstance(actual, str) and isinstance(expected, str) and actual.startswith(expected)
    if operator == FilterOperator.ENDS_WITH:
        return isinstance(actual, str) and isinstance(expected, str) and actual.endswith(expected)

    pair = _ordered_pair(actual, expected)
    if pair is None:
        return False
    left, right = pair
    if operator == FilterOperator.GT:
        return left > right  # type: ignore[operator]
    if operator == FilterOperator.GTE:
        return left >= right  # type: ignore[operator]
    if operator == FilterOperator.LT:
        return left < right  # type: ignore[operator]
    if operator == FilterOperator.LTE:
        return left <= right  # type: ignore[operator]
    raise PlanningError(f"unsupported filter operator {operator!r}")


def apply_filter(
    outputs: Sequence[OutputRef], edge_filter: EdgeFilter | None
) -> tuple[OutputRef, ...]:
    """All rules are applied without changing the source order."""

    if edge_filter is None or not edge_filter.rules:
        return tuple(outputs)
    return tuple(
        output for output in outputs if all(_matches(rule, output) for rule in edge_filter.rules)
    )


def apply_selection(
    outputs: Sequence[OutputRef], selection: Selection | None
) -> tuple[OutputRef, ...]:
    """Selection operates on one edge or correlation group.

    Missing and null values are excluded. Strings and non-boolean numbers are
    sortable, but an edge cannot mix those two categories. Ties are resolved
    by output ID in ascending order.
    """

    if selection is None:
        return tuple(outputs)

    sortable: list[tuple[OutputRef, str | int | float]] = []
    value_kind: str | None = None
    for output in outputs:
        value = output.fields.get(selection.field, _MISSING)
        if value is _MISSING or value is None:
            continue
        value = cast(JsonValue, value)
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise PlanningError(
                f"selection field {selection.field!r} on output {output.id!r} "
                f"is not a sortable string or number"
            )
        kind = "string" if isinstance(value, str) else "number"
        if value_kind is not None and kind != value_kind:
            raise PlanningError(
                f"selection field {selection.field!r} contains mixed string and number values"
            )
        value_kind = kind
        sortable.append((output, value))

    # Stable two-pass sorting keeps output ID ascending for equal values even
    # when the primary value order is descending.
    sortable.sort(key=lambda pair: pair[0].id)
    sortable.sort(key=lambda pair: pair[1], reverse=selection.order == SortOrder.DESC)
    return tuple(output for output, _value in sortable[: selection.limit])
