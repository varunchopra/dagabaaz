"""Filtering and selection for ``OutputRef`` routing fields.

Every rule in an ``EdgeFilter`` must match. Filtering retains source order.
Equality preserves JSON types, while ordered comparisons retain numeric
coercion.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from typing import Final, cast

from dagabaaz.constants import FilterOperator, SortOrder
from dagabaaz.json import JsonValue, json_equality_key, json_values_equal, routing_text
from dagabaaz.models import EdgeFilter, FilterRule, OutputRef, PlanningError, Selection

_MISSING: Final = object()


def _numeric(value: JsonValue) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return number if number.is_finite() else None


def _ordered_pair(actual: JsonValue, expected: JsonValue) -> tuple[Decimal, Decimal] | None:
    left = _numeric(actual)
    right = _numeric(expected)
    return None if left is None or right is None else (left, right)


def _contains(actual: JsonValue, expected: JsonValue) -> bool:
    if isinstance(actual, tuple):
        return any(json_values_equal(item, expected) for item in actual)
    return routing_text(expected) in routing_text(actual)


def _membership(rule: FilterRule) -> frozenset[object] | None:
    if rule.operator not in (FilterOperator.IN, FilterOperator.NOT_IN):
        return None
    candidates = rule.value if isinstance(rule.value, tuple) else (rule.value,)
    return frozenset(json_equality_key(candidate) for candidate in candidates)


def _matches(
    rule: FilterRule,
    output: OutputRef,
    membership: frozenset[object] | None = None,
) -> bool:
    actual = output.fields.get(rule.field, _MISSING)
    expected = rule.value
    operator = rule.operator

    if operator == FilterOperator.EXISTS:
        return actual is not _MISSING and actual is not None and actual != ""
    if operator == FilterOperator.NOT_EXISTS:
        return actual is _MISSING or actual is None or actual == ""
    if actual is _MISSING:
        return operator in (
            FilterOperator.NEQ,
            FilterOperator.NOT_IN,
            FilterOperator.NOT_CONTAINS,
        )
    actual = cast(JsonValue, actual)
    if operator == FilterOperator.EQ:
        return json_values_equal(actual, expected)
    if operator == FilterOperator.NEQ:
        return not json_values_equal(actual, expected)
    if operator in (FilterOperator.IN, FilterOperator.NOT_IN):
        if membership is None:
            candidates = expected if isinstance(expected, tuple) else (expected,)
            membership = frozenset(json_equality_key(candidate) for candidate in candidates)
        found = json_equality_key(actual) in membership
        return found if operator == FilterOperator.IN else not found
    if operator == FilterOperator.CONTAINS:
        return _contains(actual, expected)
    if operator == FilterOperator.NOT_CONTAINS:
        return not _contains(actual, expected)
    if operator == FilterOperator.STARTS_WITH:
        return routing_text(actual).startswith(routing_text(expected))
    if operator == FilterOperator.ENDS_WITH:
        return routing_text(actual).endswith(routing_text(expected))

    pair = _ordered_pair(actual, expected)
    if pair is None:
        return False
    left, right = pair
    if operator == FilterOperator.GT:
        return left > right
    if operator == FilterOperator.GTE:
        return left >= right
    if operator == FilterOperator.LT:
        return left < right
    if operator == FilterOperator.LTE:
        return left <= right
    raise PlanningError(f"unsupported filter operator {operator!r}")


def apply_filter(
    outputs: Sequence[OutputRef], edge_filter: EdgeFilter | None
) -> tuple[OutputRef, ...]:
    """All rules are applied without changing the source order."""

    if edge_filter is None or not edge_filter.rules:
        return tuple(outputs)
    compiled = tuple((rule, _membership(rule)) for rule in edge_filter.rules)
    return tuple(
        output
        for output in outputs
        if all(_matches(rule, output, membership) for rule, membership in compiled)
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
