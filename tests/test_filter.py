import pytest

from dagabaaz.constants import (
    MAX_FILTER_MEMBERSHIP_VALUES,
    MAX_FILTER_RULES_PER_EDGE,
    FilterOperator,
    SortOrder,
)
from dagabaaz.filter import apply_filter, apply_selection
from dagabaaz.models import EdgeFilter, FilterRule, OutputRef, Selection


def _output(output_id: str, **fields: object) -> OutputRef:
    return OutputRef(id=output_id, fields=fields)


@pytest.mark.parametrize(
    ("operator", "actual", "expected", "matches"),
    [
        (FilterOperator.EQ, "open", "open", True),
        (FilterOperator.EQ, 1, "1", False),
        (FilterOperator.NEQ, "open", "closed", True),
        (FilterOperator.GT, 3, 2, True),
        (FilterOperator.GT, "3", 2, True),
        (FilterOperator.GTE, 3, 3.0, True),
        (FilterOperator.LT, "a", "b", False),
        (FilterOperator.LTE, "3", 3, True),
        (FilterOperator.IN, "b", ["a", "b"], True),
        (FilterOperator.NOT_IN, "c", ["a", "b"], True),
        (FilterOperator.CONTAINS, ["a", "b"], "b", True),
        (FilterOperator.NOT_CONTAINS, ["a", "b"], "c", True),
        (FilterOperator.CONTAINS, "alphabet", "pha", True),
        (FilterOperator.STARTS_WITH, "alphabet", "alpha", True),
        (FilterOperator.ENDS_WITH, "alphabet", "bet", True),
        (FilterOperator.CONTAINS, 1234, 23, True),
        (FilterOperator.STARTS_WITH, 1234, 12, True),
        (FilterOperator.ENDS_WITH, 1234, 34, True),
    ],
)
def test_filter_operators(
    operator: FilterOperator,
    actual: object,
    expected: object,
    matches: bool,
) -> None:
    result = apply_filter(
        (_output("one", value=actual),),
        EdgeFilter(rules=(FilterRule(field="value", operator=operator, value=expected),)),
    )
    assert bool(result) is matches


def test_exists_and_not_exists_treat_null_as_absent() -> None:
    outputs = (
        _output("missing"),
        _output("null", value=None),
        _output("empty-string", value=""),
        _output("present", value=False),
        _output("zero", value=0),
        _output("empty-list", value=[]),
        _output("empty-object", value={}),
    )
    exists = EdgeFilter(rules=(FilterRule(field="value", operator=FilterOperator.EXISTS),))
    not_exists = EdgeFilter(rules=(FilterRule(field="value", operator=FilterOperator.NOT_EXISTS),))
    assert [output.id for output in apply_filter(outputs, exists)] == [
        "present",
        "zero",
        "empty-list",
        "empty-object",
    ]
    assert [output.id for output in apply_filter(outputs, not_exists)] == [
        "missing",
        "null",
        "empty-string",
    ]


@pytest.mark.parametrize(
    ("operator", "matches"),
    [
        (FilterOperator.EQ, False),
        (FilterOperator.NEQ, True),
        (FilterOperator.GT, False),
        (FilterOperator.GTE, False),
        (FilterOperator.LT, False),
        (FilterOperator.LTE, False),
        (FilterOperator.IN, False),
        (FilterOperator.NOT_IN, True),
        (FilterOperator.CONTAINS, False),
        (FilterOperator.NOT_CONTAINS, True),
        (FilterOperator.STARTS_WITH, False),
        (FilterOperator.ENDS_WITH, False),
    ],
)
def test_missing_fields_follow_positive_and_negative_predicate_semantics(
    operator: FilterOperator,
    matches: bool,
) -> None:
    result = apply_filter(
        (_output("missing"),),
        EdgeFilter(rules=(FilterRule(field="value", operator=operator, value="expected"),)),
    )

    assert bool(result) is matches


def test_filter_rules_are_conjoined_and_preserve_source_order() -> None:
    outputs = (
        _output("first", enabled=True, score=3),
        _output("second", enabled=False, score=5),
        _output("third", enabled=True, score=4),
    )
    edge_filter = EdgeFilter(
        rules=(
            FilterRule(field="enabled", operator=FilterOperator.EQ, value=True),
            FilterRule(field="score", operator=FilterOperator.GT, value=3),
        )
    )
    assert [output.id for output in apply_filter(outputs, edge_filter)] == ["third"]


def test_filter_rule_and_membership_cardinality_limits() -> None:
    rule = FilterRule(field="value", operator=FilterOperator.EQ, value=True)
    assert len(EdgeFilter(rules=(rule,) * MAX_FILTER_RULES_PER_EDGE).rules) == (
        MAX_FILTER_RULES_PER_EDGE
    )
    with pytest.raises(ValueError, match="too_long"):
        EdgeFilter(rules=(rule,) * (MAX_FILTER_RULES_PER_EDGE + 1))

    values = list(range(MAX_FILTER_MEMBERSHIP_VALUES))
    assert FilterRule(
        field="value",
        operator=FilterOperator.IN,
        value=values,
    ).value == tuple(values)
    with pytest.raises(ValueError, match="membership filter"):
        FilterRule(
            field="value",
            operator=FilterOperator.IN,
            value=values + [MAX_FILTER_MEMBERSHIP_VALUES],
        )


def test_filtering_precedes_selection() -> None:
    outputs = (
        _output("excluded", accepted=False, score=100),
        _output("lower", accepted=True, score=1),
        _output("chosen", accepted=True, score=2),
    )
    filtered = apply_filter(
        outputs,
        EdgeFilter(
            rules=(
                FilterRule(
                    field="accepted",
                    operator=FilterOperator.EQ,
                    value=True,
                ),
            )
        ),
    )
    selected = apply_selection(
        filtered,
        Selection(field="score", order=SortOrder.DESC, limit=1),
    )
    assert [output.id for output in selected] == ["chosen"]


def test_invalid_ordered_types_do_not_match() -> None:
    result = apply_filter(
        (_output("one", value={"nested": 1}),),
        EdgeFilter(rules=(FilterRule(field="value", operator=FilterOperator.GT, value=0),)),
    )
    assert result == ()


def test_ordered_filter_preserves_integer_precision() -> None:
    result = apply_filter(
        (_output("one", value=2**53 + 1),),
        EdgeFilter(
            rules=(
                FilterRule(
                    field="value",
                    operator=FilterOperator.GT,
                    value=2**53,
                ),
            )
        ),
    )

    assert [output.id for output in result] == ["one"]


@pytest.mark.parametrize(
    ("operator", "actual", "expected", "matches"),
    [
        (FilterOperator.EQ, {"enabled": True}, {"enabled": 1}, False),
        (FilterOperator.NEQ, {"enabled": True}, {"enabled": 1}, True),
        (FilterOperator.IN, {"enabled": True}, [{"enabled": 1}], False),
        (FilterOperator.NOT_IN, {"enabled": True}, [{"enabled": 1}], True),
        (FilterOperator.CONTAINS, [{"enabled": True}], {"enabled": 1}, False),
        (FilterOperator.NOT_CONTAINS, [{"enabled": True}], {"enabled": 1}, True),
    ],
)
def test_filter_operators_compare_nested_json_types(
    operator: FilterOperator,
    actual: object,
    expected: object,
    matches: bool,
) -> None:
    result = apply_filter(
        (_output("one", value=actual),),
        EdgeFilter(
            rules=(
                FilterRule(
                    field="value",
                    operator=operator,
                    value=expected,
                ),
            )
        ),
    )

    assert bool(result) is matches
