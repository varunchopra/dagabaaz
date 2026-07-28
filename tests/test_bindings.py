import pytest

from dagabaaz.bindings import extract_edge_field, resolve_plan_bindings
from dagabaaz.models import (
    DagNode,
    EdgeSource,
    ExpressionSource,
    LiteralSource,
    OutputRef,
    PlanningError,
    RuntimeSource,
    SecretSource,
)


def _outputs() -> tuple[OutputRef, ...]:
    return (
        OutputRef(
            id="one",
            fields={"title": "First", "metadata": {"rank": 1}},
        ),
        OutputRef(
            id="two",
            fields={"title": "Second", "metadata": {"rank": 2}},
        ),
    )


def test_edge_field_preserves_native_values_and_order() -> None:
    outputs = _outputs()
    assert extract_edge_field(outputs, "missing") is None
    assert extract_edge_field(outputs[:1], "metadata") == {"rank": 1}
    assert extract_edge_field(outputs, "title") == ("First", "Second")


def test_all_binding_sources_resolve_without_merging_namespaces() -> None:
    node = DagNode(
        plugin="consume",
        bindings={
            "titles": EdgeSource(edge="records", field="title"),
            "literal": LiteralSource(value={"enabled": True}),
            "limit": RuntimeSource(key="limit", default=5),
            "first": ExpressionSource(expression="{records.title | first}"),
            "joined": ExpressionSource(expression="{records.title | join(;)}"),
            "rank": ExpressionSource(
                expression="{records.metadata | first | json_get(rank)}"
            ),
            "token": SecretSource(name="service-token"),
        },
    )
    parameters, secret_refs = resolve_plan_bindings(
        node,
        {"records": _outputs()},
        {"limit": 2},
    )
    assert parameters == {
        "titles": ("First", "Second"),
        "literal": {"enabled": True},
        "limit": 2,
        "first": "First",
        "joined": "First;Second",
        "rank": 1,
    }
    assert secret_refs == {"token": "service-token"}


def test_nested_frozen_objects_work_with_json_get() -> None:
    node = DagNode(
        plugin="consume",
        bindings={
            "rank": ExpressionSource(expression="{records.metadata | json_get(rank)}"),
        },
    )
    parameters, _secret_refs = resolve_plan_bindings(
        node,
        {"records": _outputs()[:1]},
        {},
    )
    assert parameters["rank"] == 1


def test_expression_text_pipes_persist_thawed_json_containers() -> None:
    outputs = (
        OutputRef(id="one", fields={"metadata": {"rank": 1}}),
        OutputRef(
            id="two",
            fields={"metadata": {"rank": 2, "tags": ["reviewed"]}},
        ),
    )
    node = DagNode(
        plugin="consume",
        bindings={
            "single": ExpressionSource(expression="{single.metadata | string}"),
            "stringified": ExpressionSource(expression="{records.metadata | string}"),
            "joined": ExpressionSource(expression="{records.metadata | join(;)}"),
        },
    )

    parameters, _secret_refs = resolve_plan_bindings(
        node,
        {"single": outputs[:1], "records": outputs},
        {},
    )

    assert parameters == {
        "single": "{'rank': 1}",
        "stringified": "{'rank': 1}, {'rank': 2, 'tags': ['reviewed']}",
        "joined": "{'rank': 1};{'rank': 2, 'tags': ['reviewed']}",
    }


def test_when_can_use_edge_and_runtime_values() -> None:
    node = DagNode(
        plugin="consume",
        bindings={
            "edge_value": EdgeSource(
                edge="records",
                field="title",
                when="{records.enabled}",
            ),
            "runtime_value": LiteralSource(
                value="included",
                when="{input.enabled}",
            ),
            "secret": SecretSource(
                name="service-token",
                when="{input.include_secret}",
            ),
        },
    )
    parameters, secret_refs = resolve_plan_bindings(
        node,
        {
            "records": (
                OutputRef(
                    id="one",
                    fields={"title": "First", "enabled": True},
                ),
            )
        },
        {"enabled": True, "include_secret": False},
    )
    assert parameters == {
        "edge_value": "First",
        "runtime_value": "included",
    }
    assert secret_refs == {}


def test_when_not_pipe_uses_python_truth_value_testing() -> None:
    node = DagNode(
        plugin="consume",
        bindings={
            "value": LiteralSource(
                value="included",
                when="{input.disabled | not}",
            ),
        },
    )

    included, _secret_refs = resolve_plan_bindings(node, {}, {"disabled": False})
    excluded, _secret_refs = resolve_plan_bindings(node, {}, {"disabled": True})

    assert included == {"value": "included"}
    assert excluded == {}


@pytest.mark.parametrize("missing", [None, ""])
def test_required_runtime_input_rejects_missing_values(missing: object) -> None:
    node = DagNode(
        plugin="consume",
        bindings={"value": RuntimeSource(key="value", required=True)},
    )
    with pytest.raises(PlanningError, match="required runtime input"):
        resolve_plan_bindings(node, {}, {"value": missing})


def test_runtime_defaults_preserve_zero_and_false() -> None:
    node = DagNode(
        plugin="consume",
        bindings={
            "zero": RuntimeSource(key="zero", required=True, default=0),
            "false": RuntimeSource(key="false", required=True, default=False),
        },
    )
    parameters, _secret_refs = resolve_plan_bindings(node, {}, {})
    assert parameters == {"zero": 0, "false": False}


def test_base_parameters_are_preserved_until_an_active_binding_replaces_them() -> None:
    node = DagNode(
        plugin="root",
        bindings={
            "replaced": LiteralSource(value="binding"),
            "unchanged": LiteralSource(value="binding", when="{input.disabled}"),
        },
    )

    parameters, secret_refs = resolve_plan_bindings(
        node,
        {},
        {"disabled": False},
        base_parameters={
            "replaced": "runtime",
            "unchanged": "runtime",
            "extra": {"values": [0, False]},
        },
    )

    assert parameters == {
        "replaced": "binding",
        "unchanged": "runtime",
        "extra": {"values": (0, False)},
    }
    assert secret_refs == {}


def test_active_secret_binding_replaces_a_same_named_base_parameter() -> None:
    node = DagNode(
        plugin="root",
        bindings={
            "token": SecretSource(name="service-token"),
            "conditional": SecretSource(
                name="conditional-token",
                when="{input.disabled}",
            ),
        },
    )

    parameters, secret_refs = resolve_plan_bindings(
        node,
        {},
        {"disabled": False},
        base_parameters={"token": "runtime", "conditional": "runtime"},
    )

    assert parameters == {"conditional": "runtime"}
    assert secret_refs == {"token": "service-token"}
