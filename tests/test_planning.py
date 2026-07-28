import re

import pytest

import dagabaaz.bindings as binding_module
import dagabaaz.models as model_module
import dagabaaz.pipes as pipe_module
import dagabaaz.planning as planning
from dagabaaz.constants import (
    MAX_RUN_INPUT_BYTES,
    MAX_TASK_PLANS_PER_LAUNCH,
    CorrelationMode,
    FilterOperator,
    InputMode,
    InputRole,
    NodeDisposition,
    SortOrder,
)
from dagabaaz.filter import apply_filter, apply_selection
from dagabaaz.json import canonical_json_bytes
from dagabaaz.models import (
    DagNode,
    EdgeFilter,
    EdgeSource,
    ExpressionSource,
    FilterRule,
    InputEdge,
    LiteralSource,
    OutputRef,
    PlanningError,
    PlanningSnapshot,
    RuntimeSource,
    Selection,
    output_ref_routing_size,
)
from dagabaaz.planning import (
    construct_task_plans,
    correlation_id_for_output,
    resolve_correlation_mode,
)


def output(output_id: str, correlation_id: str | None = "c1", **fields: object) -> OutputRef:
    return OutputRef(id=output_id, fields=fields, correlation_id=correlation_id)


def snapshot(
    outputs_by_edge: dict[str, tuple[OutputRef, ...]],
    *,
    dispositions: dict[str, NodeDisposition] | None = None,
    runtime: dict[str, object] | None = None,
) -> PlanningSnapshot:
    return PlanningSnapshot(
        token="snapshot",
        generation=4,
        outputs_by_edge=outputs_by_edge,
        source_dispositions=(
            dispositions
            if dispositions is not None
            else {name: NodeDisposition.LAUNCHED for name in outputs_by_edge}
        ),
        runtime_inputs=runtime or {},
    )


def test_output_fields_are_deeply_immutable() -> None:
    original = {"nested": {"values": [1, 2]}}
    ref = OutputRef(id="one", fields=original)
    original["nested"] = {}
    assert ref.fields["nested"] == {"values": (1, 2)}
    with pytest.raises(TypeError, match="immutable"):
        ref.fields["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError, match="immutable"):
        ref.fields["nested"]["new"] = 1  # type: ignore[index]


@pytest.mark.parametrize(
    "name",
    [
        "",
        "input",
        "has space",
        "9records",
        "-records",
        "records.value",
        "records/child",
        "récords",
        "records\n",
    ],
)
def test_edge_names_follow_the_namespace_rules(name: str) -> None:
    with pytest.raises(ValueError):
        InputEdge(name=name, source="scan")


@pytest.mark.parametrize("name", ["records", "_records", "Records_1", "records-1"])
def test_valid_edge_names_round_trip(name: str) -> None:
    edge = InputEdge(name=name, source="scan")
    assert InputEdge.model_validate_json(edge.model_dump_json()) == edge


def test_filter_uses_strict_json_comparison() -> None:
    outputs = (output("a", value=1), output("b", value="1"), output("c", value=True))
    rule = EdgeFilter(rules=(FilterRule(field="value", operator=FilterOperator.EQ, value=1),))
    assert [item.id for item in apply_filter(outputs, rule)] == ["a"]


def test_selection_excludes_missing_and_null_and_uses_id_tiebreak() -> None:
    outputs = (
        output("z", score=10),
        output("a", score=10),
        output("none", score=None),
        output("missing"),
    )
    selected = apply_selection(outputs, Selection(field="score", order=SortOrder.DESC, limit=2))
    assert [item.id for item in selected] == ["a", "z"]


@pytest.mark.parametrize(
    "outputs",
    [
        (),
        (output("none", score=None), output("missing")),
    ],
)
def test_selection_returns_empty_when_no_values_are_sortable(
    outputs: tuple[OutputRef, ...],
) -> None:
    assert apply_selection(outputs, Selection(field="score")) == ()


def test_selection_orders_values_ascending() -> None:
    outputs = (
        output("three", score=3),
        output("one", score=1),
        output("two", score=2),
    )

    selected = apply_selection(
        outputs,
        Selection(field="score", order=SortOrder.ASC, limit=3),
    )

    assert [item.id for item in selected] == ["one", "two", "three"]


@pytest.mark.parametrize("bad", [True, {"x": 1}, [1, 2]])
def test_selection_rejects_unsortable_values(bad: object) -> None:
    # Arrays and objects are valid routing JSON, but not sortable selection keys.
    ref = output("a", score=bad)
    with pytest.raises(Exception, match="not a sortable"):
        apply_selection((ref,), Selection(field="score"))


def test_selection_rejects_mixed_strings_and_numbers() -> None:
    with pytest.raises(Exception, match="mixed"):
        apply_selection(
            (output("a", score=1), output("b", score="2")),
            Selection(field="score"),
        )


def test_root_creates_one_empty_plan() -> None:
    node = DagNode(slug="root", plugin="scan")
    assert node.input_mode is None
    result = construct_task_plans(node, source_indices={}, snapshot=snapshot({}))
    assert result.disposition == NodeDisposition.LAUNCHED
    assert len(result.plans) == 1
    assert result.plans[0].generation == 4
    assert result.plans[0].edges == ()


def test_root_plan_carries_the_complete_runtime_input() -> None:
    node = DagNode(
        slug="root",
        plugin="scan",
        bindings={"renamed": RuntimeSource(key="source")},
    )
    result = construct_task_plans(
        node,
        source_indices={},
        snapshot=snapshot(
            {},
            runtime={
                "source": "https://example.test/input",
                "enabled": False,
                "limit": 0,
                "options": {"values": [1, 2]},
                "extra": "kept",
            },
        ),
    )

    assert result.disposition == NodeDisposition.LAUNCHED
    assert result.plans[0].parameters == {
        "source": "https://example.test/input",
        "enabled": False,
        "limit": 0,
        "options": {"values": (1, 2)},
        "extra": "kept",
        "renamed": "https://example.test/input",
    }


def test_non_root_plan_includes_only_runtime_values_used_by_bindings() -> None:
    node = DagNode(
        slug="consume",
        plugin="consume",
        input_mode=InputMode.ALL,
        edges=(InputEdge(name="items", source="root"),),
        bindings={"limit": RuntimeSource(key="limit")},
    )
    result = construct_task_plans(
        node,
        source_indices={"items": 0},
        snapshot=snapshot(
            {"items": (output("item"),)},
            runtime={"limit": 2, "unrelated": "not copied"},
        ),
    )

    assert result.disposition == NodeDisposition.LAUNCHED
    assert result.plans[0].parameters == {"limit": 2}


def test_non_root_without_input_mode_fails_planning() -> None:
    node = DagNode(
        slug="consume",
        plugin="consume",
        edges=(InputEdge(name="items", source="root"),),
    )

    result = construct_task_plans(
        node,
        source_indices={"items": 0},
        snapshot=snapshot({"items": (output("item"),)}),
    )

    assert result.disposition == NodeDisposition.FAILED
    assert result.error == "non-root node 'consume' must declare an input mode"


def test_runtime_when_error_fails_planning() -> None:
    node = DagNode(
        slug="root",
        plugin="consume",
        bindings={
            "value": RuntimeSource(
                key="value",
                when="{input.enabled | required}",
            ),
        },
    )

    result = construct_task_plans(
        node,
        source_indices={},
        snapshot=snapshot({}, runtime={"value": "kept"}),
    )

    assert result.disposition == NodeDisposition.FAILED
    assert "Required value is missing" in result.error


def test_large_integer_filter_is_compared_without_overflow() -> None:
    node = DagNode(
        slug="consume",
        plugin="consume",
        input_mode=InputMode.ALL,
        edges=(
            InputEdge(
                name="items",
                source="root",
                filter=EdgeFilter(
                    rules=(
                        FilterRule(
                            field="value",
                            operator=FilterOperator.GT,
                            value=0,
                        ),
                    )
                ),
            ),
        ),
    )

    result = construct_task_plans(
        node,
        source_indices={"items": 0},
        snapshot=snapshot({"items": (output("large", value=10**1000),)}),
    )

    assert result.disposition == NodeDisposition.LAUNCHED
    assert result.plans[0].edges[0].output_ids == ("large",)


def test_large_integer_comparison_pipe_does_not_escape_planning() -> None:
    node = DagNode(
        slug="consume",
        plugin="consume",
        input_mode=InputMode.ALL,
        edges=(InputEdge(name="items", source="root"),),
        bindings={
            "large": ExpressionSource(expression="{items.value | gt(0)}"),
        },
    )

    result = construct_task_plans(
        node,
        source_indices={"items": 0},
        snapshot=snapshot({"items": (output("large", value=10**1000),)}),
    )

    assert result.disposition == NodeDisposition.LAUNCHED
    assert result.plans[0].parameters["large"] is True


def test_deep_json_string_does_not_escape_planning() -> None:
    node = DagNode(
        slug="consume",
        plugin="consume",
        input_mode=InputMode.ALL,
        edges=(InputEdge(name="items", source="root"),),
        bindings={
            "nested": ExpressionSource(expression="{items.value | json_get(key)}"),
        },
    )
    nested = "[" * 20_000 + "0" + "]" * 20_000

    result = construct_task_plans(
        node,
        source_indices={"items": 0},
        snapshot=snapshot({"items": (output("nested", value=nested),)}),
    )

    assert result.disposition == NodeDisposition.LAUNCHED
    assert "nested" not in result.plans[0].parameters


def test_deep_standard_regex_does_not_escape_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipe_module, "_re_engine", re)
    pattern = "(" * 1_000 + "a" + ")" * 1_000
    node = DagNode(
        slug="consume",
        plugin="consume",
        input_mode=InputMode.ALL,
        edges=(InputEdge(name="items", source="root"),),
        bindings={
            "matched": ExpressionSource(expression=f"{{items.value | match({pattern})}}"),
        },
    )

    result = construct_task_plans(
        node,
        source_indices={"items": 0},
        snapshot=snapshot({"items": (output("item", value="a"),)}),
    )

    assert result.disposition == NodeDisposition.LAUNCHED
    assert result.plans[0].parameters["matched"] == ""


def test_each_rejects_more_plans_than_the_configured_maximum() -> None:
    node = DagNode(
        slug="consume",
        plugin="consume",
        input_mode=InputMode.EACH,
        edges=(InputEdge(name="items", source="scan"),),
        bindings={
            "required": RuntimeSource(key="required", required=True),
        },
    )
    outputs = tuple(
        output(f"item-{index}", f"correlation-{index}")
        for index in range(MAX_TASK_PLANS_PER_LAUNCH + 1)
    )
    result = construct_task_plans(
        node,
        source_indices={"items": 0},
        snapshot=snapshot({"items": outputs}),
    )
    assert result.disposition == NodeDisposition.FAILED
    assert f"maximum is {MAX_TASK_PLANS_PER_LAUNCH}" in result.error


def test_by_correlation_rejects_too_many_plans_before_resolving_bindings() -> None:
    node = DagNode(
        slug="consume",
        plugin="consume",
        input_mode=InputMode.BY_CORRELATION,
        edges=(InputEdge(name="items", source="scan"),),
        bindings={
            "required": RuntimeSource(key="required", required=True),
        },
    )
    outputs = tuple(
        output(f"item-{index}", f"correlation-{index}")
        for index in range(MAX_TASK_PLANS_PER_LAUNCH + 1)
    )

    result = construct_task_plans(
        node,
        source_indices={"items": 0},
        snapshot=snapshot({"items": outputs}),
    )

    assert result.disposition == NodeDisposition.FAILED
    assert f"maximum is {MAX_TASK_PLANS_PER_LAUNCH}" in result.error


def test_launch_binding_work_is_checked_before_plan_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(planning, "MAX_LAUNCH_BINDING_EVALUATIONS", 1)
    monkeypatch.setattr(
        planning,
        "resolve_plan_bindings",
        lambda *_args, **_kwargs: pytest.fail("bindings were resolved"),
    )
    node = DagNode(
        slug="consume",
        plugin="consume",
        input_mode=InputMode.EACH,
        edges=(InputEdge(name="items", source="scan"),),
        bindings={"value": LiteralSource(value=True)},
    )

    result = construct_task_plans(
        node,
        source_indices={"items": 0},
        snapshot=snapshot(
            {
                "items": (
                    output("first", "first"),
                    output("second", "second"),
                )
            }
        ),
    )

    assert result.disposition == NodeDisposition.FAILED
    assert "binding evaluations" in result.error


def test_binding_byte_budget_is_shared_across_plans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(planning, "MAX_LAUNCH_BINDING_EVALUATED_BYTES", 10)
    node = DagNode(
        slug="consume",
        plugin="consume",
        input_mode=InputMode.EACH,
        edges=(InputEdge(name="items", source="scan"),),
        bindings={
            "value": ExpressionSource(expression="{items.value}"),
        },
    )

    exact = construct_task_plans(
        node,
        source_indices={"items": 0},
        snapshot=snapshot({"items": (output("first", "first", value="abc"),)}),
    )
    assert exact.disposition == NodeDisposition.LAUNCHED

    over = construct_task_plans(
        node,
        source_indices={"items": 0},
        snapshot=snapshot(
            {
                "items": (
                    output("first", "first", value="abc"),
                    output("second", "second", value="abc"),
                )
            }
        ),
    )
    assert over.disposition == NodeDisposition.FAILED
    assert "10-byte launch limit" in over.error


def test_side_field_extraction_is_shared_across_each_plans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    extract = binding_module.extract_edge_field

    def counted_extract(outputs: tuple[OutputRef, ...], field: str) -> object:
        nonlocal calls
        calls += 1
        return extract(outputs, field)

    monkeypatch.setattr(binding_module, "extract_edge_field", counted_extract)
    node = DagNode(
        slug="consume",
        plugin="consume",
        input_mode=InputMode.EACH,
        edges=(
            InputEdge(name="items", source="scan"),
            InputEdge(
                name="settings",
                source="settings",
                role=InputRole.SIDE,
                required=False,
            ),
        ),
        bindings={
            "missing": ExpressionSource(expression="{settings.missing}"),
        },
    )

    result = construct_task_plans(
        node,
        source_indices={"items": 0, "settings": 1},
        snapshot=snapshot(
            {
                "items": (
                    output("first", "first"),
                    output("second", "second"),
                ),
                "settings": (
                    output("setting-1", None, present=1),
                    output("setting-2", None, present=2),
                ),
            }
        ),
    )

    assert result.disposition == NodeDisposition.LAUNCHED
    assert len(result.plans) == 2
    assert calls == 1


def test_each_creates_edge_addressed_plans_and_broadcasts_side() -> None:
    node = DagNode(
        slug="consume",
        plugin="consume",
        input_mode=InputMode.EACH,
        edges=(
            InputEdge(name="items", source="scan"),
            InputEdge(
                name="settings",
                source="config",
                role=InputRole.SIDE,
                required=False,
            ),
        ),
        bindings={
            "title": EdgeSource(edge="items", field="title"),
            "limit": RuntimeSource(key="limit", default=5),
            "literal": LiteralSource(value={"ok": True}),
        },
    )
    result = construct_task_plans(
        node,
        source_indices={"items": 0, "settings": 1},
        snapshot=snapshot(
            {
                "items": (
                    output("i2", "corr-2", title="Second"),
                    output("i1", "corr-1", title="First"),
                ),
                "settings": (output("s", None, locale="en"),),
            },
            runtime={"limit": 10},
        ),
    )
    assert [plan.correlation_id for plan in result.plans] == ["corr-2", "corr-1"]
    assert result.plans[0].parameters == {
        "title": "Second",
        "limit": 10,
        "literal": {"ok": True},
    }
    assert result.plans[0].edges[1].output_ids == ("s",)


def test_required_skipped_edge_skips_but_optional_skipped_edge_is_empty() -> None:
    required = DagNode(
        slug="required",
        plugin="p",
        edges=(InputEdge(name="in", source="source"),),
        input_mode=InputMode.EACH,
    )
    result = construct_task_plans(
        required,
        source_indices={"in": 0},
        snapshot=snapshot({"in": ()}, dispositions={"in": NodeDisposition.SKIPPED}),
    )
    assert result.disposition == NodeDisposition.SKIPPED

    optional = DagNode(
        slug="optional",
        plugin="p",
        edges=(InputEdge(name="in", source="source", required=False),),
        input_mode=InputMode.ALL,
    )
    result = construct_task_plans(
        optional,
        source_indices={"in": 0},
        snapshot=snapshot({"in": ()}, dispositions={"in": NodeDisposition.SKIPPED}),
    )
    assert result.disposition == NodeDisposition.LAUNCHED
    assert result.plans[0].edges[0].output_ids == ()


@pytest.mark.parametrize(
    ("source_disposition", "expected"),
    [
        (NodeDisposition.SKIPPED, NodeDisposition.SKIPPED),
        (NodeDisposition.FILTERED, NodeDisposition.FILTERED),
    ],
)
def test_required_zero_task_disposition_precedes_optional_output_overflow(
    monkeypatch: pytest.MonkeyPatch,
    source_disposition: NodeDisposition,
    expected: NodeDisposition,
) -> None:
    monkeypatch.setattr(planning, "MAX_SNAPSHOT_OUTPUTS_PER_EDGE", 1)
    node = DagNode(
        slug="consumer",
        plugin="p",
        edges=(
            InputEdge(name="required", source="required"),
            InputEdge(name="optional", source="optional", required=False),
        ),
        input_mode=InputMode.ALL,
    )

    result = construct_task_plans(
        node,
        source_indices={"required": 0, "optional": 1},
        snapshot=snapshot(
            {
                "required": (),
                "optional": (output("one"), output("overflow")),
            },
            dispositions={
                "required": source_disposition,
                "optional": NodeDisposition.LAUNCHED,
            },
        ),
    )

    assert result.disposition == expected


@pytest.mark.parametrize(
    "disposition",
    [
        NodeDisposition.SKIPPED,
        NodeDisposition.FILTERED,
        NodeDisposition.FAILED,
    ],
)
def test_zero_task_source_dispositions_cannot_expose_outputs(
    disposition: NodeDisposition,
) -> None:
    node = DagNode(
        slug="consumer",
        plugin="p",
        input_mode=InputMode.ALL,
        edges=(InputEdge(name="source", source="source", required=False),),
    )
    result = construct_task_plans(
        node,
        source_indices={"source": 0},
        snapshot=snapshot(
            {"source": (output("unexpected"),)},
            dispositions={"source": disposition},
        ),
    )

    assert result.disposition == NodeDisposition.FAILED
    assert "zero-task source disposition" in result.error


def test_snapshot_per_edge_limit_is_a_planning_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_module,
        "MAX_SNAPSHOT_OUTPUTS_PER_EDGE",
        1,
        raising=False,
    )
    monkeypatch.setattr(planning, "MAX_SNAPSHOT_OUTPUTS_PER_EDGE", 1)
    monkeypatch.setattr(planning, "MAX_SNAPSHOT_OUTPUTS_PER_NODE", 10)
    monkeypatch.setattr(
        planning,
        "output_ref_routing_size",
        lambda _output: pytest.fail("routing size was calculated before the edge limit"),
    )
    node = DagNode(
        slug="consumer",
        plugin="consumer",
        input_mode=InputMode.ALL,
        edges=(InputEdge(name="items", source="source"),),
    )

    result = construct_task_plans(
        node,
        source_indices={"items": 0},
        snapshot=snapshot({"items": (output("first"), output("second"))}),
    )

    assert result.disposition == NodeDisposition.FAILED
    assert result.error == "edge 'items' has 2 outputs; maximum is 1"


def test_snapshot_total_limit_is_a_planning_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_module,
        "MAX_SNAPSHOT_OUTPUTS_PER_NODE",
        1,
        raising=False,
    )
    monkeypatch.setattr(planning, "MAX_SNAPSHOT_OUTPUTS_PER_EDGE", 10)
    monkeypatch.setattr(
        planning,
        "MAX_SNAPSHOT_OUTPUTS_PER_NODE",
        1,
        raising=False,
    )
    node = DagNode(
        slug="consumer",
        plugin="consumer",
        input_mode=InputMode.ALL,
        edges=(
            InputEdge(name="left", source="left"),
            InputEdge(name="right", source="right"),
        ),
    )

    result = construct_task_plans(
        node,
        source_indices={"left": 0, "right": 1},
        snapshot=snapshot(
            {
                "left": (output("left"),),
                "right": (output("right"),),
            }
        ),
    )

    assert result.disposition == NodeDisposition.FAILED
    assert result.error == "snapshot contains 2 outputs; maximum is 1"


def test_snapshot_accepts_its_exact_output_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(planning, "MAX_SNAPSHOT_OUTPUTS_PER_EDGE", 2)
    monkeypatch.setattr(planning, "MAX_SNAPSHOT_OUTPUTS_PER_NODE", 2)
    node = DagNode(
        slug="consumer",
        plugin="consumer",
        input_mode=InputMode.ALL,
        edges=(InputEdge(name="items", source="source"),),
    )

    result = construct_task_plans(
        node,
        source_indices={"items": 0},
        snapshot=snapshot({"items": (output("first"), output("second"))}),
    )

    assert result.disposition == NodeDisposition.LAUNCHED
    assert result.plans[0].edges[0].output_ids == ("first", "second")


def test_snapshot_accepts_its_exact_routing_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = output("item", value="payload")
    routing_size = output_ref_routing_size(item)
    monkeypatch.setattr(planning, "MAX_SNAPSHOT_ROUTING_BYTES", routing_size)
    node = DagNode(
        slug="consumer",
        plugin="consumer",
        input_mode=InputMode.ALL,
        edges=(InputEdge(name="items", source="source"),),
    )

    result = construct_task_plans(
        node,
        source_indices={"items": 0},
        snapshot=snapshot({"items": (item,)}),
    )

    assert result.disposition == NodeDisposition.LAUNCHED
    assert result.plans[0].edges[0].output_ids == ("item",)


def test_snapshot_routing_byte_overflow_is_a_planning_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = output("first", value="one")
    second = output("second", value="two")
    routing_size = output_ref_routing_size(first) + output_ref_routing_size(second)
    monkeypatch.setattr(planning, "MAX_SNAPSHOT_ROUTING_BYTES", routing_size - 1)
    node = DagNode(
        slug="consumer",
        plugin="consumer",
        input_mode=InputMode.ALL,
        edges=(InputEdge(name="items", source="source"),),
    )

    result = construct_task_plans(
        node,
        source_indices={"items": 0},
        snapshot=snapshot({"items": (first, second)}),
    )

    assert result.disposition == NodeDisposition.FAILED
    assert result.error == (
        f"snapshot routing data is {routing_size} bytes; maximum is {routing_size - 1}"
    )


def test_runtime_input_overflow_becomes_a_failed_plan() -> None:
    node = DagNode(slug="root", plugin="p")
    result = construct_task_plans(
        node,
        source_indices={},
        snapshot=snapshot({}, runtime={"payload": "x" * MAX_RUN_INPUT_BYTES}),
    )

    assert result.disposition == NodeDisposition.FAILED
    assert "runtime input exceeds" in result.error


def test_root_plan_accepts_the_exact_run_input_limit() -> None:
    envelope_size = len(canonical_json_bytes({"payload": ""}))
    payload = "x" * (MAX_RUN_INPUT_BYTES - envelope_size)
    result = construct_task_plans(
        DagNode(slug="root", plugin="p"),
        source_indices={},
        snapshot=snapshot({}, runtime={"payload": payload}),
    )

    assert result.disposition == NodeDisposition.LAUNCHED
    assert result.plans[0].parameters["payload"] == payload


def test_snapshot_rejects_duplicate_output_ids_within_an_edge() -> None:
    node = DagNode(
        slug="consumer",
        plugin="consumer",
        input_mode=InputMode.ALL,
        edges=(InputEdge(name="items", source="source"),),
    )
    shared = output("shared")

    result = construct_task_plans(
        node,
        source_indices={"items": 0},
        snapshot=snapshot({"items": (shared, shared)}),
    )

    assert result.disposition == NodeDisposition.FAILED
    assert "duplicate output IDs" in result.error


def test_snapshot_repeated_output_ids_must_have_the_same_routing_values() -> None:
    node = DagNode(
        slug="consumer",
        plugin="consumer",
        input_mode=InputMode.ALL,
        edges=(
            InputEdge(name="left", source="left"),
            InputEdge(name="right", source="right"),
        ),
    )
    source_indices = {"left": 0, "right": 1}
    shared = output(
        "shared",
        "correlation",
        nested={"score": 1, "enabled": True},
    )

    matching = construct_task_plans(
        node,
        source_indices=source_indices,
        snapshot=snapshot(
            {
                "left": (shared,),
                "right": (shared.model_copy(),),
            }
        ),
    )
    assert matching.disposition == NodeDisposition.LAUNCHED

    conflicting_number_representation = construct_task_plans(
        node,
        source_indices=source_indices,
        snapshot=snapshot(
            {
                "left": (shared,),
                "right": (
                    shared.model_copy(
                        update={"fields": {"nested": {"score": 1.0, "enabled": True}}}
                    ),
                ),
            }
        ),
    )
    assert conflicting_number_representation.disposition == NodeDisposition.FAILED

    conflicting_object_order = construct_task_plans(
        node,
        source_indices=source_indices,
        snapshot=snapshot(
            {
                "left": (shared,),
                "right": (
                    shared.model_copy(update={"fields": {"nested": {"enabled": True, "score": 1}}}),
                ),
            }
        ),
    )
    assert conflicting_object_order.disposition == NodeDisposition.FAILED

    zero = output("zero", "correlation", value=0.0)
    conflicting_signed_zero = construct_task_plans(
        node,
        source_indices=source_indices,
        snapshot=snapshot(
            {
                "left": (zero,),
                "right": (zero.model_copy(update={"fields": {"value": -0.0}}),),
            }
        ),
    )
    assert conflicting_signed_zero.disposition == NodeDisposition.FAILED

    conflicting_value = construct_task_plans(
        node,
        source_indices=source_indices,
        snapshot=snapshot(
            {
                "left": (shared,),
                "right": (
                    shared.model_copy(update={"fields": {"nested": {"score": 2, "enabled": True}}}),
                ),
            }
        ),
    )
    assert conflicting_value.disposition == NodeDisposition.FAILED
    assert "conflicting values for output 'shared'" in conflicting_value.error

    conflicting_type = construct_task_plans(
        node,
        source_indices=source_indices,
        snapshot=snapshot(
            {
                "left": (shared,),
                "right": (
                    shared.model_copy(update={"fields": {"nested": {"score": 1, "enabled": 1}}}),
                ),
            }
        ),
    )
    assert conflicting_type.disposition == NodeDisposition.FAILED
    assert "conflicting values for output 'shared'" in conflicting_type.error

    conflicting_correlation = construct_task_plans(
        node,
        source_indices=source_indices,
        snapshot=snapshot(
            {
                "left": (shared,),
                "right": (shared.model_copy(update={"correlation_id": "another"}),),
            }
        ),
    )
    assert conflicting_correlation.disposition == NodeDisposition.FAILED
    assert "conflicting values for output 'shared'" in conflicting_correlation.error


def test_all_required_empty_filters_and_all_optional_empty_runs() -> None:
    required = DagNode(
        slug="all",
        plugin="p",
        input_mode=InputMode.ALL,
        edges=(InputEdge(name="in", source="source"),),
    )
    result = construct_task_plans(required, source_indices={"in": 0}, snapshot=snapshot({"in": ()}))
    assert result.disposition == NodeDisposition.FILTERED

    optional = required.model_copy(
        update={"edges": (InputEdge(name="in", source="source", required=False),)}
    )
    result = construct_task_plans(optional, source_indices={"in": 0}, snapshot=snapshot({"in": ()}))
    assert len(result.plans) == 1


def test_by_correlation_required_edges_retain_ids_and_optional_edges_may_be_empty() -> None:
    node = DagNode(
        slug="join",
        plugin="join",
        input_mode=InputMode.BY_CORRELATION,
        edges=(
            InputEdge(name="left", source="left"),
            InputEdge(name="right", source="right", required=False),
        ),
    )
    result = construct_task_plans(
        node,
        source_indices={"left": 0, "right": 1},
        snapshot=snapshot(
            {
                "left": (output("l1", "c1"), output("l2", "c2")),
                "right": (output("r2", "c2"), output("r3", "c3")),
            }
        ),
    )
    assert [plan.correlation_id for plan in result.plans] == ["c1", "c2"]
    assert result.plans[0].edges[1].output_ids == ()
    assert result.plans[1].edges[1].output_ids == ("r2",)

    inner = node.model_copy(
        update={
            "edges": (
                InputEdge(name="left", source="left"),
                InputEdge(name="right", source="right"),
            )
        }
    )
    result = construct_task_plans(
        inner,
        source_indices={"left": 0, "right": 1},
        snapshot=snapshot(
            {
                "left": (output("l1", "c1"), output("l2", "c2")),
                "right": (output("r2", "c2"), output("r3", "c3")),
            }
        ),
    )
    assert [plan.correlation_id for plan in result.plans] == ["c2"]


def test_by_correlation_with_optional_main_edges_is_a_full_outer_join() -> None:
    node = DagNode(
        slug="join",
        plugin="join",
        input_mode=InputMode.BY_CORRELATION,
        edges=(
            InputEdge(name="left", source="left", required=False),
            InputEdge(name="right", source="right", required=False),
        ),
    )
    result = construct_task_plans(
        node,
        source_indices={"left": 0, "right": 1},
        snapshot=snapshot(
            {
                "left": (output("l1", "c1"),),
                "right": (output("r2", "c2"),),
            }
        ),
    )

    assert [plan.correlation_id for plan in result.plans] == ["c1", "c2"]
    assert result.plans[0].edges[1].output_ids == ()
    assert result.plans[1].edges[0].output_ids == ()

    empty = construct_task_plans(
        node,
        source_indices={"left": 0, "right": 1},
        snapshot=snapshot({"left": (), "right": ()}),
    )
    assert empty.disposition == NodeDisposition.FILTERED


def test_by_correlation_plan_order_uses_correlation_id_order() -> None:
    node = DagNode(
        slug="join",
        plugin="join",
        input_mode=InputMode.BY_CORRELATION,
        edges=(InputEdge(name="items", source="items"),),
    )
    result = construct_task_plans(
        node,
        source_indices={"items": 0},
        snapshot=snapshot(
            {
                "items": (
                    output("third", "c2"),
                    output("second", "c10"),
                    output("first", "c1"),
                )
            }
        ),
    )

    assert [plan.correlation_id for plan in result.plans] == ["c1", "c10", "c2"]


def test_side_selection_is_global_and_the_result_is_broadcast() -> None:
    node = DagNode(
        slug="join",
        plugin="join",
        input_mode=InputMode.BY_CORRELATION,
        edges=(
            InputEdge(name="items", source="items"),
            InputEdge(
                name="settings",
                source="settings",
                role=InputRole.SIDE,
                filter=EdgeFilter(
                    rules=(
                        FilterRule(
                            field="enabled",
                            operator=FilterOperator.EQ,
                            value=True,
                        ),
                    )
                ),
                selection=Selection(field="priority", order=SortOrder.DESC, limit=1),
            ),
        ),
    )
    result = construct_task_plans(
        node,
        source_indices={"items": 0, "settings": 1},
        snapshot=snapshot(
            {
                "items": (output("a", "c1"), output("b", "c2")),
                "settings": (
                    output("disabled", None, enabled=False, priority=100),
                    output("selected", None, enabled=True, priority=2),
                    output("lower", None, enabled=True, priority=1),
                ),
            }
        ),
    )

    assert [plan.edges[1].output_ids for plan in result.plans] == [
        ("selected",),
        ("selected",),
    ]

    empty_side = construct_task_plans(
        node,
        source_indices={"items": 0, "settings": 1},
        snapshot=snapshot(
            {
                "items": (output("a", "c1"),),
                "settings": (output("disabled", None, enabled=False, priority=1),),
            }
        ),
    )
    assert empty_side.disposition == NodeDisposition.FILTERED


def test_filter_precedes_selection_on_main_edges() -> None:
    node = DagNode(
        slug="consume",
        plugin="consume",
        input_mode=InputMode.ALL,
        edges=(
            InputEdge(
                name="items",
                source="items",
                filter=EdgeFilter(
                    rules=(
                        FilterRule(
                            field="enabled",
                            operator=FilterOperator.EQ,
                            value=True,
                        ),
                    ),
                ),
                selection=Selection(field="score", order=SortOrder.DESC, limit=1),
            ),
        ),
    )

    result = construct_task_plans(
        node,
        source_indices={"items": 0},
        snapshot=snapshot(
            {
                "items": (
                    output("filtered", enabled=False, score=100),
                    output("selected", enabled=True, score=10),
                    output("lower", enabled=True, score=1),
                ),
            }
        ),
    )

    assert result.disposition == NodeDisposition.LAUNCHED
    assert result.plans[0].edges[0].output_ids == ("selected",)


def test_by_correlation_selection_is_per_group() -> None:
    node = DagNode(
        slug="join",
        plugin="join",
        input_mode=InputMode.BY_CORRELATION,
        edges=(
            InputEdge(
                name="items",
                source="items",
                selection=Selection(field="score", order="desc", limit=1),
            ),
        ),
    )
    result = construct_task_plans(
        node,
        source_indices={"items": 0},
        snapshot=snapshot(
            {
                "items": (
                    output("a", "c1", score=1),
                    output("b", "c1", score=2),
                    output("c", "c2", score=3),
                    output("d", "c2", score=1),
                )
            }
        ),
    )
    assert [plan.edges[0].output_ids for plan in result.plans] == [("b",), ("c",)]


def test_by_correlation_rejects_uncorrelated_main_output() -> None:
    node = DagNode(
        slug="join",
        plugin="join",
        input_mode=InputMode.BY_CORRELATION,
        edges=(InputEdge(name="items", source="items"),),
    )
    result = construct_task_plans(
        node,
        source_indices={"items": 0},
        snapshot=snapshot({"items": (output("a", None),)}),
    )
    assert result.disposition == NodeDisposition.FAILED
    assert "uncorrelated" in result.error


@pytest.mark.parametrize(
    "source_indices",
    [{}, {"items": 0, "unexpected": 1}],
)
def test_source_indices_must_match_named_edges(
    source_indices: dict[str, int],
) -> None:
    node = DagNode(
        slug="consume",
        plugin="consume",
        input_mode=InputMode.ALL,
        edges=(InputEdge(name="items", source="scan"),),
    )
    result = construct_task_plans(
        node,
        source_indices=source_indices,
        snapshot=snapshot({"items": (output("item"),)}),
    )
    assert result.disposition == NodeDisposition.FAILED
    assert "source index edge set" in result.error


@pytest.mark.parametrize("runtime_value", [None, ""])
def test_required_runtime_input_rejects_empty_values(runtime_value: object) -> None:
    node = DagNode(
        slug="root",
        plugin="consume",
        bindings={
            "required": RuntimeSource(key="required", required=True),
        },
    )
    result = construct_task_plans(
        node,
        source_indices={},
        snapshot=snapshot({}, runtime={"required": runtime_value}),
    )
    assert result.disposition == NodeDisposition.FAILED
    assert "required runtime input" in result.error


@pytest.mark.parametrize("runtime_value", [0, False])
def test_required_runtime_input_accepts_nonempty_falsy_values(
    runtime_value: object,
) -> None:
    node = DagNode(
        slug="root",
        plugin="consume",
        bindings={
            "required": RuntimeSource(key="required", required=True),
        },
    )
    result = construct_task_plans(
        node,
        source_indices={},
        snapshot=snapshot({}, runtime={"required": runtime_value}),
    )
    assert result.disposition == NodeDisposition.LAUNCHED
    assert result.plans[0].parameters["required"] is runtime_value


def test_correlation_default_resolution_and_emission() -> None:
    root = DagNode(slug="root", plugin="p")
    each = DagNode(slug="each", plugin="p", input_mode=InputMode.EACH)
    all_node = DagNode(slug="all", plugin="p", input_mode=InputMode.ALL)
    assert resolve_correlation_mode(root, is_root=True) == CorrelationMode.NEW
    assert resolve_correlation_mode(each, is_root=False) == CorrelationMode.INHERIT
    assert resolve_correlation_mode(all_node, is_root=False) == CorrelationMode.NONE
    assert (
        correlation_id_for_output(root, is_root=True, task_correlation_id=None, output_id="out")
        == "out"
    )
    with pytest.raises(PlanningError, match="no correlation"):
        correlation_id_for_output(each, is_root=False, task_correlation_id=None, output_id="out")
    with pytest.raises(PlanningError, match="no correlation"):
        correlation_id_for_output(each, is_root=False, task_correlation_id="", output_id="out")
    with pytest.raises(PlanningError, match="output ID"):
        correlation_id_for_output(root, is_root=True, task_correlation_id=None, output_id="")


def test_each_plan_cannot_inherit_an_absent_correlation_id() -> None:
    node = DagNode(
        slug="each",
        plugin="p",
        input_mode=InputMode.EACH,
        edges=(InputEdge(name="source", source="source"),),
    )

    result = construct_task_plans(
        node,
        source_indices={"source": 0},
        snapshot=snapshot({"source": (output("uncorrelated", None),)}),
    )

    assert result.disposition == NodeDisposition.FAILED
    assert result.plans == ()
    assert "task has no correlation ID" in result.error


def test_explicit_correlation_modes_override_task_mode_defaults() -> None:
    restart = DagNode(
        slug="restart",
        plugin="p",
        correlation_mode=CorrelationMode.NEW,
    )
    inherit = DagNode(
        slug="inherit",
        plugin="p",
        correlation_mode=CorrelationMode.INHERIT,
    )
    reset = DagNode(
        slug="reset",
        plugin="p",
        correlation_mode=CorrelationMode.NONE,
    )

    assert (
        correlation_id_for_output(
            restart,
            is_root=False,
            task_correlation_id=None,
            output_id="first",
        )
        == "first"
    )
    assert (
        correlation_id_for_output(
            restart,
            is_root=False,
            task_correlation_id=None,
            output_id="second",
        )
        == "second"
    )
    assert (
        correlation_id_for_output(
            inherit,
            is_root=False,
            task_correlation_id="group",
            output_id="output",
        )
        == "group"
    )
    assert (
        correlation_id_for_output(
            reset,
            is_root=True,
            task_correlation_id="group",
            output_id="output",
        )
        is None
    )
