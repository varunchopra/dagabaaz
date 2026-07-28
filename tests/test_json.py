import copy
import json
import warnings

import pytest

from dagabaaz.constants import (
    MAX_INPUT_EDGES_PER_NODE,
    MAX_JSON_DEPTH,
    MAX_JSON_KEYS,
    MAX_JSON_VALUES,
    MAX_OUTPUT_REFS_PER_PLAN,
    MAX_TASK_PLAN_BYTES,
    NodeDisposition,
)
from dagabaaz.json import (
    FrozenDict,
    canonical_json_bytes,
    freeze_json,
    json_values_equal,
    thaw_json,
)
from dagabaaz.models import (
    DagNode,
    EdgeSource,
    InputEdge,
    LiteralSource,
    NodeLaunch,
    OutputRef,
    PlannedEdgeInput,
    PlanningSnapshot,
    ResolvedOutput,
    TaskInputPlan,
    TaskInputs,
)
from dagabaaz.pipes import BUILTIN_PIPES


@pytest.mark.parametrize("name", ["records", "_settings", "records-2"])
def test_edge_names_use_one_addressable_grammar(name: str) -> None:
    assert InputEdge(name=name, source="source").name == name
    assert EdgeSource(edge=name, field="value").edge == name
    assert PlannedEdgeInput(edge=name, source_index=0, role="main").edge == name


@pytest.mark.parametrize("name", ["", "input", "2records", "record.value", "record value"])
def test_invalid_or_reserved_edge_names_are_rejected(name: str) -> None:
    for constructor in (
        lambda: InputEdge(name=name, source="source"),
        lambda: EdgeSource(edge=name, field="value"),
        lambda: PlannedEdgeInput(edge=name, source_index=0, role="main"),
    ):
        with pytest.raises(ValueError):
            constructor()


def test_binding_names_and_when_expressions_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="binding names"):
        DagNode(plugin="consume", bindings={"": LiteralSource(value=True)})
    with pytest.raises(ValueError, match="too_short"):
        LiteralSource(value=True, when="")


def test_routing_json_rejects_normal_and_inherited_dict_mutation() -> None:
    original = {"nested": {"items": [1, 2]}}
    ref = OutputRef(id="output", fields=original)
    fields = ref.fields
    nested = fields["nested"]

    with pytest.raises(TypeError, match="immutable"):
        fields |= {"changed": True}
    with pytest.raises(TypeError, match="immutable"):
        nested |= {"changed": True}  # type: ignore[operator]
    with pytest.raises(TypeError):
        dict.__setitem__(fields, "changed", True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="immutable"):
        fields._values = {}  # type: ignore[attr-defined]

    original["nested"] = {"changed": True}
    assert thaw_json(ref.fields) == {"nested": {"items": [1, 2]}}


def test_frozen_dict_constructor_copies_and_freezes_nested_values() -> None:
    original = {"nested": {"items": [1, 2]}}
    frozen = FrozenDict(original)

    original["nested"] = {"changed": True}
    assert thaw_json(frozen) == {"nested": {"items": [1, 2]}}
    with pytest.raises(TypeError, match="immutable"):
        frozen["nested"] |= {"changed": True}  # type: ignore[index,operator]


@pytest.mark.parametrize(
    ("left", "right", "equal"),
    [
        (None, None, True),
        (None, "null", False),
        (True, True, True),
        (True, 1, False),
        (False, 0, False),
        (1, 1.0, True),
        (1, "1", False),
        ([1, True], [1.0, 1], False),
        ([1, 2], [2, 1], False),
        (
            {"outer": {"enabled": True, "scores": [1, 2]}},
            {"outer": {"scores": [1.0, 2.0], "enabled": True}},
            True,
        ),
        (
            {"outer": {"enabled": True}},
            {"outer": {"enabled": 1}},
            False,
        ),
    ],
)
def test_json_value_equality_is_recursive_and_type_aware(
    left: object,
    right: object,
    equal: bool,
) -> None:
    assert json_values_equal(freeze_json(left), freeze_json(right)) is equal


def test_json_container_boundary_freezes_core_and_thaws_serialisation() -> None:
    frozen = freeze_json({"items": [{"enabled": True}]})

    assert isinstance(frozen, FrozenDict)
    items = frozen["items"]
    assert isinstance(items, tuple)
    assert isinstance(items[0], FrozenDict)

    serialisable = thaw_json(frozen)
    assert isinstance(serialisable, dict)
    assert isinstance(serialisable["items"], list)
    assert isinstance(serialisable["items"][0], dict)


def test_routing_json_rejects_invalid_numbers_keys_values_and_depth() -> None:
    with pytest.raises(ValueError, match="finite"):
        LiteralSource(value=float("nan"))
    with pytest.raises(ValueError, match="keys must be strings"):
        LiteralSource(value={1: "value"})
    with pytest.raises(ValueError, match="not JSON-compatible"):
        LiteralSource(value=object())
    with pytest.raises(ValueError, match="key count"):
        LiteralSource(value={f"key-{index}": index for index in range(MAX_JSON_KEYS + 1)})
    with pytest.raises(ValueError, match="value count"):
        LiteralSource(value=list(range(MAX_JSON_VALUES + 1)))

    nested: object = "value"
    for _index in range(MAX_JSON_DEPTH + 1):
        nested = [nested]
    with pytest.raises(ValueError, match="maximum depth"):
        LiteralSource(value=nested)


def test_node_and_plan_cardinality_limits_are_enforced() -> None:
    edges = tuple(
        InputEdge(name=f"edge_{index}", source="source")
        for index in range(MAX_INPUT_EDGES_PER_NODE + 1)
    )
    with pytest.raises(ValueError, match="too_long"):
        DagNode(plugin="consume", edges=edges)

    planned_edges = tuple(
        PlannedEdgeInput(edge=f"edge_{index}", source_index=0, role="main")
        for index in range(MAX_INPUT_EDGES_PER_NODE + 1)
    )
    with pytest.raises(ValueError, match="too_long"):
        TaskInputPlan(generation=1, edges=planned_edges)


@pytest.mark.parametrize(
    "parameters",
    [
        {f"key-{index}": index for index in range(MAX_JSON_KEYS)},
        {"items": list(range(MAX_JSON_VALUES - 1))},
    ],
    ids=["key-limit", "value-limit"],
)
def test_plan_accepts_parameters_at_exact_routing_json_limits(
    parameters: dict[str, object],
) -> None:
    plan = TaskInputPlan(generation=1, parameters=parameters)

    assert len(plan.parameters) == len(parameters)


def test_routing_mappings_are_recursively_immutable() -> None:
    plan = TaskInputPlan(generation=1, parameters={"nested": {"value": 1}})
    snapshot = PlanningSnapshot(
        token="snapshot",
        generation=1,
        outputs_by_edge={},
        source_dispositions={},
        runtime_inputs={"nested": {"value": 1}},
    )
    literal = LiteralSource(value={"nested": {"value": 1}})

    for mapping in (
        plan.parameters,
        snapshot.runtime_inputs,
        literal.value,
    ):
        assert isinstance(mapping, FrozenDict)
        with pytest.raises(TypeError, match="immutable"):
            mapping |= {"changed": True}  # type: ignore[operator]
        child = mapping["nested"]
        with pytest.raises(TypeError, match="immutable"):
            child |= {"changed": True}  # type: ignore[operator]


def test_frozen_json_copy_and_serialisation_preserve_json_wire_shape() -> None:
    ref = OutputRef(
        id="output",
        fields={"nested": {"items": [1, 2]}, "enabled": True},
    )
    plan = TaskInputPlan(
        generation=3,
        parameters={"output": ref.fields, "values": [1, 2]},
        secret_refs={"token": "service-token"},
    )

    assert copy.copy(ref.fields) is ref.fields
    assert copy.deepcopy(ref.fields) is ref.fields
    assert plan.parameters["values"] == (1, 2)
    dumped = plan.model_dump(mode="json")
    assert dumped["parameters"] == {
        "output": {"nested": {"items": [1, 2]}, "enabled": True},
        "values": [1, 2],
    }
    assert dumped["secret_refs"] == {"token": "service-token"}
    assert canonical_json_bytes(plan.parameters) == canonical_json_bytes(dumped["parameters"])
    assert TaskInputPlan.model_validate(dumped).model_dump(mode="json") == dumped


def test_model_copy_validates_updates_to_frozen_models() -> None:
    ref = OutputRef(id="output").model_copy(update={"fields": {"items": [1, 2]}})
    plan = TaskInputPlan(generation=1).model_copy(update={"parameters": {"nested": {"value": 1}}})

    assert isinstance(ref.fields, FrozenDict)
    assert ref.fields["items"] == (1, 2)
    assert isinstance(plan.parameters, FrozenDict)
    assert isinstance(plan.parameters["nested"], FrozenDict)

    with pytest.raises(TypeError, match="immutable"):
        ref.fields |= {"changed": True}
    with pytest.raises(TypeError, match="immutable"):
        plan.parameters["nested"] |= {"changed": True}  # type: ignore[index,operator]


def test_default_mappings_on_frozen_models_cannot_be_changed() -> None:
    plan = TaskInputPlan(generation=1)
    output = ResolvedOutput(id="output")

    with pytest.raises(TypeError, match="immutable"):
        plan.secret_refs |= {"token": "secret"}
    with pytest.raises(TypeError):
        output.data["path"] = "/tmp/file"  # type: ignore[index]


def test_routing_models_dump_and_round_trip_without_warnings() -> None:
    output = OutputRef(
        id="output",
        fields={"nested": {"items": [1, 2]}},
        correlation_id="correlation",
    )
    edge = PlannedEdgeInput(
        edge="records",
        source_index=0,
        role="main",
        output_ids=("output",),
    )
    plan = TaskInputPlan(
        generation=1,
        edges=(edge,),
        parameters={"items": [1, 2]},
        secret_refs={"token": "service-token"},
        correlation_id="correlation",
    )
    snapshot = PlanningSnapshot(
        token="snapshot",
        generation=1,
        outputs_by_edge={"records": (output,)},
        source_dispositions={"records": NodeDisposition.LAUNCHED},
        runtime_inputs={"items": [1, 2]},
    )
    launch = NodeLaunch(
        id="launch",
        run_id="run",
        node_index=1,
        plugin_name="consume",
        generation=1,
        disposition=NodeDisposition.LAUNCHED,
        task_ids=("task",),
    )
    inputs = TaskInputs(
        edges={
            "records": (
                ResolvedOutput(
                    id="output",
                    fields=output.fields,
                    data={"uri": "object://output"},
                ),
            )
        },
        parameters=plan.parameters,
    )

    for model in (output, plan, snapshot, launch, inputs):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            python_dump = model.model_dump(mode="python")
            json_dump = model.model_dump(mode="json")
            json_text = model.model_dump_json()

        model_type = type(model)
        assert model_type.model_validate(python_dump).model_dump(mode="json") == json_dump
        assert model_type.model_validate(json_dump).model_dump(mode="json") == json_dump
        assert model_type.model_validate_json(json_text).model_dump(mode="json") == json_dump


def test_task_plan_rejects_duplicate_edges_and_output_ids() -> None:
    edge = PlannedEdgeInput(
        edge="records",
        source_index=0,
        role="main",
        output_ids=("first",),
    )
    with pytest.raises(ValueError, match="edge names must be unique"):
        TaskInputPlan(generation=1, edges=(edge, edge))

    with pytest.raises(ValueError, match="duplicate output IDs"):
        PlannedEdgeInput(
            edge="records",
            source_index=0,
            role="main",
            output_ids=("first", "first"),
        )

    with pytest.raises(ValueError, match="must not be empty"):
        PlannedEdgeInput(
            edge="records",
            source_index=0,
            role="main",
            output_ids=("",),
        )


def test_task_plan_rejects_an_unsupported_schema_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        TaskInputPlan.model_validate({"schema_version": 2, "generation": 1})


def test_task_plan_allows_the_same_output_on_different_edges() -> None:
    plan = TaskInputPlan(
        generation=1,
        edges=(
            PlannedEdgeInput(
                edge="primary",
                source_index=0,
                role="main",
                output_ids=("shared",),
            ),
            PlannedEdgeInput(
                edge="secondary",
                source_index=1,
                role="side",
                output_ids=("shared",),
            ),
        ),
    )

    assert plan.edges[0].output_ids == plan.edges[1].output_ids == ("shared",)


def test_task_plan_rejects_parameter_and_secret_target_overlap() -> None:
    with pytest.raises(ValueError, match="targets overlap"):
        TaskInputPlan(
            generation=1,
            parameters={"token": "literal"},
            secret_refs={"token": "service-token"},
        )


@pytest.mark.parametrize(
    "secret_refs",
    [{"": "service-token"}, {"token": ""}],
)
def test_task_plan_rejects_empty_secret_targets_and_names(
    secret_refs: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        TaskInputPlan(generation=1, secret_refs=secret_refs)


def test_task_plan_enforces_reference_and_serialized_size_limits_on_load() -> None:
    output_ids = tuple(f"output-{index}" for index in range(MAX_OUTPUT_REFS_PER_PLAN + 1))
    with pytest.raises(ValueError, match="references .* outputs"):
        TaskInputPlan.model_validate(
            {
                "generation": 1,
                "edges": [
                    {
                        "edge": "records",
                        "source_index": 0,
                        "role": "main",
                        "output_ids": output_ids,
                    }
                ],
            }
        )

    with pytest.raises(ValueError, match="bytes; maximum"):
        TaskInputPlan.model_validate_json(
            json.dumps(
                {
                    "generation": 1,
                    "parameters": {"payload": "x" * MAX_TASK_PLAN_BYTES},
                }
            )
        )


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (OutputRef(id="output"), "correlation_id"),
        (TaskInputPlan(generation=1), "correlation_id"),
    ],
)
def test_correlation_ids_must_not_be_empty(model: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        model.model_copy(update={"correlation_id": ""})  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "values",
    [
        {
            "disposition": NodeDisposition.LAUNCHED,
            "task_ids": (),
            "complete": False,
            "error": "",
        },
        {
            "disposition": NodeDisposition.SKIPPED,
            "task_ids": ("task",),
            "complete": True,
            "error": "",
        },
        {
            "disposition": NodeDisposition.FILTERED,
            "task_ids": (),
            "complete": False,
            "error": "",
        },
        {
            "disposition": NodeDisposition.FAILED,
            "task_ids": (),
            "complete": True,
            "error": "",
        },
    ],
)
def test_node_launch_rejects_invalid_disposition_state(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        NodeLaunch(
            id="launch",
            run_id="run",
            node_index=0,
            plugin_name="consume",
            generation=1,
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "disposition",
    [NodeDisposition.SKIPPED, NodeDisposition.FILTERED, NodeDisposition.FAILED],
)
def test_zero_task_node_launches_are_complete(
    disposition: NodeDisposition,
) -> None:
    launch = NodeLaunch(
        id="launch",
        run_id="run",
        node_index=0,
        plugin_name="consume",
        generation=1,
        disposition=disposition,
        complete=True,
        error="source failed" if disposition == NodeDisposition.FAILED else "",
    )
    assert launch.task_ids == ()


def test_frozen_json_values_work_with_sequence_and_mapping_pipes() -> None:
    value = LiteralSource(
        value={
            "items": [["first"], None, ["second"]],
            "record": {"name": "value"},
        }
    ).value
    assert isinstance(value, FrozenDict)
    items = value["items"]
    record = value["record"]

    assert BUILTIN_PIPES["first"](items) == ("first",)
    assert BUILTIN_PIPES["last"](items) == ("second",)
    assert BUILTIN_PIPES["nth"](items, "1") is None
    assert BUILTIN_PIPES["join"](("first", None, "second"), ":") == "first:second"
    assert BUILTIN_PIPES["string"](("first", "second")) == "first, second"
    assert BUILTIN_PIPES["flatten"](items) == ("first", None, "second")
    assert BUILTIN_PIPES["compact"](items) == (("first",), ("second",))
    assert BUILTIN_PIPES["json_get"](record, "name") == "value"
