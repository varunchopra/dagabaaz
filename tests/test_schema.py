import pytest

from dagabaaz.constants import InputMode
from dagabaaz.json import FrozenDict
from dagabaaz.models import (
    DagNode,
    EdgeSource,
    ExpressionSource,
    InputEdge,
    LiteralSource,
    RuntimeSource,
)
from dagabaaz.plugins import PluginInputMeta, PluginMeta
from dagabaaz.schema import (
    InputFieldSpec,
    get_pipeline_input_schema,
    merge_run_input,
    validate_binding_references,
)


class Plugin:
    name = "plugin"
    worker_only = False

    def get_effective_inputs(self) -> list[PluginInputMeta]:
        return [
            PluginInputMeta(
                name="url",
                description="URL",
                source="runtime",
                required=True,
                default="https://example.test",
            )
        ]


def test_plugin_protocol_requires_only_input_metadata() -> None:
    class MinimalPlugin:
        def get_effective_inputs(self) -> list[PluginInputMeta]:
            return []

    assert isinstance(MinimalPlugin(), PluginMeta)


def test_plugin_and_schema_defaults_are_frozen_json() -> None:
    metadata = PluginInputMeta(
        name="options",
        description="Options",
        source="runtime",
        default={"values": [1, 2]},
    )
    field = InputFieldSpec(
        name="options",
        label="Options",
        default=metadata.default,
    )

    assert isinstance(metadata.default, FrozenDict)
    assert isinstance(field.default, FrozenDict)
    assert field.default["values"] == (1, 2)

    with pytest.raises(ValueError, match="not JSON-compatible"):
        PluginInputMeta(
            name="callback",
            description="Callback",
            source="runtime",
            default=object(),
        )
    with pytest.raises(ValueError, match="name must not be empty"):
        PluginInputMeta(name="", description="Missing", source="runtime")


def test_input_schema_keeps_typed_defaults_and_expression_inputs() -> None:
    node = DagNode(
        slug="root",
        plugin="plugin",
        bindings={
            "count": RuntimeSource(key="count", default=3),
            "computed": ExpressionSource(expression="{input.locale}"),
        },
    )
    fields = get_pipeline_input_schema([node], lambda _name: Plugin())
    assert [(field.name, field.default) for field in fields] == [
        ("url", "https://example.test"),
        ("count", 3),
        ("locale", None),
    ]


def test_input_schema_includes_runtime_inputs_used_by_when_clauses() -> None:
    node = DagNode(
        slug="root",
        plugin="plugin",
        bindings={
            "value": LiteralSource(value="enabled", when="{input.feature_enabled}"),
        },
    )
    fields = get_pipeline_input_schema([node], lambda _name: None)
    assert fields == [InputFieldSpec(name="feature_enabled", label="Feature Enabled")]

    duplicate = DagNode(
        slug="root",
        plugin="plugin",
        bindings={
            "url": RuntimeSource(key="url", when="{input.use_default_url}"),
        },
    )
    fields = get_pipeline_input_schema([duplicate], lambda _name: Plugin())
    assert [field.name for field in fields] == ["url", "use_default_url"]


def test_merge_precedence_and_required_values() -> None:
    fields = [
        InputFieldSpec("count", "Count", required=True, default=1),
        InputFieldSpec("enabled", "Enabled", required=True, default=True),
    ]
    assert merge_run_input(
        fields,
        {"count": 2, "enabled": True},
        {"count": 3, "enabled": False, "extra": "kept"},
    ) == {"count": 3, "enabled": False, "extra": "kept"}
    assert merge_run_input(
        fields,
        {"count": 2, "enabled": True},
        {"count": "", "enabled": None},
    ) == {"count": 2, "enabled": True}
    assert merge_run_input(
        [InputFieldSpec("count", "Count", required=True)],
        {},
        {"count": 0},
    ) == {"count": 0}
    with pytest.raises(ValueError, match="required input"):
        merge_run_input([InputFieldSpec("x", "X", required=True)], {}, {})


def test_merge_validates_and_freezes_runtime_input() -> None:
    defaults = {"options": {"values": [1, 2]}}
    supplied = {"labels": ["one", "two"]}
    merged = merge_run_input(
        [],
        defaults,
        supplied,
    )
    assert isinstance(merged, FrozenDict)
    assert isinstance(merged["options"], FrozenDict)
    assert merged["options"]["values"] == (1, 2)
    assert merged["labels"] == ("one", "two")

    defaults["options"]["values"].append(3)
    supplied["labels"].append("three")
    assert merged == {
        "options": {"values": (1, 2)},
        "labels": ("one", "two"),
    }
    with pytest.raises(TypeError):
        merged["added"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        merged |= {"added": True}
    with pytest.raises(TypeError):
        merged["options"]["added"] = True  # type: ignore[index]

    with pytest.raises(ValueError, match="not JSON-compatible"):
        merge_run_input([], {}, {"callback": object()})


@pytest.mark.parametrize("pipeline_default", [None, ""])
def test_pipeline_defaults_override_schema_defaults_even_when_empty(
    pipeline_default: object,
) -> None:
    merged = merge_run_input(
        [InputFieldSpec("value", "Value", default="schema")],
        {"value": pipeline_default},
        {"value": None},
    )
    assert merged["value"] == pipeline_default


def test_binding_validation_uses_edge_names() -> None:
    nodes = [
        DagNode(slug="root", plugin="p"),
        DagNode(
            slug="child",
            plugin="p",
            input_mode=InputMode.EACH,
            edges=(InputEdge(name="records", source="root"),),
            bindings={
                "title": EdgeSource(edge="records", field="title"),
                "slug": ExpressionSource(expression="{records.title | lower}"),
            },
        ),
    ]
    assert validate_binding_references(nodes) is None
    bad = nodes[1].model_copy(
        update={"bindings": {"title": EdgeSource(edge="missing", field="title")}}
    )
    error = validate_binding_references([nodes[0], bad])
    assert error and "unknown edge" in error


def test_expression_unknown_edge_is_rejected() -> None:
    node = DagNode(
        slug="child",
        plugin="p",
        bindings={"x": ExpressionSource(expression="{ghost.x}")},
    )
    assert "unknown edges" in (validate_binding_references([node]) or "")


@pytest.mark.parametrize(
    "binding",
    [
        ExpressionSource(expression="{records.value | unknown}"),
        LiteralSource(value=True, when="{records.value"),
    ],
)
def test_malformed_expression_and_when_are_rejected(binding: object) -> None:
    root = DagNode(slug="root", plugin="p")
    node = DagNode(
        slug="child",
        plugin="p",
        input_mode=InputMode.EACH,
        edges=(InputEdge(name="records", source="root"),),
        bindings={"value": binding},
    )
    error = validate_binding_references([root, node])
    assert error is not None
    assert "error" in error
