"""Runtime input schema construction and binding validation.

Schema fields come from plugin runtime declarations, ``RuntimeSource``
bindings and ``input.*`` references in expressions or ``when`` clauses.
"""

from __future__ import annotations

from dataclasses import dataclass

from dagabaaz.expressions import extract_refs, validate_expression
from dagabaaz.json import FrozenDict, JsonInput, JsonValue, freeze_json, freeze_object
from dagabaaz.models import DagNode, EdgeSource, ExpressionError, ExpressionSource, RuntimeSource
from dagabaaz.plugins import PluginLookup


@dataclass(frozen=True, slots=True)
class InputFieldSpec:
    """One field in a pipeline's run-input schema.

    ``default`` accepts JSON input and is frozen during construction.
    """

    name: str
    label: str
    placeholder: str = ""
    required: bool = False
    default: JsonInput = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("input field name must not be empty")
        object.__setattr__(self, "default", freeze_json(self.default))


def get_pipeline_input_schema(
    nodes: list[DagNode], plugin_lookup: PluginLookup
) -> list[InputFieldSpec]:
    """The schema contains each run-input field once.

    Plugin declarations are considered before fields inferred from bindings,
    and the first declaration of a name is retained. Defaults are stored as
    immutable JSON values.
    """

    fields: list[InputFieldSpec] = []
    seen: set[str] = set()

    for node in nodes:
        plugin = plugin_lookup(node.plugin)
        if plugin is None:
            continue
        for item in plugin.get_effective_inputs():
            if item.source == "runtime" and item.name not in seen:
                seen.add(item.name)
                fields.append(
                    InputFieldSpec(
                        name=item.name,
                        label=item.description,
                        placeholder=item.placeholder,
                        required=item.required,
                        default=item.default,
                    )
                )

    for node in nodes:
        for parameter_name, binding in node.bindings.items():
            if isinstance(binding, RuntimeSource) and binding.key not in seen:
                seen.add(binding.key)
                fields.append(
                    InputFieldSpec(
                        name=binding.key,
                        label=binding.label or parameter_name.replace("_", " ").title(),
                        placeholder=binding.placeholder,
                        required=binding.required,
                        default=binding.default,
                    )
                )
            expressions: list[str] = []
            if isinstance(binding, ExpressionSource):
                expressions.append(binding.expression)
            if binding.when is not None:
                expressions.append(binding.when)
            for expression in expressions:
                try:
                    _edges, input_keys = extract_refs(expression)
                except ExpressionError:
                    continue
                for key in sorted(input_keys):
                    if key not in seen:
                        seen.add(key)
                        fields.append(InputFieldSpec(name=key, label=key.replace("_", " ").title()))
    return fields


def merge_run_input(
    schema_fields: list[InputFieldSpec],
    default_input: dict[str, object],
    run_input: dict[str, object],
) -> FrozenDict:
    """Schema defaults and supplied run input are validated and merged.

    Precedence runs from schema defaults to pipeline defaults to supplied run
    input. ``None`` and an empty string in supplied run input do not replace a
    lower-precedence value; zero and ``False`` do. The returned JSON object is
    a frozen copy.
    """

    binding_defaults = {
        field.name: field.default
        for field in schema_fields
        if field.default is not None and field.default != ""
    }
    merged = {**binding_defaults, **default_input}
    for key, value in run_input.items():
        if value is not None and value != "":
            merged[key] = value
    missing = [
        field.label
        for field in schema_fields
        if field.required and merged.get(field.name) in (None, "")
    ]
    if missing:
        raise ValueError(f"missing required input: {', '.join(missing)}")

    frozen: dict[str, JsonValue] = {}
    for key, value in merged.items():
        if not isinstance(key, str):
            raise ValueError("run input keys must be strings")
        try:
            frozen[key] = freeze_json(value)
        except ValueError as exc:
            raise ValueError(f"run input {key!r} is not JSON-compatible: {exc}") from exc
    return freeze_object(frozen)


def validate_binding_references(nodes: list[DagNode]) -> str | None:
    """Validation covers named-edge references in bindings, expressions and conditions."""

    for index, node in enumerate(nodes):
        label = node.name or f"Node {index + 1}"
        edge_names = {edge.name for edge in node.edges}
        for parameter_name, binding in node.bindings.items():
            if isinstance(binding, EdgeSource) and binding.edge not in edge_names:
                return (
                    f"{label}: binding {parameter_name!r} references unknown edge {binding.edge!r}"
                )

            expressions: list[tuple[str, str]] = []
            if isinstance(binding, ExpressionSource):
                expressions.append(("expression", binding.expression))
            if binding.when:
                expressions.append(("when-clause", binding.when))
            for kind, expression in expressions:
                error = validate_expression(expression)
                if error:
                    return f"{label}: binding {parameter_name!r} {kind} error: {error}"
                referenced_edges, _runtime_keys = extract_refs(expression)
                unknown = referenced_edges - edge_names
                if unknown:
                    return (
                        f"{label}: binding {parameter_name!r} {kind} references "
                        f"unknown edges {sorted(unknown)!r}"
                    )
    return None
