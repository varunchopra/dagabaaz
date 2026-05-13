"""Pipeline input schema generation and binding validation.

Two sources contribute to the runtime input schema:
  1. Static plugin declarations (plugin inputs with source="runtime")
  2. Dynamic bindings on nodes (RuntimeSource and expression input.* refs)
"""

from dataclasses import dataclass
from typing import Any

from dagabaaz.expressions import extract_refs, validate_expression
from dagabaaz.models import (
    DagNode,
    ExpressionError,
    ExpressionSource,
    NodeSource,
    RuntimeSource,
)
from dagabaaz.plugins import PluginLookup


@dataclass(frozen=True, slots=True)
class InputFieldSpec:
    """A single field in a pipeline's runtime input schema."""

    name: str
    label: str
    placeholder: str = ""
    required: bool = False
    default: str = ""


def get_pipeline_input_schema(
    nodes: list[DagNode],
    plugin_lookup: PluginLookup,
) -> list[InputFieldSpec]:
    """Collect runtime inputs from all nodes in a pipeline.

    Two sources of runtime inputs:
    1. Static plugin declarations (PluginInput with source="runtime")
    2. Dynamic bindings on nodes (RuntimeSource bindings)

    Static declarations take precedence when names collide.
    Binding-derived inputs are always optional since upstream steps
    may provide the value instead.

    ``nodes`` must have ``.plugin`` and ``.bindings`` attributes.
    ``plugin_lookup`` resolves plugin names to metadata (fan_in, inputs).
    """
    fields: list[InputFieldSpec] = []
    seen_names: set[str] = set()

    for node in nodes:
        plugin = plugin_lookup(node.plugin)
        if not plugin:
            continue
        for inp in plugin.get_effective_inputs():
            if inp.source == "runtime" and inp.name not in seen_names:
                seen_names.add(inp.name)
                fields.append(
                    InputFieldSpec(
                        name=inp.name,
                        label=inp.description,
                        placeholder=inp.placeholder,
                        required=inp.required,
                    )
                )

    for node in nodes:
        for binding_key, binding in node.bindings.items():
            if not isinstance(binding, RuntimeSource):
                continue
            input_key = binding.key or binding_key
            if input_key in seen_names:
                continue
            seen_names.add(input_key)
            label = binding.label or binding_key.replace("_", " ").title()
            fields.append(
                InputFieldSpec(
                    name=input_key,
                    label=label,
                    placeholder=binding.placeholder,
                    required=binding.required,
                    default=binding.default,
                )
            )

    for node in nodes:
        for binding in node.bindings.values():
            if not isinstance(binding, ExpressionSource):
                continue
            try:
                _, runtime_keys = extract_refs(binding.expression)
            except ExpressionError:
                continue
            for runtime_key in runtime_keys:
                if runtime_key in seen_names:
                    continue
                seen_names.add(runtime_key)
                fields.append(
                    InputFieldSpec(
                        name=runtime_key,
                        label=runtime_key.replace("_", " ").title(),
                    )
                )

    return fields


def merge_run_input(
    schema_fields: list[InputFieldSpec],
    default_input: dict[str, Any],
    run_input: dict[str, Any],
) -> dict[str, Any]:
    """Merge binding defaults, pipeline defaults, and user input.

    Precedence (lowest to highest):
      1. Binding defaults (from ``InputFieldSpec.default``)
      2. Pipeline default_input (set by pipeline author)
      3. User run_input (provided at run start)

    User input values of ``None`` or ``""`` are treated as absent and
    do not override lower-precedence values.

    Raises ``ValueError`` if any required field is missing after merge.
    """
    binding_defaults = {
        f.name: f.default for f in schema_fields if f.default not in (None, "")
    }
    merged: dict[str, Any] = {**binding_defaults, **default_input}
    for k, v in run_input.items():
        if v not in (None, ""):
            merged[k] = v

    missing = [
        f.label
        for f in schema_fields
        if f.required and merged.get(f.name) in (None, "")
    ]
    if missing:
        raise ValueError(f"Required input missing: {', '.join(missing)}")

    return merged


def validate_binding_references(
    nodes: list[DagNode],
    slug_to_index: dict[str, int],
) -> str | None:
    """Validate structural binding references (slugs, expressions).

    Returns an error message or None.
    """
    for i, node in enumerate(nodes):
        label = node.name or f"Node {i + 1}"
        dep_set = set(node.depends_on)
        bindings = node.bindings

        for field_name, binding in bindings.items():
            if isinstance(binding, NodeSource):
                # NodeSource must reference a slug that is in this node's
                # depends_on — otherwise it silently resolves to None at
                # runtime, which is almost never intentional.
                if binding.node not in dep_set:
                    return (
                        f"{label}: binding '{field_name}' references node "
                        f"'{binding.node}' which is not in depends_on"
                    )
            elif isinstance(binding, ExpressionSource):
                expr_err = validate_expression(binding.expression)
                if expr_err:
                    return (
                        f"{label}: binding '{field_name}' expression error: {expr_err}"
                    )
                # Safe: validate_expression above already caught tokenization errors
                expr_slugs, _ = extract_refs(binding.expression)
                for slug in expr_slugs:
                    if slug not in dep_set:
                        return f"{label}: expression references '{slug}' which is not in depends_on"

            if binding.when:
                when_err = validate_expression(binding.when)
                if when_err:
                    return (
                        f"{label}: binding '{field_name}' when-clause error: {when_err}"
                    )
                when_slugs, _ = extract_refs(binding.when)
                for slug in when_slugs:
                    if slug not in dep_set:
                        return (
                            f"{label}: when-clause on binding '{field_name}' "
                            f"references '{slug}' which is not in depends_on"
                        )

    return None
