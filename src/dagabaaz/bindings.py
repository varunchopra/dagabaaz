"""Task-plan parameter resolution from routing fields and run inputs.

During planning, expressions can read named input edges and the reserved
``input`` namespace. The resulting values are stored in the task plan.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from dagabaaz.expressions import (
    EvaluationBudget,
    Lookup,
    expression_evaluation_steps,
    resolve_expression,
)
from dagabaaz.json import FrozenDict, JsonValue, freeze_json, freeze_object
from dagabaaz.models import (
    EdgeSource,
    ExpressionSource,
    LiteralSource,
    NodeDefinition,
    OutputRef,
    PlanningError,
    RuntimeSource,
    SecretSource,
)

_MISSING = object()

type FieldValueCache = dict[
    tuple[str, str, int],
    tuple[Sequence[OutputRef], JsonValue],
]


def extract_edge_field(outputs: Sequence[OutputRef], field: str) -> JsonValue:
    """Field extraction yields ``None``, one value or an ordered tuple."""

    values = [output.fields[field] for output in outputs if field in output.fields]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return tuple(values)


def _extract_cached_edge_field(
    edge: str,
    outputs: Sequence[OutputRef],
    field: str,
    cache: FieldValueCache,
) -> JsonValue:
    key = (edge, field, id(outputs))
    cached = cache.get(key)
    if cached is not None and cached[0] is outputs:
        return cached[1]
    value = extract_edge_field(outputs, field)
    cache[key] = (outputs, value)
    return value


def build_expression_lookup(
    outputs_by_edge: Mapping[str, Sequence[OutputRef]],
    runtime_inputs: Mapping[str, JsonValue],
    *,
    field_cache: FieldValueCache | None = None,
) -> Lookup:
    """The lookup reads named-edge routing fields and frozen run input."""

    cache = field_cache if field_cache is not None else {}

    def lookup(namespace: str, key: str) -> object | None:
        if namespace == "input":
            return runtime_inputs.get(key)
        outputs = outputs_by_edge.get(namespace)
        if outputs is None:
            return None
        return _extract_cached_edge_field(namespace, outputs, key, cache)

    return lookup


def binding_evaluation_steps(node: NodeDefinition) -> int:
    """Count binding visits, reference lookups and pipe calls for one plan."""

    steps = len(node.bindings)
    for binding in node.bindings.values():
        if isinstance(binding, ExpressionSource):
            steps += expression_evaluation_steps(binding.expression)
        if binding.when is not None:
            steps += expression_evaluation_steps(binding.when)
    return steps


def resolve_plan_bindings(
    node: NodeDefinition,
    outputs_by_edge: Mapping[str, Sequence[OutputRef]],
    runtime_inputs: Mapping[str, JsonValue],
    *,
    base_parameters: Mapping[str, JsonValue] | None = None,
    field_cache: FieldValueCache | None = None,
    evaluation_budget: EvaluationBudget | None = None,
) -> tuple[FrozenDict, Mapping[str, str]]:
    """Plan bindings yield parameters and secret references.

    ``base_parameters`` provides the initial parameters for a root task. An
    active binding replaces a parameter with the same name. An inactive
    conditional binding leaves it unchanged.
    """

    lookup = build_expression_lookup(
        outputs_by_edge,
        runtime_inputs,
        field_cache=field_cache,
    )
    parameters = dict(base_parameters or {})
    secret_refs: dict[str, str] = {}

    for name, binding in node.bindings.items():
        if binding.when is not None and not bool(
            resolve_expression(binding.when, lookup, budget=evaluation_budget)
        ):
            continue

        dynamic = True
        if isinstance(binding, EdgeSource):
            value = lookup(binding.edge, binding.field)
        elif isinstance(binding, LiteralSource):
            dynamic = False
            value = binding.value
        elif isinstance(binding, RuntimeSource):
            value = runtime_inputs.get(binding.key, _MISSING)
            if value is _MISSING:
                value = binding.default
            if binding.required and (value is None or value == ""):
                raise PlanningError(f"required runtime input {binding.key!r} is missing")
        elif isinstance(binding, ExpressionSource):
            value = resolve_expression(
                binding.expression,
                lookup,
                budget=evaluation_budget,
            )
        elif isinstance(binding, SecretSource):
            parameters.pop(name, None)
            secret_refs[name] = binding.name
            continue
        else:
            raise PlanningError(f"unsupported binding for parameter {name!r}")

        try:
            frozen_value = freeze_json(value)
        except ValueError as exc:
            raise PlanningError(
                f"parameter {name!r} did not resolve to JSON-compatible data: {exc}"
            ) from exc
        if dynamic and evaluation_budget is not None:
            evaluation_budget.charge(frozen_value)
        if dynamic and frozen_value is None:
            parameters.pop(name, None)
            continue
        parameters[name] = frozen_value

    return freeze_object(parameters), MappingProxyType(secret_refs)
