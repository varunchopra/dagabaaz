"""Task input plan construction.

Planning applies source dispositions, filters and selection to a validated
snapshot. It then groups main-edge outputs by input mode, resolves parameters
from routing fields and records the selected output IDs. Workers materialise
output data after the plan has been stored.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dagabaaz.bindings import (
    FieldValueCache,
    binding_evaluation_steps,
    resolve_plan_bindings,
)
from dagabaaz.constants import (
    MAX_LAUNCH_BINDING_EVALUATED_BYTES,
    MAX_LAUNCH_BINDING_EVALUATIONS,
    MAX_NODE_BINDING_EVALUATIONS,
    MAX_OUTPUT_REFS_PER_PLAN,
    MAX_RUN_INPUT_BYTES,
    MAX_SNAPSHOT_OUTPUTS_PER_EDGE,
    MAX_SNAPSHOT_OUTPUTS_PER_NODE,
    MAX_SNAPSHOT_ROUTING_BYTES,
    MAX_TASK_PLANS_PER_LAUNCH,
    CorrelationMode,
    InputMode,
    InputRole,
    NodeDisposition,
)
from dagabaaz.expressions import EvaluationBudget
from dagabaaz.filter import apply_filter, apply_selection
from dagabaaz.json import JsonValue, bounded_json_size, json_representations_equal
from dagabaaz.models import (
    NodeDefinition,
    OutputRef,
    PlannedEdgeInput,
    PlanningError,
    PlanningSnapshot,
    TaskInputPlan,
    output_ref_routing_size,
)


@dataclass(frozen=True, slots=True)
class PlanConstruction:
    """The plans and node disposition produced by one planning operation."""

    plans: tuple[TaskInputPlan, ...]
    disposition: NodeDisposition
    error: str = ""


def resolve_correlation_mode(node: NodeDefinition, *, is_root: bool) -> CorrelationMode:
    """``DEFAULT`` is resolved from root status and input mode."""

    if node.correlation_mode != CorrelationMode.DEFAULT:
        return node.correlation_mode
    if is_root:
        return CorrelationMode.NEW
    if node.input_mode in (InputMode.EACH, InputMode.BY_CORRELATION):
        return CorrelationMode.INHERIT
    return CorrelationMode.NONE


def correlation_id_for_output(
    node: NodeDefinition,
    *,
    is_root: bool,
    task_correlation_id: str | None,
    output_id: str,
) -> str | None:
    """Output emission derives its correlation ID from the configured mode."""

    if not output_id:
        raise PlanningError("output ID must not be empty")
    mode = resolve_correlation_mode(node, is_root=is_root)
    if mode == CorrelationMode.NEW:
        return output_id
    if mode == CorrelationMode.NONE:
        return None
    if mode == CorrelationMode.INHERIT:
        if not task_correlation_id:
            raise PlanningError(
                f"node {node.slug!r} uses INHERIT but the task has no correlation ID"
            )
        return task_correlation_id
    raise AssertionError("DEFAULT must be resolved before output emission")


def _failed(error: str) -> PlanConstruction:
    return PlanConstruction((), NodeDisposition.FAILED, error)


def _zero(disposition: NodeDisposition) -> PlanConstruction:
    return PlanConstruction((), disposition)


def _validate_snapshot_shape(node: NodeDefinition, snapshot: PlanningSnapshot) -> None:
    edge_names = tuple(edge.name for edge in node.edges)
    expected = set(edge_names)
    output_keys = set(snapshot.outputs_by_edge)
    disposition_keys = set(snapshot.source_dispositions)
    if output_keys != expected or disposition_keys != expected:
        raise PlanningError(
            "planning snapshot edge set does not match node definition: "
            f"expected={sorted(expected)!r}, outputs={sorted(output_keys)!r}, "
            f"dispositions={sorted(disposition_keys)!r}"
        )
    for edge_name in edge_names:
        outputs = snapshot.outputs_by_edge[edge_name]
        if snapshot.source_dispositions[edge_name] != NodeDisposition.LAUNCHED and outputs:
            raise PlanningError(
                f"edge {edge_name!r} has a zero-task source disposition but exposes outputs"
            )


def _validate_snapshot_outputs(node: NodeDefinition, snapshot: PlanningSnapshot) -> None:
    edge_names = tuple(edge.name for edge in node.edges)
    if bounded_json_size(snapshot.runtime_inputs, MAX_RUN_INPUT_BYTES) > MAX_RUN_INPUT_BYTES:
        raise PlanningError(f"runtime input exceeds the {MAX_RUN_INPUT_BYTES}-byte limit")
    output_count = sum(len(outputs) for outputs in snapshot.outputs_by_edge.values())
    if output_count > MAX_SNAPSHOT_OUTPUTS_PER_NODE:
        raise PlanningError(
            f"snapshot contains {output_count} outputs; maximum is {MAX_SNAPSHOT_OUTPUTS_PER_NODE}"
        )
    for edge_name in edge_names:
        outputs = snapshot.outputs_by_edge[edge_name]
        if len(outputs) > MAX_SNAPSHOT_OUTPUTS_PER_EDGE:
            raise PlanningError(
                f"edge {edge_name!r} has {len(outputs)} outputs; maximum is "
                f"{MAX_SNAPSHOT_OUTPUTS_PER_EDGE}"
            )
        ids = [output.id for output in outputs]
        if len(ids) != len(set(ids)):
            raise PlanningError(f"edge {edge_name!r} contains duplicate output IDs")

    routing_size = 0
    for edge_name in edge_names:
        for output in snapshot.outputs_by_edge[edge_name]:
            routing_size += output_ref_routing_size(output)
            if routing_size > MAX_SNAPSHOT_ROUTING_BYTES:
                raise PlanningError(
                    f"snapshot routing data is {routing_size} bytes; maximum is "
                    f"{MAX_SNAPSHOT_ROUTING_BYTES}"
                )

    outputs_by_id: dict[str, OutputRef] = {}
    for edge_name in edge_names:
        for output in snapshot.outputs_by_edge[edge_name]:
            existing = outputs_by_id.setdefault(output.id, output)
            if existing.correlation_id != output.correlation_id or not json_representations_equal(
                existing.fields, output.fields
            ):
                raise PlanningError(
                    f"snapshot contains conflicting values for output {output.id!r}"
                )


def _validate_source_indices(node: NodeDefinition, source_indices: Mapping[str, int]) -> None:
    expected = {edge.name for edge in node.edges}
    actual = set(source_indices)
    if actual != expected:
        raise PlanningError(
            "source index edge set does not match node definition: "
            f"expected={sorted(expected)!r}, actual={sorted(actual)!r}"
        )


def _check_plan_count(count: int) -> None:
    if count > MAX_TASK_PLANS_PER_LAUNCH:
        raise PlanningError(f"node produced {count} plans; maximum is {MAX_TASK_PLANS_PER_LAUNCH}")


def _check_binding_work(plan_count: int, binding_steps: int) -> None:
    evaluations = plan_count * binding_steps
    if evaluations > MAX_LAUNCH_BINDING_EVALUATIONS:
        raise PlanningError(
            f"launch requires {evaluations} binding evaluations; maximum is "
            f"{MAX_LAUNCH_BINDING_EVALUATIONS}"
        )


def _make_plan(
    node: NodeDefinition,
    *,
    generation: int,
    source_indices: Mapping[str, int],
    selected: Mapping[str, Sequence[OutputRef]],
    runtime_inputs: Mapping[str, JsonValue],
    correlation_id: str | None,
    field_cache: FieldValueCache,
    evaluation_budget: EvaluationBudget,
) -> TaskInputPlan:
    if (
        resolve_correlation_mode(node, is_root=not node.edges) == CorrelationMode.INHERIT
        and correlation_id is None
    ):
        raise PlanningError(f"node {node.slug!r} uses INHERIT but the task has no correlation ID")
    reference_count = sum(len(selected[edge.name]) for edge in node.edges)
    if reference_count > MAX_OUTPUT_REFS_PER_PLAN:
        raise PlanningError(
            f"task plan references {reference_count} outputs; maximum is {MAX_OUTPUT_REFS_PER_PLAN}"
        )
    planned_edges = tuple(
        PlannedEdgeInput(
            edge=edge.name,
            source_index=source_indices[edge.name],
            role=edge.role,
            output_ids=tuple(output.id for output in selected[edge.name]),
        )
        for edge in node.edges
    )
    parameters, secret_refs = resolve_plan_bindings(
        node,
        selected,
        runtime_inputs,
        base_parameters=runtime_inputs if not node.edges else None,
        field_cache=field_cache,
        evaluation_budget=evaluation_budget,
    )
    return TaskInputPlan(
        generation=generation,
        edges=planned_edges,
        parameters=parameters,
        secret_refs=secret_refs,
        correlation_id=correlation_id,
    )


def _source_disposition(
    node: NodeDefinition,
    snapshot: PlanningSnapshot,
) -> PlanConstruction | None:
    failed = next(
        (
            edge
            for edge in node.edges
            if snapshot.source_dispositions[edge.name] == NodeDisposition.FAILED
        ),
        None,
    )
    if failed is not None:
        return _failed(f"edge {failed.name!r} source launch failed before planning")
    if any(
        edge.required and snapshot.source_dispositions[edge.name] == NodeDisposition.SKIPPED
        for edge in node.edges
    ):
        return _zero(NodeDisposition.SKIPPED)
    if any(
        edge.required and snapshot.source_dispositions[edge.name] == NodeDisposition.FILTERED
        for edge in node.edges
    ):
        return _zero(NodeDisposition.FILTERED)
    return None


def _prepare_edges(
    node: NodeDefinition, snapshot: PlanningSnapshot
) -> dict[str, tuple[OutputRef, ...]]:
    prepared: dict[str, tuple[OutputRef, ...]] = {}
    for edge in node.edges:
        disposition = snapshot.source_dispositions[edge.name]
        if disposition == NodeDisposition.SKIPPED:
            prepared[edge.name] = ()
            continue
        outputs = snapshot.outputs_by_edge[edge.name]
        prepared[edge.name] = apply_filter(outputs, edge.filter)
    return prepared


def construct_task_plans(
    node: NodeDefinition,
    *,
    source_indices: Mapping[str, int],
    snapshot: PlanningSnapshot,
) -> PlanConstruction:
    """Plan construction yields task plans or a disposition without tasks.

    Side-edge selection occurs once, and every plan receives its result.
    ``EACH`` follows edge-selection order. ``ALL`` creates one plan.
    ``BY_CORRELATION`` selects within each group, then creates plans in
    correlation-ID order. Required main edges determine which IDs remain.
    Invalid routing data produces a ``FAILED`` disposition.
    """

    try:
        _validate_snapshot_shape(node, snapshot)
        _validate_source_indices(node, source_indices)
        early = _source_disposition(node, snapshot)
        if early is not None:
            return early
        _validate_snapshot_outputs(node, snapshot)
        input_mode = node.input_mode
        if input_mode is None:
            if node.edges:
                raise PlanningError(f"non-root node {node.slug!r} must declare an input mode")
            input_mode = InputMode.ALL
        binding_steps = binding_evaluation_steps(node)
        if binding_steps > MAX_NODE_BINDING_EVALUATIONS:
            raise PlanningError(
                f"node bindings require {binding_steps} evaluations; maximum is "
                f"{MAX_NODE_BINDING_EVALUATIONS}"
            )
        prepared = _prepare_edges(node, snapshot)

        side_edges = [edge for edge in node.edges if edge.role == InputRole.SIDE]
        main_edges = [edge for edge in node.edges if edge.role == InputRole.MAIN]
        selected_sides: dict[str, tuple[OutputRef, ...]] = {}
        for edge in side_edges:
            selected_sides[edge.name] = apply_selection(prepared[edge.name], edge.selection)
            if edge.required and not selected_sides[edge.name]:
                return _zero(NodeDisposition.FILTERED)

        plans: list[TaskInputPlan] = []
        field_cache: FieldValueCache = {}
        evaluation_budget = EvaluationBudget(MAX_LAUNCH_BINDING_EVALUATED_BYTES)
        selected: dict[str, tuple[OutputRef, ...]]
        if input_mode == InputMode.EACH:
            main = main_edges[0]
            selected_main = apply_selection(prepared[main.name], main.selection)
            if not selected_main:
                return _zero(NodeDisposition.FILTERED)
            _check_plan_count(len(selected_main))
            _check_binding_work(len(selected_main), binding_steps)
            for output in selected_main:
                selected = {edge.name: () for edge in node.edges}
                selected.update(selected_sides)
                selected[main.name] = (output,)
                plans.append(
                    _make_plan(
                        node,
                        generation=snapshot.generation,
                        source_indices=source_indices,
                        selected=selected,
                        runtime_inputs=snapshot.runtime_inputs,
                        correlation_id=output.correlation_id,
                        field_cache=field_cache,
                        evaluation_budget=evaluation_budget,
                    )
                )

        elif input_mode == InputMode.ALL:
            _check_binding_work(1, binding_steps)
            selected = {edge.name: () for edge in node.edges}
            selected.update(selected_sides)
            for edge in main_edges:
                outputs = apply_selection(prepared[edge.name], edge.selection)
                if edge.required and not outputs:
                    return _zero(NodeDisposition.FILTERED)
                selected[edge.name] = outputs
            plans.append(
                _make_plan(
                    node,
                    generation=snapshot.generation,
                    source_indices=source_indices,
                    selected=selected,
                    runtime_inputs=snapshot.runtime_inputs,
                    correlation_id=None,
                    field_cache=field_cache,
                    evaluation_budget=evaluation_budget,
                )
            )

        elif input_mode == InputMode.BY_CORRELATION:
            grouped: dict[str, dict[str, tuple[OutputRef, ...]]] = {}
            candidate_ids: set[str] = set()
            for edge in main_edges:
                raw_groups: dict[str, list[OutputRef]] = defaultdict(list)
                for output in prepared[edge.name]:
                    if output.correlation_id is None:
                        raise PlanningError(
                            f"edge {edge.name!r} contains uncorrelated output {output.id!r}"
                        )
                    raw_groups[output.correlation_id].append(output)
                edge_groups: dict[str, tuple[OutputRef, ...]] = {}
                for correlation_id, outputs in raw_groups.items():
                    chosen = apply_selection(outputs, edge.selection)
                    if chosen:
                        edge_groups[correlation_id] = chosen
                        candidate_ids.add(correlation_id)
                grouped[edge.name] = edge_groups

            retained = [
                correlation_id
                for correlation_id in sorted(candidate_ids)
                if all(
                    not edge.required or correlation_id in grouped.get(edge.name, {})
                    for edge in main_edges
                )
            ]
            if not retained:
                return _zero(NodeDisposition.FILTERED)
            _check_plan_count(len(retained))
            _check_binding_work(len(retained), binding_steps)
            for correlation_id in retained:
                selected = {edge.name: () for edge in node.edges}
                selected.update(selected_sides)
                for edge in main_edges:
                    selected[edge.name] = grouped[edge.name].get(correlation_id, ())
                plans.append(
                    _make_plan(
                        node,
                        generation=snapshot.generation,
                        source_indices=source_indices,
                        selected=selected,
                        runtime_inputs=snapshot.runtime_inputs,
                        correlation_id=correlation_id,
                        field_cache=field_cache,
                        evaluation_budget=evaluation_budget,
                    )
                )
        else:
            raise PlanningError(f"unsupported input mode {input_mode!r}")

        _check_plan_count(len(plans))
        return PlanConstruction(tuple(plans), NodeDisposition.LAUNCHED)
    except (PlanningError, ValueError) as exc:
        return _failed(str(exc))
