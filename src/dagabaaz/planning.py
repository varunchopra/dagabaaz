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

from dagabaaz.bindings import resolve_plan_bindings
from dagabaaz.constants import (
    MAX_SNAPSHOT_OUTPUTS_PER_EDGE,
    MAX_SNAPSHOT_OUTPUTS_PER_NODE,
    MAX_TASK_PLANS_PER_LAUNCH,
    CorrelationMode,
    InputMode,
    InputRole,
    NodeDisposition,
)
from dagabaaz.filter import apply_filter, apply_selection
from dagabaaz.json import JsonValue, json_values_equal
from dagabaaz.models import (
    NodeDefinition,
    OutputRef,
    PlannedEdgeInput,
    PlanningError,
    PlanningSnapshot,
    TaskInputPlan,
)


@dataclass(frozen=True, slots=True)
class PlanConstruction:
    """The plans and node disposition produced by one planning operation."""

    plans: tuple[TaskInputPlan, ...]
    disposition: NodeDisposition
    error: str = ""


def resolve_correlation_mode(
    node: NodeDefinition, *, is_root: bool
) -> CorrelationMode:
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


def _validate_snapshot(node: NodeDefinition, snapshot: PlanningSnapshot) -> None:
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
    output_count = sum(len(outputs) for outputs in snapshot.outputs_by_edge.values())
    if output_count > MAX_SNAPSHOT_OUTPUTS_PER_NODE:
        raise PlanningError(
            f"snapshot contains {output_count} outputs; maximum is {MAX_SNAPSHOT_OUTPUTS_PER_NODE}"
        )

    outputs_by_id: dict[str, OutputRef] = {}
    for edge_name in edge_names:
        outputs = snapshot.outputs_by_edge[edge_name]
        if (
            snapshot.source_dispositions[edge_name] != NodeDisposition.LAUNCHED
            and outputs
        ):
            raise PlanningError(
                f"edge {edge_name!r} has a zero-task source disposition but exposes outputs"
            )
        if len(outputs) > MAX_SNAPSHOT_OUTPUTS_PER_EDGE:
            raise PlanningError(
                f"edge {edge_name!r} has {len(outputs)} outputs; maximum is "
                f"{MAX_SNAPSHOT_OUTPUTS_PER_EDGE}"
            )
        ids = [output.id for output in outputs]
        if len(ids) != len(set(ids)):
            raise PlanningError(f"edge {edge_name!r} contains duplicate output IDs")
        for output in outputs:
            existing = outputs_by_id.setdefault(output.id, output)
            if existing.correlation_id != output.correlation_id or not json_values_equal(
                existing.fields, output.fields
            ):
                raise PlanningError(
                    f"snapshot contains conflicting values for output {output.id!r}"
                )


def _validate_source_indices(
    node: NodeDefinition, source_indices: Mapping[str, int]
) -> None:
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


def _make_plan(
    node: NodeDefinition,
    *,
    generation: int,
    source_indices: Mapping[str, int],
    selected: Mapping[str, Sequence[OutputRef]],
    runtime_inputs: Mapping[str, JsonValue],
    correlation_id: str | None,
) -> TaskInputPlan:
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
    )
    return TaskInputPlan(
        generation=generation,
        edges=planned_edges,
        parameters=parameters,
        secret_refs=secret_refs,
        correlation_id=correlation_id,
    )


def _prepare_edges(
    node: NodeDefinition, snapshot: PlanningSnapshot
) -> tuple[dict[str, tuple[OutputRef, ...]], PlanConstruction | None]:
    prepared: dict[str, tuple[OutputRef, ...]] = {}
    for edge in node.edges:
        disposition = snapshot.source_dispositions[edge.name]
        if disposition == NodeDisposition.FAILED:
            return {}, _failed(f"edge {edge.name!r} source launch failed before planning")
        if disposition == NodeDisposition.SKIPPED:
            if edge.required:
                return {}, _zero(NodeDisposition.SKIPPED)
            prepared[edge.name] = ()
            continue
        outputs = snapshot.outputs_by_edge[edge.name]
        prepared[edge.name] = apply_filter(outputs, edge.filter)
    return prepared, None


def construct_task_plans(
    node: NodeDefinition,
    *,
    source_indices: Mapping[str, int],
    snapshot: PlanningSnapshot,
) -> PlanConstruction:
    """Plan construction yields task plans or a disposition without tasks.

    Side-edge selection occurs once and its result is included in every plan.
    ``EACH`` follows the order returned by edge selection, ``ALL`` creates one plan, and
    ``BY_CORRELATION`` selects within each group before creating plans in
    correlation-ID order. Required main edges determine which IDs remain.
    Invalid routing data produces a ``FAILED`` disposition.
    """

    try:
        _validate_snapshot(node, snapshot)
        _validate_source_indices(node, source_indices)
        prepared, early = _prepare_edges(node, snapshot)
        if early is not None:
            return early

        side_edges = [edge for edge in node.edges if edge.role == InputRole.SIDE]
        main_edges = [edge for edge in node.edges if edge.role == InputRole.MAIN]
        selected_sides: dict[str, tuple[OutputRef, ...]] = {}
        for edge in side_edges:
            selected_sides[edge.name] = apply_selection(prepared[edge.name], edge.selection)
            if edge.required and not selected_sides[edge.name]:
                return _zero(NodeDisposition.FILTERED)

        plans: list[TaskInputPlan] = []
        selected: dict[str, tuple[OutputRef, ...]]
        if node.input_mode == InputMode.EACH:
            main = main_edges[0]
            selected_main = apply_selection(prepared[main.name], main.selection)
            if not selected_main:
                return _zero(NodeDisposition.FILTERED)
            _check_plan_count(len(selected_main))
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
                    )
                )

        elif node.input_mode == InputMode.ALL:
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
                )
            )

        elif node.input_mode == InputMode.BY_CORRELATION:
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
                    )
                )
        else:
            raise PlanningError(f"unsupported input mode {node.input_mode!r}")

        _check_plan_count(len(plans))
        return PlanConstruction(tuple(plans), NodeDisposition.LAUNCHED)
    except (PlanningError, ValueError) as exc:
        return _failed(str(exc))
