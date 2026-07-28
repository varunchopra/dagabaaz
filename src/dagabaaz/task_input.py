"""Worker input resolution from a stored task plan.

The worker adapter validates the task attempt and generation before resolution.
Workers do not repeat routing, filtering, selection or binding evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping

from dagabaaz.models import (
    MaterializationError,
    ResolvedOutput,
    TaskInputPlan,
    TaskInputs,
)
from dagabaaz.store import OutputResolver


def resolve_task_inputs(
    store: OutputResolver,
    *,
    run_id: str,
    plan: TaskInputPlan,
) -> TaskInputs:
    """Planned output IDs are resolved in the recorded edge order.

    IDs shared by several edges are requested once. The resolver must return
    every requested ID from the run and no others, with each mapping key equal
    to its ``ResolvedOutput.id``. Parameters pass through unchanged; the worker
    adapter resolves secret references separately.
    """

    requested_order = tuple(output_id for edge in plan.edges for output_id in edge.output_ids)
    unique_ids = tuple(dict.fromkeys(requested_order))
    try:
        resolved = store.resolve_outputs(run_id, unique_ids)
    except Exception as exc:
        raise MaterializationError(
            f"materializer failed while resolving outputs for run {run_id!r}"
        ) from exc
    if not isinstance(resolved, Mapping):
        raise MaterializationError("materializer did not return an output mapping")

    for key, output in resolved.items():
        if not isinstance(key, str) or not isinstance(output, ResolvedOutput):
            raise MaterializationError("materializer returned a value outside the output protocol")
        if output.id != key:
            raise MaterializationError(f"materializer key {key!r} contains output {output.id!r}")

    expected = set(unique_ids)
    actual = set(resolved)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise MaterializationError(
            f"materializer returned a different output ID set; missing={missing!r}, extra={extra!r}"
        )

    edges = {
        edge.edge: tuple(resolved[output_id] for output_id in edge.output_ids)
        for edge in plan.edges
    }
    return TaskInputs(edges=edges, parameters=plan.parameters)
