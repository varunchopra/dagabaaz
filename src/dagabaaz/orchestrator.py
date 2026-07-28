"""Persisted-state reconciliation and run lifecycle transitions.

The orchestrator plans nodes whose source launches are complete. Stores create
launches, tasks, plans and queue outbox rows together, so repeated reconciliation
does not duplicate work.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from dagabaaz.constants import (
    MAX_SNAPSHOT_OUTPUTS_PER_EDGE,
    MAX_SNAPSHOT_OUTPUTS_PER_NODE,
    MAX_SNAPSHOT_RETRIES,
    MAX_SNAPSHOT_ROUTING_BYTES,
    LaunchCreateStatus,
    NodeDisposition,
    RunStatus,
    TaskStatus,
)
from dagabaaz.models import EmittedOutput, LaunchCreateResult, NodeLaunch, TaskInputPlan
from dagabaaz.planning import construct_task_plans
from dagabaaz.store import (
    DagStore,
    NodeLaunchRef,
    OutputPublicationError,
    StoreContractError,
    TaskAttemptRef,
    TaskCompletionResult,
    TaskFailureResult,
)
from dagabaaz.topology import RunTopology

logger = logging.getLogger(__name__)

RunCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class OrchestratorCallbacks:
    """Callbacks invoked after a successful terminal transition.

    The application serialises terminal handlers and run retries for a run
    until the callback returns.
    """

    on_run_completed: RunCallback
    on_run_failed: RunCallback
    on_run_crashed: RunCallback
    on_run_cancelled: RunCallback


class RunStartRejectedError(RuntimeError):
    """The stored run cannot enter or remain in the running state."""


def _terminal_failure(
    store: DagStore,
    *,
    run_id: str,
    status: RunStatus,
    callbacks: OrchestratorCallbacks,
    cause: TaskAttemptRef | NodeLaunchRef | None = None,
    reason: str | None = None,
) -> bool:
    """Finalisation, including cancellation, completes before the callback."""

    if not store.try_finalize_run(
        run_id,
        status,
        cause=cause,
        reason=reason,
    ):
        return False
    if status == RunStatus.CRASHED:
        callbacks.on_run_crashed(run_id)
    elif status == RunStatus.CANCELLED:
        callbacks.on_run_cancelled(run_id)
    else:
        callbacks.on_run_failed(run_id)
    return True


def start_run(
    store: DagStore,
    run_id: str,
    *,
    callbacks: OrchestratorCallbacks,
) -> list[int]:
    """Stored topology is validated and reconciled before root indices are returned."""

    nodes = store.get_run_nodes(run_id)
    if nodes is None:
        raise ValueError(f"run {run_id!r} has no stored node definition")
    topology = RunTopology.build(nodes)
    roots = [index for index, dependencies in enumerate(topology.dependencies) if not dependencies]
    if not store.try_start_run(run_id):
        raise RunStartRejectedError(f"run {run_id!r} cannot be started")
    _reconcile_topology(store, run_id, topology=topology, callbacks=callbacks)
    return roots


def _validate_launch_result(
    *,
    status: LaunchCreateStatus,
    run_id: str,
    node_index: int,
    plugin_name: str,
    generation: int,
    plans: tuple[TaskInputPlan, ...],
    zero_disposition: NodeDisposition | None,
    launch: object,
) -> NodeLaunch:
    """A store-created launch must match the accepted operation."""

    if not isinstance(launch, NodeLaunch):
        raise StoreContractError("launch creation did not return a NodeLaunch")
    expected_disposition = NodeDisposition.LAUNCHED if plans else zero_disposition
    mismatches: list[str] = []
    if launch.run_id != run_id:
        mismatches.append("run_id")
    if launch.node_index != node_index:
        mismatches.append("node_index")
    if launch.plugin_name != plugin_name:
        mismatches.append("plugin_name")
    if launch.generation != generation:
        mismatches.append("generation")
    if launch.disposition != expected_disposition:
        mismatches.append("disposition")
    if len(launch.task_ids) != len(plans):
        mismatches.append("task_ids")
    elif any(not task_id for task_id in launch.task_ids):
        mismatches.append("empty task_id")
    elif len(set(launch.task_ids)) != len(launch.task_ids):
        mismatches.append("duplicate task_ids")
    if launch.disposition == NodeDisposition.LAUNCHED:
        if not launch.task_ids:
            mismatches.append("launched without tasks")
        if status == LaunchCreateStatus.CREATED and launch.complete:
            mismatches.append("new task launch is complete")
    else:
        if launch.task_ids:
            mismatches.append("zero-task disposition with tasks")
        if not launch.complete:
            mismatches.append("incomplete zero-task disposition")
    if launch.disposition == NodeDisposition.FAILED and not launch.error:
        mismatches.append("failed without error")
    if mismatches:
        raise StoreContractError(
            f"launch creation returned mismatched fields: {', '.join(mismatches)}"
        )
    return launch


def reconcile_run(
    store: DagStore,
    run_id: str,
    *,
    callbacks: OrchestratorCallbacks,
) -> None:
    """Launches are created for nodes whose source launches have finished.

    The function may follow any committed state change and may be called
    repeatedly. Terminal runs and nodes with active launches require no further
    work. Stale snapshots are retried up to ``MAX_SNAPSHOT_RETRIES``. A
    completed launch without tasks may make another graph level ready during
    the same call; a task-bearing launch becomes complete after a worker event.
    """

    status = store.get_run_status(run_id)
    if status != RunStatus.RUNNING:
        return
    nodes = store.get_run_nodes(run_id)
    if nodes is None:
        raise StoreContractError(f"running run {run_id!r} has no stored node definition")
    topology = RunTopology.build(nodes)
    _reconcile_topology(store, run_id, topology=topology, callbacks=callbacks)


def _reconcile_topology(
    store: DagStore,
    run_id: str,
    *,
    topology: RunTopology,
    callbacks: OrchestratorCallbacks,
) -> None:
    """A validated running topology is reconciled from one launch-state read."""

    launches = dict(store.list_node_launches(run_id))
    failed_launch = min(
        (launch for launch in launches.values() if launch.disposition == NodeDisposition.FAILED),
        key=lambda launch: launch.node_index,
        default=None,
    )
    if failed_launch is not None:
        _terminal_failure(
            store,
            run_id=run_id,
            status=RunStatus.FAILED,
            callbacks=callbacks,
            cause=NodeLaunchRef(
                launch_id=failed_launch.id,
                node_index=failed_launch.node_index,
                generation=failed_launch.generation,
            ),
        )
        return

    completed = {index for index, launch in launches.items() if launch.complete}
    launched = set(launches)
    remaining_parents = [
        sum(parent_index not in completed for parent_index in dependencies)
        for dependencies in topology.dependencies
    ]
    ready = deque(
        index
        for index, parent_count in enumerate(remaining_parents)
        if index not in launched and parent_count == 0
    )
    queued = set(ready)

    while ready:
        node_index = ready.popleft()
        queued.discard(node_index)
        if node_index in launched:
            continue
        node = topology.nodes[node_index]
        source_indices = {edge.name: topology.slug_to_index[edge.source] for edge in node.edges}
        launch = None
        for _attempt in range(MAX_SNAPSHOT_RETRIES):
            snapshot = store.get_planning_snapshot(
                run_id,
                node_index,
                source_indices,
                max_outputs_per_edge=MAX_SNAPSHOT_OUTPUTS_PER_EDGE,
                max_outputs_total=MAX_SNAPSHOT_OUTPUTS_PER_NODE,
                max_routing_bytes=MAX_SNAPSHOT_ROUTING_BYTES,
            )
            construction = construct_task_plans(
                node,
                source_indices=source_indices,
                snapshot=snapshot,
            )
            zero_disposition = None if construction.plans else construction.disposition
            result = store.try_create_node_launch(
                run_id,
                node_index,
                node.plugin,
                snapshot.token,
                construction.plans,
                zero_disposition,
                construction.error,
            )
            if not isinstance(result, LaunchCreateResult):
                raise StoreContractError("launch creation did not return a LaunchCreateResult")
            if result.status == LaunchCreateStatus.STALE:
                continue
            if (
                result.status == LaunchCreateStatus.ALREADY_EXISTS
                and result.launch is not None
                and result.launch.generation != snapshot.generation
            ):
                continue
            launch = _validate_launch_result(
                status=result.status,
                run_id=run_id,
                node_index=node_index,
                plugin_name=node.plugin,
                generation=snapshot.generation,
                plans=construction.plans,
                zero_disposition=zero_disposition,
                launch=result.launch,
            )
            break

        if launch is None:
            logger.info(
                "planning for run %s node %d was deferred after %d stale snapshots",
                run_id,
                node_index,
                MAX_SNAPSHOT_RETRIES,
            )
            continue

        launches[node_index] = launch
        launched.add(node_index)
        if launch.disposition == NodeDisposition.FAILED:
            _terminal_failure(
                store,
                run_id=run_id,
                status=RunStatus.FAILED,
                callbacks=callbacks,
                cause=NodeLaunchRef(
                    launch_id=launch.id,
                    node_index=launch.node_index,
                    generation=launch.generation,
                ),
            )
            return
        if not launch.complete:
            continue

        completed.add(node_index)
        for child_index in topology.children[node_index]:
            remaining_parents[child_index] -= 1
            if remaining_parents[child_index] < 0:
                raise StoreContractError("a source launch was completed more than once")
            if (
                remaining_parents[child_index] == 0
                and child_index not in launched
                and child_index not in queued
            ):
                ready.append(child_index)
                queued.add(child_index)

    completed_count = sum(launch.complete for launch in launches.values())
    if (
        len(launches) == len(topology.nodes)
        and completed_count == len(topology.nodes)
        and store.try_finalize_run(run_id, RunStatus.COMPLETED)
    ):
        callbacks.on_run_completed(run_id)


def on_task_complete(
    store: DagStore,
    *,
    task_id: str,
    expected_attempt_id: str,
    outputs: tuple[EmittedOutput, ...] = (),
    callbacks: OrchestratorCallbacks,
) -> None:
    """Task completion and output publication precede any required reconciliation."""

    try:
        result = store.try_complete_task(
            task_id,
            outputs,
            expected_attempt_id=expected_attempt_id,
        )
    except OutputPublicationError as exc:
        on_task_failed(
            store,
            task_id=task_id,
            expected_attempt_id=expected_attempt_id,
            error_message=str(exc),
            callbacks=callbacks,
        )
        return
    if result is None:
        return
    if not isinstance(result, TaskCompletionResult):
        raise StoreContractError("task completion did not return a TaskCompletionResult")
    if result.task_id != task_id or result.attempt_id != expected_attempt_id:
        raise StoreContractError("task completion returned mismatched attempt")
    if not result.launch_complete:
        return
    reconcile_run(store, result.run_id, callbacks=callbacks)


def on_task_failed(
    store: DagStore,
    *,
    task_id: str,
    expected_attempt_id: str,
    error_message: str,
    callbacks: OrchestratorCallbacks,
) -> None:
    """A current matching attempt is marked ``FAILED`` before a run transition is attempted."""

    result = store.mark_task_failed(
        task_id,
        error_message,
        expected_attempt_id=expected_attempt_id,
    )
    if result is None:
        return
    if not isinstance(result, TaskFailureResult):
        raise StoreContractError("task failure did not return a TaskFailureResult")
    if (
        result.task_id != task_id
        or result.attempt_id != expected_attempt_id
        or result.status != TaskStatus.FAILED
    ):
        raise StoreContractError("task failure returned mismatched state")
    _terminal_failure(
        store,
        run_id=result.run_id,
        status=RunStatus.FAILED,
        callbacks=callbacks,
        cause=TaskAttemptRef(
            task_id=task_id,
            attempt_id=expected_attempt_id,
        ),
    )


def on_task_crashed(
    store: DagStore,
    *,
    task_id: str,
    expected_attempt_id: str,
    error_message: str,
    callbacks: OrchestratorCallbacks,
) -> None:
    """A current matching attempt is marked ``CRASHED`` before a run transition is attempted."""

    result = store.mark_task_crashed(
        task_id,
        error_message,
        expected_attempt_id=expected_attempt_id,
    )
    if result is None:
        return
    if not isinstance(result, TaskFailureResult):
        raise StoreContractError("task crash did not return a TaskFailureResult")
    if (
        result.task_id != task_id
        or result.attempt_id != expected_attempt_id
        or result.status != TaskStatus.CRASHED
    ):
        raise StoreContractError("task crash returned mismatched state")
    _terminal_failure(
        store,
        run_id=result.run_id,
        status=RunStatus.CRASHED,
        callbacks=callbacks,
        cause=TaskAttemptRef(
            task_id=task_id,
            attempt_id=expected_attempt_id,
        ),
    )


def abort_run(
    store: DagStore,
    *,
    run_id: str,
    reason: str,
    callbacks: OrchestratorCallbacks,
    status: RunStatus = RunStatus.FAILED,
) -> bool:
    """The function attempts a ``FAILED``, ``CRASHED`` or ``CANCELLED`` transition."""

    if status not in (RunStatus.FAILED, RunStatus.CRASHED, RunStatus.CANCELLED):
        raise ValueError(f"invalid abort status {status!r}")
    return _terminal_failure(
        store,
        run_id=run_id,
        status=status,
        callbacks=callbacks,
        reason=reason,
    )
