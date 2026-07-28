"""Persisted-state reconciliation and run lifecycle transitions.

The orchestrator plans nodes whose source launches are complete. Stores create
launches, tasks, plans and queue outbox rows together, so repeated reconciliation
does not duplicate work.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from dagabaaz.constants import (
    MAX_SNAPSHOT_RETRIES,
    RUN_TERMINAL_STATUSES,
    LaunchCreateStatus,
    NodeDisposition,
    RunStatus,
)
from dagabaaz.graph import find_ready_nodes, find_root_nodes
from dagabaaz.models import NodeLaunch, TaskInputPlan
from dagabaaz.planning import construct_task_plans
from dagabaaz.store import DagStore, StoreContractError, TaskAttemptRef
from dagabaaz.topology import RunTopology

logger = logging.getLogger(__name__)

RunCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class OrchestratorCallbacks:
    """Callbacks invoked after a successful terminal transition.

    The application serialises terminal handlers and boundary retries for a run
    until the callback returns.
    """

    on_run_completed: RunCallback
    on_run_failed: RunCallback
    on_run_crashed: RunCallback
    on_run_cancelled: RunCallback


def _terminal_failure(
    store: DagStore,
    *,
    run_id: str,
    status: RunStatus,
    error: str,
    callbacks: OrchestratorCallbacks,
    cause: TaskAttemptRef | None = None,
) -> bool:
    """Finalisation, including cancellation, completes before the callback."""

    if not store.try_finalize_run(run_id, status, error, cause=cause):
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
    roots = find_root_nodes(nodes)
    reconcile_run(store, run_id, callbacks=callbacks)
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
    if status is None or status in RUN_TERMINAL_STATUSES:
        return
    nodes = store.get_run_nodes(run_id)
    if nodes is None:
        logger.error("run %s has no snapshotted node definition", run_id)
        return
    topology = RunTopology.build(nodes)

    while True:
        launches = dict(store.list_node_launches(run_id))
        failed_launch = min(
            (
                launch
                for launch in launches.values()
                if launch.disposition == NodeDisposition.FAILED
            ),
            key=lambda launch: launch.node_index,
            default=None,
        )
        if failed_launch is not None:
            _terminal_failure(
                store,
                run_id=run_id,
                status=RunStatus.FAILED,
                error=failed_launch.error or "node planning failed",
                callbacks=callbacks,
            )
            return

        completed = {index for index, launch in launches.items() if launch.complete}
        launched = set(launches)
        ready = find_ready_nodes(topology.dependencies, completed, launched)
        if not ready:
            break

        created_complete_launch = False
        for node_index in ready:
            node = topology.nodes[node_index]
            source_indices = {edge.name: topology.slug_to_index[edge.source] for edge in node.edges}
            launch = None
            for _attempt in range(MAX_SNAPSHOT_RETRIES):
                snapshot = store.get_planning_snapshot(run_id, node_index, node.edges)
                construction = construct_task_plans(
                    node, source_indices=source_indices, snapshot=snapshot
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
            if launch.disposition == NodeDisposition.FAILED:
                _terminal_failure(
                    store,
                    run_id=run_id,
                    status=RunStatus.FAILED,
                    error=launch.error or "node planning failed",
                    callbacks=callbacks,
                )
                return
            if launch.complete:
                created_complete_launch = True

        # A complete launch can make the next graph level ready. This includes
        # launches without tasks and launches completed concurrently.
        refreshed = dict(store.list_node_launches(run_id))
        new_indices = set(refreshed) - launched
        if not new_indices or not created_complete_launch:
            break

    launches = dict(store.list_node_launches(run_id))
    completed_count = sum(launch.complete for launch in launches.values())
    if (
        len(launches) == len(nodes)
        and completed_count == len(nodes)
        and store.try_finalize_run(run_id, RunStatus.COMPLETED, None)
    ):
        callbacks.on_run_completed(run_id)


def on_task_complete(
    store: DagStore,
    *,
    task_id: str,
    callbacks: OrchestratorCallbacks,
) -> None:
    """Reconciliation follows committed task completion and output publication."""

    context = store.get_task_context(task_id)
    if context is None:
        logger.error("task %s not found", task_id)
        return
    reconcile_run(store, context.run_id, callbacks=callbacks)


def on_task_failed(
    store: DagStore,
    *,
    task_id: str,
    expected_attempt_id: str,
    error_message: str,
    callbacks: OrchestratorCallbacks,
) -> None:
    """A current matching attempt is marked ``FAILED`` before a run transition is attempted."""

    run_id = store.mark_task_failed(
        task_id,
        error_message,
        expected_attempt_id=expected_attempt_id,
    )
    if run_id is not None:
        _terminal_failure(
            store,
            run_id=run_id,
            status=RunStatus.FAILED,
            error=error_message,
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

    run_id = store.mark_task_crashed(
        task_id,
        error_message,
        expected_attempt_id=expected_attempt_id,
    )
    if run_id is not None:
        _terminal_failure(
            store,
            run_id=run_id,
            status=RunStatus.CRASHED,
            error=error_message,
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
        error=reason,
        callbacks=callbacks,
    )
