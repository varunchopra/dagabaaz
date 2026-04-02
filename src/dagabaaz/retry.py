"""Retry logic for the DAG execution engine.

Computes which nodes need re-execution after a failure and drives the
re-dispatch sequence.
"""

from dataclasses import dataclass

from dagabaaz.models import DagNode
from dagabaaz.store import DagRetryStore


@dataclass(frozen=True, slots=True)
class RetryBoundary:
    """Which nodes need re-execution after a failure.

    ``failed``: launched, non-completed nodes with retryable task failures.
    These get their non-completed tasks deleted and re-dispatched.

    ``downstream``: launched non-completed nodes without failures, plus
    unlaunched nodes. These get all tasks and node_launches wiped so the
    orchestrator can rediscover them as upstream tasks complete.
    """

    failed: list[int]
    downstream: list[int]


@dataclass(frozen=True, slots=True)
class RetryableTaskInfo:
    """Minimal task info needed for retry re-dispatch decisions."""

    task_id: str
    node_index: int
    plugin_name: str
    input_artifact_id: str | None
    origin_artifact_id: str | None
    status: str


@dataclass(frozen=True, slots=True)
class RetryResult:
    """Outcome of engine-level retry: old task ID → new task ID."""

    replacement_map: dict[str, str]


def compute_retry_boundary(
    completed_nodes: set[int],
    launched_nodes: set[int],
    node_count: int,
    node_has_retryable_failure: dict[int, bool],
) -> RetryBoundary:
    """Compute which nodes need re-execution from a failure point.

    Partitions the graph into three zones:
    - Completed: nodes where all tasks succeeded. Left untouched.
    - Failed: launched, non-completed nodes with retryable failures.
    - Downstream: everything else. Wiped and rediscovered.
    """
    non_completed = launched_nodes - completed_nodes
    failed: list[int] = []
    downstream: list[int] = []

    for idx in sorted(non_completed):
        if node_has_retryable_failure.get(idx, False):
            failed.append(idx)
        else:
            downstream.append(idx)

    unlaunched = [
        i
        for i in range(node_count)
        if i not in launched_nodes and i not in completed_nodes
    ]
    downstream.extend(unlaunched)

    return RetryBoundary(failed=failed, downstream=downstream)


def retry_run(
    store: DagRetryStore,
    run_id: str,
    nodes: list[DagNode],
    boundary: RetryBoundary,
) -> RetryResult:
    """Delete failed/downstream tasks and re-dispatch failed ones.

    Uses the pre-computed ``RetryBoundary`` to know which nodes to touch.
    Returns a ``RetryResult`` mapping old_task_id → new_task_id for
    caller bookkeeping (attempt archival, billing adjustment).

    Preserves the dispatch method: tasks with ``origin_artifact_id``
    are re-dispatched via ``dispatch_grouped_task``; others via
    ``dispatch_task``.
    """
    replacement_map: dict[str, str] = {}

    if boundary.failed:
        store.delete_node_launches(run_id, boundary.failed)
        all_tasks = store.get_retryable_tasks(run_id, boundary.failed)
        store.delete_non_completed_tasks_at_nodes(run_id, boundary.failed)

        for task in all_tasks:
            plugin_name = nodes[task.node_index].plugin
            if task.origin_artifact_id:
                new_id = store.dispatch_grouped_task(
                    run_id, task.node_index, plugin_name, task.origin_artifact_id
                )
            else:
                new_id = store.dispatch_task(
                    run_id, task.node_index, plugin_name, task.input_artifact_id
                )
            replacement_map[task.task_id] = new_id

    if boundary.downstream:
        store.delete_node_launches(run_id, boundary.downstream)
        store.delete_tasks_at_nodes(run_id, boundary.downstream)

    return RetryResult(replacement_map=replacement_map)
