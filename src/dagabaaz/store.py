"""Storage protocol for the DAG execution engine.

Defines the interface the engine needs for persistence and job dispatch.
"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from dagabaaz.constants import NodeSummaryStatus, RunStatus
from dagabaaz.models import DagArtifact, DagNode, TaskArtifact, TaskContext

if TYPE_CHECKING:
    from dagabaaz.retry import RetryableTaskInfo


@runtime_checkable
class DagStore(Protocol):
    """Storage and dispatch interface for the DAG execution engine.

    Methods are grouped by concern: run state, task queries,
    task dispatch, and artifact queries.

    **Concurrency contract**: Callers must serialize ``on_task_complete``
    per run (e.g. via ``pg_advisory_xact_lock``) so that barrier sync
    reads see committed state from prior completions.
    """

    def get_run_status(self, run_id: str) -> RunStatus | None:
        """Return the run's current lifecycle status, or None if not found."""
        ...

    def get_barrier_state(
        self, run_id: str, node_index: int
    ) -> tuple[RunStatus | None, int, int]:
        """Return (run_status, total_tasks, completed_tasks) in one round-trip.

        Combines the run-status terminal check and the barrier-sync task
        count into a single call.
        """
        ...

    def get_run_nodes(self, run_id: str) -> list[DagNode] | None:
        """Return the snapshotted node definitions for a run, or None if not found.

        Extra fields beyond DagNode (config, credentials) are silently ignored.
        """
        ...

    def try_claim_run_terminal(
        self, run_id: str, status: RunStatus, error: str | None = None
    ) -> bool:
        """Atomically transition a run to a terminal status.

        Returns True if the transition succeeded, False if the run was
        already in a terminal state. Must be atomic — if two callers
        race (e.g., one completing, one failing), exactly one wins.
        """
        ...

    def set_run_progress(self, run_id: str, completed_count: int) -> None:
        """Update the run's progress to the count of completed nodes."""
        ...

    def get_task_context(self, task_id: str) -> TaskContext | None:
        """Return the run_id and node_index for a task, or None if not found."""
        ...

    def get_completed_node_indices(self, run_id: str) -> set[int]:
        """Return indices of nodes where ALL tasks are terminal.

        A node is complete when every task has reached a terminal state
        (completed, skipped, or filtered).
        """
        ...

    def get_launched_node_indices(self, run_id: str) -> set[int]:
        """Return indices of nodes that have at least one task."""
        ...

    def get_node_summary(self, run_id: str) -> dict[int, NodeSummaryStatus]:
        """Return per-node aggregate status for skip cascade decisions.

        Maps node_index to one of:
        - ``"completed"``: all tasks finished (completed, filtered, or mixed)
        - ``"skipped"``:   ALL tasks are skipped (upstream dead — cascading)
        - ``"partial"``:   some tasks still pending/running/failed

        The orchestrator uses ``"skipped"`` to propagate skip cascades:
        if any dependency is fully skipped, the downstream node is also
        skipped without executing. Filtered nodes (edge filter produced 0
        artifacts) map to ``"completed"`` and do NOT cascade.
        """
        ...

    def try_claim_node_launch(self, run_id: str, node_index: int) -> bool:
        """Attempt to claim exclusive right to launch a node.

        Returns True if this caller claimed the node, False if another
        caller already claimed it. Must be atomic — exactly one caller
        wins when called concurrently for the same (run_id, node_index).
        """
        ...

    def dispatch_task(
        self,
        run_id: str,
        node_index: int,
        plugin_name: str,
        input_artifact_id: str | None,
    ) -> str:
        """Create and enqueue a task for execution. Returns the task ID.

        Called during fan-out (one call per artifact) or for fan-in/root
        nodes (one call with input_artifact_id=None).

        **Async dispatch contract**: The dispatched task MUST NOT complete
        before this method returns. The orchestrator's launch loop assumes
        dispatched tasks are pending/queued, not completed.
        """
        ...

    def dispatch_skipped_task(
        self,
        run_id: str,
        node_index: int,
        plugin_name: str,
    ) -> str:
        """Insert a pre-marked 'skipped' task (no queue job). Returns the task ID.

        The marker row lets dependency resolution count this node as
        satisfied and skip cascades propagate to downstream nodes.
        """
        ...

    def dispatch_filtered_task(
        self,
        run_id: str,
        node_index: int,
        plugin_name: str,
    ) -> str:
        """Insert a pre-marked 'filtered' task (no queue job). Returns the task ID.

        Unlike skipped, filtered does NOT cascade -- downstream nodes
        proceed normally, collecting artifacts from further upstream.
        """
        ...

    def dispatch_failed_task(
        self,
        run_id: str,
        node_index: int,
        plugin_name: str,
        error: str,
    ) -> str:
        """Insert a pre-marked 'failed' task with error (no queue job). Returns the task ID.

        Atomic alternative to dispatch_task + mark_task_failed, which has a
        race window where a worker could pick up the dispatched task before
        the failure mark lands.
        """
        ...

    def fail_run(self, run_id: str, error: str) -> bool:
        """Claim terminal FAILED + cancel remaining tasks. Returns True if claimed."""
        ...

    def mark_task_failed(self, task_id: str, error: str) -> str | None:
        """Mark a task as failed. Returns the run_id, or None if not found."""
        ...

    def mark_task_crashed(self, task_id: str, error: str) -> str | None:
        """Mark a task as crashed (infra failure). Returns the run_id, or None if not found.

        Distinct from ``mark_task_failed``: crashed = infrastructure failure,
        failed = plugin/execution error.
        """
        ...

    def cancel_remaining_tasks(self, run_id: str, reason: str) -> int:
        """Cancel all non-terminal tasks for a run. Returns the count.

        Uses "not terminal" rather than "active" so new statuses are
        automatically covered.
        """
        ...

    def dispatch_grouped_task(
        self,
        run_id: str,
        node_index: int,
        plugin_name: str,
        origin_artifact_id: str,
    ) -> str:
        """Create a task for one origin group in grouped fan-in. Returns the task ID.

        Stores ``origin_artifact_id`` on the task instead of
        ``input_artifact_id``.
        """
        ...

    def get_artifacts_by_node_indices(
        self,
        run_id: str,
        node_indices: list[int],
    ) -> list[DagArtifact]:
        """Fetch artifacts produced by tasks at the given node indices."""
        ...


@runtime_checkable
class TaskInputStore(Protocol):
    """Storage interface for worker-side task input resolution.

    Separate from ``DagStore`` -- users who only need orchestration
    don't implement this.
    """

    def get_artifact_data(self, artifact_id: str) -> TaskArtifact | None:
        """Fetch a single artifact by ID with file path."""
        ...

    def get_task_artifacts_by_node_indices(
        self,
        run_id: str,
        node_indices: list[int],
    ) -> list[TaskArtifact]:
        """Fetch artifacts (with file_path) produced by tasks at the given node indices."""
        ...

    def get_grouped_artifacts(
        self,
        run_id: str,
        dep_indices: list[int],
        origin_id: str,
    ) -> list[TaskArtifact]:
        """Fetch artifacts sharing a specific origin from dependency nodes."""
        ...

    def get_broadcast_artifacts(
        self,
        run_id: str,
        dep_indices: list[int],
    ) -> list[TaskArtifact]:
        """Fetch artifacts with NULL origin (broadcast/side inputs)."""
        ...

    def get_run_input(self, run_id: str) -> dict[str, object]:
        """Fetch the run's user-provided input."""
        ...

    def get_artifacts_partitioned(
        self,
        run_id: str,
        node_indices: list[int],
    ) -> dict[int, list[TaskArtifact]]:
        """Fetch artifacts from nodes, partitioned by producing node index."""
        ...

    def get_artifact_producing_node(self, artifact_id: str) -> int | None:
        """Return the node_index of the task that produced a given artifact."""
        ...


@runtime_checkable
class DagRetryStore(DagStore, Protocol):
    """Extended DagStore with retry capabilities.

    Separated from ``DagStore`` so existing implementations don't need
    to implement retry methods.
    """

    def delete_node_launches(self, run_id: str, node_indices: list[int]) -> None:
        """Delete node_launches entries for given nodes.

        Clears the atomic claim so re-dispatch can re-claim them.
        """
        ...

    def delete_tasks_at_nodes(self, run_id: str, node_indices: list[int]) -> None:
        """Delete all tasks at given nodes (downstream wipe on retry)."""
        ...

    def get_retryable_tasks(
        self, run_id: str, node_indices: list[int]
    ) -> list["RetryableTaskInfo"]:
        """Get non-completed tasks at given nodes for retry re-dispatch.

        Returns minimal task info needed to decide how to re-dispatch
        (grouped vs fan-out) and to build the replacement map.
        """
        ...

    def delete_non_completed_tasks_at_nodes(
        self, run_id: str, node_indices: list[int]
    ) -> None:
        """Delete non-completed tasks at given nodes (failed node cleanup).

        Completed tasks at these nodes are preserved — only pending,
        queued, running, failed, crashed, cancelled tasks are removed.
        """
        ...
