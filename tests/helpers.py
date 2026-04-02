import itertools
from dataclasses import dataclass
from typing import NamedTuple

from dagabaaz.constants import (
    TASK_TERMINAL_STATUSES,
    FanMode,
    NodeSummaryStatus,
    RunStatus,
    TaskStatus,
)
from dagabaaz.models import DagArtifact, DagNode, EdgeFilter, TaskContext
from dagabaaz.store import DagStore


@dataclass
class TaskRecord:
    """Internal record used by MockDagStore to track task state."""

    task_id: str
    run_id: str
    node_index: int
    plugin_name: str
    status: TaskStatus
    input_artifact_id: str | None = None
    origin_artifact_id: str | None = None
    error: str = ""


class SkippedRecord(NamedTuple):
    run_id: str
    node_index: int
    plugin_name: str


class FilteredRecord(NamedTuple):
    run_id: str
    node_index: int
    plugin_name: str


class FailedRecord(NamedTuple):
    run_id: str
    node_index: int
    plugin_name: str
    error: str


class GroupedRecord(NamedTuple):
    run_id: str
    node_index: int
    plugin_name: str
    origin_artifact_id: str


class CancelledRecord(NamedTuple):
    run_id: str
    reason: str


class MockDagStore:
    """In-memory DagStore for testing."""

    def __init__(self) -> None:
        # Internal state
        self._run_status: dict[str, RunStatus] = {}
        self._run_nodes: dict[str, list[DagNode]] = {}
        self._run_progress: dict[str, int] = {}
        self._tasks: dict[str, TaskRecord] = {}
        self._node_launches: set[tuple[str, int]] = set()
        self._terminal_claimed: set[str] = set()
        self._artifacts: dict[str, dict[int, list[DagArtifact]]] = {}
        self._task_counter = itertools.count(1)

        # Call-tracking for test assertions
        self.dispatched_tasks: list[TaskRecord] = []
        self.skipped_nodes: list[SkippedRecord] = []
        self.filtered_nodes: list[FilteredRecord] = []
        self.failed_nodes: list[FailedRecord] = []
        self.grouped_tasks: list[GroupedRecord] = []
        self.cancelled_runs: list[CancelledRecord] = []

    def setup_run(
        self,
        run_id: str,
        nodes: list[DagNode],
        status: RunStatus = RunStatus.RUNNING,
    ) -> None:
        self._run_status[run_id] = status
        self._run_nodes[run_id] = nodes

    def setup_task(
        self,
        task_id: str,
        run_id: str,
        node_index: int,
        plugin_name: str,
        status: TaskStatus = TaskStatus.COMPLETED,
    ) -> None:
        record = TaskRecord(
            task_id=task_id,
            run_id=run_id,
            node_index=node_index,
            plugin_name=plugin_name,
            status=status,
        )
        self._tasks[task_id] = record
        self._node_launches.add((run_id, node_index))

    def setup_artifacts(
        self, run_id: str, node_index: int, artifacts: list[DagArtifact]
    ) -> None:
        if run_id not in self._artifacts:
            self._artifacts[run_id] = {}
        self._artifacts[run_id][node_index] = list(artifacts)

    def get_run_status(self, run_id: str) -> RunStatus | None:
        return self._run_status.get(run_id)

    def get_barrier_state(
        self, run_id: str, node_index: int
    ) -> tuple[RunStatus | None, int, int]:
        tasks = [
            t
            for t in self._tasks.values()
            if t.run_id == run_id and t.node_index == node_index
        ]
        total = len(tasks)
        completed = sum(1 for t in tasks if t.status in TASK_TERMINAL_STATUSES)
        return (self._run_status.get(run_id), total, completed)

    def get_run_nodes(self, run_id: str) -> list[DagNode] | None:
        return self._run_nodes.get(run_id)

    def try_claim_run_terminal(
        self, run_id: str, status: RunStatus, error: str | None = None
    ) -> bool:
        if run_id in self._terminal_claimed:
            return False
        self._terminal_claimed.add(run_id)
        self._run_status[run_id] = status
        return True

    def set_run_progress(self, run_id: str, completed_count: int) -> None:
        self._run_progress[run_id] = completed_count

    def get_task_context(self, task_id: str) -> TaskContext | None:
        record = self._tasks.get(task_id)
        if record is None:
            return None
        return TaskContext(run_id=record.run_id, node_index=record.node_index)

    def get_completed_node_indices(self, run_id: str) -> set[int]:
        nodes_tasks: dict[int, list[TaskRecord]] = {}
        for t in self._tasks.values():
            if t.run_id == run_id:
                nodes_tasks.setdefault(t.node_index, []).append(t)
        result: set[int] = set()
        for node_idx, tasks in nodes_tasks.items():
            if tasks and all(t.status in TASK_TERMINAL_STATUSES for t in tasks):
                result.add(node_idx)
        return result

    def get_launched_node_indices(self, run_id: str) -> set[int]:
        return {ni for (rid, ni) in self._node_launches if rid == run_id}

    def get_node_summary(self, run_id: str) -> dict[int, NodeSummaryStatus]:
        nodes_tasks: dict[int, list[TaskRecord]] = {}
        for t in self._tasks.values():
            if t.run_id == run_id:
                nodes_tasks.setdefault(t.node_index, []).append(t)
        summary: dict[int, NodeSummaryStatus] = {}
        for node_idx, tasks in nodes_tasks.items():
            if any(t.status not in TASK_TERMINAL_STATUSES for t in tasks):
                summary[node_idx] = NodeSummaryStatus.PARTIAL
            elif all(t.status == TaskStatus.SKIPPED for t in tasks):
                summary[node_idx] = NodeSummaryStatus.SKIPPED
            else:
                summary[node_idx] = NodeSummaryStatus.COMPLETED
        return summary

    def try_claim_node_launch(self, run_id: str, node_index: int) -> bool:
        key = (run_id, node_index)
        if key in self._node_launches:
            return False
        self._node_launches.add(key)
        return True

    def dispatch_task(
        self,
        run_id: str,
        node_index: int,
        plugin_name: str,
        input_artifact_id: str | None,
    ) -> str:
        task_id = f"task-{next(self._task_counter)}"
        record = TaskRecord(
            task_id=task_id,
            run_id=run_id,
            node_index=node_index,
            plugin_name=plugin_name,
            status=TaskStatus.QUEUED,
            input_artifact_id=input_artifact_id,
        )
        self._tasks[task_id] = record
        self._node_launches.add((run_id, node_index))
        self.dispatched_tasks.append(record)
        return task_id

    def dispatch_skipped_task(
        self, run_id: str, node_index: int, plugin_name: str
    ) -> str:
        task_id = f"task-{next(self._task_counter)}"
        record = TaskRecord(
            task_id=task_id,
            run_id=run_id,
            node_index=node_index,
            plugin_name=plugin_name,
            status=TaskStatus.SKIPPED,
        )
        self._tasks[task_id] = record
        self._node_launches.add((run_id, node_index))
        self.skipped_nodes.append(SkippedRecord(run_id, node_index, plugin_name))
        return task_id

    def dispatch_filtered_task(
        self, run_id: str, node_index: int, plugin_name: str
    ) -> str:
        task_id = f"task-{next(self._task_counter)}"
        record = TaskRecord(
            task_id=task_id,
            run_id=run_id,
            node_index=node_index,
            plugin_name=plugin_name,
            status=TaskStatus.FILTERED,
        )
        self._tasks[task_id] = record
        self._node_launches.add((run_id, node_index))
        self.filtered_nodes.append(FilteredRecord(run_id, node_index, plugin_name))
        return task_id

    def dispatch_failed_task(
        self, run_id: str, node_index: int, plugin_name: str, error: str
    ) -> str:
        task_id = f"task-{next(self._task_counter)}"
        record = TaskRecord(
            task_id=task_id,
            run_id=run_id,
            node_index=node_index,
            plugin_name=plugin_name,
            status=TaskStatus.FAILED,
            error=error,
        )
        self._tasks[task_id] = record
        self._node_launches.add((run_id, node_index))
        self.failed_nodes.append(FailedRecord(run_id, node_index, plugin_name, error))
        return task_id

    def fail_run(self, run_id: str, error: str) -> bool:
        claimed = self.try_claim_run_terminal(run_id, RunStatus.FAILED, error)
        if claimed:
            self.cancel_remaining_tasks(run_id, error)
        return claimed

    def mark_task_failed(self, task_id: str, error: str) -> str | None:
        record = self._tasks.get(task_id)
        if record is None:
            return None
        record.status = TaskStatus.FAILED
        record.error = error
        return record.run_id

    def mark_task_crashed(self, task_id: str, error: str) -> str | None:
        record = self._tasks.get(task_id)
        if record is None:
            return None
        record.status = TaskStatus.CRASHED
        record.error = error
        return record.run_id

    def cancel_remaining_tasks(self, run_id: str, reason: str) -> int:
        count = 0
        for t in self._tasks.values():
            if t.run_id == run_id and t.status not in TASK_TERMINAL_STATUSES:
                t.status = TaskStatus.CANCELLED
                count += 1
        self.cancelled_runs.append(CancelledRecord(run_id, reason))
        return count

    def dispatch_grouped_task(
        self,
        run_id: str,
        node_index: int,
        plugin_name: str,
        origin_artifact_id: str,
    ) -> str:
        task_id = f"task-{next(self._task_counter)}"
        record = TaskRecord(
            task_id=task_id,
            run_id=run_id,
            node_index=node_index,
            plugin_name=plugin_name,
            status=TaskStatus.QUEUED,
            origin_artifact_id=origin_artifact_id,
        )
        self._tasks[task_id] = record
        self._node_launches.add((run_id, node_index))
        self.grouped_tasks.append(
            GroupedRecord(run_id, node_index, plugin_name, origin_artifact_id)
        )
        return task_id

    def get_artifacts_by_node_indices(
        self, run_id: str, node_indices: list[int]
    ) -> list[DagArtifact]:
        node_map = self._artifacts.get(run_id, {})
        result: list[DagArtifact] = []
        for idx in node_indices:
            result.extend(node_map.get(idx, []))
        return result


assert isinstance(
    MockDagStore(), DagStore
), "MockDagStore does not satisfy DagStore protocol"


_dag_art_counter = itertools.count(1)


def make_node(
    plugin: str,
    slug: str = "",
    depends_on: list[str] | None = None,
    edge_filters: dict[str, EdgeFilter] | None = None,
    fan_mode: FanMode = FanMode.SINGLE,
) -> DagNode:
    return DagNode(
        plugin=plugin,
        slug=slug,
        depends_on=depends_on or [],
        edge_filters=edge_filters or {},
        fan_mode=fan_mode,
    )


def make_dag_artifact(
    name: str,
    size: int | None = 1000,
    mime: str | None = None,
    metadata: dict[str, object] | None = None,
    origin_artifact_id: str | None = None,
) -> DagArtifact:
    return DagArtifact(
        id=f"art-{next(_dag_art_counter)}",
        file_name=name,
        file_size=size,
        mime_type=mime,
        metadata=metadata or {},
        origin_artifact_id=origin_artifact_id,
    )
