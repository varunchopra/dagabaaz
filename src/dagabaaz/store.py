"""Protocols for persistence and output resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from dagabaaz.constants import RUN_TERMINAL_STATUSES, RunStatus
from dagabaaz.models import (
    DagNode,
    InputEdge,
    LaunchCreateResult,
    NodeDisposition,
    NodeLaunch,
    PlanningSnapshot,
    ResolvedOutput,
    TaskContext,
    TaskInputPlan,
)


class StoreContractError(RuntimeError):
    """A store returned state that does not match the requested operation."""


@dataclass(frozen=True, slots=True)
class TaskAttemptRef:
    """The task attempt associated with a requested terminal transition."""

    task_id: str
    attempt_id: str

    def __post_init__(self) -> None:
        if not self.task_id or not self.attempt_id:
            raise ValueError("task and attempt IDs must not be empty")


@dataclass(frozen=True, slots=True)
class RunReopenResult:
    """The retry count and plan generations committed when a run is reopened."""

    retry_count: int
    generations: Mapping[int, int]

    def __post_init__(self) -> None:
        if type(self.retry_count) is not int or self.retry_count < 1:
            raise ValueError("retry_count must be a positive integer")
        if not isinstance(self.generations, Mapping):
            raise ValueError("run generations must be a mapping")
        copied: dict[int, int] = {}
        for index, generation in self.generations.items():
            if type(index) is not int or type(generation) is not int:
                raise ValueError("node indices and generations must be integers")
            copied[index] = generation
        if any(index < 0 or generation < 1 for index, generation in copied.items()):
            raise ValueError("node indices must be non-negative and generations must be positive")
        object.__setattr__(self, "generations", MappingProxyType(copied))


@runtime_checkable
class DagStore(Protocol):
    """Persistence operations used by the orchestrator.

    Planning validates snapshot contents before launch creation. In one
    transaction, the store validates the snapshot token and run state, then
    writes the launch. The application's worker adapter commits task completion
    and output publication together. Published routing fields remain unchanged.
    """

    def get_run_status(self, run_id: str) -> RunStatus | None:
        """The run status is returned, or ``None`` when the run does not exist."""
        ...

    def get_run_nodes(self, run_id: str) -> list[DagNode] | None:
        """A copy of the run's stored node definitions is returned.

        The stored topology must not change during a run. Boundary retries use
        the same definitions rather than the pipeline's current definition.
        """
        ...

    def list_node_launches(self, run_id: str) -> Mapping[int, NodeLaunch]:
        """Launches that have not been invalidated are keyed by node index."""
        ...

    def get_planning_snapshot(
        self,
        run_id: str,
        node_index: int,
        edges: tuple[InputEdge, ...],
    ) -> PlanningSnapshot:
        """A planning snapshot represents one consistent source state.

        The snapshot contains the exact named-edge sets, source dispositions,
        generation and frozen run input. Its token must become stale when source
        state changes. Skipped, filtered and failed source launches expose no
        outputs.
        """
        ...

    def try_create_node_launch(
        self,
        run_id: str,
        node_index: int,
        plugin_name: str,
        snapshot_token: str,
        plans: tuple[TaskInputPlan, ...],
        zero_task_disposition: NodeDisposition | None,
        error: str = "",
    ) -> LaunchCreateResult:
        """A node launch is created if the snapshot and run are still current.

        Validation of the token and run state, launch creation, task and plan
        persistence, and queue outbox writes occur in one transaction. A
        disposition without tasks creates no task rows. ``STALE`` takes
        precedence when an existing launch belongs to another generation;
        ``ALREADY_EXISTS`` is reserved for the generation represented by the
        snapshot token.
        """
        ...

    def get_task_context(self, task_id: str) -> TaskContext | None:
        """The task's routing context is returned, or ``None`` when unavailable."""
        ...

    def mark_task_failed(
        self,
        task_id: str,
        error: str,
        *,
        expected_attempt_id: str,
    ) -> str | None:
        """The current task attempt moves to ``FAILED`` and yields its run ID.

        The attempt comparison and state change share a transaction. Stale,
        missing, invalidated or terminal tasks return ``None`` without a change.
        """
        ...

    def mark_task_crashed(
        self,
        task_id: str,
        error: str,
        *,
        expected_attempt_id: str,
    ) -> str | None:
        """The current task attempt moves to ``CRASHED`` and yields its run ID.

        The attempt comparison and state change share a transaction. Stale,
        missing, invalidated or terminal tasks return ``None`` without a change.
        """
        ...

    def try_finalize_run(
        self,
        run_id: str,
        status: RunStatus,
        error: str | None,
        *,
        cause: TaskAttemptRef | None = None,
    ) -> bool:
        """The terminal transition is validated and applied in one transaction.

        When ``cause`` is present, its task attempt must still be current in its
        active launch. Completion requires active launch keys for every stored
        node index, complete active launches, no failed active launch and
        successful completion of every task owned by those launches.
        Invalidated launches, tasks and attempt history do not count. Other
        terminal transitions cancel unfinished active work. Validation,
        cancellation, progress and the status change share a transaction.
        Missing and terminal runs return ``False`` without a change.
        """
        if status not in RUN_TERMINAL_STATUSES:
            raise ValueError(f"status {status!r} is not terminal")
        ...


@runtime_checkable
class OutputResolver(Protocol):
    def resolve_outputs(
        self, run_id: str, output_ids: tuple[str, ...]
    ) -> Mapping[str, ResolvedOutput]:
        """The requested output IDs are resolved within the given run.

        The mapping contains every requested ID and no others, keyed by output
        ID. Each ``fields`` mapping must match the routing fields published for
        that output. Data may contain ordinary values, paths, URIs or durable
        references that remain valid throughout task execution. Resolved data
        must not contain live handles, cursors or other resource-owning objects;
        the worker adapter acquires and cleans up such resources separately.
        """
        ...


@runtime_checkable
class DagRetryStore(DagStore, Protocol):
    def try_create_task_retry(
        self, task_id: str, *, expected_attempt_id: str
    ) -> str | None:
        """A replacement is created only for a current crashed attempt on an active run.

        The transaction compares ``expected_attempt_id``, retains the task plan
        and generation, and creates the replacement attempt and queue outbox
        row. Success returns the replacement attempt ID. Rejection returns
        ``None`` without changing state.
        """
        ...

    def try_reopen_run(
        self,
        run_id: str,
        boundary_indices: tuple[int, ...] | None,
        affected_indices: tuple[int, ...],
    ) -> RunReopenResult | None:
        """The run is reopened only if the supplied retry state remains current.

        The store acquires its run lock before reading status, topology,
        launches, tasks or generations. A failed or crashed run requires a
        non-empty active failure boundary. A cancelled run requires ``None``.
        The store adds every incomplete or failed active launch to the restart
        roots, recomputes their downstream closure and requires it to equal
        ``affected_indices``.

        Reopening, retry-count increment, generation changes and invalidation
        share that transaction. All planning snapshots obtained before the
        retry become stale. Success returns the written retry count and every
        affected generation. A cancelled run with no affected nodes returns an
        empty generation mapping. Rejection returns ``None`` without changing
        state.
        """
        ...
