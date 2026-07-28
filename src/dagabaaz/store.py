"""Protocols for persistence and output resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from dagabaaz.constants import (
    MAX_OUTPUTS_PER_TASK_COMPLETION,
    MAX_TASK_COMPLETION_ROUTING_BYTES,
    RUN_TERMINAL_STATUSES,
    RunStatus,
    TaskStatus,
)
from dagabaaz.models import (
    DagNode,
    EmittedOutput,
    LaunchCreateResult,
    NodeDisposition,
    NodeLaunch,
    PlanningSnapshot,
    ResolvedOutput,
    TaskInputPlan,
)


class StoreContractError(RuntimeError):
    """A store returned state that does not match the requested operation."""


class OutputPublicationError(ValueError):
    """A current task attempt supplied outputs that cannot be published."""


def validate_output_batch_size(output_count: int) -> None:
    """Reject a task completion whose output batch exceeds the store contract."""

    if type(output_count) is not int or output_count < 0:
        raise ValueError("output count must be a non-negative integer")
    if output_count > MAX_OUTPUTS_PER_TASK_COMPLETION:
        raise OutputPublicationError(
            f"task completion contains {output_count} outputs; maximum is "
            f"{MAX_OUTPUTS_PER_TASK_COMPLETION}"
        )


def validate_output_batch_routing_size(routing_size: int) -> None:
    """Reject a task completion whose routing state exceeds the store contract."""

    if type(routing_size) is not int or routing_size < 0:
        raise ValueError("routing size must be a non-negative integer")
    if routing_size > MAX_TASK_COMPLETION_ROUTING_BYTES:
        raise OutputPublicationError(
            f"task completion routing data is {routing_size} bytes; maximum is "
            f"{MAX_TASK_COMPLETION_ROUTING_BYTES}"
        )


@dataclass(frozen=True, slots=True)
class TaskAttemptRef:
    """The task attempt associated with a requested terminal transition."""

    task_id: str
    attempt_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.task_id, str)
            or not self.task_id
            or not isinstance(self.attempt_id, str)
            or not self.attempt_id
        ):
            raise ValueError("task and attempt IDs must be non-empty strings")


@dataclass(frozen=True, slots=True)
class NodeLaunchRef:
    """The planning launch associated with a requested terminal transition."""

    launch_id: str
    node_index: int
    generation: int

    def __post_init__(self) -> None:
        if not isinstance(self.launch_id, str) or not self.launch_id:
            raise ValueError("launch ID must be a non-empty string")
        if type(self.node_index) is not int or self.node_index < 0:
            raise ValueError("node index must be a non-negative integer")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class TaskCompletionResult:
    """The launch state committed with a task and its emitted outputs."""

    task_id: str
    attempt_id: str
    run_id: str
    node_index: int
    generation: int
    launch_complete: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.task_id, str)
            or not self.task_id
            or not isinstance(self.attempt_id, str)
            or not self.attempt_id
        ):
            raise ValueError("task and attempt IDs must be non-empty strings")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run ID must be a non-empty string")
        if type(self.node_index) is not int or self.node_index < 0:
            raise ValueError("node index must be a non-negative integer")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if type(self.launch_complete) is not bool:
            raise ValueError("launch_complete must be a boolean")


@dataclass(frozen=True, slots=True)
class TaskFailureResult:
    """The failure state committed for one task attempt."""

    task_id: str
    attempt_id: str
    run_id: str
    node_index: int
    generation: int
    status: TaskStatus

    def __post_init__(self) -> None:
        if (
            not isinstance(self.task_id, str)
            or not self.task_id
            or not isinstance(self.attempt_id, str)
            or not self.attempt_id
        ):
            raise ValueError("task and attempt IDs must be non-empty strings")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run ID must be a non-empty string")
        if type(self.node_index) is not int or self.node_index < 0:
            raise ValueError("node index must be a non-negative integer")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if not isinstance(self.status, TaskStatus) or self.status not in (
            TaskStatus.FAILED,
            TaskStatus.CRASHED,
        ):
            raise ValueError("task failure status must be FAILED or CRASHED")


@dataclass(frozen=True, slots=True)
class RunReopenResult:
    """The affected nodes and state committed when a run is reopened."""

    affected_nodes: tuple[int, ...]
    retry_count: int
    generations: Mapping[int, int]

    def __post_init__(self) -> None:
        if any(type(index) is not int or index < 0 for index in self.affected_nodes):
            raise ValueError("affected node indices must be non-negative integers")
        if tuple(sorted(set(self.affected_nodes))) != self.affected_nodes:
            raise ValueError("affected node indices must be sorted and unique")
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
        if set(copied) != set(self.affected_nodes):
            raise ValueError("generation keys must match the affected node indices")
        object.__setattr__(self, "generations", MappingProxyType(copied))


@runtime_checkable
class DagStore(Protocol):
    """Persistence operations used by the orchestrator.

    Planning validates snapshot contents before launch creation. The store owns
    each state transition and the records written with it. State changes for a
    run use the same run-level serialisation boundary. Published routing fields
    remain unchanged.
    """

    def get_run_status(self, run_id: str) -> RunStatus | None:
        """The run status is returned, or ``None`` when the run does not exist."""
        ...

    def get_run_nodes(self, run_id: str) -> list[DagNode] | None:
        """A copy of the run's stored node definitions is returned.

        The stored topology must not change after the run record is created.
        Run retries use the same definitions rather than the pipeline's
        current definition.
        """
        ...

    def try_start_run(self, run_id: str) -> bool:
        """A validated run moves from ``PENDING`` to ``RUNNING``.

        The run lock protects the transition. A run that is already
        ``RUNNING`` also returns ``True``. Missing and terminal runs return
        ``False`` without changing state. The transition does not create node
        launches. The integration must redeliver start requests or reconcile
        running runs after a process failure.
        """
        ...

    def list_node_launches(self, run_id: str) -> Mapping[int, NodeLaunch]:
        """Launches that have not been invalidated are keyed by node index."""
        ...

    def get_planning_snapshot(
        self,
        run_id: str,
        node_index: int,
        source_indices: Mapping[str, int],
        *,
        max_outputs_per_edge: int,
        max_outputs_total: int,
        max_routing_bytes: int,
    ) -> PlanningSnapshot:
        """A planning snapshot represents one consistent source state.

        The snapshot contains every named edge, each source disposition, the
        generation and the frozen run input. Its token must become stale when
        any source state changes. ``source_indices`` comes from the validated
        run topology, so stores do not have to resolve edge sources. Skipped,
        filtered and failed source launches expose no outputs.

        All three maxima are non-negative. Every edge key remains present. An
        edge contains at most ``max_outputs_per_edge + 1`` outputs and the
        snapshot contains at most ``max_outputs_total + 1`` outputs in all.
        Canonical routing size includes each output's ID, fields and correlation
        ID. The first output that takes the total beyond ``max_routing_bytes``
        is retained as the overflow signal. Overflow outputs are never routed.
        Loading stops after any overflow, so later edge values may be empty.
        The token still covers the full source state.
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

    def try_complete_task(
        self,
        task_id: str,
        outputs: tuple[EmittedOutput, ...],
        *,
        expected_attempt_id: str,
    ) -> TaskCompletionResult | None:
        """The current task attempt and its outputs are committed together.

        The store validates the active run, launch, generation, plan and
        attempt; derives output correlation from the stored node and plan;
        enforces run-lifetime output ID uniqueness; publishes routing fields and
        materialised data; completes the task; and updates launch state,
        progress and snapshot freshness in one transaction. The attempt ID is
        the completion idempotency key: a redelivery of the completed attempt
        returns its committed result and cannot replace its outputs. An invalid
        first publication raises ``OutputPublicationError`` without changing
        state. For a current first publication, the store first calls
        ``validate_output_batch_size``. It then derives ``OutputRef`` values,
        sums their canonical routing size and calls
        ``validate_output_batch_routing_size`` before it writes any output.
        Materialised data is not included in this byte limit. Stale, missing
        and invalidated attempts return ``None`` without changing state.
        """
        ...

    def mark_task_failed(
        self,
        task_id: str,
        error: str,
        *,
        expected_attempt_id: str,
    ) -> TaskFailureResult | None:
        """The current task attempt moves to ``FAILED``.

        The attempt comparison and state change share a transaction. A
        redelivery of the same failed attempt returns its committed result.
        Stale, missing and invalidated attempts return ``None`` without a
        change.
        """
        ...

    def mark_task_crashed(
        self,
        task_id: str,
        error: str,
        *,
        expected_attempt_id: str,
    ) -> TaskFailureResult | None:
        """The current task attempt moves to ``CRASHED``.

        The attempt comparison and state change share a transaction. A
        redelivery of the same crashed attempt returns its committed result.
        Stale, missing and invalidated attempts return ``None`` without a
        change.
        """
        ...

    def try_finalize_run(
        self,
        run_id: str,
        status: RunStatus,
        *,
        cause: TaskAttemptRef | NodeLaunchRef | None = None,
        reason: str | None = None,
    ) -> bool:
        """The terminal transition is validated and applied in one transaction.

        A task or launch cause must still be current in the active generation.
        Its persisted status and error determine the transition; a caller
        cannot replace that error on redelivery. Free-form errors are accepted
        only for an external abort. Completion requires active launch keys for
        every stored node index, complete active launches, no failed active
        launch and successful completion of every task owned by those
        launches. Invalidated history does not count.

        For failure, crash or cancellation, the transaction records launches
        that had settled before cancellation, cancels unfinished active work,
        updates progress and changes run status. Missing and terminal runs
        return ``False`` without a change.
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
    def try_create_task_retry(self, task_id: str, *, expected_attempt_id: str) -> str | None:
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
    ) -> RunReopenResult | None:
        """A terminal run and its affected node generations are reopened atomically.

        The store acquires the run lock before reading status, topology,
        launches, tasks or generations. It derives every unsuccessful active
        launch and its descendants. A supplied boundary is an additional
        assertion permitted only for failed and crashed runs; each supplied
        node must still contain an active planning or task failure. Omitting it
        lets the store derive the boundary for any retryable terminal status.

        Every affected node generation is invalidated as a unit, including
        successful sibling plans and outputs. Reopening, retry-count increment,
        generation changes, invalidation, progress and global snapshot
        revision share one transaction. Invalidated output IDs remain reserved
        for the lifetime of the run and no longer resolve. Success returns the
        exact affected nodes, generations and retry count. Rejection returns
        ``None`` without changing state.
        """
        ...
