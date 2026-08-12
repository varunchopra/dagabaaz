"""Protocols for persistence and output resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from dagabaaz.constants import (
    MAX_OUTPUTS_PER_TASK_COMPLETION,
    MAX_TASK_COMPLETION_ROUTING_BYTES,
    MAX_WAIT_ID_LENGTH,
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


def _validate_non_empty_id(value: object, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")


def _validate_generation(value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("generation must be a non-negative integer")


def validate_wait_id(wait_id: object) -> None:
    """Keep wait keys bounded and safe for database text columns."""

    if not isinstance(wait_id, str) or not wait_id:
        raise ValueError("wait ID must be a non-empty string")
    if len(wait_id) > MAX_WAIT_ID_LENGTH:
        raise ValueError(f"wait ID exceeds the {MAX_WAIT_ID_LENGTH}-character limit")
    if "\x00" in wait_id:
        raise ValueError("wait ID must not contain NUL characters")


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
class TaskClaimResult:
    """Context available only to the delivery that won the claim."""

    task_id: str
    attempt_id: str
    run_id: str
    node_index: int
    generation: int
    resumed_from_wait_id: str | None

    def __post_init__(self) -> None:
        _validate_non_empty_id(self.task_id, "task ID")
        _validate_non_empty_id(self.attempt_id, "attempt ID")
        _validate_non_empty_id(self.run_id, "run ID")
        if type(self.node_index) is not int or self.node_index < 0:
            raise ValueError("node index must be a non-negative integer")
        _validate_generation(self.generation)
        if self.resumed_from_wait_id is not None:
            validate_wait_id(self.resumed_from_wait_id)


@dataclass(frozen=True, slots=True)
class TaskRecoveryResult:
    """A stable record of an abandoned attempt and its only replacement."""

    task_id: str
    abandoned_attempt_id: str
    recovered_attempt_id: str
    run_id: str
    node_index: int
    generation: int
    resumed_from_wait_id: str | None

    def __post_init__(self) -> None:
        _validate_non_empty_id(self.task_id, "task ID")
        _validate_non_empty_id(self.abandoned_attempt_id, "abandoned attempt ID")
        _validate_non_empty_id(self.recovered_attempt_id, "recovered attempt ID")
        if self.abandoned_attempt_id == self.recovered_attempt_id:
            raise ValueError("a recovered attempt must have a new attempt ID")
        _validate_non_empty_id(self.run_id, "run ID")
        if type(self.node_index) is not int or self.node_index < 0:
            raise ValueError("node index must be a non-negative integer")
        _validate_generation(self.generation)
        if self.resumed_from_wait_id is not None:
            validate_wait_id(self.resumed_from_wait_id)


@dataclass(frozen=True, slots=True)
class TaskWaitResult:
    """A stable record linking one execution attempt to one wait."""

    task_id: str
    attempt_id: str
    run_id: str
    node_index: int
    generation: int
    wait_id: str

    def __post_init__(self) -> None:
        _validate_non_empty_id(self.task_id, "task ID")
        _validate_non_empty_id(self.attempt_id, "attempt ID")
        _validate_non_empty_id(self.run_id, "run ID")
        if type(self.node_index) is not int or self.node_index < 0:
            raise ValueError("node index must be a non-negative integer")
        _validate_generation(self.generation)
        validate_wait_id(self.wait_id)


@dataclass(frozen=True, slots=True)
class TaskResumeResult:
    """A stable record linking one wait to its only replacement attempt."""

    task_id: str
    waiting_attempt_id: str
    resumed_attempt_id: str
    run_id: str
    node_index: int
    generation: int
    wait_id: str

    def __post_init__(self) -> None:
        _validate_non_empty_id(self.task_id, "task ID")
        _validate_non_empty_id(self.waiting_attempt_id, "waiting attempt ID")
        _validate_non_empty_id(self.resumed_attempt_id, "resumed attempt ID")
        if self.waiting_attempt_id == self.resumed_attempt_id:
            raise ValueError("a resumed attempt must have a new attempt ID")
        _validate_non_empty_id(self.run_id, "run ID")
        if type(self.node_index) is not int or self.node_index < 0:
            raise ValueError("node index must be a non-negative integer")
        _validate_generation(self.generation)
        validate_wait_id(self.wait_id)


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

    def try_claim_task(
        self,
        task_id: str,
        *,
        expected_attempt_id: str,
        expected_generation: int,
    ) -> TaskClaimResult | None:
        """Choose the one delivery allowed to execute the current attempt.

        Compare the run, launch, plan, generation, attempt and ``QUEUED`` state
        in one transaction. ``None`` forbids execution but does not prove that
        acknowledging or recovering the delivery is safe.
        """
        ...

    def try_recover_task(
        self,
        task_id: str,
        *,
        expected_attempt_id: str,
        expected_generation: int,
    ) -> TaskRecoveryResult | None:
        """Replace abandoned execution and make its later callbacks stale.

        Return the stored result for an exact repeat, even after later state
        changes. A first recovery accepts only the current ``RUNNING`` attempt
        in the active generation. It keeps the plan and wait correlation, then
        commits the replacement and its outbox row together.

        An expired queue lease does not prove that the old worker stopped. The
        application must establish that repeating any action outside Dagabaaz
        is safe.
        """
        ...

    def try_complete_task(
        self,
        task_id: str,
        outputs: tuple[EmittedOutput, ...],
        *,
        expected_attempt_id: str,
    ) -> TaskCompletionResult | None:
        """Publish outputs only for the attempt that owns a ``RUNNING`` task.

        Check ownership before validating outputs, so a callback from queued,
        waiting or stale work cannot fail the task. For a first completion,
        derive correlation from the stored plan, reserve output IDs for the
        life of the run, enforce the count and routing-byte limits, and commit
        outputs, completion, launch state and progress together. Materialised
        data does not count towards the routing-byte limit. An exact repeat
        returns the first result and cannot replace its outputs. An invalid
        first batch raises ``OutputPublicationError`` before any write.
        """
        ...

    def mark_task_failed(
        self,
        task_id: str,
        error: str,
        *,
        expected_attempt_id: str,
    ) -> TaskFailureResult | None:
        """Accept only the current ``RUNNING`` attempt; replay its first result."""
        ...

    def mark_task_crashed(
        self,
        task_id: str,
        error: str,
        *,
        expected_attempt_id: str,
    ) -> TaskFailureResult | None:
        """Accept only the current ``RUNNING`` attempt; replay its first result."""
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

        Failure, crash and cancellation invalidate active waits before they
        cancel unfinished work. Settled progress and the terminal run state
        commit in the same transaction. Missing and terminal runs return
        ``False`` without a change.
        """
        if status not in RUN_TERMINAL_STATUSES:
            raise ValueError(f"status {status!r} is not terminal")
        ...


@runtime_checkable
class DagWaitStore(DagStore, Protocol):
    """A store that can leave unfinished tasks without dispatch work."""

    def try_wait_task(
        self,
        task_id: str,
        wait_id: str,
        *,
        expected_attempt_id: str,
        expected_generation: int,
    ) -> TaskWaitResult | None:
        """Release a worker without promising another queue delivery.

        Return the stored result for an exact repeat. Otherwise accept only the
        current ``RUNNING`` attempt in the active generation. A task never
        reuses a wait ID. Creating the wait changes no plan or generation and
        creates no attempt or outbox row. The store does not acknowledge the
        delivery that entered the wait.
        """
        ...

    def try_resume_task(
        self,
        task_id: str,
        wait_id: str,
        *,
        expected_attempt_id: str,
        expected_generation: int,
    ) -> TaskResumeResult | None:
        """Create one replacement attempt for an unresolved active wait.

        Commit resolution, the replacement and its outbox row together. An
        exact repeat always returns the first result without creating work,
        even after later state changes. A wait invalidated before resumption
        returns ``None``.
        """
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
        """Replace a crash without losing the response that resumed its attempt.

        Accept only the current crashed attempt while its run is ``RUNNING``.
        Keep its plan, generation and wait correlation, and commit the
        replacement with its outbox row.
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
