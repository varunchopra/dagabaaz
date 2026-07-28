from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from dagabaaz.constants import (
    RUN_RETRYABLE_STATUSES,
    RUN_TERMINAL_STATUSES,
    TASK_TERMINAL_STATUSES,
    LaunchCreateStatus,
    NodeDisposition,
    RunStatus,
    TaskStatus,
)
from dagabaaz.graph import (
    build_slug_to_index_map,
    downstream_closure,
    validate_graph,
)
from dagabaaz.models import (
    DagNode,
    InputEdge,
    LaunchCreateResult,
    NodeLaunch,
    OutputRef,
    PlannedEdgeInput,
    PlanningSnapshot,
    ResolvedOutput,
    TaskContext,
    TaskInputPlan,
)
from dagabaaz.store import RunReopenResult, TaskAttemptRef


class FakeStore:
    """In-memory implementation of the store protocols and their atomic operations."""

    def __init__(self, nodes: list[DagNode], *, run_id: str = "run") -> None:
        self.run_id = run_id
        self.nodes = [node.model_copy(deep=True) for node in nodes]
        self.status = RunStatus.RUNNING
        self.run_error: str | None = None
        self.runtime_inputs: Mapping[str, object] = {}

        self.launches: dict[int, NodeLaunch] = {}
        self.invalidated_launches: list[NodeLaunch] = []
        self.outputs: dict[int, tuple[OutputRef, ...]] = {}
        self.resolved: dict[tuple[str, str], ResolvedOutput] = {}

        self.plans: dict[str, TaskInputPlan] = {}
        self.invalidated_plans: dict[str, TaskInputPlan] = {}
        self.task_contexts: dict[str, TaskContext] = {}
        self.task_statuses: dict[str, TaskStatus] = {}
        self.current_attempts: dict[str, str] = {}
        self.failed_tasks: dict[str, str] = {}

        self.queue_payloads: list[dict[str, object]] = []
        self.retry_payloads: list[dict[str, object]] = []
        self.active_deliveries: dict[str, dict[str, object]] = {}
        self.invalidated_deliveries: dict[str, dict[str, object]] = {}

        self.cancelled = 0
        self.progress = 0
        self.retry_count = 0
        self.generations: dict[int, int] = {}
        self.snapshot_tokens: dict[int, str] = {}
        self.snapshot_generations: dict[str, int] = {}
        self.stale_creations = 0

        self._revision = 0
        self._launch_counter = 0
        self._task_counter = 0
        self._attempt_counter = 0

    def _copy_nodes(self) -> list[DagNode]:
        return [node.model_copy(deep=True) for node in self.nodes]

    def _snapshot_token(self, run_id: str, node_index: int) -> str:
        generation = self.generations.get(node_index, 0)
        return f"{run_id}:{node_index}:{generation}:revision-{self._revision}"

    def _bump_revision(self) -> None:
        self._revision += 1

    def _refresh_launch_completion(self, node_index: int) -> None:
        launch = self.launches.get(node_index)
        if launch is None or not launch.task_ids:
            return
        complete = all(
            self.task_statuses[task_id] == TaskStatus.COMPLETED
            for task_id in launch.task_ids
        )
        if complete != launch.complete:
            self.launches[node_index] = launch.model_copy(update={"complete": complete})

    def _enqueue_attempt(
        self,
        *,
        task_id: str,
        launch_id: str,
        generation: int,
        retry: bool,
    ) -> str:
        self._attempt_counter += 1
        attempt_id = f"attempt-{self._attempt_counter}"
        payload: dict[str, object] = {
            "task_id": task_id,
            "attempt_id": attempt_id,
            "generation": generation,
            "launch_id": launch_id,
        }
        self.current_attempts[task_id] = attempt_id
        self.queue_payloads.append(payload)
        self.active_deliveries[attempt_id] = payload
        if retry:
            self.retry_payloads.append(payload)
        return attempt_id

    def get_run_status(self, run_id: str) -> RunStatus | None:
        return self.status if run_id == self.run_id else None

    def get_run_nodes(self, run_id: str) -> list[DagNode] | None:
        return self._copy_nodes() if run_id == self.run_id else None

    def list_node_launches(self, run_id: str) -> Mapping[int, NodeLaunch]:
        return dict(self.launches) if run_id == self.run_id else {}

    def get_planning_snapshot(
        self, run_id: str, node_index: int, edges: tuple[InputEdge, ...]
    ) -> PlanningSnapshot:
        if run_id != self.run_id:
            raise KeyError(run_id)
        slug_to_index = build_slug_to_index_map(self._copy_nodes())
        token = self._snapshot_token(run_id, node_index)
        self.snapshot_tokens[node_index] = token
        self.snapshot_generations[token] = self.generations.get(node_index, 0)
        outputs: dict[str, tuple[OutputRef, ...]] = {}
        dispositions: dict[str, NodeDisposition] = {}
        for edge in edges:
            source_index = slug_to_index[edge.source]
            source_launch = self.launches[source_index]
            outputs[edge.name] = self.outputs.get(source_index, ())
            dispositions[edge.name] = source_launch.disposition
        return PlanningSnapshot(
            token=token,
            generation=self.generations.get(node_index, 0),
            outputs_by_edge=outputs,
            source_dispositions=dispositions,
            runtime_inputs=self.runtime_inputs,
        )

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
        if run_id != self.run_id or self.status in RUN_TERMINAL_STATUSES:
            return LaunchCreateResult(status=LaunchCreateStatus.STALE)
        generation = self.generations.get(node_index, 0)
        if self.snapshot_generations.get(snapshot_token) != generation:
            return LaunchCreateResult(status=LaunchCreateStatus.STALE)
        existing = self.launches.get(node_index)
        if existing is not None:
            if existing.generation != generation:
                return LaunchCreateResult(status=LaunchCreateStatus.STALE)
            return LaunchCreateResult(
                status=LaunchCreateStatus.ALREADY_EXISTS,
                launch=existing,
            )
        if self.stale_creations:
            self.stale_creations -= 1
            return LaunchCreateResult(status=LaunchCreateStatus.STALE)
        if snapshot_token != self._snapshot_token(run_id, node_index):
            return LaunchCreateResult(status=LaunchCreateStatus.STALE)

        if plans:
            if zero_task_disposition is not None:
                raise ValueError("a task launch cannot have a zero-task disposition")
            if any(plan.generation != generation for plan in plans):
                raise ValueError("task plan generation does not match the launch generation")
            disposition = NodeDisposition.LAUNCHED
        else:
            if zero_task_disposition not in {
                NodeDisposition.SKIPPED,
                NodeDisposition.FILTERED,
                NodeDisposition.FAILED,
            }:
                raise ValueError("a zero-task launch requires a zero-task disposition")
            if zero_task_disposition == NodeDisposition.FAILED and not error:
                raise ValueError("a failed launch requires an error")
            disposition = zero_task_disposition

        self._launch_counter += 1
        launch_id = f"launch-{self._launch_counter}"
        task_ids: list[str] = []
        for plan in plans:
            self._task_counter += 1
            task_id = f"task-{self._task_counter}"
            task_ids.append(task_id)
            self.plans[task_id] = plan
            self.task_contexts[task_id] = TaskContext(
                run_id=run_id,
                node_index=node_index,
                generation=generation,
            )
            self.task_statuses[task_id] = TaskStatus.QUEUED
            self._enqueue_attempt(
                task_id=task_id,
                launch_id=launch_id,
                generation=generation,
                retry=False,
            )
        launch = NodeLaunch(
            id=launch_id,
            run_id=run_id,
            node_index=node_index,
            plugin_name=plugin_name,
            generation=generation,
            disposition=disposition,
            task_ids=tuple(task_ids),
            complete=not plans,
            error=error,
        )
        self.launches[node_index] = launch
        self._bump_revision()
        return LaunchCreateResult(status=LaunchCreateStatus.CREATED, launch=launch)

    def seed_launch(
        self,
        node_index: int,
        *,
        plans: tuple[TaskInputPlan, ...] | None = None,
        disposition: NodeDisposition | None = None,
        error: str = "",
    ) -> NodeLaunch:
        node = self.nodes[node_index]
        generation = self.generations.get(node_index, 0)
        slug_to_index = build_slug_to_index_map(self._copy_nodes())
        default_plan = TaskInputPlan(
            generation=generation,
            edges=tuple(
                PlannedEdgeInput(
                    edge=edge.name,
                    source_index=slug_to_index[edge.source],
                    role=edge.role,
                )
                for edge in node.edges
            ),
        )
        launch_plans = plans if plans is not None else (default_plan,)
        snapshot = self.get_planning_snapshot(self.run_id, node_index, node.edges)
        result = self.try_create_node_launch(
            self.run_id,
            node_index,
            node.plugin,
            snapshot.token,
            launch_plans,
            disposition,
            error,
        )
        assert result.launch is not None
        return result.launch

    def complete_task(
        self,
        task_id: str,
        *outputs: OutputRef,
        expected_attempt_id: str,
    ) -> bool:
        context = self.task_contexts.get(task_id)
        if context is None or context.run_id != self.run_id:
            return False
        output_ids = [output.id for output in outputs]
        if len(output_ids) != len(set(output_ids)) or any(
            (self.run_id, output_id) in self.resolved for output_id in output_ids
        ):
            return False
        launch = self.launches.get(context.node_index)
        if (
            launch is None
            or launch.generation != context.generation
            or self.current_attempts.get(task_id) != expected_attempt_id
            or self.status in RUN_TERMINAL_STATUSES
            or self.task_statuses.get(task_id) in TASK_TERMINAL_STATUSES
        ):
            return False

        self.task_statuses[task_id] = TaskStatus.COMPLETED
        if outputs:
            combined = self.outputs.get(context.node_index, ()) + tuple(outputs)
            self.outputs[context.node_index] = combined
            for output in outputs:
                self.resolved[(self.run_id, output.id)] = ResolvedOutput(
                    id=output.id,
                    fields=output.fields,
                    data={"value": output.id},
                )
        self._refresh_launch_completion(context.node_index)
        self._bump_revision()
        return True

    def get_task_context(self, task_id: str) -> TaskContext | None:
        return self.task_contexts.get(task_id)

    def _mark_task_terminal(
        self,
        task_id: str,
        *,
        status: TaskStatus,
        error: str,
        expected_attempt_id: str,
    ) -> str | None:
        context = self.task_contexts.get(task_id)
        launch = self.launches.get(context.node_index) if context is not None else None
        if (
            context is None
            or context.run_id != self.run_id
            or launch is None
            or launch.generation != context.generation
            or self.current_attempts.get(task_id) != expected_attempt_id
            or self.status in RUN_TERMINAL_STATUSES
            or self.task_statuses.get(task_id) in TASK_TERMINAL_STATUSES
        ):
            return None
        self.failed_tasks[task_id] = error
        self.task_statuses[task_id] = status
        self._refresh_launch_completion(context.node_index)
        self._bump_revision()
        return self.run_id

    def mark_task_failed(
        self,
        task_id: str,
        error: str,
        *,
        expected_attempt_id: str,
    ) -> str | None:
        return self._mark_task_terminal(
            task_id,
            status=TaskStatus.FAILED,
            error=error,
            expected_attempt_id=expected_attempt_id,
        )

    def mark_task_crashed(
        self,
        task_id: str,
        error: str,
        *,
        expected_attempt_id: str,
    ) -> str | None:
        return self._mark_task_terminal(
            task_id,
            status=TaskStatus.CRASHED,
            error=error,
            expected_attempt_id=expected_attempt_id,
        )

    def try_finalize_run(
        self,
        run_id: str,
        status: RunStatus,
        error: str | None,
        *,
        cause: TaskAttemptRef | None = None,
    ) -> bool:
        if status not in RUN_TERMINAL_STATUSES:
            raise ValueError(f"status {status!r} is not terminal")
        if run_id != self.run_id or self.status in RUN_TERMINAL_STATUSES:
            return False
        if cause is not None:
            context = self.task_contexts.get(cause.task_id)
            launch = (
                self.launches.get(context.node_index)
                if context is not None
                else None
            )
            expected_task_status = {
                RunStatus.FAILED: TaskStatus.FAILED,
                RunStatus.CRASHED: TaskStatus.CRASHED,
            }.get(status)
            if (
                context is None
                or launch is None
                or launch.generation != context.generation
                or self.current_attempts.get(cause.task_id) != cause.attempt_id
                or expected_task_status is None
                or self.task_statuses.get(cause.task_id) != expected_task_status
            ):
                return False

        active_task_ids = {
            task_id for launch in self.launches.values() for task_id in launch.task_ids
        }
        if status == RunStatus.COMPLETED:
            if (
                set(self.launches) != set(range(len(self.nodes)))
                or any(not launch.complete for launch in self.launches.values())
                or any(
                    launch.disposition == NodeDisposition.FAILED
                    for launch in self.launches.values()
                )
                or any(
                    self.task_statuses[task_id] != TaskStatus.COMPLETED
                    for task_id in active_task_ids
                )
            ):
                return False
        else:
            for task_id in active_task_ids:
                if self.task_statuses[task_id] not in TASK_TERMINAL_STATUSES:
                    self.task_statuses[task_id] = TaskStatus.CANCELLED
                    self.cancelled += 1
            for node_index in tuple(self.launches):
                self._refresh_launch_completion(node_index)

        self.progress = sum(launch.complete for launch in self.launches.values())
        self.status = status
        self.run_error = error
        self._bump_revision()
        return True

    def resolve_outputs(
        self, run_id: str, output_ids: tuple[str, ...]
    ) -> Mapping[str, ResolvedOutput]:
        if run_id != self.run_id:
            return {}
        return {
            output_id: self.resolved[(run_id, output_id)]
            for output_id in output_ids
            if (run_id, output_id) in self.resolved
        }

    def try_create_task_retry(
        self,
        task_id: str,
        *,
        expected_attempt_id: str,
    ) -> str | None:
        context = self.task_contexts.get(task_id)
        if (
            context is None
            or task_id not in self.plans
            or self.status != RunStatus.RUNNING
            or self.task_statuses.get(task_id) != TaskStatus.CRASHED
            or self.current_attempts.get(task_id) != expected_attempt_id
        ):
            return None
        launch = self.launches.get(context.node_index)
        if launch is None or launch.generation != context.generation:
            return None

        attempt_id = self._enqueue_attempt(
            task_id=task_id,
            launch_id=launch.id,
            generation=context.generation,
            retry=True,
        )
        self.task_statuses[task_id] = TaskStatus.QUEUED
        self.launches[context.node_index] = launch.model_copy(update={"complete": False})
        self._bump_revision()
        return attempt_id

    def try_reopen_run(
        self,
        run_id: str,
        boundary_indices: tuple[int, ...] | None,
        affected_indices: tuple[int, ...],
    ) -> RunReopenResult | None:
        if run_id != self.run_id or self.status not in RUN_RETRYABLE_STATUSES:
            return None
        if self.status == RunStatus.CANCELLED:
            if boundary_indices is not None:
                return None
        elif not boundary_indices:
            return None

        dependencies = validate_graph(self._copy_nodes())
        restart_roots = {
            index
            for index, launch in self.launches.items()
            if launch.disposition == NodeDisposition.FAILED
            or not launch.complete
            or (
                launch.disposition == NodeDisposition.LAUNCHED
                and any(
                    self.task_statuses.get(task_id) != TaskStatus.COMPLETED
                    for task_id in launch.task_ids
                )
            )
        }
        restart_roots.update(boundary_indices or ())
        expected_affected = tuple(sorted(downstream_closure(dependencies, restart_roots)))
        if tuple(affected_indices) != expected_affected:
            return None
        for index in boundary_indices or ():
            launch = self.launches.get(index)
            if launch is None:
                return None
            failed = launch.disposition == NodeDisposition.FAILED or any(
                self.task_statuses.get(task_id) in {TaskStatus.FAILED, TaskStatus.CRASHED}
                for task_id in launch.task_ids
            )
            if not failed:
                return None

        generations: dict[int, int] = {}
        self.status = RunStatus.RUNNING
        self.run_error = None
        self.retry_count += 1
        for index in affected_indices:
            launch = self.launches.pop(index, None)
            if launch is not None:
                self.invalidated_launches.append(launch)
                for task_id in launch.task_ids:
                    plan = self.plans.pop(task_id, None)
                    if plan is not None:
                        self.invalidated_plans[task_id] = plan
                    if self.task_statuses.get(task_id) not in TASK_TERMINAL_STATUSES:
                        self.task_statuses[task_id] = TaskStatus.CANCELLED
                        self.cancelled += 1
                    for attempt_id, payload in tuple(self.active_deliveries.items()):
                        if payload["task_id"] == task_id:
                            self.invalidated_deliveries[attempt_id] = payload
                            del self.active_deliveries[attempt_id]

            removed_outputs = self.outputs.pop(index, ())
            for output in removed_outputs:
                self.resolved.pop((self.run_id, output.id), None)
            generation = self.generations.get(index, 0) + 1
            self.generations[index] = generation
            generations[index] = generation

        self.progress = sum(launch.complete for launch in self.launches.values())
        self._bump_revision()
        return RunReopenResult(
            retry_count=self.retry_count,
            generations=generations,
        )


@dataclass
class CallbackRecorder:
    completed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    crashed: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)

    def callbacks(self):
        from dagabaaz.orchestrator import OrchestratorCallbacks

        return OrchestratorCallbacks(
            on_run_completed=self.completed.append,
            on_run_failed=self.failed.append,
            on_run_crashed=self.crashed.append,
            on_run_cancelled=self.cancelled.append,
        )
