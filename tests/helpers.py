from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from dagabaaz.constants import (
    MAX_SNAPSHOT_OUTPUTS_PER_EDGE,
    MAX_SNAPSHOT_OUTPUTS_PER_NODE,
    MAX_SNAPSHOT_ROUTING_BYTES,
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
    EmittedOutput,
    LaunchCreateResult,
    NodeLaunch,
    OutputRef,
    PlannedEdgeInput,
    PlanningSnapshot,
    ResolvedOutput,
    TaskContext,
    TaskInputPlan,
    output_ref_routing_size,
)
from dagabaaz.planning import correlation_id_for_output
from dagabaaz.store import (
    NodeLaunchRef,
    OutputPublicationError,
    RunReopenResult,
    TaskAttemptRef,
    TaskCompletionResult,
    TaskFailureResult,
    validate_output_batch_routing_size,
    validate_output_batch_size,
)


class FakeStore:
    """In-memory implementation of the store protocols and their atomic operations."""

    def __init__(self, nodes: list[DagNode], *, run_id: str = "run") -> None:
        self.run_id = run_id
        self.nodes = [node.model_copy(deep=True) for node in nodes]
        self.status = RunStatus.PENDING
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
        self.invalidated_attempts: dict[str, str] = {}
        self.attempt_errors: dict[str, str] = {}
        self.completion_results: dict[str, TaskCompletionResult] = {}
        self.failure_results: dict[str, TaskFailureResult] = {}

        self.queue_payloads: list[dict[str, object]] = []
        self.retry_payloads: list[dict[str, object]] = []
        self.active_deliveries: dict[str, dict[str, object]] = {}
        self.invalidated_deliveries: dict[str, dict[str, object]] = {}

        self.cancelled = 0
        self.progress = 0
        self.retry_count = 0
        self.generations: dict[int, int] = {}
        self._settled_nodes: set[int] = set()
        self._published_output_ids: set[str] = set()
        self.snapshot_tokens: dict[int, str] = {}
        self.snapshot_generations: dict[str, int] = {}
        self.stale_creations = 0
        self.get_run_nodes_calls = 0
        self.list_node_launches_calls = 0

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
            self.task_statuses[task_id] == TaskStatus.COMPLETED for task_id in launch.task_ids
        )
        if complete != launch.complete:
            self.launches[node_index] = launch.model_copy(update={"complete": complete})

    def _record_settled_launch(self, node_index: int) -> None:
        launch = self.launches.get(node_index)
        if launch is None or node_index in self._settled_nodes:
            return
        settled = launch.disposition != NodeDisposition.LAUNCHED or all(
            self.task_statuses[task_id] in TASK_TERMINAL_STATUSES for task_id in launch.task_ids
        )
        if settled:
            self._settled_nodes.add(node_index)
            self.progress = len(self._settled_nodes)

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
        self.get_run_nodes_calls += 1
        return self._copy_nodes() if run_id == self.run_id else None

    def try_start_run(self, run_id: str) -> bool:
        if run_id != self.run_id:
            return False
        if self.status == RunStatus.PENDING:
            self.status = RunStatus.RUNNING
            self._bump_revision()
            return True
        return self.status == RunStatus.RUNNING

    def list_node_launches(self, run_id: str) -> Mapping[int, NodeLaunch]:
        self.list_node_launches_calls += 1
        return dict(self.launches) if run_id == self.run_id else {}

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
        if run_id != self.run_id:
            raise KeyError(run_id)
        if type(max_outputs_per_edge) is not int or max_outputs_per_edge < 0:
            raise ValueError("per-edge output limit must be a non-negative integer")
        if type(max_outputs_total) is not int or max_outputs_total < 0:
            raise ValueError("total output limit must be a non-negative integer")
        if type(max_routing_bytes) is not int or max_routing_bytes < 0:
            raise ValueError("routing byte limit must be a non-negative integer")
        token = self._snapshot_token(run_id, node_index)
        self.snapshot_tokens[node_index] = token
        self.snapshot_generations[token] = self.generations.get(node_index, 0)
        outputs: dict[str, tuple[OutputRef, ...]] = {}
        dispositions: dict[str, NodeDisposition] = {}
        output_count = 0
        routing_size = 0
        overflow = False
        for edge_name, source_index in source_indices.items():
            source_launch = self.launches[source_index]
            dispositions[edge_name] = source_launch.disposition
            if overflow:
                outputs[edge_name] = ()
                continue
            remaining = max_outputs_total + 1 - output_count
            edge_limit = min(max_outputs_per_edge + 1, remaining)
            edge_outputs: list[OutputRef] = []
            for output in self.outputs.get(source_index, ()):
                if len(edge_outputs) == edge_limit:
                    break
                edge_outputs.append(output)
                output_count += 1
                routing_size += output_ref_routing_size(output)
                overflow = (
                    len(edge_outputs) > max_outputs_per_edge
                    or output_count > max_outputs_total
                    or routing_size > max_routing_bytes
                )
                if overflow:
                    break
            outputs[edge_name] = tuple(edge_outputs)
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
        if run_id != self.run_id or self.status != RunStatus.RUNNING:
            return LaunchCreateResult(status=LaunchCreateStatus.STALE)
        generation = self.generations.get(node_index, 0)
        if self.snapshot_generations.get(
            snapshot_token
        ) != generation or snapshot_token != self._snapshot_token(run_id, node_index):
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

        launch_id = f"launch-{self._launch_counter + 1}"
        task_ids = tuple(
            f"task-{self._task_counter + offset}" for offset in range(1, len(plans) + 1)
        )
        contexts = tuple(
            TaskContext(
                run_id=run_id,
                node_index=node_index,
                generation=generation,
            )
            for _plan in plans
        )
        launch = NodeLaunch(
            id=launch_id,
            run_id=run_id,
            node_index=node_index,
            plugin_name=plugin_name,
            generation=generation,
            disposition=disposition,
            task_ids=task_ids,
            complete=not plans,
            error=error,
        )

        self._launch_counter += 1
        self._task_counter += len(plans)
        for task_id, plan, context in zip(task_ids, plans, contexts, strict=True):
            self.plans[task_id] = plan
            self.task_contexts[task_id] = context
            self.task_statuses[task_id] = TaskStatus.QUEUED
            self._enqueue_attempt(
                task_id=task_id,
                launch_id=launch_id,
                generation=generation,
                retry=False,
            )
        self.launches[node_index] = launch
        self._record_settled_launch(node_index)
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
        if self.status == RunStatus.PENDING:
            self.status = RunStatus.RUNNING
        node = self.nodes[node_index]
        generation = self.generations.get(node_index, 0)
        copied_nodes = self._copy_nodes()
        validate_graph(copied_nodes)
        slug_to_index = build_slug_to_index_map(copied_nodes)
        source_indices = {edge.name: slug_to_index[edge.source] for edge in node.edges}
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
        snapshot = self.get_planning_snapshot(
            self.run_id,
            node_index,
            source_indices,
            max_outputs_per_edge=MAX_SNAPSHOT_OUTPUTS_PER_EDGE,
            max_outputs_total=MAX_SNAPSHOT_OUTPUTS_PER_NODE,
            max_routing_bytes=MAX_SNAPSHOT_ROUTING_BYTES,
        )
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

    def try_complete_task(
        self,
        task_id: str,
        outputs: tuple[EmittedOutput, ...],
        *,
        expected_attempt_id: str,
    ) -> TaskCompletionResult | None:
        context = self.task_contexts.get(task_id)
        if context is None or context.run_id != self.run_id:
            return None
        launch = self.launches.get(context.node_index)
        plan = self.plans.get(task_id)
        if (
            launch is None
            or launch.generation != context.generation
            or task_id not in launch.task_ids
            or plan is None
            or plan.generation != context.generation
            or self.current_attempts.get(task_id) != expected_attempt_id
        ):
            return None
        task_status = self.task_statuses.get(task_id)
        if task_status == TaskStatus.COMPLETED:
            return self.completion_results[task_id]
        if self.status != RunStatus.RUNNING or task_status in TASK_TERMINAL_STATUSES:
            return None

        validate_output_batch_size(len(outputs))
        output_ids = [output.id for output in outputs]
        if len(output_ids) != len(set(output_ids)):
            raise OutputPublicationError("a task completion contains duplicate output IDs")
        colliding = sorted(
            output_id for output_id in output_ids if output_id in self._published_output_ids
        )
        if colliding:
            raise OutputPublicationError(f"output IDs already exist in this run: {colliding!r}")

        node = self.nodes[context.node_index]
        routed_outputs: list[OutputRef] = []
        routing_size = 0
        try:
            for output in outputs:
                correlation_id = correlation_id_for_output(
                    node,
                    is_root=not node.edges,
                    task_correlation_id=plan.correlation_id,
                    output_id=output.id,
                )
                routed_output = OutputRef(
                    id=output.id,
                    fields=output.fields,
                    correlation_id=correlation_id,
                )
                routed_outputs.append(routed_output)
                routing_size += output_ref_routing_size(routed_output)
                validate_output_batch_routing_size(routing_size)
        except ValueError as exc:
            raise OutputPublicationError(str(exc)) from exc
        resolved_outputs = [
            ResolvedOutput(
                id=output.id,
                fields=output.fields,
                data=output.data,
            )
            for output in outputs
        ]

        self.task_statuses[task_id] = TaskStatus.COMPLETED
        if routed_outputs:
            combined = self.outputs.get(context.node_index, ()) + tuple(routed_outputs)
            self.outputs[context.node_index] = combined
            for output, resolved in zip(outputs, resolved_outputs, strict=True):
                self._published_output_ids.add(output.id)
                self.resolved[(self.run_id, output.id)] = resolved
        self._refresh_launch_completion(context.node_index)
        self._record_settled_launch(context.node_index)
        self._bump_revision()
        result = TaskCompletionResult(
            task_id=task_id,
            attempt_id=expected_attempt_id,
            run_id=self.run_id,
            node_index=context.node_index,
            generation=context.generation,
            launch_complete=self.launches[context.node_index].complete,
        )
        self.completion_results[task_id] = result
        return result

    def _mark_task_terminal(
        self,
        task_id: str,
        *,
        status: TaskStatus,
        error: str,
        expected_attempt_id: str,
    ) -> TaskFailureResult | None:
        existing = self.failure_results.get(expected_attempt_id)
        if existing is not None:
            launch = self.launches.get(existing.node_index)
            if (
                existing.task_id == task_id
                and existing.status == status
                and launch is not None
                and launch.generation == existing.generation
                and task_id in launch.task_ids
                and self.current_attempts.get(task_id) == expected_attempt_id
                and self.task_statuses.get(task_id) == status
            ):
                return existing
            return None

        context = self.task_contexts.get(task_id)
        launch = self.launches.get(context.node_index) if context is not None else None
        if (
            context is None
            or context.run_id != self.run_id
            or launch is None
            or launch.generation != context.generation
            or task_id not in launch.task_ids
            or self.current_attempts.get(task_id) != expected_attempt_id
            or self.status in RUN_TERMINAL_STATUSES
            or self.task_statuses.get(task_id) in TASK_TERMINAL_STATUSES
        ):
            return None
        self.attempt_errors[expected_attempt_id] = error
        self.task_statuses[task_id] = status
        self._refresh_launch_completion(context.node_index)
        self._record_settled_launch(context.node_index)
        self._bump_revision()
        result = TaskFailureResult(
            task_id=task_id,
            attempt_id=expected_attempt_id,
            run_id=self.run_id,
            node_index=context.node_index,
            generation=context.generation,
            status=status,
        )
        self.failure_results[expected_attempt_id] = result
        return result

    def mark_task_failed(
        self,
        task_id: str,
        error: str,
        *,
        expected_attempt_id: str,
    ) -> TaskFailureResult | None:
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
    ) -> TaskFailureResult | None:
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
        *,
        cause: TaskAttemptRef | NodeLaunchRef | None = None,
        reason: str | None = None,
    ) -> bool:
        if status not in RUN_TERMINAL_STATUSES:
            raise ValueError(f"status {status!r} is not terminal")
        if cause is not None and reason is not None:
            raise ValueError("a terminal transition cannot have both a cause and a reason")
        if status == RunStatus.COMPLETED and (cause is not None or reason is not None):
            raise ValueError("run completion cannot have a cause or reason")
        if run_id != self.run_id or self.status in RUN_TERMINAL_STATUSES:
            return False

        error = reason
        if isinstance(cause, TaskAttemptRef):
            failure = self.failure_results.get(cause.attempt_id)
            launch = self.launches.get(failure.node_index) if failure is not None else None
            expected_task_status = {
                RunStatus.FAILED: TaskStatus.FAILED,
                RunStatus.CRASHED: TaskStatus.CRASHED,
            }.get(status)
            if (
                failure is None
                or failure.task_id != cause.task_id
                or failure.run_id != self.run_id
                or launch is None
                or launch.generation != failure.generation
                or cause.task_id not in launch.task_ids
                or self.current_attempts.get(cause.task_id) != cause.attempt_id
                or expected_task_status is None
                or failure.status != expected_task_status
                or self.task_statuses.get(cause.task_id) != expected_task_status
            ):
                return False
            error = self.attempt_errors.get(cause.attempt_id)
            if error is None:
                return False
        elif isinstance(cause, NodeLaunchRef):
            launch = self.launches.get(cause.node_index)
            if (
                status != RunStatus.FAILED
                or launch is None
                or launch.id != cause.launch_id
                or launch.generation != cause.generation
                or launch.disposition != NodeDisposition.FAILED
            ):
                return False
            error = launch.error
        elif cause is not None:
            raise TypeError("unsupported terminal cause")

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
            for node_index in self.launches:
                self._record_settled_launch(node_index)
        else:
            for node_index in self.launches:
                self._record_settled_launch(node_index)
            for task_id in active_task_ids:
                if self.task_statuses[task_id] not in TASK_TERMINAL_STATUSES:
                    self.task_statuses[task_id] = TaskStatus.CANCELLED
                    self.cancelled += 1
            for node_index in tuple(self.launches):
                self._refresh_launch_completion(node_index)

        self.progress = len(self._settled_nodes)
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
            or context.run_id != self.run_id
            or task_id not in self.plans
            or self.status != RunStatus.RUNNING
            or self.task_statuses.get(task_id) != TaskStatus.CRASHED
            or self.current_attempts.get(task_id) != expected_attempt_id
        ):
            return None
        launch = self.launches.get(context.node_index)
        if (
            launch is None
            or launch.generation != context.generation
            or task_id not in launch.task_ids
        ):
            return None

        attempt_id = self._enqueue_attempt(
            task_id=task_id,
            launch_id=launch.id,
            generation=context.generation,
            retry=True,
        )
        self.task_statuses[task_id] = TaskStatus.QUEUED
        self.launches[context.node_index] = launch.model_copy(update={"complete": False})
        self._settled_nodes.discard(context.node_index)
        self.progress = len(self._settled_nodes)
        self._bump_revision()
        return attempt_id

    def try_reopen_run(
        self,
        run_id: str,
        boundary_indices: tuple[int, ...] | None,
    ) -> RunReopenResult | None:
        if run_id != self.run_id or self.status not in RUN_RETRYABLE_STATUSES:
            return None
        if boundary_indices is not None and (
            not boundary_indices
            or any(type(index) is not int or index < 0 for index in boundary_indices)
        ):
            return None
        if boundary_indices is not None and self.status not in {
            RunStatus.FAILED,
            RunStatus.CRASHED,
        }:
            return None

        dependencies = validate_graph(self._copy_nodes())
        restart_roots = {
            index
            for index, launch in self.launches.items()
            if launch.disposition == NodeDisposition.FAILED
            or (
                launch.disposition == NodeDisposition.LAUNCHED
                and any(
                    self.task_statuses.get(task_id) != TaskStatus.COMPLETED
                    for task_id in launch.task_ids
                )
            )
        }
        invalid_roots = sorted(
            index for index in restart_roots if index < 0 or index >= len(self.nodes)
        )
        if invalid_roots:
            return None
        for index in boundary_indices or ():
            if index < 0 or index >= len(self.nodes):
                return None
            launch = self.launches.get(index)
            if launch is None:
                return None
            failed = launch.disposition == NodeDisposition.FAILED or any(
                self.task_statuses.get(task_id) in {TaskStatus.FAILED, TaskStatus.CRASHED}
                for task_id in launch.task_ids
            )
            if not failed:
                return None

        affected_indices = tuple(sorted(downstream_closure(dependencies, restart_roots)))
        generations: dict[int, int] = {}
        for index in affected_indices:
            launch = self.launches.pop(index, None)
            if launch is not None:
                self.invalidated_launches.append(launch)
                for task_id in launch.task_ids:
                    plan = self.plans.pop(task_id, None)
                    if plan is not None:
                        self.invalidated_plans[task_id] = plan
                    attempt_id = self.current_attempts.pop(task_id, None)
                    if attempt_id is not None:
                        self.invalidated_attempts[task_id] = attempt_id
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

        self._settled_nodes.difference_update(affected_indices)
        self.progress = len(self._settled_nodes)
        self.status = RunStatus.RUNNING
        self.run_error = None
        self.retry_count += 1
        self._bump_revision()
        return RunReopenResult(
            affected_nodes=affected_indices,
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
