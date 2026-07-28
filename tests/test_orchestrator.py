import pytest

import dagabaaz.models as model_module
import dagabaaz.planning as planning
from dagabaaz.constants import (
    InputMode,
    LaunchCreateStatus,
    NodeDisposition,
    RunStatus,
    TaskStatus,
)
from dagabaaz.models import (
    DagNode,
    InputEdge,
    LaunchCreateResult,
    NodeLaunch,
    OutputRef,
    TaskInputPlan,
)
from dagabaaz.orchestrator import (
    OrchestratorCallbacks,
    _validate_launch_result,
    abort_run,
    on_task_complete,
    on_task_crashed,
    on_task_failed,
    reconcile_run,
    start_run,
)
from dagabaaz.plugins import PluginInputMeta
from dagabaaz.retry import retry_run, retry_task
from dagabaaz.schema import get_pipeline_input_schema, merge_run_input
from dagabaaz.store import (
    DagRetryStore,
    DagStore,
    OutputResolver,
    StoreContractError,
    TaskAttemptRef,
)
from dagabaaz.task_input import resolve_task_inputs

from .helpers import CallbackRecorder, FakeStore


def linear_nodes() -> list[DagNode]:
    return [
        DagNode(slug="source", plugin="source"),
        DagNode(
            slug="child",
            plugin="child",
            input_mode=InputMode.EACH,
            edges=(InputEdge(name="source", source="source"),),
        ),
    ]


def test_fake_store_implements_the_public_store_protocols() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])

    assert isinstance(store, DagStore)
    assert isinstance(store, DagRetryStore)
    assert isinstance(store, OutputResolver)


def test_start_and_completion_reconcile_exact_plans() -> None:
    store = FakeStore(linear_nodes())
    recorder = CallbackRecorder()
    roots = start_run(store, "run", callbacks=recorder.callbacks())
    assert roots == [0]
    assert store.launches[0].disposition == NodeDisposition.LAUNCHED
    root_task = store.launches[0].task_ids[0]

    store.complete_task(
        root_task,
        OutputRef(id="two", fields={"title": "Two"}, correlation_id="c2"),
        OutputRef(id="one", fields={"title": "One"}, correlation_id="c1"),
        expected_attempt_id=store.current_attempts[root_task],
    )
    on_task_complete(store, task_id=root_task, callbacks=recorder.callbacks())
    child_tasks = store.launches[1].task_ids
    assert len(child_tasks) == 2
    assert [store.plans[task].edges[0].output_ids[0] for task in child_tasks] == [
        "two",
        "one",
    ]

    for task_id in child_tasks:
        store.complete_task(
            task_id,
            expected_attempt_id=store.current_attempts[task_id],
        )
    on_task_complete(store, task_id=child_tasks[-1], callbacks=recorder.callbacks())
    assert store.status == RunStatus.COMPLETED
    assert store.progress == 2
    assert recorder.completed == ["run"]


def test_schema_declared_runtime_input_reaches_a_root_worker_plan() -> None:
    class SourcePlugin:
        def get_effective_inputs(self) -> list[PluginInputMeta]:
            return [
                PluginInputMeta(
                    name="url",
                    description="URL",
                    source="runtime",
                    required=True,
                )
            ]

    nodes = [DagNode(slug="source", plugin="source")]
    schema = get_pipeline_input_schema(nodes, lambda _name: SourcePlugin())
    merged = merge_run_input(
        schema,
        {"options": {"attempts": [1, 2]}},
        {
            "url": "https://example.test/input",
            "enabled": False,
            "limit": 0,
            "extra": "kept",
        },
    )
    store = FakeStore(nodes)
    store.runtime_inputs = merged

    start_run(store, "run", callbacks=CallbackRecorder().callbacks())
    task_id = store.launches[0].task_ids[0]
    inputs = resolve_task_inputs(store, run_id="run", plan=store.plans[task_id])

    assert inputs.parameters == {
        "options": {"attempts": (1, 2)},
        "url": "https://example.test/input",
        "enabled": False,
        "limit": 0,
        "extra": "kept",
    }


def test_start_uses_the_stored_topology_and_rejects_a_missing_one() -> None:
    caller_definition = linear_nodes()
    store = FakeStore(caller_definition)
    caller_definition.clear()
    assert start_run(store, "run", callbacks=CallbackRecorder().callbacks()) == [0]

    class MissingTopologyStore(FakeStore):
        def get_run_nodes(self, run_id: str) -> list[DagNode] | None:
            return None

    missing = MissingTopologyStore(linear_nodes())
    with pytest.raises(ValueError, match="no stored node definition"):
        start_run(missing, "run", callbacks=CallbackRecorder().callbacks())


def test_multiple_roots_launch_once() -> None:
    store = FakeStore(
        [
            DagNode(slug="left", plugin="p"),
            DagNode(slug="right", plugin="p"),
        ]
    )
    store.runtime_inputs = {"shared": {"enabled": True}}
    callbacks = CallbackRecorder().callbacks()
    assert start_run(store, "run", callbacks=callbacks) == [0, 1]
    assert set(store.launches) == {0, 1}
    assert len(store.queue_payloads) == 2
    assert [
        store.plans[task_id].parameters
        for launch in store.launches.values()
        for task_id in launch.task_ids
    ] == [
        {"shared": {"enabled": True}},
        {"shared": {"enabled": True}},
    ]

    reconcile_run(store, "run", callbacks=callbacks)
    assert len(store.queue_payloads) == 2


def test_partial_barrier_waits_for_every_source() -> None:
    nodes = [
        DagNode(slug="left", plugin="p"),
        DagNode(slug="right", plugin="p"),
        DagNode(
            slug="join",
            plugin="p",
            edges=(
                InputEdge(name="left", source="left"),
                InputEdge(name="right", source="right"),
            ),
        ),
    ]
    store = FakeStore(nodes)
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())

    left_task = store.launches[0].task_ids[0]
    store.complete_task(
        left_task,
        OutputRef(id="left-output"),
        expected_attempt_id=store.current_attempts[left_task],
    )
    on_task_complete(store, task_id=left_task, callbacks=recorder.callbacks())
    assert 2 not in store.launches

    right_task = store.launches[1].task_ids[0]
    store.complete_task(
        right_task,
        OutputRef(id="right-output"),
        expected_attempt_id=store.current_attempts[right_task],
    )
    on_task_complete(store, task_id=right_task, callbacks=recorder.callbacks())
    assert store.launches[2].disposition == NodeDisposition.LAUNCHED


def test_reconcile_launches_a_node_that_was_ready_before_the_call() -> None:
    store = FakeStore(linear_nodes())
    root = store.seed_launch(0)
    root_task = root.task_ids[0]
    store.complete_task(
        root_task,
        OutputRef(id="output", correlation_id="c1"),
        expected_attempt_id=store.current_attempts[root_task],
    )

    reconcile_run(store, "run", callbacks=CallbackRecorder().callbacks())

    assert store.launches[1].disposition == NodeDisposition.LAUNCHED


def test_zero_task_dispositions_complete_and_unlock_downstream() -> None:
    nodes = [
        DagNode(slug="source", plugin="source"),
        DagNode(
            slug="filtered",
            plugin="filtered",
            input_mode=InputMode.EACH,
            edges=(InputEdge(name="source", source="source"),),
        ),
        DagNode(
            slug="optional",
            plugin="optional",
            edges=(InputEdge(name="filtered", source="filtered", required=False),),
        ),
    ]
    store = FakeStore(nodes)
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    root_task = store.launches[0].task_ids[0]
    store.complete_task(
        root_task,
        expected_attempt_id=store.current_attempts[root_task],
    )
    on_task_complete(store, task_id=root_task, callbacks=recorder.callbacks())
    assert store.launches[1].disposition == NodeDisposition.FILTERED
    assert store.launches[1].task_ids == ()
    assert store.launches[2].disposition == NodeDisposition.LAUNCHED
    assert len(store.launches[2].task_ids) == 1


def test_required_skipped_source_cascades_but_optional_does_not() -> None:
    nodes = [
        DagNode(slug="skipped", plugin="source"),
        DagNode(
            slug="required",
            plugin="required",
            input_mode=InputMode.EACH,
            edges=(InputEdge(name="source", source="skipped"),),
        ),
        DagNode(
            slug="optional",
            plugin="optional",
            edges=(InputEdge(name="source", source="skipped", required=False),),
        ),
    ]
    store = FakeStore(nodes)
    store.seed_launch(0, plans=(), disposition=NodeDisposition.SKIPPED)
    reconcile_run(store, "run", callbacks=CallbackRecorder().callbacks())
    assert store.launches[1].disposition == NodeDisposition.SKIPPED
    assert store.launches[2].disposition == NodeDisposition.LAUNCHED


def test_stale_snapshot_is_retried_without_duplicate_launch() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    store.stale_creations = 1
    reconcile_run(store, "run", callbacks=CallbackRecorder().callbacks())
    assert len(store.launches) == 1
    assert len(store.queue_payloads) == 1


def test_existing_launch_from_a_newer_generation_retries_the_snapshot() -> None:
    class CrossGenerationStore(FakeStore):
        crossed_generation = False

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
            if not self.crossed_generation:
                self.crossed_generation = True
                self.generations[node_index] = 1
                self._bump_revision()
                fresh = self.get_planning_snapshot(
                    run_id,
                    node_index,
                    self.nodes[node_index].edges,
                )
                current_plans = tuple(
                    plan.model_copy(update={"generation": 1}) for plan in plans
                )
                created = FakeStore.try_create_node_launch(
                    self,
                    run_id,
                    node_index,
                    plugin_name,
                    fresh.token,
                    current_plans,
                    zero_task_disposition,
                    error,
                )
                assert created.launch is not None
                return LaunchCreateResult(
                    status=LaunchCreateStatus.ALREADY_EXISTS,
                    launch=created.launch,
                )
            return super().try_create_node_launch(
                run_id,
                node_index,
                plugin_name,
                snapshot_token,
                plans,
                zero_task_disposition,
                error,
            )

    store = CrossGenerationStore([DagNode(slug="root", plugin="p")])
    reconcile_run(store, "run", callbacks=CallbackRecorder().callbacks())

    assert store.launches[0].generation == 1
    assert len(store.queue_payloads) == 1


def test_output_completion_invalidates_an_older_snapshot() -> None:
    store = FakeStore(linear_nodes())
    root = store.seed_launch(0)
    stale = store.get_planning_snapshot("run", 1, store.nodes[1].edges)
    root_task = root.task_ids[0]
    store.complete_task(
        root_task,
        OutputRef(id="new-output"),
        expected_attempt_id=store.current_attempts[root_task],
    )

    result = store.try_create_node_launch(
        "run",
        1,
        "child",
        stale.token,
        (TaskInputPlan(generation=0),),
        None,
    )
    assert result.status == LaunchCreateStatus.STALE
    assert 1 not in store.launches


def test_task_completion_rejects_duplicate_output_ids_atomically() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    launch = store.seed_launch(0)
    task_id = launch.task_ids[0]

    assert not store.complete_task(
        task_id,
        OutputRef(id="duplicate", fields={"value": 1}),
        OutputRef(id="duplicate", fields={"value": 2}),
        expected_attempt_id=store.current_attempts[task_id],
    )
    assert store.task_statuses[task_id] == TaskStatus.QUEUED
    assert store.outputs == {}
    assert store.resolved == {}


def test_terminal_run_rejects_launch_creation_after_snapshot() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    snapshot = store.get_planning_snapshot("run", 0, ())
    assert store.try_finalize_run("run", RunStatus.CANCELLED, "cancelled")

    result = store.try_create_node_launch(
        "run",
        0,
        "p",
        snapshot.token,
        (TaskInputPlan(generation=0),),
        None,
    )
    assert result.status == LaunchCreateStatus.STALE
    assert store.launches == {}
    assert store.queue_payloads == []


def test_already_existing_launch_from_a_race_is_accepted_once() -> None:
    class ConcurrentLaunchStore(FakeStore):
        def try_create_node_launch(self, *args, **kwargs) -> LaunchCreateResult:
            result = super().try_create_node_launch(*args, **kwargs)
            if result.status == LaunchCreateStatus.CREATED:
                return LaunchCreateResult(
                    status=LaunchCreateStatus.ALREADY_EXISTS,
                    launch=result.launch,
                )
            return result

    store = ConcurrentLaunchStore([DagNode(slug="root", plugin="p")])
    reconcile_run(store, "run", callbacks=CallbackRecorder().callbacks())
    assert len(store.launches) == 1
    assert len(store.queue_payloads) == 1


def test_created_task_launch_cannot_report_completion() -> None:
    class PrematureCompletionStore(FakeStore):
        def try_create_node_launch(self, *args, **kwargs) -> LaunchCreateResult:
            result = super().try_create_node_launch(*args, **kwargs)
            if result.status != LaunchCreateStatus.CREATED or result.launch is None:
                return result
            completed = result.launch.model_copy(update={"complete": True})
            self.launches[completed.node_index] = completed
            return LaunchCreateResult(
                status=LaunchCreateStatus.CREATED,
                launch=completed,
            )

    store = PrematureCompletionStore(linear_nodes())
    with pytest.raises(StoreContractError, match="new task launch is complete"):
        reconcile_run(store, "run", callbacks=CallbackRecorder().callbacks())
    assert 1 not in store.launches


def test_completed_existing_launch_can_unlock_its_consumer() -> None:
    class ConcurrentCompletionStore(FakeStore):
        def try_create_node_launch(self, *args, **kwargs) -> LaunchCreateResult:
            result = super().try_create_node_launch(*args, **kwargs)
            if (
                result.status != LaunchCreateStatus.CREATED
                or result.launch is None
                or result.launch.node_index != 0
            ):
                return result
            task_id = result.launch.task_ids[0]
            self.task_statuses[task_id] = TaskStatus.COMPLETED
            completed = result.launch.model_copy(update={"complete": True})
            self.launches[0] = completed
            return LaunchCreateResult(
                status=LaunchCreateStatus.ALREADY_EXISTS,
                launch=completed,
            )

    nodes = [
        DagNode(slug="source", plugin="source"),
        DagNode(
            slug="consumer",
            plugin="consumer",
            edges=(InputEdge(name="source", source="source", required=False),),
        ),
    ]
    store = ConcurrentCompletionStore(nodes)
    reconcile_run(store, "run", callbacks=CallbackRecorder().callbacks())

    assert store.launches[0].complete
    assert store.launches[1].disposition == NodeDisposition.LAUNCHED


def _unvalidated_launch(**updates: object) -> NodeLaunch:
    values: dict[str, object] = {
        "id": "launch",
        "run_id": "run",
        "node_index": 0,
        "plugin_name": "p",
        "generation": 0,
        "disposition": NodeDisposition.LAUNCHED,
        "task_ids": ("task",),
        "complete": False,
        "error": "",
    }
    values.update(updates)
    return NodeLaunch.model_construct(**values)


@pytest.mark.parametrize(
    ("updates", "mismatch"),
    [
        ({"run_id": "other"}, "run_id"),
        ({"node_index": 1}, "node_index"),
        ({"plugin_name": "other"}, "plugin_name"),
        ({"generation": 1}, "generation"),
        ({"disposition": NodeDisposition.FILTERED}, "disposition"),
        ({"task_ids": ()}, "task_ids"),
        ({"task_ids": ("",)}, "empty task_id"),
    ],
)
def test_launch_creation_result_must_match_the_request(
    updates: dict[str, object],
    mismatch: str,
) -> None:
    with pytest.raises(StoreContractError, match=mismatch):
        _validate_launch_result(
            status=LaunchCreateStatus.CREATED,
            run_id="run",
            node_index=0,
            plugin_name="p",
            generation=0,
            plans=(TaskInputPlan(generation=0),),
            zero_disposition=None,
            launch=_unvalidated_launch(**updates),
        )


def test_launch_creation_result_rejects_duplicate_task_ids() -> None:
    with pytest.raises(StoreContractError, match="duplicate task_ids"):
        _validate_launch_result(
            status=LaunchCreateStatus.CREATED,
            run_id="run",
            node_index=0,
            plugin_name="p",
            generation=0,
            plans=(TaskInputPlan(generation=0), TaskInputPlan(generation=0)),
            zero_disposition=None,
            launch=_unvalidated_launch(task_ids=("task", "task")),
        )


@pytest.mark.parametrize(
    ("launch", "disposition", "mismatch"),
    [
        (
            _unvalidated_launch(
                disposition=NodeDisposition.FILTERED,
                task_ids=(),
                complete=False,
            ),
            NodeDisposition.FILTERED,
            "incomplete zero-task disposition",
        ),
        (
            _unvalidated_launch(
                disposition=NodeDisposition.FAILED,
                task_ids=(),
                complete=True,
                error="",
            ),
            NodeDisposition.FAILED,
            "failed without error",
        ),
    ],
)
def test_zero_task_launch_result_must_be_complete_and_failed_has_an_error(
    launch: NodeLaunch,
    disposition: NodeDisposition,
    mismatch: str,
) -> None:
    with pytest.raises(StoreContractError, match=mismatch):
        _validate_launch_result(
            status=LaunchCreateStatus.CREATED,
            run_id="run",
            node_index=0,
            plugin_name="p",
            generation=0,
            plans=(),
            zero_disposition=disposition,
            launch=launch,
        )


def test_launch_creation_result_requires_a_consistent_payload() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    created = store.seed_launch(
        0,
        plans=(),
        disposition=NodeDisposition.FILTERED,
    )

    with pytest.raises(ValueError, match="must contain a launch"):
        LaunchCreateResult(status=LaunchCreateStatus.CREATED)
    with pytest.raises(ValueError, match="cannot contain a launch"):
        LaunchCreateResult(status=LaunchCreateStatus.STALE, launch=created)


def test_planning_failure_creates_failed_launch_and_fails_run() -> None:
    nodes = [
        DagNode(slug="source", plugin="source"),
        DagNode(
            slug="join",
            plugin="join",
            input_mode=InputMode.BY_CORRELATION,
            edges=(InputEdge(name="source", source="source"),),
        ),
    ]
    store = FakeStore(nodes)
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    task_id = store.launches[0].task_ids[0]
    store.complete_task(
        task_id,
        OutputRef(id="bad", correlation_id=None),
        expected_attempt_id=store.current_attempts[task_id],
    )
    on_task_complete(store, task_id=task_id, callbacks=recorder.callbacks())
    assert store.launches[1].disposition == NodeDisposition.FAILED
    assert store.status == RunStatus.FAILED
    assert recorder.failed == ["run"]


def test_reconcile_reports_the_lowest_failed_node_independent_of_store_order() -> None:
    store = FakeStore(
        [
            DagNode(slug="first", plugin="first"),
            DagNode(slug="second", plugin="second"),
        ]
    )
    store.seed_launch(
        1,
        plans=(),
        disposition=NodeDisposition.FAILED,
        error="second failed",
    )
    store.seed_launch(
        0,
        plans=(),
        disposition=NodeDisposition.FAILED,
        error="first failed",
    )

    reconcile_run(store, "run", callbacks=CallbackRecorder().callbacks())

    assert store.status == RunStatus.FAILED
    assert store.run_error == "first failed"


def test_snapshot_limit_creates_a_retryable_failed_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_module,
        "MAX_SNAPSHOT_OUTPUTS_PER_EDGE",
        1,
        raising=False,
    )
    monkeypatch.setattr(planning, "MAX_SNAPSHOT_OUTPUTS_PER_EDGE", 1)
    nodes = [
        DagNode(slug="source", plugin="source"),
        DagNode(
            slug="consumer",
            plugin="consumer",
            edges=(InputEdge(name="source", source="source"),),
        ),
    ]
    store = FakeStore(nodes)
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    source_task = store.launches[0].task_ids[0]
    store.complete_task(
        source_task,
        OutputRef(id="first"),
        OutputRef(id="second"),
        expected_attempt_id=store.current_attempts[source_task],
    )

    on_task_complete(store, task_id=source_task, callbacks=recorder.callbacks())

    failed_launch = store.launches[1]
    assert failed_launch.disposition == NodeDisposition.FAILED
    assert failed_launch.complete
    assert failed_launch.task_ids == ()
    assert store.status == RunStatus.FAILED
    assert len(store.queue_payloads) == 1
    assert recorder.failed == ["run"]

    retried = retry_run(store, "run", [1])
    assert retried.generations == {1: 1}
    assert store.status == RunStatus.RUNNING
    assert 1 not in store.launches


def test_task_fail_crash_and_abort_callbacks() -> None:
    for handler, expected, callback_field in (
        (on_task_failed, RunStatus.FAILED, "failed"),
        (on_task_crashed, RunStatus.CRASHED, "crashed"),
    ):
        store = FakeStore([DagNode(slug="root", plugin="p")])
        recorder = CallbackRecorder()
        start_run(store, "run", callbacks=recorder.callbacks())
        task_id = store.launches[0].task_ids[0]
        handler(
            store,
            task_id=task_id,
            expected_attempt_id=store.current_attempts[task_id],
            error_message="boom",
            callbacks=recorder.callbacks(),
        )
        assert store.status == expected
        assert getattr(recorder, callback_field) == ["run"]

    store = FakeStore([DagNode(slug="root", plugin="p")])
    recorder = CallbackRecorder()
    assert abort_run(
        store,
        run_id="run",
        reason="stop",
        status=RunStatus.CANCELLED,
        callbacks=recorder.callbacks(),
    )
    assert recorder.cancelled == ["run"]


def test_callback_failure_cannot_strand_other_executing_tasks() -> None:
    store = FakeStore(
        [
            DagNode(slug="left", plugin="p"),
            DagNode(slug="right", plugin="p"),
        ]
    )
    start_run(store, "run", callbacks=CallbackRecorder().callbacks())
    failed_task = store.launches[0].task_ids[0]

    def explode(_run_id: str) -> None:
        raise RuntimeError("callback exploded")

    callbacks = OrchestratorCallbacks(
        on_run_completed=explode,
        on_run_failed=explode,
        on_run_crashed=explode,
        on_run_cancelled=explode,
    )
    with pytest.raises(RuntimeError, match="callback exploded"):
        on_task_failed(
            store,
            task_id=failed_task,
            expected_attempt_id=store.current_attempts[failed_task],
            error_message="boom",
            callbacks=callbacks,
        )

    assert store.status == RunStatus.FAILED
    assert store.cancelled == 1
    assert set(store.task_statuses.values()) <= {
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }
    on_task_failed(
        store,
        task_id=failed_task,
        expected_attempt_id=store.current_attempts[failed_task],
        error_message="boom",
        callbacks=CallbackRecorder().callbacks(),
    )
    assert store.cancelled == 1


def test_task_retry_winning_before_terminalisation_keeps_the_run_active() -> None:
    class RetryRaceStore(FakeStore):
        replacement_attempt: str | None = None

        def try_finalize_run(
            self,
            run_id: str,
            status: RunStatus,
            error: str | None,
            *,
            cause: TaskAttemptRef | None = None,
        ) -> bool:
            if cause is not None and self.replacement_attempt is None:
                self.replacement_attempt = retry_task(
                    self,
                    cause.task_id,
                    expected_attempt_id=cause.attempt_id,
                )
            return super().try_finalize_run(
                run_id,
                status,
                error,
                cause=cause,
            )

    store = RetryRaceStore([DagNode(slug="root", plugin="p")])
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    task_id = store.launches[0].task_ids[0]
    on_task_crashed(
        store,
        task_id=task_id,
        expected_attempt_id=store.current_attempts[task_id],
        error_message="transient",
        callbacks=recorder.callbacks(),
    )

    assert store.replacement_attempt == "attempt-2"
    assert store.task_statuses[task_id] == TaskStatus.QUEUED
    assert store.status == RunStatus.RUNNING
    assert recorder.crashed == []


def test_completion_callback_observes_committed_terminal_state() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    recorder = CallbackRecorder()
    observed: list[tuple[RunStatus, int]] = []
    callbacks = OrchestratorCallbacks(
        on_run_completed=lambda _run_id: observed.append((store.status, store.progress)),
        on_run_failed=recorder.failed.append,
        on_run_crashed=recorder.crashed.append,
        on_run_cancelled=recorder.cancelled.append,
    )
    start_run(store, "run", callbacks=callbacks)
    task_id = store.launches[0].task_ids[0]
    store.complete_task(
        task_id,
        expected_attempt_id=store.current_attempts[task_id],
    )

    on_task_complete(store, task_id=task_id, callbacks=callbacks)

    assert observed == [(RunStatus.COMPLETED, 1)]


def test_completion_terminal_transaction_rechecks_task_state() -> None:
    class CompletionRaceStore(FakeStore):
        lose_once = True

        def try_finalize_run(
            self,
            run_id: str,
            status: RunStatus,
            error: str | None,
            *,
            cause: TaskAttemptRef | None = None,
        ) -> bool:
            if status == RunStatus.COMPLETED and self.lose_once:
                self.lose_once = False
                task_id = self.launches[0].task_ids[0]
                self.task_statuses[task_id] = TaskStatus.QUEUED
                self.launches[0] = self.launches[0].model_copy(
                    update={"complete": False}
                )
            return super().try_finalize_run(
                run_id,
                status,
                error,
                cause=cause,
            )

    store = CompletionRaceStore([DagNode(slug="root", plugin="p")])
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    task_id = store.launches[0].task_ids[0]
    store.complete_task(
        task_id,
        expected_attempt_id=store.current_attempts[task_id],
    )
    on_task_complete(store, task_id=task_id, callbacks=recorder.callbacks())
    assert store.status == RunStatus.RUNNING
    assert recorder.completed == []

    store.complete_task(
        task_id,
        expected_attempt_id=store.current_attempts[task_id],
    )
    on_task_complete(store, task_id=task_id, callbacks=recorder.callbacks())
    assert store.status == RunStatus.COMPLETED
    assert recorder.completed == ["run"]


def test_completion_requires_an_active_launch_for_every_stored_node() -> None:
    store = FakeStore(linear_nodes())
    root = store.seed_launch(0)
    root_task = root.task_ids[0]
    assert store.complete_task(
        root_task,
        expected_attempt_id=store.current_attempts[root_task],
    )

    assert not store.try_finalize_run("run", RunStatus.COMPLETED, None)
    assert store.status == RunStatus.RUNNING


def test_completion_rejects_the_wrong_active_launch_indices() -> None:
    store = FakeStore(linear_nodes())
    root = store.seed_launch(0)
    child = store.seed_launch(1)
    for task_id in (root.task_ids[0], child.task_ids[0]):
        assert store.complete_task(
            task_id,
            expected_attempt_id=store.current_attempts[task_id],
        )
    store.launches[99] = store.launches.pop(1)

    assert not store.try_finalize_run("run", RunStatus.COMPLETED, None)
    assert store.status == RunStatus.RUNNING


def test_abort_returns_the_terminal_claim_result() -> None:
    callbacks = CallbackRecorder().callbacks()
    missing = FakeStore([DagNode(slug="root", plugin="p")])
    assert not abort_run(
        missing,
        run_id="missing",
        reason="not found",
        callbacks=callbacks,
    )

    class ClaimLosingStore(FakeStore):
        def try_finalize_run(
            self,
            run_id: str,
            status: RunStatus,
            error: str | None,
            *,
            cause: TaskAttemptRef | None = None,
        ) -> bool:
            return False

    race = ClaimLosingStore([DagNode(slug="root", plugin="p")])
    assert not abort_run(
        race,
        run_id="run",
        reason="lost race",
        callbacks=callbacks,
    )


def test_fake_output_resolution_is_run_scoped() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    launch = store.seed_launch(0)
    task_id = launch.task_ids[0]
    store.complete_task(
        task_id,
        OutputRef(id="output"),
        expected_attempt_id=store.current_attempts[task_id],
    )
    assert tuple(store.resolve_outputs("run", ("output",))) == ("output",)
    assert store.resolve_outputs("another-run", ("output",)) == {}
