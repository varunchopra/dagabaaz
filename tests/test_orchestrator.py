from collections.abc import Mapping

import pytest

import dagabaaz.models as model_module
import dagabaaz.orchestrator as orchestrator_module
import dagabaaz.planning as planning
import dagabaaz.store as store_module
from dagabaaz.constants import (
    MAX_SNAPSHOT_OUTPUTS_PER_EDGE,
    MAX_SNAPSHOT_OUTPUTS_PER_NODE,
    MAX_SNAPSHOT_ROUTING_BYTES,
    CorrelationMode,
    InputMode,
    LaunchCreateStatus,
    NodeDisposition,
    RunStatus,
    TaskStatus,
)
from dagabaaz.models import (
    DagNode,
    EmittedOutput,
    InputEdge,
    LaunchCreateResult,
    NodeLaunch,
    OutputRef,
    PlanningSnapshot,
    TaskInputPlan,
    output_ref_routing_size,
)
from dagabaaz.orchestrator import (
    OrchestratorCallbacks,
    RunStartRejectedError,
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
    NodeLaunchRef,
    OutputPublicationError,
    OutputResolver,
    StoreContractError,
    TaskAttemptRef,
    TaskCompletionResult,
)
from dagabaaz.task_input import resolve_task_inputs

from . import helpers as helpers_module
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

    on_task_complete(
        store,
        task_id=root_task,
        expected_attempt_id=store.current_attempts[root_task],
        outputs=(
            EmittedOutput(id="two", fields={"title": "Two"}),
            EmittedOutput(id="one", fields={"title": "One"}),
        ),
        callbacks=recorder.callbacks(),
    )
    child_tasks = store.launches[1].task_ids
    assert len(child_tasks) == 2
    assert [store.plans[task].edges[0].output_ids[0] for task in child_tasks] == [
        "two",
        "one",
    ]

    for task_id in child_tasks:
        on_task_complete(
            store,
            task_id=task_id,
            expected_attempt_id=store.current_attempts[task_id],
            callbacks=recorder.callbacks(),
        )
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


def test_explicit_slugs_are_resolved_before_snapshot_planning() -> None:
    store = FakeStore(
        [
            DagNode(slug="source", plugin="source"),
            DagNode(
                slug="consumer",
                plugin="consumer",
                input_mode=InputMode.EACH,
                edges=(InputEdge(name="items", source="source"),),
            ),
        ]
    )
    recorder = CallbackRecorder()

    start_run(store, "run", callbacks=recorder.callbacks())
    source_task = store.launches[0].task_ids[0]
    on_task_complete(
        store,
        task_id=source_task,
        expected_attempt_id=store.current_attempts[source_task],
        outputs=(EmittedOutput(id="item"),),
        callbacks=recorder.callbacks(),
    )

    consumer_task = store.launches[1].task_ids[0]
    assert store.plans[consumer_task].edges[0].source_index == 0
    assert store.plans[consumer_task].edges[0].output_ids == ("item",)


def test_same_run_id_uses_each_store_instance_stored_topology() -> None:
    first = FakeStore([DagNode(slug="one", plugin="p")])
    second = FakeStore(
        [
            DagNode(slug="left", plugin="p"),
            DagNode(slug="right", plugin="p"),
        ]
    )

    assert start_run(first, "run", callbacks=CallbackRecorder().callbacks()) == [0]
    assert start_run(second, "run", callbacks=CallbackRecorder().callbacks()) == [0, 1]
    assert set(first.launches) == {0}
    assert set(second.launches) == {0, 1}


def test_start_validates_before_changing_run_state_or_writing_work() -> None:
    nodes = [
        DagNode(slug="source", plugin="source"),
        DagNode(
            slug="consumer",
            plugin="consumer",
            edges=(InputEdge(name="source", source="source"),),
        ),
    ]
    store = FakeStore(nodes)

    with pytest.raises(ValueError, match="must declare an input mode"):
        start_run(store, "run", callbacks=CallbackRecorder().callbacks())

    assert store.status == RunStatus.PENDING
    assert store.launches == {}
    assert store.plans == {}
    assert store.queue_payloads == []


def test_start_is_idempotent_for_a_running_run() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    callbacks = CallbackRecorder().callbacks()

    assert start_run(store, "run", callbacks=callbacks) == [0]
    assert start_run(store, "run", callbacks=callbacks) == [0]

    assert store.status == RunStatus.RUNNING
    assert len(store.launches) == 1
    assert len(store.queue_payloads) == 1


def test_start_rejects_a_terminal_run_without_writing_work() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    store.status = RunStatus.CANCELLED

    with pytest.raises(RunStartRejectedError, match="cannot be started"):
        start_run(store, "run", callbacks=CallbackRecorder().callbacks())

    assert store.launches == {}
    assert store.plans == {}
    assert store.queue_payloads == []


def test_reconcile_requires_topology_only_for_a_running_run() -> None:
    class MissingTopologyStore(FakeStore):
        def get_run_nodes(self, run_id: str) -> list[DagNode] | None:
            return None

    running = MissingTopologyStore([DagNode(slug="root", plugin="p")])
    running.status = RunStatus.RUNNING
    with pytest.raises(StoreContractError, match="running run.*no stored node"):
        reconcile_run(running, "run", callbacks=CallbackRecorder().callbacks())

    class TopologyMustNotBeReadStore(FakeStore):
        def get_run_nodes(self, run_id: str) -> list[DagNode] | None:
            raise AssertionError("terminal reconciliation read the topology")

    terminal = TopologyMustNotBeReadStore([DagNode(slug="root", plugin="p")])
    terminal.status = RunStatus.FAILED
    reconcile_run(terminal, "run", callbacks=CallbackRecorder().callbacks())


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
            input_mode=InputMode.ALL,
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
    on_task_complete(
        store,
        task_id=left_task,
        expected_attempt_id=store.current_attempts[left_task],
        outputs=(EmittedOutput(id="left-output"),),
        callbacks=recorder.callbacks(),
    )
    assert 2 not in store.launches

    right_task = store.launches[1].task_ids[0]
    on_task_complete(
        store,
        task_id=right_task,
        expected_attempt_id=store.current_attempts[right_task],
        outputs=(EmittedOutput(id="right-output"),),
        callbacks=recorder.callbacks(),
    )
    assert store.launches[2].disposition == NodeDisposition.LAUNCHED


def test_reconcile_launches_a_node_that_was_ready_before_the_call() -> None:
    store = FakeStore(linear_nodes())
    root = store.seed_launch(0)
    root_task = root.task_ids[0]
    assert store.try_complete_task(
        root_task,
        (EmittedOutput(id="output"),),
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
            input_mode=InputMode.ALL,
            edges=(InputEdge(name="filtered", source="filtered", required=False),),
        ),
    ]
    store = FakeStore(nodes)
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    root_task = store.launches[0].task_ids[0]
    on_task_complete(
        store,
        task_id=root_task,
        expected_attempt_id=store.current_attempts[root_task],
        callbacks=recorder.callbacks(),
    )
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
            input_mode=InputMode.ALL,
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
    assert store.try_start_run("run")
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
                    {},
                    max_outputs_per_edge=MAX_SNAPSHOT_OUTPUTS_PER_EDGE,
                    max_outputs_total=MAX_SNAPSHOT_OUTPUTS_PER_NODE,
                    max_routing_bytes=MAX_SNAPSHOT_ROUTING_BYTES,
                )
                current_plans = tuple(plan.model_copy(update={"generation": 1}) for plan in plans)
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
    assert store.try_start_run("run")
    reconcile_run(store, "run", callbacks=CallbackRecorder().callbacks())

    assert store.launches[0].generation == 1
    assert len(store.queue_payloads) == 1


def test_output_completion_invalidates_an_older_snapshot() -> None:
    store = FakeStore(linear_nodes())
    root = store.seed_launch(0)
    stale = store.get_planning_snapshot(
        "run",
        1,
        {"source": 0},
        max_outputs_per_edge=MAX_SNAPSHOT_OUTPUTS_PER_EDGE,
        max_outputs_total=MAX_SNAPSHOT_OUTPUTS_PER_NODE,
        max_routing_bytes=MAX_SNAPSHOT_ROUTING_BYTES,
    )
    root_task = root.task_ids[0]
    assert store.try_complete_task(
        root_task,
        (EmittedOutput(id="new-output"),),
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

    with pytest.raises(OutputPublicationError, match="duplicate output IDs"):
        store.try_complete_task(
            task_id,
            (
                EmittedOutput(id="duplicate", fields={"value": 1}),
                EmittedOutput(id="duplicate", fields={"value": 2}),
            ),
            expected_attempt_id=store.current_attempts[task_id],
        )
    assert store.task_statuses[task_id] == TaskStatus.QUEUED
    assert store.outputs == {}
    assert store.resolved == {}


def test_invalid_output_publication_fails_the_task_and_run() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    task_id = store.launches[0].task_ids[0]

    on_task_complete(
        store,
        task_id=task_id,
        expected_attempt_id=store.current_attempts[task_id],
        outputs=(EmittedOutput(id="same"), EmittedOutput(id="same")),
        callbacks=recorder.callbacks(),
    )

    assert store.task_statuses[task_id] == TaskStatus.FAILED
    assert store.status == RunStatus.FAILED
    assert store.run_error == "a task completion contains duplicate output IDs"
    assert recorder.failed == ["run"]


def test_completion_redelivery_repairs_reconciliation_without_republishing() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    task_id = store.launches[0].task_ids[0]
    attempt_id = store.current_attempts[task_id]
    committed = (EmittedOutput(id="committed", fields={"value": 1}),)

    result = store.try_complete_task(
        task_id,
        committed,
        expected_attempt_id=attempt_id,
    )
    assert result is not None
    assert store.status == RunStatus.RUNNING

    on_task_complete(
        store,
        task_id=task_id,
        expected_attempt_id=attempt_id,
        outputs=(EmittedOutput(id="ignored", fields={"value": 2}),),
        callbacks=recorder.callbacks(),
    )

    assert [output.id for output in store.outputs[0]] == ["committed"]
    assert store.status == RunStatus.COMPLETED
    assert recorder.completed == ["run"]


def test_task_completion_derives_correlation_and_keeps_data_namespaced() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    launch = store.seed_launch(0)
    task_id = launch.task_ids[0]

    result = store.try_complete_task(
        task_id,
        (
            EmittedOutput(
                id="output",
                fields={"path": "routing-path", "value": 1},
                data={"path": "materialised-path", "value": 2},
            ),
        ),
        expected_attempt_id=store.current_attempts[task_id],
    )

    assert result is not None
    assert result.launch_complete
    assert store.outputs[0][0].correlation_id == "output"
    resolved = store.resolve_outputs("run", ("output",))["output"]
    assert resolved.fields == {"path": "routing-path", "value": 1}
    assert resolved.data == {"path": "materialised-path", "value": 2}


def test_output_ids_are_unique_across_tasks_in_a_run() -> None:
    store = FakeStore(linear_nodes())
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    root_task = store.launches[0].task_ids[0]
    on_task_complete(
        store,
        task_id=root_task,
        expected_attempt_id=store.current_attempts[root_task],
        outputs=(EmittedOutput(id="left"), EmittedOutput(id="right")),
        callbacks=recorder.callbacks(),
    )
    first, second = store.launches[1].task_ids
    on_task_complete(
        store,
        task_id=first,
        expected_attempt_id=store.current_attempts[first],
        outputs=(EmittedOutput(id="shared"),),
        callbacks=recorder.callbacks(),
    )
    on_task_complete(
        store,
        task_id=second,
        expected_attempt_id=store.current_attempts[second],
        outputs=(EmittedOutput(id="shared"),),
        callbacks=recorder.callbacks(),
    )

    assert store.task_statuses[first] == TaskStatus.COMPLETED
    assert store.task_statuses[second] == TaskStatus.FAILED
    assert store.status == RunStatus.FAILED
    assert recorder.failed == ["run"]


def test_task_operations_require_active_launch_ownership() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    launch = store.seed_launch(
        0,
        plans=(TaskInputPlan(generation=0), TaskInputPlan(generation=0)),
    )
    detached, active = launch.task_ids
    attempt_id = store.current_attempts[detached]
    store.launches[0] = launch.model_copy(update={"task_ids": (active,)})

    assert (
        store.try_complete_task(
            detached,
            (),
            expected_attempt_id=attempt_id,
        )
        is None
    )
    assert (
        store.mark_task_failed(
            detached,
            "failed",
            expected_attempt_id=attempt_id,
        )
        is None
    )
    store.task_statuses[detached] = TaskStatus.CRASHED
    assert (
        store.try_create_task_retry(
            detached,
            expected_attempt_id=attempt_id,
        )
        is None
    )
    store.task_statuses[detached] = TaskStatus.FAILED
    store.attempt_errors[attempt_id] = "failed"
    assert not store.try_finalize_run(
        "run",
        RunStatus.FAILED,
        cause=TaskAttemptRef(task_id=detached, attempt_id=attempt_id),
    )
    assert store.status == RunStatus.RUNNING


def test_terminal_run_rejects_launch_creation_after_snapshot() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    assert store.try_start_run("run")
    snapshot = store.get_planning_snapshot(
        "run",
        0,
        {},
        max_outputs_per_edge=MAX_SNAPSHOT_OUTPUTS_PER_EDGE,
        max_outputs_total=MAX_SNAPSHOT_OUTPUTS_PER_NODE,
        max_routing_bytes=MAX_SNAPSHOT_ROUTING_BYTES,
    )
    assert store.try_finalize_run(
        "run",
        RunStatus.CANCELLED,
        reason="cancelled",
    )

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


def test_stale_snapshot_precedes_an_existing_launch_result() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    launch = store.seed_launch(0)
    snapshot = store.get_planning_snapshot(
        "run",
        0,
        {},
        max_outputs_per_edge=MAX_SNAPSHOT_OUTPUTS_PER_EDGE,
        max_outputs_total=MAX_SNAPSHOT_OUTPUTS_PER_NODE,
        max_routing_bytes=MAX_SNAPSHOT_ROUTING_BYTES,
    )
    store._bump_revision()

    result = store.try_create_node_launch(
        "run",
        0,
        "p",
        snapshot.token,
        (store.plans[launch.task_ids[0]],),
        None,
    )

    assert result.status == LaunchCreateStatus.STALE


def test_launch_validation_failure_writes_no_partial_task_state() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    assert store.try_start_run("run")
    snapshot = store.get_planning_snapshot(
        "run",
        0,
        {},
        max_outputs_per_edge=MAX_SNAPSHOT_OUTPUTS_PER_EDGE,
        max_outputs_total=MAX_SNAPSHOT_OUTPUTS_PER_NODE,
        max_routing_bytes=MAX_SNAPSHOT_ROUTING_BYTES,
    )

    with pytest.raises(ValueError, match="plugin_name"):
        store.try_create_node_launch(
            "run",
            0,
            "",
            snapshot.token,
            (TaskInputPlan(generation=0),),
            None,
        )

    assert store.launches == {}
    assert store.plans == {}
    assert store.task_contexts == {}
    assert store.current_attempts == {}
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
    assert store.try_start_run("run")
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
    assert store.try_start_run("run")
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
            input_mode=InputMode.ALL,
            edges=(InputEdge(name="source", source="source", required=False),),
        ),
    ]
    store = ConcurrentCompletionStore(nodes)
    assert store.try_start_run("run")
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


def test_reconcile_rejects_an_invalid_launch_creation_result() -> None:
    class InvalidResultStore(FakeStore):
        def try_create_node_launch(self, *args, **kwargs) -> LaunchCreateResult:
            return object()  # type: ignore[return-value]

    store = InvalidResultStore([DagNode(slug="root", plugin="p")])
    assert store.try_start_run("run")

    with pytest.raises(StoreContractError, match="LaunchCreateResult"):
        reconcile_run(store, "run", callbacks=CallbackRecorder().callbacks())


def test_planning_failure_creates_failed_launch_and_fails_run() -> None:
    nodes = [
        DagNode(
            slug="source",
            plugin="source",
            correlation_mode=CorrelationMode.NONE,
        ),
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
    on_task_complete(
        store,
        task_id=task_id,
        expected_attempt_id=store.current_attempts[task_id],
        outputs=(EmittedOutput(id="bad"),),
        callbacks=recorder.callbacks(),
    )
    assert store.launches[1].disposition == NodeDisposition.FAILED
    assert store.status == RunStatus.FAILED
    assert recorder.failed == ["run"]


def test_each_with_an_uncorrelated_input_fails_during_planning() -> None:
    nodes = [
        DagNode(
            slug="source",
            plugin="source",
            correlation_mode=CorrelationMode.NONE,
        ),
        DagNode(
            slug="consumer",
            plugin="consumer",
            input_mode=InputMode.EACH,
            edges=(InputEdge(name="source", source="source"),),
        ),
    ]
    store = FakeStore(nodes)
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    source_task = store.launches[0].task_ids[0]

    on_task_complete(
        store,
        task_id=source_task,
        expected_attempt_id=store.current_attempts[source_task],
        outputs=(EmittedOutput(id="uncorrelated"),),
        callbacks=recorder.callbacks(),
    )

    assert store.launches[1].disposition == NodeDisposition.FAILED
    assert "task has no correlation ID" in store.launches[1].error
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
    monkeypatch.setattr(orchestrator_module, "MAX_SNAPSHOT_OUTPUTS_PER_EDGE", 1)
    nodes = [
        DagNode(slug="source", plugin="source"),
        DagNode(
            slug="consumer",
            plugin="consumer",
            input_mode=InputMode.ALL,
            edges=(InputEdge(name="source", source="source"),),
        ),
    ]
    store = FakeStore(nodes)
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    source_task = store.launches[0].task_ids[0]
    on_task_complete(
        store,
        task_id=source_task,
        expected_attempt_id=store.current_attempts[source_task],
        outputs=(
            EmittedOutput(id="first"),
            EmittedOutput(id="second"),
            EmittedOutput(id="third"),
        ),
        callbacks=recorder.callbacks(),
    )

    failed_launch = store.launches[1]
    assert failed_launch.disposition == NodeDisposition.FAILED
    assert failed_launch.complete
    assert failed_launch.task_ids == ()
    assert failed_launch.error == "edge 'source' has 2 outputs; maximum is 1"
    assert store.status == RunStatus.FAILED
    assert len(store.queue_payloads) == 1
    assert recorder.failed == ["run"]

    retried = retry_run(store, "run", [1])
    assert retried.generations == {1: 1}
    assert store.status == RunStatus.RUNNING
    assert 1 not in store.launches


def test_snapshot_read_returns_one_extra_output_per_edge() -> None:
    store = FakeStore(linear_nodes())
    source = store.seed_launch(0)
    source_task = source.task_ids[0]
    assert store.try_complete_task(
        source_task,
        tuple(EmittedOutput(id=f"output-{index}") for index in range(4)),
        expected_attempt_id=store.current_attempts[source_task],
    )

    snapshot = store.get_planning_snapshot(
        "run",
        1,
        {"source": 0},
        max_outputs_per_edge=2,
        max_outputs_total=10,
        max_routing_bytes=MAX_SNAPSHOT_ROUTING_BYTES,
    )

    assert tuple(output.id for output in snapshot.outputs_by_edge["source"]) == (
        "output-0",
        "output-1",
        "output-2",
    )


def test_snapshot_read_returns_one_extra_output_in_total() -> None:
    nodes = [
        DagNode(slug="left", plugin="left"),
        DagNode(slug="right", plugin="right"),
        DagNode(slug="third", plugin="third"),
        DagNode(
            slug="consumer",
            plugin="consumer",
            input_mode=InputMode.ALL,
            edges=(
                InputEdge(name="left", source="left"),
                InputEdge(name="right", source="right"),
                InputEdge(name="third", source="third"),
            ),
        ),
    ]
    store = FakeStore(nodes)
    for node_index in range(3):
        launch = store.seed_launch(node_index)
        task_id = launch.task_ids[0]
        assert store.try_complete_task(
            task_id,
            tuple(
                EmittedOutput(id=f"{node_index}-output-{output_index}") for output_index in range(2)
            ),
            expected_attempt_id=store.current_attempts[task_id],
        )

    snapshot = store.get_planning_snapshot(
        "run",
        3,
        {"left": 0, "right": 1, "third": 2},
        max_outputs_per_edge=10,
        max_outputs_total=2,
        max_routing_bytes=MAX_SNAPSHOT_ROUTING_BYTES,
    )

    assert set(snapshot.outputs_by_edge) == {"left", "right", "third"}
    assert tuple(len(outputs) for outputs in snapshot.outputs_by_edge.values()) == (2, 1, 0)


def test_snapshot_read_accepts_exact_limits() -> None:
    store = FakeStore(linear_nodes())
    source = store.seed_launch(0)
    source_task = source.task_ids[0]
    assert store.try_complete_task(
        source_task,
        (EmittedOutput(id="first"), EmittedOutput(id="second")),
        expected_attempt_id=store.current_attempts[source_task],
    )

    snapshot = store.get_planning_snapshot(
        "run",
        1,
        {"source": 0},
        max_outputs_per_edge=2,
        max_outputs_total=2,
        max_routing_bytes=MAX_SNAPSHOT_ROUTING_BYTES,
    )
    construction = planning.construct_task_plans(
        store.nodes[1],
        source_indices={"source": 0},
        snapshot=snapshot,
    )

    assert construction.disposition == NodeDisposition.LAUNCHED
    assert tuple(plan.edges[0].output_ids for plan in construction.plans) == (
        ("first",),
        ("second",),
    )


def test_snapshot_read_stops_after_routing_byte_overflow() -> None:
    nodes = [
        DagNode(slug="source", plugin="source"),
        DagNode(slug="later", plugin="later"),
        DagNode(
            slug="consumer",
            plugin="consumer",
            input_mode=InputMode.ALL,
            edges=(
                InputEdge(name="source", source="source"),
                InputEdge(name="later", source="later"),
            ),
        ),
    ]
    store = FakeStore(nodes)
    source = store.seed_launch(0)
    source_task = source.task_ids[0]
    assert store.try_complete_task(
        source_task,
        tuple(EmittedOutput(id=f"source-{index}") for index in range(3)),
        expected_attempt_id=store.current_attempts[source_task],
    )
    later = store.seed_launch(1)
    later_task = later.task_ids[0]
    assert store.try_complete_task(
        later_task,
        (EmittedOutput(id="later-output"),),
        expected_attempt_id=store.current_attempts[later_task],
    )
    first_size = output_ref_routing_size(store.outputs[0][0])

    snapshot = store.get_planning_snapshot(
        "run",
        2,
        {"source": 0, "later": 1},
        max_outputs_per_edge=10,
        max_outputs_total=10,
        max_routing_bytes=first_size,
    )

    assert tuple(output.id for output in snapshot.outputs_by_edge["source"]) == (
        "source-0",
        "source-1",
    )
    assert snapshot.outputs_by_edge["later"] == ()


def test_routing_byte_overflow_creates_a_failed_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = OutputRef(id="first", correlation_id="first")
    routing_limit = output_ref_routing_size(first)
    monkeypatch.setattr(orchestrator_module, "MAX_SNAPSHOT_ROUTING_BYTES", routing_limit)
    monkeypatch.setattr(planning, "MAX_SNAPSHOT_ROUTING_BYTES", routing_limit)
    nodes = [
        DagNode(slug="source", plugin="source"),
        DagNode(
            slug="consumer",
            plugin="consumer",
            input_mode=InputMode.ALL,
            edges=(InputEdge(name="source", source="source"),),
        ),
    ]
    store = FakeStore(nodes)
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    source_task = store.launches[0].task_ids[0]

    on_task_complete(
        store,
        task_id=source_task,
        expected_attempt_id=store.current_attempts[source_task],
        outputs=(EmittedOutput(id="first"), EmittedOutput(id="second")),
        callbacks=recorder.callbacks(),
    )

    failed_launch = store.launches[1]
    routed_size = sum(output_ref_routing_size(output) for output in store.outputs[0])
    assert failed_launch.disposition == NodeDisposition.FAILED
    assert failed_launch.error == (
        f"snapshot routing data is {routed_size} bytes; maximum is {routing_limit}"
    )
    assert store.status == RunStatus.FAILED
    assert recorder.failed == ["run"]


def test_task_completion_limit_fails_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store_module, "MAX_OUTPUTS_PER_TASK_COMPLETION", 1)
    monkeypatch.setattr(
        helpers_module,
        "output_ref_routing_size",
        lambda _output: pytest.fail("routing size was calculated before the output count"),
    )
    store = FakeStore([DagNode(slug="root", plugin="p")])
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    task_id = store.launches[0].task_ids[0]

    on_task_complete(
        store,
        task_id=task_id,
        expected_attempt_id=store.current_attempts[task_id],
        outputs=(EmittedOutput(id="first"), EmittedOutput(id="second")),
        callbacks=recorder.callbacks(),
    )

    assert store.outputs == {}
    assert store.resolved == {}
    assert store.task_statuses[task_id] == TaskStatus.FAILED
    assert store.status == RunStatus.FAILED
    assert store.run_error == "task completion contains 2 outputs; maximum is 1"


def test_store_rejects_an_oversized_completion_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store_module, "MAX_OUTPUTS_PER_TASK_COMPLETION", 1)
    store = FakeStore([DagNode(slug="root", plugin="p")])
    launch = store.seed_launch(0)
    task_id = launch.task_ids[0]
    oversized = (EmittedOutput(id="first"), EmittedOutput(id="second"))

    assert (
        store.try_complete_task(
            task_id,
            oversized,
            expected_attempt_id="stale-attempt",
        )
        is None
    )

    with pytest.raises(
        OutputPublicationError,
        match="task completion contains 2 outputs; maximum is 1",
    ):
        store.try_complete_task(
            task_id,
            oversized,
            expected_attempt_id=store.current_attempts[task_id],
        )

    assert store.task_statuses[task_id] == TaskStatus.QUEUED
    assert store.outputs == {}
    assert store.resolved == {}


def test_task_completion_accepts_its_exact_routing_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routed = OutputRef(
        id="output",
        fields={"kind": "record"},
        correlation_id="output",
    )
    routing_size = output_ref_routing_size(routed)
    monkeypatch.setattr(store_module, "MAX_TASK_COMPLETION_ROUTING_BYTES", routing_size)
    store = FakeStore([DagNode(slug="root", plugin="p")])
    launch = store.seed_launch(0)
    task_id = launch.task_ids[0]

    result = store.try_complete_task(
        task_id,
        (
            EmittedOutput(
                id="output",
                fields={"kind": "record"},
                data={"payload": "x" * 10_000},
            ),
        ),
        expected_attempt_id=store.current_attempts[task_id],
    )

    assert result is not None
    assert store.outputs[0] == (routed,)
    assert store.resolved[("run", "output")].data["payload"] == "x" * 10_000


def test_task_completion_routing_byte_overflow_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = OutputRef(id="first", correlation_id="first")
    second = OutputRef(id="second", correlation_id="second")
    routing_size = output_ref_routing_size(first) + output_ref_routing_size(second)
    monkeypatch.setattr(
        store_module,
        "MAX_TASK_COMPLETION_ROUTING_BYTES",
        routing_size - 1,
    )
    store = FakeStore([DagNode(slug="root", plugin="p")])
    launch = store.seed_launch(0)
    task_id = launch.task_ids[0]

    with pytest.raises(
        OutputPublicationError,
        match=(
            f"task completion routing data is {routing_size} bytes; maximum is {routing_size - 1}"
        ),
    ):
        store.try_complete_task(
            task_id,
            (EmittedOutput(id="first"), EmittedOutput(id="second")),
            expected_attempt_id=store.current_attempts[task_id],
        )

    assert store.task_statuses[task_id] == TaskStatus.QUEUED
    assert store.outputs == {}
    assert store.resolved == {}
    assert store._published_output_ids == set()


def test_oversized_completion_redelivery_repairs_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store_module, "MAX_OUTPUTS_PER_TASK_COMPLETION", 1)
    store = FakeStore([DagNode(slug="root", plugin="p")])
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    task_id = store.launches[0].task_ids[0]
    attempt_id = store.current_attempts[task_id]
    committed = store.try_complete_task(
        task_id,
        (EmittedOutput(id="committed"),),
        expected_attempt_id=attempt_id,
    )
    assert committed is not None
    assert store.status == RunStatus.RUNNING

    on_task_complete(
        store,
        task_id=task_id,
        expected_attempt_id=attempt_id,
        outputs=(EmittedOutput(id="ignored-1"), EmittedOutput(id="ignored-2")),
        callbacks=recorder.callbacks(),
    )

    assert store.status == RunStatus.COMPLETED
    assert store.outputs[0][0].id == "committed"
    assert ("run", "ignored-1") not in store.resolved
    assert ("run", "ignored-2") not in store.resolved
    assert recorder.completed == ["run"]


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


def test_failure_redelivery_finalises_from_the_persisted_attempt_error() -> None:
    class InterruptedFinalisationStore(FakeStore):
        reject_once = True

        def try_finalize_run(
            self,
            run_id: str,
            status: RunStatus,
            *,
            cause: TaskAttemptRef | NodeLaunchRef | None = None,
            reason: str | None = None,
        ) -> bool:
            if isinstance(cause, TaskAttemptRef) and self.reject_once:
                self.reject_once = False
                self.task_contexts.pop(cause.task_id, None)
                return False
            return super().try_finalize_run(
                run_id,
                status,
                cause=cause,
                reason=reason,
            )

    store = InterruptedFinalisationStore([DagNode(slug="root", plugin="p")])
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    task_id = store.launches[0].task_ids[0]
    attempt_id = store.current_attempts[task_id]

    on_task_failed(
        store,
        task_id=task_id,
        expected_attempt_id=attempt_id,
        error_message="original error",
        callbacks=recorder.callbacks(),
    )
    assert store.status == RunStatus.RUNNING
    assert store.task_statuses[task_id] == TaskStatus.FAILED
    assert task_id not in store.task_contexts

    on_task_failed(
        store,
        task_id=task_id,
        expected_attempt_id=attempt_id,
        error_message="different redelivery error",
        callbacks=recorder.callbacks(),
    )

    assert store.status == RunStatus.FAILED
    assert store.run_error == "original error"
    assert recorder.failed == ["run"]


def test_terminal_progress_excludes_tasks_cancelled_by_the_transition() -> None:
    store = FakeStore(
        [
            DagNode(slug="completed", plugin="p"),
            DagNode(slug="failed", plugin="p"),
            DagNode(slug="cancelled", plugin="p"),
        ]
    )
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    completed_task = store.launches[0].task_ids[0]
    failed_task = store.launches[1].task_ids[0]
    cancelled_task = store.launches[2].task_ids[0]

    on_task_complete(
        store,
        task_id=completed_task,
        expected_attempt_id=store.current_attempts[completed_task],
        callbacks=recorder.callbacks(),
    )
    on_task_failed(
        store,
        task_id=failed_task,
        expected_attempt_id=store.current_attempts[failed_task],
        error_message="failed",
        callbacks=recorder.callbacks(),
    )

    assert store.status == RunStatus.FAILED
    assert store.progress == 2
    assert store.task_statuses[cancelled_task] == TaskStatus.CANCELLED


def test_task_retry_winning_before_terminalisation_keeps_the_run_active() -> None:
    class RetryRaceStore(FakeStore):
        replacement_attempt: str | None = None

        def try_finalize_run(
            self,
            run_id: str,
            status: RunStatus,
            *,
            cause: TaskAttemptRef | NodeLaunchRef | None = None,
            reason: str | None = None,
        ) -> bool:
            if isinstance(cause, TaskAttemptRef) and self.replacement_attempt is None:
                self.replacement_attempt = retry_task(
                    self,
                    cause.task_id,
                    expected_attempt_id=cause.attempt_id,
                )
            return super().try_finalize_run(
                run_id,
                status,
                cause=cause,
                reason=reason,
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
    on_task_complete(
        store,
        task_id=task_id,
        expected_attempt_id=store.current_attempts[task_id],
        callbacks=callbacks,
    )

    assert observed == [(RunStatus.COMPLETED, 1)]


def test_task_completion_does_not_require_retained_task_context() -> None:
    class ArchivingStore(FakeStore):
        def try_complete_task(
            self,
            task_id: str,
            outputs: tuple[EmittedOutput, ...],
            *,
            expected_attempt_id: str,
        ) -> TaskCompletionResult | None:
            result = super().try_complete_task(
                task_id,
                outputs,
                expected_attempt_id=expected_attempt_id,
            )
            if result is not None:
                self.task_contexts.pop(task_id, None)
            return result

    store = ArchivingStore([DagNode(slug="root", plugin="p")])
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    task_id = store.launches[0].task_ids[0]

    on_task_complete(
        store,
        task_id=task_id,
        expected_attempt_id=store.current_attempts[task_id],
        callbacks=recorder.callbacks(),
    )

    assert store.status == RunStatus.COMPLETED
    assert recorder.completed == ["run"]


def test_completion_terminal_transaction_rechecks_task_state() -> None:
    class CompletionRaceStore(FakeStore):
        lose_once = True

        def try_finalize_run(
            self,
            run_id: str,
            status: RunStatus,
            *,
            cause: TaskAttemptRef | NodeLaunchRef | None = None,
            reason: str | None = None,
        ) -> bool:
            if status == RunStatus.COMPLETED and self.lose_once:
                self.lose_once = False
                task_id = self.launches[0].task_ids[0]
                self.task_statuses[task_id] = TaskStatus.QUEUED
                self.launches[0] = self.launches[0].model_copy(update={"complete": False})
            return super().try_finalize_run(
                run_id,
                status,
                cause=cause,
                reason=reason,
            )

    store = CompletionRaceStore([DagNode(slug="root", plugin="p")])
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    task_id = store.launches[0].task_ids[0]
    on_task_complete(
        store,
        task_id=task_id,
        expected_attempt_id=store.current_attempts[task_id],
        callbacks=recorder.callbacks(),
    )
    assert store.status == RunStatus.RUNNING
    assert recorder.completed == []

    on_task_complete(
        store,
        task_id=task_id,
        expected_attempt_id=store.current_attempts[task_id],
        callbacks=recorder.callbacks(),
    )
    assert store.status == RunStatus.COMPLETED
    assert recorder.completed == ["run"]


def test_completion_requires_an_active_launch_for_every_stored_node() -> None:
    store = FakeStore(linear_nodes())
    root = store.seed_launch(0)
    root_task = root.task_ids[0]
    assert store.try_complete_task(
        root_task,
        (),
        expected_attempt_id=store.current_attempts[root_task],
    )

    assert not store.try_finalize_run("run", RunStatus.COMPLETED)
    assert store.status == RunStatus.RUNNING


def test_completion_rejects_the_wrong_active_launch_indices() -> None:
    store = FakeStore(linear_nodes())
    root = store.seed_launch(0)
    child = store.seed_launch(1)
    for task_id in (root.task_ids[0], child.task_ids[0]):
        assert store.try_complete_task(
            task_id,
            (),
            expected_attempt_id=store.current_attempts[task_id],
        )
    store.launches[99] = store.launches.pop(1)

    assert not store.try_finalize_run("run", RunStatus.COMPLETED)
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
            *,
            cause: TaskAttemptRef | NodeLaunchRef | None = None,
            reason: str | None = None,
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
    assert store.try_complete_task(
        task_id,
        (EmittedOutput(id="output"),),
        expected_attempt_id=store.current_attempts[task_id],
    )
    assert tuple(store.resolve_outputs("run", ("output",))) == ("output",)
    assert store.resolve_outputs("another-run", ("output",)) == {}


def test_non_final_task_completions_do_not_reconcile_the_graph() -> None:
    store = FakeStore(linear_nodes())
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    root_task = store.launches[0].task_ids[0]
    outputs = tuple(EmittedOutput(id=f"output-{index}") for index in range(200))
    on_task_complete(
        store,
        task_id=root_task,
        expected_attempt_id=store.current_attempts[root_task],
        outputs=outputs,
        callbacks=recorder.callbacks(),
    )
    child_tasks = store.launches[1].task_ids
    assert len(child_tasks) == 200
    store.get_run_nodes_calls = 0
    store.list_node_launches_calls = 0

    for task_id in child_tasks[:-1]:
        on_task_complete(
            store,
            task_id=task_id,
            expected_attempt_id=store.current_attempts[task_id],
            callbacks=recorder.callbacks(),
        )

    assert store.get_run_nodes_calls == 0
    assert store.list_node_launches_calls == 0

    final_task = child_tasks[-1]
    on_task_complete(
        store,
        task_id=final_task,
        expected_attempt_id=store.current_attempts[final_task],
        callbacks=recorder.callbacks(),
    )

    assert store.get_run_nodes_calls == 1
    assert store.list_node_launches_calls == 1
    assert store.status == RunStatus.COMPLETED


def test_zero_task_chain_is_reconciled_without_repeated_graph_scans() -> None:
    class CountingSnapshotStore(FakeStore):
        snapshot_calls = 0

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
            self.snapshot_calls += 1
            return super().get_planning_snapshot(
                run_id,
                node_index,
                source_indices,
                max_outputs_per_edge=max_outputs_per_edge,
                max_outputs_total=max_outputs_total,
                max_routing_bytes=max_routing_bytes,
            )

    node_count = 1_100
    nodes = [DagNode(slug="node-0", plugin="p")]
    nodes.extend(
        DagNode(
            slug=f"node-{index}",
            plugin="p",
            input_mode=InputMode.ALL,
            edges=(
                InputEdge(
                    name="source",
                    source=f"node-{index - 1}",
                ),
            ),
        )
        for index in range(1, node_count)
    )
    store = CountingSnapshotStore(nodes)
    store.seed_launch(0, plans=(), disposition=NodeDisposition.FILTERED)

    reconcile_run(store, "run", callbacks=CallbackRecorder().callbacks())

    assert len(store.launches) == node_count
    assert store.snapshot_calls == node_count
    assert store.get_run_nodes_calls == 1
    assert store.list_node_launches_calls == 1
    assert store.progress == node_count
    assert store.status == RunStatus.COMPLETED
