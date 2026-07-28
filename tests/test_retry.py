import pytest

from dagabaaz.constants import InputMode, LaunchCreateStatus, RunStatus, TaskStatus
from dagabaaz.models import DagNode, InputEdge, OutputRef, TaskInputPlan
from dagabaaz.orchestrator import (
    on_task_crashed,
    on_task_failed,
    reconcile_run,
    start_run,
)
from dagabaaz.retry import RetryRejectedError, retry_run, retry_task
from dagabaaz.store import RunReopenResult, StoreContractError

from .helpers import CallbackRecorder, FakeStore


def nodes() -> list[DagNode]:
    return [
        DagNode(slug="root", plugin="p"),
        DagNode(
            slug="left",
            plugin="p",
            input_mode=InputMode.EACH,
            edges=(InputEdge(name="root", source="root"),),
        ),
        DagNode(
            slug="right",
            plugin="p",
            input_mode=InputMode.EACH,
            edges=(InputEdge(name="root", source="root"),),
        ),
        DagNode(
            slug="leaf",
            plugin="p",
            input_mode=InputMode.EACH,
            edges=(InputEdge(name="left", source="left"),),
        ),
    ]


def seed_graph(store: FakeStore) -> None:
    for node_index in range(len(store.nodes)):
        store.seed_launch(node_index)


def fail_run_at(store: FakeStore, *node_indices: int) -> None:
    for node_index in node_indices:
        task_id = store.launches[node_index].task_ids[0]
        assert (
            store.mark_task_failed(
                task_id,
                f"node {node_index} failed",
                expected_attempt_id=store.current_attempts[task_id],
            )
            == "run"
        )
    assert store.try_finalize_run("run", RunStatus.FAILED, "failed")


def test_task_retry_preserves_plan_generation_and_enqueues_a_new_attempt() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    start_run(store, "run", callbacks=CallbackRecorder().callbacks())
    task_id = store.launches[0].task_ids[0]
    plan = store.plans[task_id]
    predecessor = store.current_attempts[task_id]
    assert predecessor == "attempt-1"
    store.mark_task_crashed(
        task_id,
        "transient",
        expected_attempt_id=predecessor,
    )

    attempt = retry_task(
        store,
        task_id,
        expected_attempt_id=predecessor,
    )
    assert attempt == "attempt-2"
    assert store.plans[task_id] is plan
    assert store.task_statuses[task_id] == TaskStatus.QUEUED
    assert store.retry_payloads == [
        {
            "task_id": task_id,
            "attempt_id": "attempt-2",
            "generation": plan.generation,
            "launch_id": store.launches[0].id,
        }
    ]


def test_task_retry_rejects_a_stale_attempt_token() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    start_run(store, "run", callbacks=CallbackRecorder().callbacks())
    task_id = store.launches[0].task_ids[0]
    first_attempt = store.current_attempts[task_id]
    store.mark_task_crashed(
        task_id,
        "transient",
        expected_attempt_id=first_attempt,
    )
    second_attempt = retry_task(
        store,
        task_id,
        expected_attempt_id=first_attempt,
    )
    assert second_attempt is not None

    store.mark_task_crashed(
        task_id,
        "transient again",
        expected_attempt_id=second_attempt,
    )
    before = list(store.queue_payloads)
    assert (
        retry_task(
            store,
            task_id,
            expected_attempt_id=first_attempt,
        )
        is None
    )
    assert store.queue_payloads == before
    assert (
        retry_task(
            store,
            task_id,
            expected_attempt_id=second_attempt,
        )
        == "attempt-3"
    )


def test_late_attempt_callbacks_cannot_terminate_a_replacement_attempt() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    task_id = store.launches[0].task_ids[0]
    first_attempt = store.current_attempts[task_id]
    assert (
        store.mark_task_crashed(
            task_id,
            "transient",
            expected_attempt_id=first_attempt,
        )
        == "run"
    )
    second_attempt = retry_task(
        store,
        task_id,
        expected_attempt_id=first_attempt,
    )
    assert second_attempt is not None

    on_task_failed(
        store,
        task_id=task_id,
        expected_attempt_id=first_attempt,
        error_message="late failure",
        callbacks=recorder.callbacks(),
    )
    on_task_crashed(
        store,
        task_id=task_id,
        expected_attempt_id=first_attempt,
        error_message="late crash",
        callbacks=recorder.callbacks(),
    )

    assert store.task_statuses[task_id] == TaskStatus.QUEUED
    assert store.status == RunStatus.RUNNING
    assert recorder.failed == []
    assert recorder.crashed == []

    on_task_failed(
        store,
        task_id=task_id,
        expected_attempt_id=second_attempt,
        error_message="current failure",
        callbacks=recorder.callbacks(),
    )
    assert store.status == RunStatus.FAILED
    assert recorder.failed == ["run"]


def test_late_attempt_cannot_complete_a_replacement_attempt() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    start_run(store, "run", callbacks=CallbackRecorder().callbacks())
    task_id = store.launches[0].task_ids[0]
    first_attempt = store.current_attempts[task_id]
    assert (
        store.mark_task_crashed(
            task_id,
            "transient",
            expected_attempt_id=first_attempt,
        )
        == "run"
    )
    second_attempt = retry_task(
        store,
        task_id,
        expected_attempt_id=first_attempt,
    )
    assert second_attempt is not None

    assert not store.complete_task(
        task_id,
        OutputRef(id="stale-output"),
        expected_attempt_id=first_attempt,
    )
    assert store.task_statuses[task_id] == TaskStatus.QUEUED
    assert store.outputs == {}
    assert store.complete_task(
        task_id,
        OutputRef(id="current-output"),
        expected_attempt_id=second_attempt,
    )


def test_task_retry_rejects_terminal_run() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    start_run(store, "run", callbacks=CallbackRecorder().callbacks())
    task_id = store.launches[0].task_ids[0]
    predecessor = store.current_attempts[task_id]
    store.mark_task_crashed(
        task_id,
        "transient",
        expected_attempt_id=predecessor,
    )
    assert store.try_finalize_run("run", RunStatus.CRASHED, "transient")
    assert (
        retry_task(
            store,
            task_id,
            expected_attempt_id=predecessor,
        )
        is None
    )


def test_boundary_retry_invalidates_only_boundary_and_descendants() -> None:
    store = FakeStore(nodes())
    seed_graph(store)
    root_task = store.launches[0].task_ids[0]
    right_task = store.launches[2].task_ids[0]
    leaf_task = store.launches[3].task_ids[0]
    assert store.complete_task(
        root_task,
        OutputRef(id="root-output"),
        expected_attempt_id=store.current_attempts[root_task],
    )
    assert store.complete_task(
        right_task,
        OutputRef(id="right-output"),
        expected_attempt_id=store.current_attempts[right_task],
    )
    assert store.complete_task(
        leaf_task,
        OutputRef(id="leaf-output"),
        expected_attempt_id=store.current_attempts[leaf_task],
    )
    preserved_launches = {index: store.launches[index] for index in (0, 2)}
    preserved_plans = {
        task_id: store.plans[task_id]
        for index in (0, 2)
        for task_id in store.launches[index].task_ids
    }
    affected_tasks = {
        task_id
        for index in (1, 3)
        for task_id in store.launches[index].task_ids
    }
    fail_run_at(store, 1)

    result = retry_run(store, "run", [1])
    assert result.affected_nodes == (1, 3)
    assert result.generations == {1: 1, 3: 1}
    assert result.retry_count == 1
    assert store.status == RunStatus.RUNNING

    assert set(store.launches) == {0, 2}
    assert all(store.launches[index] is launch for index, launch in preserved_launches.items())
    assert all(store.plans[task_id] is plan for task_id, plan in preserved_plans.items())
    assert {launch.node_index for launch in store.invalidated_launches} == {1, 3}
    assert set(store.invalidated_plans) == affected_tasks
    assert {
        payload["task_id"] for payload in store.invalidated_deliveries.values()
    } == affected_tasks
    assert store.resolve_outputs("run", ("root-output", "right-output")) == {
        "root-output": store.resolved[("run", "root-output")],
        "right-output": store.resolved[("run", "right-output")],
    }


def test_boundary_retry_replans_work_cancelled_on_a_parallel_branch() -> None:
    store = FakeStore(nodes())
    recorder = CallbackRecorder()
    seed_graph(store)
    root_task = store.launches[0].task_ids[0]
    assert store.complete_task(
        root_task,
        OutputRef(id="root-output"),
        expected_attempt_id=store.current_attempts[root_task],
    )
    fail_run_at(store, 1)
    cancelled_task = store.launches[2].task_ids[0]
    assert store.task_statuses[cancelled_task] == TaskStatus.CANCELLED

    result = retry_run(store, "run", [1])

    assert result.affected_nodes == (1, 2, 3)
    assert result.generations == {1: 1, 2: 1, 3: 1}
    assert set(store.launches) == {0}

    reconcile_run(store, "run", callbacks=recorder.callbacks())
    left_task = store.launches[1].task_ids[0]
    right_task = store.launches[2].task_ids[0]
    assert store.complete_task(
        left_task,
        OutputRef(id="left-output"),
        expected_attempt_id=store.current_attempts[left_task],
    )
    assert store.complete_task(
        right_task,
        expected_attempt_id=store.current_attempts[right_task],
    )
    reconcile_run(store, "run", callbacks=recorder.callbacks())
    leaf_task = store.launches[3].task_ids[0]
    assert store.complete_task(
        leaf_task,
        expected_attempt_id=store.current_attempts[leaf_task],
    )
    reconcile_run(store, "run", callbacks=recorder.callbacks())

    assert store.status == RunStatus.COMPLETED
    assert recorder.completed == ["run"]


def test_invalidated_attempt_history_does_not_block_a_successful_replan() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    old_task = store.launches[0].task_ids[0]
    old_attempt = store.current_attempts[old_task]
    on_task_failed(
        store,
        task_id=old_task,
        expected_attempt_id=old_attempt,
        error_message="failed",
        callbacks=recorder.callbacks(),
    )

    retry_run(store, "run", [0])
    reconcile_run(store, "run", callbacks=recorder.callbacks())
    new_task = store.launches[0].task_ids[0]
    assert new_task != old_task
    assert (
        store.mark_task_crashed(
            old_task,
            "late crash",
            expected_attempt_id=old_attempt,
        )
        is None
    )

    assert store.complete_task(
        new_task,
        expected_attempt_id=store.current_attempts[new_task],
    )
    reconcile_run(store, "run", callbacks=recorder.callbacks())
    assert store.status == RunStatus.COMPLETED
    assert recorder.completed == ["run"]


def test_snapshot_from_before_a_boundary_retry_is_stale_for_the_new_launch() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    old_snapshot = store.get_planning_snapshot("run", 0, ())
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    old_task = store.launches[0].task_ids[0]
    on_task_failed(
        store,
        task_id=old_task,
        expected_attempt_id=store.current_attempts[old_task],
        error_message="failed",
        callbacks=recorder.callbacks(),
    )
    retry_run(store, "run", [0])
    reconcile_run(store, "run", callbacks=recorder.callbacks())

    result = store.try_create_node_launch(
        "run",
        0,
        "p",
        old_snapshot.token,
        (TaskInputPlan(generation=old_snapshot.generation),),
        None,
    )
    assert result.status == LaunchCreateStatus.STALE
    assert store.launches[0].generation == 1


def test_boundary_retry_uses_stored_topology_when_caller_definition_changes() -> None:
    caller_definition = nodes()
    store = FakeStore(caller_definition)
    caller_definition[3].edges = (InputEdge(name="right", source="right"),)
    seed_graph(store)
    for node_index in (0, 2):
        task_id = store.launches[node_index].task_ids[0]
        assert store.complete_task(
            task_id,
            expected_attempt_id=store.current_attempts[task_id],
        )
    fail_run_at(store, 1)

    result = retry_run(store, "run", [1])
    assert result.affected_nodes == (1, 3)


def test_boundary_retry_supports_several_boundaries_with_one_generation_map() -> None:
    store = FakeStore(nodes())
    seed_graph(store)
    root_task = store.launches[0].task_ids[0]
    assert store.complete_task(
        root_task,
        expected_attempt_id=store.current_attempts[root_task],
    )
    fail_run_at(store, 1, 2)

    result = retry_run(store, "run", [2, 1])
    assert result.affected_nodes == (1, 2, 3)
    assert result.generations == {1: 1, 2: 1, 3: 1}
    assert set(store.launches) == {0}


def test_boundary_retry_rejects_stale_nonretryable_or_missing_state() -> None:
    store = FakeStore(nodes())
    before = (
        store.status,
        dict(store.launches),
        dict(store.generations),
        store.retry_count,
    )
    with pytest.raises(RetryRejectedError):
        retry_run(store, "run", [1])
    assert (
        store.status,
        dict(store.launches),
        dict(store.generations),
        store.retry_count,
    ) == before

    class MissingTopologyStore(FakeStore):
        def get_run_nodes(self, run_id: str) -> list[DagNode] | None:
            return None

    with pytest.raises(RetryRejectedError, match="no stored topology"):
        retry_run(MissingTopologyStore(nodes()), "run", [1])


def test_only_one_boundary_retry_can_win() -> None:
    store = FakeStore(nodes())
    seed_graph(store)
    fail_run_at(store, 1)

    first = retry_run(store, "run", [1])
    assert first.retry_count == 1
    with pytest.raises(RetryRejectedError):
        retry_run(store, "run", [1])
    assert store.retry_count == 1


def test_cancelled_sibling_is_not_a_retry_boundary() -> None:
    store = FakeStore(nodes())
    seed_graph(store)
    fail_run_at(store, 1)
    cancelled_task = store.launches[2].task_ids[0]
    assert store.task_statuses[cancelled_task] == TaskStatus.CANCELLED

    with pytest.raises(RetryRejectedError):
        retry_run(store, "run", [2])

    assert store.status == RunStatus.FAILED
    assert store.retry_count == 0


def test_cancelled_run_restarts_interrupted_work_and_preserves_completed_work() -> None:
    store = FakeStore(
        [
            DagNode(slug="complete", plugin="p"),
            DagNode(slug="interrupted", plugin="p"),
        ]
    )
    recorder = CallbackRecorder()
    start_run(store, "run", callbacks=recorder.callbacks())
    complete_task = store.launches[0].task_ids[0]
    assert store.complete_task(
        complete_task,
        expected_attempt_id=store.current_attempts[complete_task],
    )
    preserved_launch = store.launches[0]
    interrupted_task = store.launches[1].task_ids[0]
    assert store.try_finalize_run("run", RunStatus.CANCELLED, "cancelled")
    assert store.task_statuses[interrupted_task] == TaskStatus.CANCELLED

    result = retry_run(store, "run")

    assert result.affected_nodes == (1,)
    assert result.generations == {1: 1}
    assert result.retry_count == 1
    assert store.launches == {0: preserved_launch}

    reconcile_run(store, "run", callbacks=recorder.callbacks())
    replacement_task = store.launches[1].task_ids[0]
    assert replacement_task != interrupted_task
    assert store.complete_task(
        replacement_task,
        expected_attempt_id=store.current_attempts[replacement_task],
    )
    reconcile_run(store, "run", callbacks=recorder.callbacks())

    assert store.status == RunStatus.COMPLETED
    assert recorder.completed == ["run"]


def test_cancelled_run_can_reopen_before_any_launch_exists() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    recorder = CallbackRecorder()
    snapshot = store.get_planning_snapshot("run", 0, ())
    assert store.try_finalize_run("run", RunStatus.CANCELLED, "cancelled")

    result = retry_run(store, "run")

    assert result.affected_nodes == ()
    assert result.generations == {}
    assert result.retry_count == 1
    assert store.status == RunStatus.RUNNING
    stale = store.try_create_node_launch(
        "run",
        0,
        "p",
        snapshot.token,
        (TaskInputPlan(generation=snapshot.generation),),
        None,
    )
    assert stale.status == LaunchCreateStatus.STALE

    reconcile_run(store, "run", callbacks=recorder.callbacks())
    task_id = store.launches[0].task_ids[0]
    assert store.complete_task(
        task_id,
        expected_attempt_id=store.current_attempts[task_id],
    )
    reconcile_run(store, "run", callbacks=recorder.callbacks())
    assert store.status == RunStatus.COMPLETED
    assert recorder.completed == ["run"]


def test_retry_mode_must_match_the_terminal_status() -> None:
    failed = FakeStore([DagNode(slug="root", plugin="p")])
    failed.seed_launch(0)
    fail_run_at(failed, 0)
    with pytest.raises(RetryRejectedError):
        retry_run(failed, "run")

    cancelled = FakeStore([DagNode(slug="root", plugin="p")])
    cancelled.seed_launch(0)
    assert cancelled.try_finalize_run("run", RunStatus.CANCELLED, "cancelled")
    with pytest.raises(RetryRejectedError):
        retry_run(cancelled, "run", [0])


def test_crashed_run_accepts_a_crashed_boundary() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    store.seed_launch(0)
    task_id = store.launches[0].task_ids[0]
    assert (
        store.mark_task_crashed(
            task_id,
            "crashed",
            expected_attempt_id=store.current_attempts[task_id],
        )
        == "run"
    )
    assert store.try_finalize_run("run", RunStatus.CRASHED, "crashed")

    result = retry_run(store, "run", [0])

    assert result.affected_nodes == (0,)
    assert result.generations == {0: 1}
    assert store.status == RunStatus.RUNNING


def test_store_rejects_an_affected_set_that_omits_cancelled_work() -> None:
    store = FakeStore(nodes())
    seed_graph(store)
    root_task = store.launches[0].task_ids[0]
    assert store.complete_task(
        root_task,
        expected_attempt_id=store.current_attempts[root_task],
    )
    fail_run_at(store, 1)
    before = (
        store.status,
        dict(store.launches),
        dict(store.generations),
        store.retry_count,
    )

    assert store.try_reopen_run("run", (1,), (1, 3)) is None

    assert (
        store.status,
        dict(store.launches),
        dict(store.generations),
        store.retry_count,
    ) == before


@pytest.mark.parametrize(
    "generations",
    [
        {1: 1},
        {1: 1, 3: 1, 99: 1},
    ],
)
def test_boundary_retry_rejects_an_inexact_generation_map(
    generations: dict[int, int],
) -> None:
    class WrongGenerationStore(FakeStore):
        def try_reopen_run(
            self,
            run_id: str,
            boundary_indices: tuple[int, ...] | None,
            affected_indices: tuple[int, ...],
        ) -> RunReopenResult | None:
            return RunReopenResult(retry_count=1, generations=generations)

    with pytest.raises(StoreContractError, match="different generation set"):
        retry_run(WrongGenerationStore(nodes()), "run", [1])


def test_boundary_retry_validates_input() -> None:
    store = FakeStore(nodes())
    with pytest.raises(ValueError, match="must not be empty"):
        retry_run(store, "run", [])
    with pytest.raises(ValueError, match="invalid"):
        retry_run(store, "run", [99])
    for invalid in (True, 1.0, "1"):
        with pytest.raises(ValueError, match="integers"):
            retry_run(store, "run", [invalid])  # type: ignore[list-item]


def test_run_reopen_result_copies_and_freezes_generations() -> None:
    generations = {3: 4}
    result = RunReopenResult(retry_count=2, generations=generations)
    generations[3] = 9
    assert result.generations == {3: 4}
    with pytest.raises(TypeError):
        result.generations[3] = 9  # type: ignore[index]

    with pytest.raises(ValueError, match="positive"):
        RunReopenResult(retry_count=2, generations={3: 0})
    assert RunReopenResult(retry_count=2, generations={}).generations == {}


@pytest.mark.parametrize(
    ("retry_count", "generations"),
    [
        (True, {1: 1}),
        (1, {True: 1}),
        (1, {"1": 1}),
        (1, {1: True}),
        (1, {1: "1"}),
        (1, {1: 1.0}),
    ],
)
def test_run_reopen_result_rejects_non_integer_state(
    retry_count: object,
    generations: dict[object, object],
) -> None:
    with pytest.raises(ValueError, match="integer"):
        RunReopenResult(  # type: ignore[arg-type]
            retry_count=retry_count,
            generations=generations,
        )
