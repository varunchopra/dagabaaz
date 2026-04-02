import pytest

from dagabaaz.constants import FanMode, RunStatus, TaskStatus
from dagabaaz.models import EdgeFilter, FilterRule
from dagabaaz.orchestrator import (
    OrchestratorCallbacks,
    on_task_complete,
    on_task_crashed,
    on_task_failed,
    start_run,
)
from tests.helpers import MockDagStore, make_dag_artifact, make_node

pytestmark = pytest.mark.usefixtures("reset_topology_cache")


def _make_callbacks() -> tuple[OrchestratorCallbacks, dict[str, list[str]]]:
    tracker: dict[str, list[str]] = {"completed": [], "failed": [], "crashed": []}
    callbacks = OrchestratorCallbacks(
        on_run_completed=lambda rid: tracker["completed"].append(rid),
        on_run_failed=lambda rid: tracker["failed"].append(rid),
        on_run_crashed=lambda rid: tracker["crashed"].append(rid),
    )
    return callbacks, tracker


def _no_passthrough(plugin: str) -> bool:
    return False


class TestStartRun:
    def test_dispatches_root_nodes(self) -> None:
        nodes = [
            make_node("fetch", slug="a"),
            make_node("transform", slug="b", depends_on=["a"]),
            make_node("export", slug="c", depends_on=["b"]),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)

        result = start_run(store, "run-1", nodes)

        assert result == [0]
        assert len(store.dispatched_tasks) == 1
        assert store.dispatched_tasks[0].node_index == 0
        assert store.dispatched_tasks[0].plugin_name == "fetch"
        assert store.dispatched_tasks[0].input_artifact_id is None

    def test_multiple_roots_all_dispatched(self) -> None:
        nodes = [
            make_node("src_a", slug="a"),
            make_node("src_b", slug="b"),
            make_node("merge", slug="c", depends_on=["a", "b"]),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)

        result = start_run(store, "run-1", nodes)

        assert result == [0, 1]
        assert len(store.dispatched_tasks) == 2
        assert {t.node_index for t in store.dispatched_tasks} == {0, 1}

    def test_no_root_nodes_raises(self) -> None:
        nodes = [
            make_node("a", slug="a", depends_on=["b"]),
            make_node("b", slug="b", depends_on=["a"]),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)

        with pytest.raises(ValueError, match="no root nodes"):
            start_run(store, "run-1", nodes)

    def test_claim_race_skips_already_claimed(self) -> None:
        nodes = [make_node("fetch", slug="a")]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store._node_launches.add(("run-1", 0))

        start_run(store, "run-1", nodes)

        assert len(store.dispatched_tasks) == 0


class TestOnTaskComplete:
    def test_barrier_not_met_waits(self) -> None:
        nodes = [make_node("fetch", slug="a")]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-1", "run-1", 0, "fetch", TaskStatus.COMPLETED)
        store.setup_task("task-2", "run-1", 0, "fetch", TaskStatus.RUNNING)
        callbacks, tracker = _make_callbacks()

        on_task_complete(
            store,
            task_id="task-1",
            callbacks=callbacks,
            resolve_passthrough=_no_passthrough,
        )

        assert len(store.dispatched_tasks) == 0
        assert tracker["completed"] == []

    def test_barrier_met_launches_downstream(self) -> None:
        """A→B chain: completing A's only task satisfies the barrier,
        dispatching B with A's artifact."""
        nodes = [
            make_node("fetch", slug="a"),
            make_node("process", slug="b", depends_on=["a"]),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "fetch", TaskStatus.COMPLETED)
        art = make_dag_artifact("output.dat")
        store.setup_artifacts("run-1", 0, [art])
        callbacks, tracker = _make_callbacks()

        on_task_complete(
            store,
            task_id="task-a",
            callbacks=callbacks,
            resolve_passthrough=_no_passthrough,
        )

        assert len(store.dispatched_tasks) == 1
        assert store.dispatched_tasks[0].node_index == 1
        assert store.dispatched_tasks[0].plugin_name == "process"
        assert store.dispatched_tasks[0].input_artifact_id == art.id
        assert tracker["completed"] == []

    def test_skip_cascade_propagates(self) -> None:
        """When one upstream is skipped and the other completes, the merge node
        and all its descendants are skip-cascaded. Run completes."""
        nodes = [
            make_node("root", slug="r"),
            make_node("source", slug="s"),
            make_node("merge", slug="a", depends_on=["r", "s"]),
            make_node("step_b", slug="b", depends_on=["a"]),
            make_node("step_c", slug="c", depends_on=["b"]),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-r", "run-1", 0, "root", TaskStatus.COMPLETED)
        store.setup_task("task-s", "run-1", 1, "source", TaskStatus.SKIPPED)
        callbacks, tracker = _make_callbacks()

        on_task_complete(
            store,
            task_id="task-r",
            callbacks=callbacks,
            resolve_passthrough=_no_passthrough,
        )

        assert len(store.skipped_nodes) == 3
        assert [s.node_index for s in store.skipped_nodes] == [2, 3, 4]
        assert len(store.dispatched_tasks) == 0
        assert tracker["completed"] == ["run-1"]

    def test_filtered_does_not_cascade(self) -> None:
        nodes = [
            make_node("root", slug="r"),
            make_node("step_a", slug="a", depends_on=["r"]),
            make_node("step_b", slug="b", depends_on=["a"]),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-r", "run-1", 0, "root", TaskStatus.COMPLETED)
        # No artifacts for node 0.
        callbacks, tracker = _make_callbacks()

        on_task_complete(
            store,
            task_id="task-r",
            callbacks=callbacks,
            resolve_passthrough=_no_passthrough,
        )

        assert len(store.filtered_nodes) == 2
        assert [f.node_index for f in store.filtered_nodes] == [1, 2]
        assert len(store.skipped_nodes) == 0
        assert tracker["completed"] == ["run-1"]

    def test_fan_out_limit_exceeded_fails_run(self) -> None:
        nodes = [
            make_node("source", slug="a"),
            make_node("process", slug="b", depends_on=["a"]),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "source", TaskStatus.COMPLETED)
        arts = [make_dag_artifact(f"file_{i}.dat") for i in range(201)]
        store.setup_artifacts("run-1", 0, arts)
        callbacks, tracker = _make_callbacks()

        on_task_complete(
            store,
            task_id="task-a",
            callbacks=callbacks,
            resolve_passthrough=_no_passthrough,
        )

        assert len(store.failed_nodes) == 1
        assert store.failed_nodes[0].node_index == 1
        assert "Fan-out limit" in store.failed_nodes[0].error
        assert tracker["failed"] == ["run-1"]
        assert len(store.cancelled_runs) == 1
        assert len(store.dispatched_tasks) == 0

    def test_grouped_fallback_to_aggregate(self) -> None:
        nodes = [
            make_node("source", slug="a"),
            make_node("merge", slug="b", depends_on=["a"], fan_mode=FanMode.GROUPED),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "source", TaskStatus.COMPLETED)
        arts = [
            make_dag_artifact("f1.dat", origin_artifact_id=None),
            make_dag_artifact("f2.dat", origin_artifact_id=None),
        ]
        store.setup_artifacts("run-1", 0, arts)
        callbacks, tracker = _make_callbacks()

        on_task_complete(
            store,
            task_id="task-a",
            callbacks=callbacks,
            resolve_passthrough=_no_passthrough,
        )

        assert len(store.grouped_tasks) == 0
        assert len(store.dispatched_tasks) == 1
        assert store.dispatched_tasks[0].node_index == 1
        assert store.dispatched_tasks[0].input_artifact_id is None
        assert tracker["completed"] == []

    def test_grouped_dispatches_per_origin(self) -> None:
        """GROUPED fan mode with two distinct origin IDs dispatches two grouped
        tasks — one per origin — rather than a single aggregate."""
        nodes = [
            make_node("source", slug="a"),
            make_node("merge", slug="b", depends_on=["a"], fan_mode=FanMode.GROUPED),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "source", TaskStatus.COMPLETED)
        arts = [
            make_dag_artifact("f1.dat", origin_artifact_id="origin-1"),
            make_dag_artifact("f2.dat", origin_artifact_id="origin-1"),
            make_dag_artifact("f3.dat", origin_artifact_id="origin-2"),
        ]
        store.setup_artifacts("run-1", 0, arts)
        callbacks, tracker = _make_callbacks()

        on_task_complete(
            store,
            task_id="task-a",
            callbacks=callbacks,
            resolve_passthrough=_no_passthrough,
        )

        assert len(store.grouped_tasks) == 2
        assert {g.origin_artifact_id for g in store.grouped_tasks} == {
            "origin-1",
            "origin-2",
        }
        assert len(store.dispatched_tasks) == 0
        assert tracker["completed"] == []

    def test_terminal_run_short_circuits(self) -> None:
        nodes = [make_node("fetch", slug="a")]
        store = MockDagStore()
        store.setup_run("run-1", nodes, status=RunStatus.FAILED)
        store.setup_task("task-a", "run-1", 0, "fetch", TaskStatus.COMPLETED)
        callbacks, tracker = _make_callbacks()

        on_task_complete(
            store,
            task_id="task-a",
            callbacks=callbacks,
            resolve_passthrough=_no_passthrough,
        )

        assert len(store.dispatched_tasks) == 0
        assert tracker["completed"] == []
        assert tracker["failed"] == []

    def test_run_completes_when_all_nodes_done(self) -> None:
        nodes = [
            make_node("fetch", slug="a"),
            make_node("process", slug="b", depends_on=["a"]),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "fetch", TaskStatus.COMPLETED)
        store.setup_task("task-b", "run-1", 1, "process", TaskStatus.COMPLETED)
        callbacks, tracker = _make_callbacks()

        on_task_complete(
            store,
            task_id="task-b",
            callbacks=callbacks,
            resolve_passthrough=_no_passthrough,
        )

        assert tracker["completed"] == ["run-1"]

    def test_claim_race_returns_false(self) -> None:
        nodes = [
            make_node("fetch", slug="a"),
            make_node("process", slug="b", depends_on=["a"]),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "fetch", TaskStatus.COMPLETED)
        store._node_launches.add(("run-1", 1))
        callbacks, tracker = _make_callbacks()

        on_task_complete(
            store,
            task_id="task-a",
            callbacks=callbacks,
            resolve_passthrough=_no_passthrough,
        )

        assert len(store.dispatched_tasks) == 0
        assert len(store.skipped_nodes) == 0
        assert tracker["completed"] == []

    def test_gate_rejection_aggregate_skips(self) -> None:
        """AGGREGATE node with an edge filter that rejects all upstream artifacts
        produces a skip (not a filter), completing the run."""
        nodes = [
            make_node("source", slug="a"),
            make_node(
                "sink",
                slug="b",
                depends_on=["a"],
                fan_mode=FanMode.AGGREGATE,
                edge_filters={
                    "a": EdgeFilter(
                        rules=[
                            FilterRule(
                                field="file_type", operator="eq", value="subtitle"
                            )
                        ]
                    )
                },
            ),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "source", TaskStatus.COMPLETED)
        arts = [make_dag_artifact("movie.mp4", mime="video/mp4")]
        store.setup_artifacts("run-1", 0, arts)
        callbacks, tracker = _make_callbacks()

        on_task_complete(
            store,
            task_id="task-a",
            callbacks=callbacks,
            resolve_passthrough=_no_passthrough,
        )

        assert len(store.skipped_nodes) == 1
        assert store.skipped_nodes[0].node_index == 1
        assert len(store.filtered_nodes) == 0
        assert tracker["completed"] == ["run-1"]

    def test_gate_rejection_grouped_does_not_skip(self) -> None:
        """GROUPED node with an edge filter that rejects all artifacts produces
        a filter (not a skip) — grouped nodes don't skip-cascade."""
        nodes = [
            make_node("source", slug="a"),
            make_node(
                "sink",
                slug="b",
                depends_on=["a"],
                fan_mode=FanMode.GROUPED,
                edge_filters={
                    "a": EdgeFilter(
                        rules=[
                            FilterRule(
                                field="file_type", operator="eq", value="subtitle"
                            )
                        ]
                    )
                },
            ),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "source", TaskStatus.COMPLETED)
        arts = [make_dag_artifact("movie.mp4", mime="video/mp4")]
        store.setup_artifacts("run-1", 0, arts)
        callbacks, tracker = _make_callbacks()

        on_task_complete(
            store,
            task_id="task-a",
            callbacks=callbacks,
            resolve_passthrough=_no_passthrough,
        )

        assert len(store.skipped_nodes) == 0
        assert len(store.filtered_nodes) == 1
        assert store.filtered_nodes[0].node_index == 1
        assert tracker["completed"] == ["run-1"]


class TestOnTaskFailed:
    def test_marks_run_failed_and_cancels(self) -> None:
        nodes = [
            make_node("fetch", slug="a"),
            make_node("process", slug="b", depends_on=["a"]),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "fetch", TaskStatus.COMPLETED)
        store.setup_task("task-b", "run-1", 1, "process", TaskStatus.RUNNING)
        callbacks, tracker = _make_callbacks()

        on_task_failed(
            store,
            task_id="task-b",
            error_message="plugin crashed",
            callbacks=callbacks,
        )

        assert store._tasks["task-b"].status == TaskStatus.FAILED
        assert store._run_status["run-1"] == RunStatus.FAILED
        assert tracker["failed"] == ["run-1"]
        assert len(store.cancelled_runs) == 1


class TestOnTaskCrashed:
    def test_marks_run_crashed_and_cancels(self) -> None:
        nodes = [make_node("fetch", slug="a")]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "fetch", TaskStatus.RUNNING)
        callbacks, tracker = _make_callbacks()

        on_task_crashed(
            store,
            task_id="task-a",
            error_message="OOM killed",
            callbacks=callbacks,
        )

        assert store._tasks["task-a"].status == TaskStatus.CRASHED
        assert store._run_status["run-1"] == RunStatus.CRASHED
        assert tracker["crashed"] == ["run-1"]
        assert len(store.cancelled_runs) == 1


@pytest.mark.parametrize(
    "func,status,tracker_key",
    [
        (on_task_failed, RunStatus.FAILED, "failed"),
        (on_task_crashed, RunStatus.CRASHED, "crashed"),
    ],
)
def test_already_terminal_no_op(
    func: object, status: RunStatus, tracker_key: str
) -> None:
    nodes = [make_node("fetch", slug="a")]
    store = MockDagStore()
    store.setup_run("run-1", nodes, status=status)
    store._terminal_claimed.add("run-1")
    store.setup_task("task-a", "run-1", 0, "fetch", TaskStatus.RUNNING)
    callbacks, tracker = _make_callbacks()

    func(
        store,
        task_id="task-a",
        error_message="too late",
        callbacks=callbacks,
    )

    assert tracker[tracker_key] == []
