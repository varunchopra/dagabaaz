import pytest

from dagabaaz.constants import ArtifactSelector, FanMode, RunStatus, TaskStatus
from dagabaaz.models import DagNode, EdgeFilter, FilterRule
from dagabaaz.orchestrator import (
    OrchestratorCallbacks,
    abort_run,
    on_task_complete,
    on_task_crashed,
    on_task_failed,
    reconcile_run,
    start_run,
)
from tests.helpers import MockDagStore, make_dag_artifact, make_node

pytestmark = pytest.mark.usefixtures("reset_topology_cache")


def _make_callbacks() -> tuple[OrchestratorCallbacks, dict[str, list[str]]]:
    tracker: dict[str, list[str]] = {
        "completed": [],
        "failed": [],
        "crashed": [],
        "cancelled": [],
    }
    callbacks = OrchestratorCallbacks(
        on_run_completed=lambda rid: tracker["completed"].append(rid),
        on_run_failed=lambda rid: tracker["failed"].append(rid),
        on_run_crashed=lambda rid: tracker["crashed"].append(rid),
        on_run_cancelled=lambda rid: tracker["cancelled"].append(rid),
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

    def test_missing_slug_is_assigned_before_dispatch(self) -> None:
        nodes = [make_node("source")]
        store = MockDagStore()
        store.setup_run("run-1", nodes)

        result = start_run(store, "run-1", nodes)

        assert result == [0]
        assert nodes[0].slug == "source_1"
        assert len(store.dispatched_tasks) == 1

    def test_dependency_cycle_rejected_before_dispatch(self) -> None:
        nodes = [
            make_node("root", slug="root"),
            make_node("a", slug="a", depends_on=["b"]),
            make_node("b", slug="b", depends_on=["a"]),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)

        with pytest.raises(ValueError, match="dependency cycle"):
            start_run(store, "run-1", nodes)

        assert store.dispatched_tasks == []
        assert store.get_launched_node_indices("run-1") == set()

    def test_unknown_dependency_rejected_before_dispatch(self) -> None:
        nodes = [
            make_node("publish", slug="publish", depends_on=["authenticate_typo"]),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)

        with pytest.raises(ValueError, match="Unknown dependency slug"):
            start_run(store, "run-1", nodes)

        assert store.dispatched_tasks == []
        assert store.get_launched_node_indices("run-1") == set()

    @pytest.mark.parametrize(
        "nodes",
        [
            [make_node("first", slug="same"), make_node("second", slug="same")],
            [
                make_node("source", slug="source"),
                make_node(
                    "sink",
                    slug="sink",
                    depends_on=["source", "source"],
                ),
            ],
            [
                make_node("source", slug="source"),
                make_node("other", slug="other"),
                make_node(
                    "sink",
                    slug="sink",
                    depends_on=["source"],
                    edge_filters={"other": EdgeFilter()},
                ),
            ],
        ],
        ids=[
            "duplicate-slug",
            "repeated-dependency",
            "filter-on-non-dependency",
        ],
    )
    def test_ambiguous_graph_rejected_before_dispatch(
        self, nodes: list[DagNode]
    ) -> None:
        store = MockDagStore()
        store.setup_run("run-1", nodes)

        with pytest.raises(ValueError):
            start_run(store, "run-1", nodes)

        assert store.dispatched_tasks == []
        assert store.get_launched_node_indices("run-1") == set()

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

    def test_grouped_selector_dispatches_each_matching_origin(self) -> None:
        nodes = [
            make_node("source", slug="a"),
            make_node(
                "merge",
                slug="b",
                depends_on=["a"],
                edge_filters={
                    "a": EdgeFilter(select=ArtifactSelector.LARGEST),
                },
                fan_mode=FanMode.GROUPED,
            ),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "source", TaskStatus.COMPLETED)
        store.setup_artifacts(
            "run-1",
            0,
            [
                make_dag_artifact(
                    "origin-1-small.dat", size=100, origin_artifact_id="origin-1"
                ),
                make_dag_artifact(
                    "origin-1-large.dat", size=200, origin_artifact_id="origin-1"
                ),
                make_dag_artifact(
                    "origin-2-small.dat", size=300, origin_artifact_id="origin-2"
                ),
                make_dag_artifact(
                    "origin-2-large.dat", size=400, origin_artifact_id="origin-2"
                ),
            ],
        )
        callbacks, _ = _make_callbacks()

        on_task_complete(
            store,
            task_id="task-a",
            callbacks=callbacks,
            resolve_passthrough=_no_passthrough,
        )

        assert {task.origin_artifact_id for task in store.grouped_tasks} == {
            "origin-1",
            "origin-2",
        }

    def test_grouped_selector_compares_broadcast_per_origin(self) -> None:
        nodes = [
            make_node("source", slug="a"),
            make_node(
                "merge",
                slug="b",
                depends_on=["a"],
                edge_filters={
                    "a": EdgeFilter(select=ArtifactSelector.LARGEST),
                },
                fan_mode=FanMode.GROUPED,
            ),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "source", TaskStatus.COMPLETED)
        store.setup_artifacts(
            "run-1",
            0,
            [
                make_dag_artifact(
                    "origin-1.dat", size=100, origin_artifact_id="origin-1"
                ),
                make_dag_artifact(
                    "origin-2.dat", size=300, origin_artifact_id="origin-2"
                ),
                make_dag_artifact("broadcast.dat", size=200),
            ],
        )
        callbacks, _ = _make_callbacks()

        on_task_complete(
            store,
            task_id="task-a",
            callbacks=callbacks,
            resolve_passthrough=_no_passthrough,
        )

        assert [task.origin_artifact_id for task in store.grouped_tasks] == [
            "origin-2"
        ]

    def test_grouped_fan_out_limit_fails_run(self) -> None:
        nodes = [
            make_node("source", slug="a"),
            make_node(
                "merge",
                slug="b",
                depends_on=["a"],
                edge_filters={
                    "a": EdgeFilter(select=ArtifactSelector.LARGEST),
                },
                fan_mode=FanMode.GROUPED,
            ),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "source", TaskStatus.COMPLETED)
        artifacts = [
            make_dag_artifact(
                f"origin-{index}.dat",
                size=1000 + index,
                origin_artifact_id=f"origin-{index}",
            )
            for index in range(201)
        ]
        artifacts.append(make_dag_artifact("broadcast.dat", size=1))
        store.setup_artifacts("run-1", 0, artifacts)
        callbacks, tracker = _make_callbacks()

        on_task_complete(
            store,
            task_id="task-a",
            callbacks=callbacks,
            resolve_passthrough=_no_passthrough,
        )

        assert store.grouped_tasks == []
        assert len(store.failed_nodes) == 1
        assert "Grouped fan-out limit" in store.failed_nodes[0].error
        assert tracker["failed"] == ["run-1"]

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

    def test_stranded_downstream_discovered_after_unrelated_event(self) -> None:
        """Just-completed node has no children; readiness must still discover
        a node whose parent completed in an earlier turn."""
        nodes = [
            make_node("root", slug="a"),
            make_node("parallel_left", slug="b", depends_on=["a"]),
            make_node("retry_target", slug="c", depends_on=["a"]),
            make_node("downstream_of_b", slug="d", depends_on=["b"]),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "root", TaskStatus.COMPLETED)
        store.setup_task("task-b", "run-1", 1, "parallel_left", TaskStatus.COMPLETED)
        store.setup_task("task-c", "run-1", 2, "retry_target", TaskStatus.COMPLETED)
        store.setup_artifacts("run-1", 0, [make_dag_artifact("from_a.dat")])
        store.setup_artifacts("run-1", 1, [make_dag_artifact("from_b.dat")])
        store.setup_artifacts("run-1", 2, [make_dag_artifact("from_c.dat")])
        callbacks, tracker = _make_callbacks()

        on_task_complete(
            store,
            task_id="task-c",
            callbacks=callbacks,
            resolve_passthrough=_no_passthrough,
        )

        assert len(store.dispatched_tasks) == 1
        assert store.dispatched_tasks[0].node_index == 3
        assert tracker["completed"] == []

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


class TestReconcileRun:
    def test_dispatches_ready_nodes_without_preceding_event(self) -> None:
        """Direct invocation (e.g. from retry caller) discovers nodes whose
        parents completed before any event the engine has seen."""
        nodes = [
            make_node("root", slug="a"),
            make_node("child", slug="b", depends_on=["a"]),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "root", TaskStatus.COMPLETED)
        store.setup_artifacts("run-1", 0, [make_dag_artifact("output.dat")])
        callbacks, tracker = _make_callbacks()

        reconcile_run(
            store,
            "run-1",
            callbacks=callbacks,
            resolve_passthrough=_no_passthrough,
        )

        assert len(store.dispatched_tasks) == 1
        assert store.dispatched_tasks[0].node_index == 1


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


class TestAbortRun:
    def test_failed_fires_failed_callback(self) -> None:
        nodes = [
            make_node("fetch", slug="a"),
            make_node("process", slug="b", depends_on=["a"]),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "fetch", TaskStatus.RUNNING)
        callbacks, tracker = _make_callbacks()

        result = abort_run(
            store,
            run_id="run-1",
            reason="run deadline exceeded",
            callbacks=callbacks,
        )

        assert result is True
        assert store._run_status["run-1"] == RunStatus.FAILED
        assert tracker["failed"] == ["run-1"]
        assert len(store.cancelled_runs) == 1

    def test_crashed_fires_crashed_callback(self) -> None:
        nodes = [make_node("fetch", slug="a")]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "fetch", TaskStatus.RUNNING)
        callbacks, tracker = _make_callbacks()

        result = abort_run(
            store,
            run_id="run-1",
            reason="infra failure",
            callbacks=callbacks,
            status=RunStatus.CRASHED,
        )

        assert result is True
        assert store._run_status["run-1"] == RunStatus.CRASHED
        assert tracker["crashed"] == ["run-1"]
        assert len(store.cancelled_runs) == 1

    def test_cancelled_fires_cancelled_callback(self) -> None:
        nodes = [make_node("fetch", slug="a")]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "fetch", TaskStatus.RUNNING)
        callbacks, tracker = _make_callbacks()

        result = abort_run(
            store,
            run_id="run-1",
            reason="Cancelled by user",
            callbacks=callbacks,
            status=RunStatus.CANCELLED,
        )

        assert result is True
        assert store._run_status["run-1"] == RunStatus.CANCELLED
        assert tracker["cancelled"] == ["run-1"]
        assert len(store.cancelled_runs) == 1

    def test_already_terminal_returns_false(self) -> None:
        nodes = [make_node("fetch", slug="a")]
        store = MockDagStore()
        store.setup_run("run-1", nodes, status=RunStatus.FAILED)
        store._terminal_claimed.add("run-1")
        callbacks, tracker = _make_callbacks()

        result = abort_run(
            store,
            run_id="run-1",
            reason="too late",
            callbacks=callbacks,
        )

        assert result is False
        assert tracker["failed"] == []

    def test_rejects_invalid_status(self) -> None:
        store = MockDagStore()
        callbacks, _ = _make_callbacks()

        with pytest.raises(ValueError, match="FAILED, CRASHED, or CANCELLED"):
            abort_run(
                store,
                run_id="run-1",
                reason="nope",
                callbacks=callbacks,
                status=RunStatus.COMPLETED,
            )

    def test_evicts_topology_cache(self) -> None:
        nodes = [make_node("fetch", slug="a")]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "fetch", TaskStatus.RUNNING)

        from dagabaaz.topology import _topology_cache, get_or_build

        get_or_build("run-1", lambda: nodes, lambda _: False)
        assert "run-1" in _topology_cache

        callbacks, _ = _make_callbacks()
        abort_run(store, run_id="run-1", reason="timeout", callbacks=callbacks)

        assert "run-1" not in _topology_cache


class _FailingProgressStore(MockDagStore):
    """MockDagStore where the first set_run_progress call raises."""

    def __init__(self) -> None:
        super().__init__()
        self._fail_next = True

    def set_run_progress(self, run_id: str, completed_count: int) -> None:
        if self._fail_next:
            self._fail_next = False
            raise RuntimeError("simulated DB failure")
        super().set_run_progress(run_id, completed_count)


class TestProgressContract:
    """Progress is current at every terminal callback, and progress-write
    failure leaves the run recoverable rather than permanently terminal."""

    def test_completion_callback_sees_final_progress(self) -> None:
        nodes = [
            make_node("fetch", slug="a"),
            make_node("process", slug="b", depends_on=["a"]),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "fetch", TaskStatus.COMPLETED)
        store.setup_task("task-b", "run-1", 1, "process", TaskStatus.COMPLETED)
        captured: list[int | None] = []
        callbacks = OrchestratorCallbacks(
            on_run_completed=lambda rid: captured.append(
                store._run_progress.get(rid)
            ),
            on_run_failed=lambda _: None,
            on_run_crashed=lambda _: None,
            on_run_cancelled=lambda _: None,
        )

        on_task_complete(
            store,
            task_id="task-b",
            callbacks=callbacks,
            resolve_passthrough=_no_passthrough,
        )

        assert captured == [2]

    def test_in_loop_failure_callback_sees_current_progress(self) -> None:
        nodes = [
            make_node("source", slug="a"),
            make_node("process", slug="b", depends_on=["a"]),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "source", TaskStatus.COMPLETED)
        arts = [make_dag_artifact(f"file_{i}.dat") for i in range(201)]
        store.setup_artifacts("run-1", 0, arts)
        captured: list[int | None] = []
        callbacks = OrchestratorCallbacks(
            on_run_completed=lambda _: None,
            on_run_failed=lambda rid: captured.append(
                store._run_progress.get(rid)
            ),
            on_run_crashed=lambda _: None,
            on_run_cancelled=lambda _: None,
        )

        on_task_complete(
            store,
            task_id="task-a",
            callbacks=callbacks,
            resolve_passthrough=_no_passthrough,
        )

        assert captured == [1]

    def test_on_task_failed_callback_sees_current_progress(self) -> None:
        nodes = [
            make_node("fetch", slug="a"),
            make_node("process", slug="b", depends_on=["a"]),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "fetch", TaskStatus.COMPLETED)
        store.setup_task("task-b", "run-1", 1, "process", TaskStatus.RUNNING)
        captured: list[int | None] = []
        callbacks = OrchestratorCallbacks(
            on_run_completed=lambda _: None,
            on_run_failed=lambda rid: captured.append(
                store._run_progress.get(rid)
            ),
            on_run_crashed=lambda _: None,
            on_run_cancelled=lambda _: None,
        )

        on_task_failed(
            store,
            task_id="task-b",
            error_message="plugin crashed",
            callbacks=callbacks,
        )

        assert captured == [2]

    def test_on_task_crashed_callback_sees_current_progress(self) -> None:
        nodes = [
            make_node("fetch", slug="a"),
            make_node("process", slug="b", depends_on=["a"]),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "fetch", TaskStatus.COMPLETED)
        store.setup_task("task-b", "run-1", 1, "process", TaskStatus.RUNNING)
        captured: list[int | None] = []
        callbacks = OrchestratorCallbacks(
            on_run_completed=lambda _: None,
            on_run_failed=lambda _: None,
            on_run_crashed=lambda rid: captured.append(
                store._run_progress.get(rid)
            ),
            on_run_cancelled=lambda _: None,
        )

        on_task_crashed(
            store,
            task_id="task-b",
            error_message="OOM killed",
            callbacks=callbacks,
        )

        assert captured == [2]

    @pytest.mark.parametrize(
        "abort_status,callback_field",
        [
            (RunStatus.FAILED, "on_run_failed"),
            (RunStatus.CRASHED, "on_run_crashed"),
            (RunStatus.CANCELLED, "on_run_cancelled"),
        ],
    )
    def test_abort_callback_sees_current_progress(
        self, abort_status: RunStatus, callback_field: str
    ) -> None:
        nodes = [
            make_node("fetch", slug="a"),
            make_node("process", slug="b", depends_on=["a"]),
        ]
        store = MockDagStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "fetch", TaskStatus.COMPLETED)
        store.setup_task("task-b", "run-1", 1, "process", TaskStatus.RUNNING)
        captured: list[int | None] = []
        capture = lambda rid: captured.append(store._run_progress.get(rid))  # noqa: E731
        callback_kwargs = {
            "on_run_completed": lambda _: None,
            "on_run_failed": lambda _: None,
            "on_run_crashed": lambda _: None,
            "on_run_cancelled": lambda _: None,
            callback_field: capture,
        }
        callbacks = OrchestratorCallbacks(**callback_kwargs)

        abort_run(
            store,
            run_id="run-1",
            reason="external trigger",
            callbacks=callbacks,
            status=abort_status,
        )

        assert captured == [1]

    def test_completion_progress_failure_is_recoverable(self) -> None:
        nodes = [
            make_node("fetch", slug="a"),
            make_node("process", slug="b", depends_on=["a"]),
        ]
        store = _FailingProgressStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "fetch", TaskStatus.COMPLETED)
        store.setup_task("task-b", "run-1", 1, "process", TaskStatus.COMPLETED)
        callbacks, tracker = _make_callbacks()

        with pytest.raises(RuntimeError, match="simulated DB failure"):
            on_task_complete(
                store,
                task_id="task-b",
                callbacks=callbacks,
                resolve_passthrough=_no_passthrough,
            )

        assert store._run_status["run-1"] == RunStatus.RUNNING
        assert tracker["completed"] == []

        on_task_complete(
            store,
            task_id="task-b",
            callbacks=callbacks,
            resolve_passthrough=_no_passthrough,
        )

        assert store._run_status["run-1"] == RunStatus.COMPLETED
        assert tracker["completed"] == ["run-1"]
        assert store._run_progress["run-1"] == 2

    def test_on_task_failed_progress_failure_is_recoverable(self) -> None:
        nodes = [
            make_node("fetch", slug="a"),
            make_node("process", slug="b", depends_on=["a"]),
        ]
        store = _FailingProgressStore()
        store.setup_run("run-1", nodes)
        store.setup_task("task-a", "run-1", 0, "fetch", TaskStatus.COMPLETED)
        store.setup_task("task-b", "run-1", 1, "process", TaskStatus.RUNNING)
        callbacks, tracker = _make_callbacks()

        with pytest.raises(RuntimeError, match="simulated DB failure"):
            on_task_failed(
                store,
                task_id="task-b",
                error_message="plugin crashed",
                callbacks=callbacks,
            )

        assert store._run_status["run-1"] == RunStatus.RUNNING
        assert tracker["failed"] == []

        on_task_failed(
            store,
            task_id="task-b",
            error_message="plugin crashed",
            callbacks=callbacks,
        )

        assert store._run_status["run-1"] == RunStatus.FAILED
        assert tracker["failed"] == ["run-1"]
        assert store._run_progress["run-1"] == 2
