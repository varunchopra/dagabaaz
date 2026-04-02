from dagabaaz.models import TaskArtifact
from dagabaaz.store import TaskInputStore
from dagabaaz.task_input import build_task_input, collect_upstream_task_artifacts_bfs
from tests.helpers import make_node as _node


def _art(
    file_path: str = "/data/file.dat",
    file_name: str = "file.dat",
    file_size: int = 1000,
    mime_type: str = "application/octet-stream",
    metadata: dict | None = None,
    origin_artifact_id: str | None = None,
) -> TaskArtifact:
    return TaskArtifact(
        file_path=file_path,
        file_name=file_name,
        file_size=file_size,
        mime_type=mime_type,
        metadata=metadata or {},
        origin_artifact_id=origin_artifact_id,
    )


class MockTaskInputStore:
    """In-memory TaskInputStore for testing."""

    def __init__(
        self,
        artifacts_by_id: dict[str, TaskArtifact] | None = None,
        artifacts_by_node: dict[str, dict[int, list[TaskArtifact]]] | None = None,
        run_inputs: dict[str, dict[str, object]] | None = None,
        grouped: dict[str, list[TaskArtifact]] | None = None,
        broadcast: dict[str, list[TaskArtifact]] | None = None,
        producing_node: dict[str, int] | None = None,
    ) -> None:
        self._artifacts_by_id = artifacts_by_id or {}
        self._artifacts_by_node = artifacts_by_node or {}
        self._run_inputs = run_inputs or {}
        self._grouped = grouped or {}
        self._broadcast = broadcast or {}
        self._producing_node = producing_node or {}

    def get_artifact_data(self, artifact_id: str) -> TaskArtifact | None:
        return self._artifacts_by_id.get(artifact_id)

    def get_task_artifacts_by_node_indices(
        self, run_id: str, node_indices: list[int]
    ) -> list[TaskArtifact]:
        node_map = self._artifacts_by_node.get(run_id, {})
        result: list[TaskArtifact] = []
        for idx in node_indices:
            result.extend(node_map.get(idx, []))
        return result

    def get_grouped_artifacts(
        self, run_id: str, dep_indices: list[int], origin_id: str
    ) -> list[TaskArtifact]:
        return self._grouped.get(origin_id, [])

    def get_broadcast_artifacts(
        self, run_id: str, dep_indices: list[int]
    ) -> list[TaskArtifact]:
        return self._broadcast.get(run_id, [])

    def get_run_input(self, run_id: str) -> dict[str, object]:
        return self._run_inputs.get(run_id, {})

    def get_artifacts_partitioned(
        self, run_id: str, node_indices: list[int]
    ) -> dict[int, list[TaskArtifact]]:
        node_map = self._artifacts_by_node.get(run_id, {})
        return {idx: node_map.get(idx, []) for idx in node_indices if idx in node_map}

    def get_artifact_producing_node(self, artifact_id: str) -> int | None:
        return self._producing_node.get(artifact_id)


assert isinstance(
    MockTaskInputStore(), TaskInputStore
), "MockTaskInputStore does not satisfy TaskInputStore protocol"


class TestBuildTaskInputFanOut:
    """Fan-out: single artifact -> flat dict."""

    def test_returns_artifact_fields(self) -> None:
        art = _art(file_path="/data/output.dat", file_name="output.dat", file_size=5000)
        store = MockTaskInputStore(artifacts_by_id={"art-1": art})
        nodes = [_node("process", slug="p_1")]

        result = build_task_input(
            store, run_id="run-1", node_index=0, input_artifact_id="art-1", nodes=nodes
        )

        assert result["file_path"] == "/data/output.dat"
        assert result["file_name"] == "output.dat"
        assert result["file_size"] == 5000

    def test_missing_artifact_returns_empty(self) -> None:
        store = MockTaskInputStore()
        nodes = [_node("process", slug="p_1")]

        result = build_task_input(
            store,
            run_id="run-1",
            node_index=0,
            input_artifact_id="missing",
            nodes=nodes,
        )

        assert result == {}

    def test_metadata_spread_into_dict(self) -> None:
        art = _art(metadata={"category": "type_a", "external_id": "id_123"})
        store = MockTaskInputStore(artifacts_by_id={"art-1": art})
        nodes = [_node("transform", slug="t_1")]

        result = build_task_input(
            store, run_id="run-1", node_index=0, input_artifact_id="art-1", nodes=nodes
        )

        assert result["category"] == "type_a"
        assert result["external_id"] == "id_123"

    def test_standard_fields_win_over_metadata(self) -> None:
        """Metadata keys must not overwrite file_path/file_name."""
        art = _art(
            file_path="/data/real.dat",
            file_name="real.dat",
            metadata={"file_path": "/fake/path", "file_name": "fake.dat"},
        )
        store = MockTaskInputStore(artifacts_by_id={"art-1": art})
        nodes = [_node("process", slug="p_1")]

        result = build_task_input(
            store, run_id="run-1", node_index=0, input_artifact_id="art-1", nodes=nodes
        )

        assert result["file_path"] == "/data/real.dat"
        assert result["file_name"] == "real.dat"


class TestBuildTaskInputRoot:
    """Root node: no dependencies -> run input."""

    def test_returns_run_input(self) -> None:
        store = MockTaskInputStore(run_inputs={"run-1": {"source": "input://example"}})
        nodes = [_node("process", slug="p_1")]

        result = build_task_input(
            store,
            run_id="run-1",
            node_index=0,
            input_artifact_id=None,
            nodes=nodes,
        )

        assert result["source"] == "input://example"


class TestBuildTaskInputAggregate:
    """Aggregate/dependency: all upstream artifacts as list."""

    def test_returns_artifacts_list(self) -> None:
        arts = [_art(file_name="item_01.dat"), _art(file_name="item_02.dat")]
        store = MockTaskInputStore(
            artifacts_by_node={"run-1": {0: arts}},
        )
        nodes = [
            _node("process", slug="p_1"),
            _node("bundle", slug="bd_1", depends_on=["p_1"]),
        ]

        result = build_task_input(
            store,
            run_id="run-1",
            node_index=1,
            input_artifact_id=None,
            nodes=nodes,
        )

        assert "artifacts" in result
        artifact_list = result["artifacts"]
        assert isinstance(artifact_list, list)
        assert len(artifact_list) == 2
        assert artifact_list[0]["file_name"] == "item_01.dat"

    def test_no_artifacts_returns_empty_artifacts(self) -> None:
        """Dependency nodes with no artifacts return empty list, not run_input."""
        store = MockTaskInputStore(
            artifacts_by_node={"run-1": {}},
            run_inputs={"run-1": {"fallback": "yes"}},
        )
        nodes = [
            _node("process", slug="p_1"),
            _node("bundle", slug="bd_1", depends_on=["p_1"]),
        ]

        result = build_task_input(
            store,
            run_id="run-1",
            node_index=1,
            input_artifact_id=None,
            nodes=nodes,
        )

        assert result == {"artifacts": []}


class TestBuildTaskInputGrouped:
    """Grouped: correlated subset + broadcast."""

    def test_grouped_with_broadcast(self) -> None:
        grouped_arts = [
            _art(file_name="item_01_dest1.out", origin_artifact_id="origin-1")
        ]
        broadcast_arts = [_art(file_name="metadata.json", origin_artifact_id=None)]
        store = MockTaskInputStore(
            grouped={"origin-1": grouped_arts},
            broadcast={"run-1": broadcast_arts},
        )
        nodes = [
            _node("export", slug="ex_1"),
            _node("sink", slug="sk_1", depends_on=["ex_1"]),
        ]

        result = build_task_input(
            store,
            run_id="run-1",
            node_index=1,
            input_artifact_id=None,
            origin_artifact_id="origin-1",
            nodes=nodes,
        )

        assert "artifacts" in result
        assert len(result["artifacts"]) == 2  # type: ignore[arg-type]

    def test_empty_grouped_returns_empty_artifacts(self) -> None:
        """Grouped tasks with no artifacts get empty list."""
        store = MockTaskInputStore(
            run_inputs={"run-1": {"should_not": "appear"}},
        )
        nodes = [
            _node("export", slug="ex_1"),
            _node("sink", slug="sk_1", depends_on=["ex_1"]),
        ]

        result = build_task_input(
            store,
            run_id="run-1",
            node_index=1,
            input_artifact_id=None,
            origin_artifact_id="origin-1",
            nodes=nodes,
        )

        assert result == {"artifacts": []}


class TestCollectUpstreamBfsPassthrough:
    """BFS respects passthrough_indices — stops at non-passthrough nodes."""

    def test_stops_at_non_passthrough(self) -> None:
        """Node 1 (non-passthrough) has no artifacts. BFS should NOT walk
        through it to find node 0's artifacts."""
        store = MockTaskInputStore(
            artifacts_by_node={
                "run-1": {
                    0: [_art(file_name="root.dat")],
                    # Node 1 has no artifacts (non-passthrough processor)
                }
            },
        )
        deps = [[], [0], [1]]  # 0->1->2
        passthrough_indices: set[int] = set()  # no passthrough nodes

        result = collect_upstream_task_artifacts_bfs(
            store, "run-1", [1], deps, passthrough_indices=passthrough_indices
        )

        # Node 1 has no artifacts, and it's NOT passthrough, so BFS stops.
        assert result == []

    def test_walks_through_passthrough(self) -> None:
        """Node 1 (passthrough, e.g. Gate) has no artifacts. BFS should walk
        through it to find node 0's artifacts."""
        store = MockTaskInputStore(
            artifacts_by_node={
                "run-1": {
                    0: [_art(file_name="root.dat")],
                    # Node 1 (Gate) has no artifacts
                }
            },
        )
        deps = [[], [0], [1]]  # 0->1->2
        passthrough_indices = {1}  # node 1 is a Gate

        result = collect_upstream_task_artifacts_bfs(
            store, "run-1", [1], deps, passthrough_indices=passthrough_indices
        )

        assert len(result) == 1
        assert result[0].file_name == "root.dat"
