from dagabaaz.graph import (
    assign_slugs,
    bfs_collect,
    build_slug_to_index_map,
    find_ready_nodes,
    find_root_nodes,
    rekey_edge_filters_by_index,
    resolve_dependency_indices,
)
from dagabaaz.models import EdgeFilter, FilterRule
from tests.helpers import make_node as _node


class TestAssignSlugs:
    def test_assigns_missing_slugs(self) -> None:
        nodes = [_node("step_a"), _node("step_b"), _node("step_c")]
        assign_slugs(nodes)
        assert nodes[0].slug == "step_a_1"
        assert nodes[1].slug == "step_b_1"
        assert nodes[2].slug == "step_c_1"

    def test_preserves_existing_slugs(self) -> None:
        nodes = [_node("step_a", slug="my_dl"), _node("step_b")]
        assign_slugs(nodes)
        assert nodes[0].slug == "my_dl"
        assert nodes[1].slug == "step_b_1"

    def test_avoids_collision_with_existing(self) -> None:
        nodes = [_node("dl", slug="dl_1"), _node("dl")]
        assign_slugs(nodes)
        assert nodes[0].slug == "dl_1"
        assert nodes[1].slug == "dl_2"

    def test_idempotent(self) -> None:
        nodes = [_node("a"), _node("b")]
        assign_slugs(nodes)
        slugs_first = [n.slug for n in nodes]
        assign_slugs(nodes)
        assert [n.slug for n in nodes] == slugs_first


class TestBuildSlugIndex:
    def test_basic_index(self) -> None:
        nodes = [_node("a", slug="a_1"), _node("b", slug="b_1")]
        assert build_slug_to_index_map(nodes) == {"a_1": 0, "b_1": 1}

    def test_skips_empty_slugs(self) -> None:
        nodes = [_node("a", slug="a_1"), _node("b")]
        assert build_slug_to_index_map(nodes) == {"a_1": 0}


class TestNormalizeDependencies:
    def test_linear_chain(self) -> None:
        nodes = [
            _node("a", slug="a_1"),
            _node("b", slug="b_1", depends_on=["a_1"]),
            _node("c", slug="c_1", depends_on=["b_1"]),
        ]
        deps = resolve_dependency_indices(nodes)
        assert deps == [[], [0], [1]]

    def test_diamond(self) -> None:
        nodes = [
            _node("root", slug="root_1"),
            _node("left", slug="left_1", depends_on=["root_1"]),
            _node("right", slug="right_1", depends_on=["root_1"]),
            _node("join", slug="join_1", depends_on=["left_1", "right_1"]),
        ]
        deps = resolve_dependency_indices(nodes)
        assert deps == [[], [0], [0], [1, 2]]

    def test_no_deps(self) -> None:
        nodes = [_node("a", slug="a_1"), _node("b", slug="b_1")]
        deps = resolve_dependency_indices(nodes)
        assert deps == [[], []]


class TestResolveEdgeFilters:
    def test_resolves_slug_to_index(self) -> None:
        ef = EdgeFilter(
            rules=[FilterRule(field="file_type", operator="eq", value="video")]
        )
        node = _node("encode", depends_on=["dl_1"], edge_filters={"dl_1": ef})
        slug_index = {"dl_1": 0, "encode_1": 1}
        resolved = rekey_edge_filters_by_index(node, slug_index)
        assert 0 in resolved
        assert resolved[0] == ef

    def test_unknown_slug_dropped(self) -> None:
        ef = EdgeFilter(
            rules=[FilterRule(field="file_type", operator="eq", value="video")]
        )
        node = _node("encode", edge_filters={"ghost": ef})
        resolved = rekey_edge_filters_by_index(node, {"dl_1": 0})
        assert resolved == {}


class TestFindReadyNodes:
    def test_root_nodes_ready_initially(self) -> None:
        deps = [[], [0], [0]]  # node 0 is root
        ready = find_ready_nodes(deps, completed=set(), launched=set())
        assert ready == [0]

    def test_multiple_roots(self) -> None:
        deps = [[], [], [0, 1]]
        ready = find_ready_nodes(deps, completed=set(), launched=set())
        assert ready == [0, 1]

    def test_dep_satisfied_unlocks(self) -> None:
        deps = [[], [0]]
        ready = find_ready_nodes(deps, completed={0}, launched={0})
        assert ready == [1]

    def test_partial_deps_not_ready(self) -> None:
        deps = [[], [], [0, 1]]
        ready = find_ready_nodes(deps, completed={0}, launched={0})
        assert ready == [1]  # node 2 needs both 0 AND 1

    def test_all_deps_satisfied(self) -> None:
        deps = [[], [], [0, 1]]
        ready = find_ready_nodes(deps, completed={0, 1}, launched={0, 1})
        assert ready == [2]

    def test_already_launched_not_returned(self) -> None:
        deps = [[], [0]]
        ready = find_ready_nodes(deps, completed={0}, launched={0, 1})
        assert ready == []

    def test_empty_graph(self) -> None:
        assert find_ready_nodes([], set(), set()) == []

    def test_single_node_no_deps(self) -> None:
        ready = find_ready_nodes([[]], completed=set(), launched=set())
        assert ready == [0]


class TestFindRootNodes:
    """find_root_nodes returns indices of nodes with no dependencies."""

    def test_single_root(self) -> None:
        nodes = [_node("a", slug="a_1"), _node("b", slug="b_1", depends_on=["a_1"])]
        assert find_root_nodes(nodes) == [0]

    def test_multiple_roots(self) -> None:
        nodes = [
            _node("a", slug="a_1"),
            _node("b", slug="b_1"),
            _node("c", slug="c_1", depends_on=["a_1", "b_1"]),
        ]
        assert find_root_nodes(nodes) == [0, 1]

    def test_all_roots(self) -> None:
        nodes = [_node("a", slug="a_1"), _node("b", slug="b_1")]
        assert find_root_nodes(nodes) == [0, 1]

    def test_empty_graph(self) -> None:
        assert find_root_nodes([]) == []


class TestBfsDepthGuard:
    """BFS depth and cycle handling."""

    @staticmethod
    def _fetch_nothing(run_id: str, indices: list[int]) -> list:
        """Fetch function that always returns empty — forces BFS to keep walking."""
        return []

    def test_cycle_returns_empty(self) -> None:
        """Cycle: 0->1->0. BFS visited set prevents infinite loop."""
        deps = [[1], [0]]
        result = bfs_collect(
            self._fetch_nothing,
            "run-1",
            [0],
            deps,
            passthrough_indices={0, 1},
        )
        assert result == []

    def test_max_depth_exceeded_returns_empty(self) -> None:
        """Chain longer than max_depth returns empty."""
        # Chain: 0->1->2->3->4
        deps = [[], [0], [1], [2], [3]]
        result = bfs_collect(
            self._fetch_nothing,
            "run-1",
            [4],
            deps,
            passthrough_indices={0, 1, 2, 3, 4},
            max_depth=2,
        )
        assert result == []
