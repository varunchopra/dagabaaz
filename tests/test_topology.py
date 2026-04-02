import pytest

from dagabaaz.topology import RunTopology, configure_cache_size, evict, get_or_build
from tests.helpers import make_node

pytestmark = pytest.mark.usefixtures("reset_topology_cache")


class TestTopologyCache:
    def test_get_or_build_caches_result(self) -> None:
        nodes = [make_node("a", slug="a"), make_node("b", slug="b", depends_on=["a"])]
        call_count = 0

        def fetch():
            nonlocal call_count
            call_count += 1
            return nodes

        t1 = get_or_build("run-1", fetch, lambda p: False)
        t2 = get_or_build("run-1", fetch, lambda p: False)
        assert t1 is t2
        assert call_count == 1

    def test_evict_removes_from_cache(self) -> None:
        nodes = [make_node("a", slug="a")]
        call_count = 0

        def fetch():
            nonlocal call_count
            call_count += 1
            return nodes

        get_or_build("run-1", fetch, lambda p: False)
        assert call_count == 1
        evict("run-1")
        get_or_build("run-1", fetch, lambda p: False)
        assert call_count == 2

    @pytest.mark.parametrize("size", [0, -1])
    def test_configure_cache_size_rejects_non_positive(self, size: int) -> None:
        with pytest.raises(ValueError):
            configure_cache_size(size)

    def test_fifo_eviction(self) -> None:
        from dagabaaz import topology

        topology._MAX_TOPOLOGY_CACHE = 2

        nodes = [make_node("a", slug="a")]
        for rid in ["run-1", "run-2", "run-3"]:
            get_or_build(rid, lambda n=nodes: n, lambda p: False)

        assert "run-1" not in topology._topology_cache
        assert "run-2" in topology._topology_cache
        assert "run-3" in topology._topology_cache


class TestRunTopologyBuild:
    def test_build_computes_all_fields(self) -> None:
        nodes = [
            make_node("gate", slug="g"),
            make_node("process", slug="p", depends_on=["g"]),
        ]

        topo = RunTopology.build(nodes, resolve_passthrough=lambda p: p == "gate")

        assert topo.nodes is nodes
        assert topo.deps == [[], [0]]
        assert topo.slug_to_index == {"g": 0, "p": 1}
        assert topo.passthrough_indices == {0}
        assert topo.children == [[1], []]
        assert len(topo.edge_filters) == 2
