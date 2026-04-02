"""Run topology computation and caching.

Pre-computes structural graph data (dependencies, slug map, passthrough
set, children adjacency, edge filters) for a run's immutable node
definitions. The topology is computed once per run and cached for reuse
across multiple ``on_task_complete`` invocations.

Cache uses OrderedDict with FIFO eviction — rebuild is cheap (pure
computation from immutable node data), so evicting an active run just
costs one recomputation.
"""

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from dagabaaz.graph import (
    build_children,
    build_slug_to_index_map,
    rekey_edge_filters_by_index,
    resolve_dependency_indices,
)
from dagabaaz.models import DagNode, EdgeFilter

logger = logging.getLogger(__name__)

ResolvePassthrough = Callable[[str], bool]
"""Given a plugin name, return whether it allows BFS artifact passthrough."""


@dataclass(frozen=True, slots=True)
class RunTopology:
    """Pre-computed structural graph data for a single run.

    Built once per run and cached — reused across all ``on_task_complete``
    calls for the same run. Avoids recomputing deps, slug map, passthrough
    set, and children adjacency on every task completion.
    """

    nodes: list[DagNode]
    deps: list[list[int]]
    slug_to_index: dict[str, int]
    passthrough_indices: set[int]
    children: list[list[int]]
    edge_filters: list[dict[int, EdgeFilter]]

    @staticmethod
    def build(
        nodes: list[DagNode],
        resolve_passthrough: ResolvePassthrough,
    ) -> "RunTopology":
        deps = resolve_dependency_indices(nodes)
        slug_to_index = build_slug_to_index_map(nodes)
        passthrough_indices = {
            i for i, n in enumerate(nodes) if resolve_passthrough(n.plugin)
        }
        children = build_children(deps)
        edge_filters = [rekey_edge_filters_by_index(n, slug_to_index) for n in nodes]
        return RunTopology(
            nodes=nodes,
            deps=deps,
            slug_to_index=slug_to_index,
            passthrough_indices=passthrough_indices,
            children=children,
            edge_filters=edge_filters,
        )


_topology_lock = threading.Lock()
_topology_cache: OrderedDict[str, RunTopology] = OrderedDict()
_MAX_TOPOLOGY_CACHE = 500


def configure_cache_size(max_entries: int) -> None:
    """Set the maximum number of cached run topologies.

    Must be called before any runs start. Default is 500.
    """
    global _MAX_TOPOLOGY_CACHE
    if max_entries < 1:
        raise ValueError("Cache size must be at least 1")
    _MAX_TOPOLOGY_CACHE = max_entries


def get_or_build(
    run_id: str,
    fetch_nodes: Callable[[], list[DagNode] | None],
    resolve_passthrough: ResolvePassthrough,
) -> RunTopology | None:
    """Return cached topology for a run, building it if necessary.

    ``fetch_nodes`` is called only on cache miss — and outside the lock
    to avoid holding the lock during a DB round-trip.

    Returns None if ``fetch_nodes`` returns None (run not found).
    """
    with _topology_lock:
        topology = _topology_cache.get(run_id)
        if topology is not None:
            return topology

    # fetch_nodes() may hit the DB — do it outside the lock.
    nodes = fetch_nodes()
    if nodes is None:
        return None

    topology = RunTopology.build(nodes, resolve_passthrough)

    with _topology_lock:
        # Double-check — another thread may have built it while we fetched.
        if run_id not in _topology_cache:
            _topology_cache[run_id] = topology
            while len(_topology_cache) > _MAX_TOPOLOGY_CACHE:
                evicted_id, _ = _topology_cache.popitem(last=False)
                logger.debug("Topology cache evicted run %s (cache full)", evicted_id)
        else:
            topology = _topology_cache[run_id]
    return topology


def evict(run_id: str) -> None:
    """Remove a run's cached topology."""
    with _topology_lock:
        _topology_cache.pop(run_id, None)
