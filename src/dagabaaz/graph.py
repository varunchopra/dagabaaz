"""Graph algorithms for DAG pipelines — slug assignment, dependency resolution,
readiness detection, and BFS artifact collection.
"""

import logging
from collections.abc import Callable

from dagabaaz.constants import MAX_BFS_DEPTH
from dagabaaz.models import DagArtifact, DagNode, EdgeFilter
from dagabaaz.store import DagStore

logger = logging.getLogger(__name__)


def assign_slugs(nodes: list[DagNode]) -> None:
    """Ensure every node has a slug. Modifies in-place. Idempotent."""
    existing = {node.slug for node in nodes if node.slug}
    counters: dict[str, int] = {}
    for node in nodes:
        if not node.slug:
            counter = counters.get(node.plugin, 0) + 1
            while f"{node.plugin}_{counter}" in existing:
                counter += 1
            node.slug = f"{node.plugin}_{counter}"
            existing.add(node.slug)
            counters[node.plugin] = counter


def build_slug_to_index_map(nodes: list[DagNode]) -> dict[str, int]:
    """Map node slugs to their array positions.

    Used at every runtime boundary where slug-based references need
    to be converted to index-based lookups.
    """
    return {node.slug: i for i, node in enumerate(nodes) if node.slug}


def resolve_dependency_indices(nodes: list[DagNode]) -> list[list[int]]:
    """Convert slug-based depends_on to index-based lists for runtime.

    Pipeline definitions use slugs for stable references. The runtime
    (orchestrator, worker, tasks DB) stays index-based. This function
    is the conversion boundary.

    Raises ``ValueError`` if a dependency references an unknown slug.
    """
    index = build_slug_to_index_map(nodes)
    result: list[list[int]] = []
    for node in nodes:
        deps: list[int] = []
        for dependency_slug in node.depends_on:
            if dependency_slug not in index:
                raise ValueError(
                    f"Unknown dependency slug {dependency_slug!r} in node {node.slug!r}"
                )
            deps.append(index[dependency_slug])
        result.append(deps)
    return result


def find_root_nodes(nodes: list[DagNode]) -> list[int]:
    """Return indices of nodes with no dependencies (root/entry nodes).

    A root node has an empty ``depends_on`` list after slug→index
    resolution. These are the starting points for a run — they receive
    the user's input and produce the first artifacts.
    """
    deps = resolve_dependency_indices(nodes)
    return [i for i, d in enumerate(deps) if not d]


def rekey_edge_filters_by_index(
    node: DagNode,
    slug_to_index: dict[str, int],
) -> dict[int, EdgeFilter]:
    """Convert slug-keyed edge_filters to index-keyed for runtime.

    Edge filters are stored keyed by dependency slug in the pipeline
    definition. At runtime, artifact queries use node indices, so
    filter keys must be converted. Unknown slugs are dropped with a
    debug log — pipeline validation should catch these at creation time.
    """
    result: dict[int, EdgeFilter] = {}
    for slug, edge_filter in node.edge_filters.items():
        if slug in slug_to_index:
            result[slug_to_index[slug]] = edge_filter
        else:
            logger.debug(
                "Edge filter slug %r not found in slug map for node %r, dropping",
                slug,
                node.slug,
            )
    return result


def build_children(deps: list[list[int]]) -> list[list[int]]:
    """Build the inverse adjacency list (parent → children).

    Given ``deps[i]`` = list of parents for node i, return ``children[j]``
    = list of nodes that depend on node j. Used by ``find_ready_nodes``
    to limit readiness checks to direct descendants of newly-completed
    nodes instead of scanning all N nodes on every call.
    """
    children: list[list[int]] = [[] for _ in range(len(deps))]
    for child_idx, parents in enumerate(deps):
        for parent_idx in parents:
            children[parent_idx].append(child_idx)
    return children


def find_ready_nodes(
    deps: list[list[int]],
    completed: set[int],
    launched: set[int],
    *,
    children: list[list[int]] | None = None,
    changed_indices: set[int] | None = None,
) -> list[int]:
    """Return node indices whose dependencies are all satisfied.

    A node is ready when:
    1. All its dependency indices are in ``completed``
    2. It hasn't been launched yet (not in ``launched``)

    When ``children`` and ``changed_indices`` are both provided, only
    direct descendants of the changed nodes are checked — O(C * D)
    where C = number of children of changed nodes. Falls back to full
    O(N * D) scan otherwise.
    """
    if children is not None and changed_indices is not None:
        # Targeted check: only inspect children of nodes that just changed.
        candidates: set[int] = set()
        for idx in changed_indices:
            if idx < len(children):
                candidates.update(children[idx])
        ready: list[int] = []
        for i in candidates:
            if i in launched:
                continue
            if all(dep in completed for dep in deps[i]):
                ready.append(i)
        # Sort for deterministic launch order — set iteration is unordered.
        return sorted(ready)

    # Full scan fallback — used on first call or when no children data.
    ready = []
    for i, node_deps in enumerate(deps):
        if i in launched:
            continue
        if all(dep in completed for dep in node_deps):
            ready.append(i)
    return ready


def bfs_collect[A](
    fetch: Callable[[str, list[int]], list[A]],
    run_id: str,
    dependency_indices: list[int],
    dep_adjacency: list[list[int]],
    *,
    passthrough_indices: set[int] | None = None,
    max_depth: int = MAX_BFS_DEPTH,
) -> list[A]:
    """Generic BFS through dependency chain to collect items.

    Searches immediate dependency nodes first. If none produced items,
    walks transitively through each dependency's own deps — but only through
    **passthrough** nodes (routing nodes like Gate that don't consume data).
    Non-passthrough nodes are dead ends: if a processing node has no items,
    BFS stops there and returns empty rather than reaching past it.

    When ``passthrough_indices`` is None, all nodes are walkable (backward
    compat). BFS guarantees shortest-path: closest ancestor with items wins.

    ``max_depth`` caps traversal to detect cycles.

    **Callers MUST invoke per-edge for independent collection** — passing
    multiple indices merges results at the first level that has any items,
    which conflates separate dependency edges.
    """
    if not dependency_indices:
        return []

    search_indices = list(dependency_indices)
    visited: set[int] = set(dependency_indices)
    depth = 0

    while search_indices:
        if depth >= max_depth:
            logger.error(
                "BFS depth limit (%d) reached for run %s — possible cycle or malformed graph",
                max_depth,
                run_id,
            )
            return []

        items = fetch(run_id, search_indices)
        if items:
            return items

        next_search: list[int] = []
        for idx in search_indices:
            if passthrough_indices is not None and idx not in passthrough_indices:
                continue
            if idx < len(dep_adjacency):
                for dep in dep_adjacency[idx]:
                    if dep not in visited:
                        next_search.append(dep)
                        visited.add(dep)
        search_indices = next_search
        depth += 1

    return []


def collect_upstream_artifacts_bfs(
    store: DagStore,
    run_id: str,
    dependency_indices: list[int],
    dep_adjacency: list[list[int]],
    *,
    passthrough_indices: set[int] | None = None,
    max_depth: int = MAX_BFS_DEPTH,
) -> list[DagArtifact]:
    """BFS through dependency chain to collect artifacts."""
    return bfs_collect(
        store.get_artifacts_by_node_indices,
        run_id,
        dependency_indices,
        dep_adjacency,
        passthrough_indices=passthrough_indices,
        max_depth=max_depth,
    )
