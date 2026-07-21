"""Run topology computation."""

from collections.abc import Callable
from dataclasses import dataclass

from dagabaaz.graph import (
    build_children,
    build_slug_to_index_map,
    rekey_edge_filters_by_index,
    resolve_dependency_indices,
)
from dagabaaz.models import DagNode, EdgeFilter

ResolvePassthrough = Callable[[str], bool]
"""Given a plugin name, return whether it allows BFS artifact passthrough."""


@dataclass(frozen=True, slots=True)
class RunTopology:
    """Structural graph data for a single run."""

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
