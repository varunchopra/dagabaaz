"""Run topology built from node edges."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from dagabaaz.constants import CorrelationMode, InputMode
from dagabaaz.graph import (
    _normalise_and_validate_graph,
    build_children,
    build_slug_to_index_map,
)
from dagabaaz.models import DagNode, InputBinding, InputEdge, NodeDefinition


@dataclass(frozen=True, slots=True)
class _TopologyNode:
    """Node data frozen after graph validation."""

    name: str
    slug: str
    plugin: str
    edges: tuple[InputEdge, ...]
    bindings: Mapping[str, InputBinding]
    input_mode: InputMode
    correlation_mode: CorrelationMode

    @classmethod
    def from_node(cls, node: DagNode) -> "_TopologyNode":
        return cls(
            name=node.name,
            slug=node.slug,
            plugin=node.plugin,
            edges=tuple(node.edges),
            bindings=MappingProxyType(dict(node.bindings)),
            input_mode=node.input_mode if node.input_mode is not None else InputMode.ALL,
            correlation_mode=node.correlation_mode,
        )


@dataclass(frozen=True, slots=True)
class RunTopology:
    """Validated structural data derived from one run's stored nodes.

    Construction copies and freezes node data so later caller mutation cannot
    alter this topology. Dependencies and children use stable run-local indices.
    """

    nodes: tuple[NodeDefinition, ...]
    slug_to_index: Mapping[str, int]
    dependencies: tuple[tuple[int, ...], ...]
    children: tuple[tuple[int, ...], ...]

    @classmethod
    def build(cls, nodes: list[DagNode]) -> "RunTopology":
        """Nodes are revalidated before the topology is frozen."""

        normalised_nodes, dependencies = _normalise_and_validate_graph(nodes)
        frozen_nodes = tuple(_TopologyNode.from_node(node) for node in normalised_nodes)
        return cls(
            nodes=frozen_nodes,
            slug_to_index=MappingProxyType(build_slug_to_index_map(normalised_nodes)),
            dependencies=tuple(tuple(items) for items in dependencies),
            children=tuple(tuple(items) for items in build_children(dependencies)),
        )
