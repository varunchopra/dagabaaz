"""Graph validation and readiness checks for named edges."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

from dagabaaz.constants import CorrelationMode, InputMode, InputRole
from dagabaaz.models import DagNode
from dagabaaz.schema import validate_binding_references


def assign_slugs(nodes: list[DagNode]) -> None:
    """Missing slugs are assigned in place. Existing slugs remain unchanged."""

    existing = {node.slug for node in nodes if node.slug}
    counters: dict[str, int] = {}
    for node in nodes:
        if node.slug:
            continue
        counter = counters.get(node.plugin, 0) + 1
        while f"{node.plugin}_{counter}" in existing:
            counter += 1
        node.slug = f"{node.plugin}_{counter}"
        existing.add(node.slug)
        counters[node.plugin] = counter


def build_slug_to_index_map(nodes: list[DagNode]) -> dict[str, int]:
    """Each non-empty, unique node slug maps to its run-local index."""

    result: dict[str, int] = {}
    for index, node in enumerate(nodes):
        if not node.slug:
            raise ValueError(f"node at index {index} has an empty slug")
        if node.slug in result:
            raise ValueError(f"duplicate node slug {node.slug!r}")
        result[node.slug] = index
    return result


def resolve_dependency_indices(nodes: list[DagNode]) -> list[list[int]]:
    """Edge sources become indices after missing slugs have been assigned.

    Multiple named edges from the same source produce one graph dependency
    while remaining separate routing inputs.
    """

    assign_slugs(nodes)
    slug_to_index = build_slug_to_index_map(nodes)
    dependencies: list[list[int]] = []
    for node in nodes:
        seen: set[int] = set()
        current: list[int] = []
        for edge in node.edges:
            source_index = slug_to_index.get(edge.source)
            if source_index is None:
                raise ValueError(
                    f"node {node.slug!r} edge {edge.name!r} references unknown "
                    f"source {edge.source!r}"
                )
            if source_index not in seen:
                seen.add(source_index)
                current.append(source_index)
        dependencies.append(current)
    return dependencies


def build_children(dependencies: Sequence[Sequence[int]]) -> list[list[int]]:
    """Parent-to-child adjacency is derived from the dependency list."""

    children: list[list[int]] = [[] for _ in dependencies]
    for child, parents in enumerate(dependencies):
        for parent in parents:
            children[parent].append(child)
    return children


def validate_graph(nodes: list[DagNode]) -> list[list[int]]:
    """Validation covers graph, edge, mode, correlation and binding rules.

    Cycle detection is iterative, so graph depth does not consume Python stack
    frames. The returned dependencies use run-local node indices.
    """

    if not nodes:
        raise ValueError("pipeline has no nodes")
    dependencies = resolve_dependency_indices(nodes)

    for node in nodes:
        edge_names: set[str] = set()
        for edge in node.edges:
            if edge.name in edge_names:
                raise ValueError(f"node {node.slug!r} has duplicate edge name {edge.name!r}")
            edge_names.add(edge.name)

        main_edges = [edge for edge in node.edges if edge.role == InputRole.MAIN]
        if node.input_mode == InputMode.EACH:
            if len(main_edges) != 1 or not main_edges[0].required:
                raise ValueError(f"EACH node {node.slug!r} requires exactly one required main edge")
        elif node.input_mode == InputMode.BY_CORRELATION and not main_edges:
            raise ValueError(f"BY_CORRELATION node {node.slug!r} requires at least one main edge")
        elif not node.edges and node.input_mode != InputMode.ALL:
            raise ValueError(f"root node {node.slug!r} must use ALL input mode")
        if (
            node.correlation_mode == CorrelationMode.INHERIT
            and node.input_mode == InputMode.ALL
        ):
            raise ValueError(
                f"ALL node {node.slug!r} cannot inherit a correlation ID"
            )

    binding_error = validate_binding_references(nodes)
    if binding_error is not None:
        raise ValueError(binding_error)

    children = build_children(dependencies)
    remaining_parents = [len(parents) for parents in dependencies]
    ready = deque(
        index for index, parent_count in enumerate(remaining_parents) if parent_count == 0
    )
    visited = 0
    while ready:
        current = ready.popleft()
        visited += 1
        for child in children[current]:
            remaining_parents[child] -= 1
            if remaining_parents[child] == 0:
                ready.append(child)
    if visited != len(nodes):
        first = next(index for index, parent_count in enumerate(remaining_parents) if parent_count)
        raise ValueError(f"cycle detected involving node {nodes[first].slug!r}")
    return dependencies


def find_root_nodes(nodes: list[DagNode]) -> list[int]:
    """Root node indices have no source dependencies."""

    dependencies = validate_graph(nodes)
    return [index for index, parents in enumerate(dependencies) if not parents]


def find_ready_nodes(
    dependencies: Sequence[Sequence[int]],
    completed: set[int],
    launched: set[int],
) -> list[int]:
    """Ready nodes are unlaunched and have complete source launches."""

    return [
        index
        for index, parents in enumerate(dependencies)
        if index not in launched and all(parent in completed for parent in parents)
    ]


def downstream_closure(
    dependencies: Sequence[Sequence[int]], boundary_indices: Iterable[int]
) -> set[int]:
    """The downstream closure contains each retry boundary and its descendants."""

    children = build_children(dependencies)
    result = set(boundary_indices)
    frontier = list(result)
    while frontier:
        current = frontier.pop()
        for child in children[current]:
            if child not in result:
                result.add(child)
                frontier.append(child)
    return result
