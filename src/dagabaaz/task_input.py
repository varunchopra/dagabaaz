"""Task input resolution — collects artifacts, applies filters, resolves
bindings, and builds the input_data dict passed to plugins. Consumer-side
complement to the orchestrator's dispatch.
"""

import logging
from typing import Any

from dagabaaz.bindings import (
    any_binding_requires_run_input,
    build_expression_lookup,
    extract_node_indices_from_bindings,
    resolve_binding,
)
from dagabaaz.expressions import resolve_expression
from dagabaaz.filter import filter_artifacts, filter_artifacts_by_origin
from dagabaaz.graph import (
    bfs_collect,
    build_slug_to_index_map,
    rekey_edge_filters_by_index,
    resolve_dependency_indices,
)
from dagabaaz.models import DagNode, EdgeFilter, ExpressionError, TaskArtifact
from dagabaaz.store import TaskInputStore

logger = logging.getLogger(__name__)


def _is_truthy(value: object) -> bool:
    """Plain bool() – False/0/[]/{} are falsy.

    Stricter than the `default` pipe and EXISTS/NOT_EXISTS filter operators
    (which only treat None/"" as absent). Must stay in sync with the `not`
    pipe.
    """
    return bool(value)


def collect_upstream_task_artifacts_bfs(
    store: TaskInputStore,
    run_id: str,
    dependency_indices: list[int],
    dep_adjacency: list[list[int]],
    *,
    passthrough_indices: set[int] | None = None,
) -> list[TaskArtifact]:
    """BFS through dependency chain to collect TaskArtifacts."""
    return bfs_collect(
        store.get_task_artifacts_by_node_indices,
        run_id,
        dependency_indices,
        dep_adjacency,
        passthrough_indices=passthrough_indices,
    )


def _collect_and_filter_per_edge(
    store: TaskInputStore,
    run_id: str,
    dep_indices: list[int],
    dep_adjacency: list[list[int]],
    edge_filters: dict[int, EdgeFilter],
    *,
    passthrough_indices: set[int] | None = None,
) -> list[TaskArtifact]:
    all_artifacts: list[TaskArtifact] = []

    for dep_idx in dep_indices:
        upstream = collect_upstream_task_artifacts_bfs(
            store,
            run_id,
            [dep_idx],
            dep_adjacency,
            passthrough_indices=passthrough_indices,
        )
        if not upstream:
            continue

        dep_filter = edge_filters.get(dep_idx)
        if dep_filter and (dep_filter.rules or dep_filter.select):
            upstream = filter_artifacts(upstream, dep_filter)

        all_artifacts.extend(upstream)

    return all_artifacts


def _collect_grouped_artifacts_bfs(
    store: TaskInputStore,
    run_id: str,
    dependency_index: int,
    dep_adjacency: list[list[int]],
    origin_artifact_id: str,
    passthrough_indices: set[int] | None,
) -> list[TaskArtifact]:
    def fetch_group(
        current_run_id: str, node_indices: list[int]
    ) -> list[TaskArtifact]:
        grouped = store.get_grouped_artifacts(
            current_run_id, node_indices, origin_artifact_id
        )
        broadcast = store.get_broadcast_artifacts(current_run_id, node_indices)
        return grouped + broadcast

    return bfs_collect(
        fetch_group,
        run_id,
        [dependency_index],
        dep_adjacency,
        passthrough_indices=passthrough_indices,
    )


def build_task_input(
    store: TaskInputStore,
    *,
    run_id: str,
    node_index: int,
    input_artifact_id: str | None,
    nodes: list[DagNode],
    deps: list[list[int]] | None = None,
    origin_artifact_id: str | None = None,
    passthrough_indices: set[int] | None = None,
) -> dict[str, object]:
    if input_artifact_id:
        art = store.get_artifact_data(input_artifact_id)
        if not art:
            return {}
        input_data: dict[str, object] = {}
        if art.metadata:
            input_data.update(art.metadata)
        # Standard fields override colliding metadata keys.
        input_data["file_path"] = art.file_path
        input_data["file_name"] = art.file_name
        if art.file_size is not None:
            input_data["file_size"] = art.file_size
        if art.mime_type is not None:
            input_data["mime_type"] = art.mime_type
        return input_data

    if deps is None:
        deps = resolve_dependency_indices(nodes) if nodes else []
    dep_indices = deps[node_index] if node_index < len(deps) else []
    node_def = nodes[node_index] if node_index < len(nodes) else None
    edge_filters = (
        rekey_edge_filters_by_index(node_def, build_slug_to_index_map(nodes))
        if node_def and node_def.edge_filters
        else {}
    )

    if origin_artifact_id and dep_indices:
        grouped_artifacts: list[TaskArtifact] = []
        for dependency_index in dep_indices:
            dependency_artifacts = _collect_grouped_artifacts_bfs(
                store,
                run_id,
                dependency_index,
                deps,
                origin_artifact_id,
                passthrough_indices,
            )
            edge_filter = edge_filters.get(dependency_index)
            if edge_filter and (edge_filter.rules or edge_filter.select):
                dependency_artifacts = filter_artifacts(
                    dependency_artifacts, edge_filter
                )
            grouped_artifacts.extend(dependency_artifacts)

        if grouped_artifacts:
            return {
                "artifacts": [
                    artifact.to_input_dict() for artifact in grouped_artifacts
                ]
            }
        logger.warning(
            "Grouped task at node %d has no artifacts (run=%s, origin=%s)",
            node_index,
            run_id,
            origin_artifact_id,
        )
        return {"artifacts": []}

    if not dep_indices:
        # Bindings mutate input_data in place.
        return dict(store.get_run_input(run_id))

    if edge_filters:
        upstream_artifacts = _collect_and_filter_per_edge(
            store,
            run_id,
            dep_indices,
            deps,
            edge_filters,
            passthrough_indices=passthrough_indices,
        )
    else:
        # Iterate per-dep to match the edge-filter path's behavior —
        # passing all deps at once mixes BFS branches.
        upstream_artifacts: list[TaskArtifact] = []
        for dep_idx in dep_indices:
            dep_arts = collect_upstream_task_artifacts_bfs(
                store,
                run_id,
                [dep_idx],
                deps,
                passthrough_indices=passthrough_indices,
            )
            upstream_artifacts.extend(dep_arts)

    if upstream_artifacts:
        return {"artifacts": [a.to_input_dict() for a in upstream_artifacts]}

    logger.warning(
        "Dependency node %d has no artifacts after BFS (run=%s)",
        node_index,
        run_id,
    )
    return {"artifacts": []}


def resolve_task_bindings(
    store: TaskInputStore,
    *,
    run_id: str,
    node_index: int,
    nodes: list[DagNode],
    input_data: dict[str, object],
    node_config: dict[str, Any] | None = None,
    deps: list[list[int]] | None = None,
    input_artifact_id: str | None = None,
    origin_artifact_id: str | None = None,
) -> None:
    """Resolve bindings into input_data in place."""
    if node_index >= len(nodes):
        return

    current_node = nodes[node_index]
    bindings = current_node.bindings
    if not bindings:
        return

    slug_to_index = build_slug_to_index_map(nodes)

    artifacts_by_node: dict[int, list[TaskArtifact]]
    if input_artifact_id:
        fan_out_artifact = store.get_artifact_data(input_artifact_id)
        artifacts_by_node = {}
        if fan_out_artifact:
            # Look up the actual producing node so the artifact is
            # placed in the correct slot, preventing cross-contamination.
            producing_idx = store.get_artifact_producing_node(input_artifact_id)
            if producing_idx is not None:
                artifacts_by_node[producing_idx] = [fan_out_artifact]
            else:
                # Fallback: assign to all dep indices (original behavior)
                if deps is None:
                    deps = resolve_dependency_indices(nodes)
                dep_indices = deps[node_index] if node_index < len(deps) else []
                for dep_idx in dep_indices:
                    artifacts_by_node[dep_idx] = [fan_out_artifact]
    else:
        dep_indices_needed = extract_node_indices_from_bindings(bindings, slug_to_index)
        if dep_indices_needed:
            artifacts_by_node = store.get_artifacts_partitioned(
                run_id, sorted(dep_indices_needed)
            )
        else:
            artifacts_by_node = {}

    if origin_artifact_id and artifacts_by_node:
        artifacts_by_node = filter_artifacts_by_origin(
            artifacts_by_node, origin_artifact_id
        )

    edge_filters = rekey_edge_filters_by_index(current_node, slug_to_index)
    for dependency_index, edge_filter in edge_filters.items():
        artifacts = artifacts_by_node.get(dependency_index)
        if artifacts and (edge_filter.rules or edge_filter.select):
            artifacts_by_node[dependency_index] = filter_artifacts(
                artifacts, edge_filter
            )

    run_input = (
        store.get_run_input(run_id) if any_binding_requires_run_input(bindings) else {}
    )

    lookup = build_expression_lookup(
        artifacts_by_node, slug_to_index, run_input, node_config or {}
    )

    # Composition is the pipe chain – no infix and/or in the language.
    for field, binding in bindings.items():
        if binding.when is not None:
            try:
                when_value = resolve_expression(binding.when, lookup)
            except ExpressionError as exc:
                logger.warning(
                    "when-clause expression error on node %d field %r: %s",
                    node_index,
                    field,
                    exc,
                )
                continue
            if not _is_truthy(when_value):
                continue

        value = resolve_binding(
            binding,
            artifacts_by_node,
            slug_to_index,
            run_input,
            node_config or {},
        )
        if value is not None:
            input_data[field] = value
