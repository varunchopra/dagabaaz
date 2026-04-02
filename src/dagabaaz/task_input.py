"""Task input resolution — collects artifacts, applies filters, resolves
bindings, and builds the input_data dict passed to plugins. Consumer-side
complement to the orchestrator's dispatch.
"""

import logging
from typing import Any

from dagabaaz.bindings import (
    any_binding_requires_run_input,
    extract_node_indices_from_bindings,
    resolve_binding,
)
from dagabaaz.filter import filter_artifacts, filter_artifacts_by_origin
from dagabaaz.graph import (
    bfs_collect,
    build_slug_to_index_map,
    rekey_edge_filters_by_index,
    resolve_dependency_indices,
)
from dagabaaz.models import DagNode, EdgeFilter, TaskArtifact
from dagabaaz.store import TaskInputStore

logger = logging.getLogger(__name__)


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


def _collect_and_filter_per_edge(  # NOTE: orchestrator.py has a parallel version for DagArtifact
    store: TaskInputStore,
    run_id: str,
    dep_indices: list[int],
    dep_adjacency: list[list[int]],
    edge_filters: dict[int, EdgeFilter],
    *,
    passthrough_indices: set[int] | None = None,
) -> list[TaskArtifact]:
    """Collect artifacts per dependency edge, apply per-edge filters, merge.

    Iterates each dependency independently so per-edge filters apply to
    the correct subset. Uses BFS directly for each edge — it checks the
    first level itself, so no redundant pre-query is needed.
    """
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
    """Build the input data dict for a task.

    Four dispatch paths based on how the task was created:

    - **Fan-out** (``input_artifact_id`` set): Single artifact's fields
      spread into a flat dict. Most common path for SINGLE fan mode.
    - **Grouped** (``origin_artifact_id`` set, has deps): Correlated
      subset of upstream artifacts plus broadcast artifacts, returned
      as ``{"artifacts": [...]}``.
    - **Root** (no dependencies): The run's user-provided input.
    - **Aggregate/dependency** (has deps, no input artifact): All
      upstream artifacts via passthrough-aware BFS, with optional
      per-edge filtering, returned as ``{"artifacts": [...]}``.
    """
    # Fan-out: single artifact → flat dict
    if input_artifact_id:
        art = store.get_artifact_data(input_artifact_id)
        if not art:
            return {}
        # Spread metadata first, then set standard fields LAST so
        # metadata keys can't overwrite file_path/file_name.
        input_data: dict[str, object] = {}
        if art.metadata:
            input_data.update(art.metadata)
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

    # Grouped: correlated artifacts + broadcast → list
    if origin_artifact_id and dep_indices:
        grouped = store.get_grouped_artifacts(run_id, dep_indices, origin_artifact_id)
        broadcast = store.get_broadcast_artifacts(run_id, dep_indices)
        all_grouped = grouped + broadcast
        if all_grouped:
            return {"artifacts": [a.to_input_dict() for a in all_grouped]}
        # Dependency node with no artifacts — return empty artifacts list
        # rather than run_input (which is meant for root nodes only).
        logger.warning(
            "Grouped task at node %d has no artifacts (run=%s, origin=%s)",
            node_index,
            run_id,
            origin_artifact_id,
        )
        return {"artifacts": []}

    # Root: no dependencies → run input.
    # Defensive copy — resolve_task_bindings mutates input_data in place.
    if not dep_indices:
        return dict(store.get_run_input(run_id))

    # Aggregate/dependency: BFS collect with optional edge filters
    node_def = nodes[node_index] if node_index < len(nodes) else None
    slug_to_index = build_slug_to_index_map(nodes)
    edge_filters = (
        rekey_edge_filters_by_index(node_def, slug_to_index) if node_def else {}
    )

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

    # Dependency node with no artifacts — return empty artifacts list
    # rather than run_input (which is meant for root nodes only).
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
    """Resolve bindings and merge into input_data (mutates in place).

    Bindings are an overlay: applied AFTER ``build_task_input``, so they
    can override values from artifacts or run input. This lets pipeline
    designers wire specific values between nodes explicitly.

    For grouped tasks, artifacts are filtered by origin so each task
    sees only its own correlated data.
    """
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

    # Grouped origin filter: each grouped task sees only its correlated data.
    if origin_artifact_id and artifacts_by_node:
        artifacts_by_node = filter_artifacts_by_origin(
            artifacts_by_node, origin_artifact_id
        )

    run_input = (
        store.get_run_input(run_id) if any_binding_requires_run_input(bindings) else {}
    )

    for field, binding in bindings.items():
        value = resolve_binding(
            binding,
            artifacts_by_node,
            slug_to_index,
            run_input,
            node_config or {},
        )
        if value is not None:
            input_data[field] = value
