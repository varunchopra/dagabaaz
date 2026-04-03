"""DAG orchestration — barrier sync, node launching, skip/filter cascade, lifecycle.

Key algorithms:
- Barrier sync: a node is ready when ALL its dependency nodes have all
  tasks completed/skipped/filtered.
- Skip cascade: if any dependency is fully skipped (upstream dead), the
  downstream node is also skipped. Cascades through the entire subgraph.
- Filtered (non-cascading): when a node has no artifacts, it is marked
  ``filtered``. Downstream nodes still attempt artifact collection.
- Fan modes: SINGLE dispatches one task per artifact (with a max fan-out
  guard of 200). AGGREGATE dispatches one task for all artifacts. GROUPED
  dispatches one task per origin group.
- External termination: ``abort_run`` terminates a run from outside the
  normal task lifecycle (e.g. run-level deadline, user cancellation).
  For task-level timeouts, use ``on_task_failed`` instead.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from dagabaaz.constants import (
    MAX_FAN_OUT,
    RUN_TERMINAL_STATUSES,
    FanMode,
    NodeSummaryStatus,
    RunStatus,
)
from dagabaaz.filter import filter_artifacts, group_by_origin
from dagabaaz.graph import (
    collect_upstream_artifacts_bfs,
    find_ready_nodes,
    find_root_nodes,
)
from dagabaaz.models import DagArtifact, DagNode, EdgeFilter
from dagabaaz.store import DagStore
from dagabaaz.topology import ResolvePassthrough, RunTopology
from dagabaaz.topology import evict as _evict_topology
from dagabaaz.topology import get_or_build as _get_or_build_topology

logger = logging.getLogger(__name__)

OnRunCompleted = Callable[[str], None]
"""Called when a run completes. Receives run_id only — the application
captures cursor/queue in a closure, keeping the engine free of those types."""

OnRunFailed = Callable[[str], None]
"""Called when a run fails. Receives run_id."""

OnRunCrashed = Callable[[str], None]
"""Called when a run crashes (infra failure). Receives run_id."""

OnRunCancelled = Callable[[str], None]
"""Called when a run is cancelled externally. Receives run_id."""


@dataclass(frozen=True, slots=True)
class OrchestratorCallbacks:
    """Grouped lifecycle callbacks for the DAG engine."""

    on_run_completed: OnRunCompleted
    on_run_failed: OnRunFailed
    on_run_crashed: OnRunCrashed
    on_run_cancelled: OnRunCancelled


@dataclass(frozen=True, slots=True)
class LaunchResult:
    """Outcome of a single node launch attempt.

    Separates "0 tasks because skip/filter" from "0 tasks because fatal error".
    The caller inspects ``disposition`` to decide whether to propagate a skip
    cascade (skipped/filtered) or abort the entire run (failed).
    """

    tasks_created: int
    disposition: Literal["launched", "skipped", "filtered", "failed"]
    error: str = ""


def start_run(
    store: DagStore,
    run_id: str,
    nodes: list[DagNode],
) -> list[int]:
    """Find root nodes, claim them, and dispatch tasks. Returns root indices.

    This is the standard way to start a DAG run. Root nodes (no
    dependencies) are identified, atomically claimed via
    ``try_claim_node_launch``, and dispatched with no input artifact
    (root tasks receive the user's run input instead).

    Raises ``ValueError`` if the pipeline has no root nodes.
    """
    root_indices = find_root_nodes(nodes)
    if not root_indices:
        raise ValueError("Pipeline has no root nodes (circular dependencies?)")

    dispatched: list[int] = []
    for node_idx in root_indices:
        if not store.try_claim_node_launch(run_id, node_idx):
            continue
        store.dispatch_task(run_id, node_idx, nodes[node_idx].plugin, None)
        dispatched.append(node_idx)

    logger.info(
        "Run %s: dispatched %d root node(s) %s", run_id, len(dispatched), dispatched
    )
    return root_indices  # Full set — callers need structural info


def _collect_and_filter_per_edge(  # NOTE: task_input.py has a parallel version for TaskArtifact
    store: DagStore,
    run_id: str,
    dependency_indices: list[int],
    dep_adjacency: list[list[int]],
    edge_filters: dict[int, EdgeFilter],
    *,
    passthrough_indices: set[int] | None = None,
) -> tuple[list[DagArtifact], bool]:
    """Collect artifacts per dependency edge, apply per-edge filters, merge.

    Returns (merged_artifacts, is_gate_rejected).

    is_gate_rejected is True when at least one filtered edge had all its
    artifacts eliminated — upstream had data, a filter was active, and
    the filter rejected everything. This distinguishes deliberate rejection
    from an empty upstream (gate passthrough).
    """
    if not dependency_indices:
        return [], False

    is_gate_rejected = False
    all_artifacts: list[DagArtifact] = []

    for dep_idx in dependency_indices:
        dep_artifacts = collect_upstream_artifacts_bfs(
            store,
            run_id,
            [dep_idx],
            dep_adjacency,
            passthrough_indices=passthrough_indices,
        )
        if not dep_artifacts:
            continue

        dep_filter = edge_filters.get(dep_idx)
        if dep_filter and (dep_filter.rules or dep_filter.select):
            before_count = len(dep_artifacts)
            dep_artifacts = filter_artifacts(dep_artifacts, dep_filter)
            if before_count != len(dep_artifacts):
                logger.info(
                    "Edge %d filter: %d -> %d artifact(s)",
                    dep_idx,
                    before_count,
                    len(dep_artifacts),
                )
            if not dep_artifacts:
                is_gate_rejected = True

        all_artifacts.extend(dep_artifacts)

    return all_artifacts, is_gate_rejected


def _launch_node(
    store: DagStore,
    *,
    run_id: str,
    node_index: int,
    topology: RunTopology,
    node_completion_map: dict[int, NodeSummaryStatus] | None,
) -> LaunchResult:
    """Launch tasks for a single node. Returns a LaunchResult.

    Skip and filter semantics (conditional branching):
    - If ANY dependency was entirely skipped -> this node is **skipped** (cascading).
    - If an aggregate/grouped node's edge filter rejects all artifacts ->
      **skipped** (cascading).
    - If a non-root node has no artifacts after collection -> **filtered** (non-cascading).
      This covers two cases: edge filters rejected everything, or BFS stopped at
      a non-passthrough node upstream. Passthrough nodes (Gate) let BFS walk
      through to find artifacts from further upstream; non-passthrough nodes
      (processors) block BFS and create a natural cascade
      through artifact absence.

    Returns a ``LaunchResult`` with ``disposition`` indicating the outcome.
    The caller inspects ``disposition == "failed"`` to abort the run, and
    ``tasks_created == 0`` (skip/filter) to propagate cascades in-memory.
    """
    nodes = topology.nodes
    deps = topology.deps
    passthrough_indices = topology.passthrough_indices

    node = nodes[node_index]
    dependency_indices = deps[node_index]
    edge_filters = topology.edge_filters[node_index]
    plugin_name = node.plugin
    fan_mode = node.fan_mode

    if node_completion_map and dependency_indices:
        for dep_idx in dependency_indices:
            if node_completion_map.get(dep_idx) == NodeSummaryStatus.SKIPPED:
                store.dispatch_skipped_task(run_id, node_index, plugin_name)
                logger.info(
                    "Node %d skipped (dependency %d was skipped)",
                    node_index,
                    dep_idx,
                )
                return LaunchResult(0, "skipped")

    is_gate_rejected = False
    if not edge_filters:
        # Iterate per-dep to match the edge-filter path's behavior — passing
        # all deps at once mixes BFS branches and can skip passthrough
        # traversal when one dep has artifacts and another is passthrough-empty.
        all_artifacts: list[DagArtifact] = []
        for dep_idx in dependency_indices:
            dep_arts = collect_upstream_artifacts_bfs(
                store,
                run_id,
                [dep_idx],
                deps,
                passthrough_indices=passthrough_indices,
            )
            all_artifacts.extend(dep_arts)
        artifacts = all_artifacts
    else:
        artifacts, is_gate_rejected = _collect_and_filter_per_edge(
            store,
            run_id,
            dependency_indices,
            deps,
            edge_filters,
            passthrough_indices=passthrough_indices,
        )

    # Gate rejection: cascading skip for AGGREGATE only. AGGREGATE merges
    # all edges into one task, so a poisoned edge corrupts the whole input.
    # GROUPED is intentionally excluded — each origin group is independent,
    # so a rejected edge only kills its own groups while other edges'
    # groups proceed normally (scatter-gather partial success).
    if is_gate_rejected and fan_mode == FanMode.AGGREGATE:
        store.dispatch_skipped_task(run_id, node_index, plugin_name)
        logger.info("Node %d skipped (gate rejected on aggregate)", node_index)
        return LaunchResult(0, "skipped")

    # No artifacts for a node with dependencies — filtered
    if not artifacts and dependency_indices:
        store.dispatch_filtered_task(run_id, node_index, plugin_name)
        logger.info("Node %d filtered (no artifacts available)", node_index)
        return LaunchResult(0, "filtered")

    task_count = 0

    match fan_mode:
        case FanMode.SINGLE:
            # One task per artifact — standard fan-out.
            if len(artifacts) > MAX_FAN_OUT:
                error = f"Fan-out limit exceeded: {len(artifacts)} artifacts (max {MAX_FAN_OUT})"
                logger.error("Node %d %s", node_index, error)
                return LaunchResult(0, "failed", error)
            for artifact in artifacts:
                store.dispatch_task(run_id, node_index, plugin_name, artifact.id)
                task_count += 1

        case FanMode.AGGREGATE:
            # One task receives ALL artifacts. Root nodes (no deps) also
            # dispatch a single task with no input artifact.
            store.dispatch_task(run_id, node_index, plugin_name, None)
            task_count = 1

        case FanMode.GROUPED:
            # One task per origin group — scatter-gather correlation.
            # Group the ALREADY-COLLECTED artifacts by origin. Artifacts were
            # collected via BFS (which handles passthrough traversal), so we
            # group the results in Python — no separate query needed.
            result = group_by_origin(artifacts)
            if not result.groups:
                # All artifacts are broadcast — no origin grouping possible.
                # Fall back to aggregate dispatch so the node still executes.
                store.dispatch_task(run_id, node_index, plugin_name, None)
                task_count = 1
            elif len(result.groups) > MAX_FAN_OUT:
                # Same guard as SINGLE mode — fail loudly rather than
                # silently creating hundreds of grouped tasks.
                error = (
                    f"Grouped fan-out limit exceeded: "
                    f"{len(result.groups)} groups (max {MAX_FAN_OUT})"
                )
                logger.error("Node %d: %s", node_index, error)
                return LaunchResult(0, "failed", error)
            else:
                for origin_id in result.groups:
                    store.dispatch_grouped_task(
                        run_id, node_index, plugin_name, origin_id
                    )
                    task_count += 1

    logger.info("Launched node %d with %d task(s)", node_index, task_count)
    return LaunchResult(task_count, "launched")


def on_task_complete(
    store: DagStore,
    *,
    task_id: str,
    callbacks: OrchestratorCallbacks,
    resolve_passthrough: ResolvePassthrough,
) -> None:
    """Handle task completion with DAG-aware scheduling.

    Called after a task finishes successfully. Checks if all sibling tasks
    at the same node are done, then finds all newly-ready downstream nodes
    and launches them. Multiple nodes can be launched in parallel.

    The launch loop handles multi-level skip cascades: skipping a node
    may unlock further downstream nodes, so we loop until no more nodes
    become ready. Each iteration is O(nodes) for readiness + O(1) store
    operations per launched/skipped node.

    Can trigger ``on_run_completed`` (all nodes done), ``on_run_failed``
    (fan-out limit exceeded or similar fatal error during downstream
    launch), or ``on_run_crashed`` (not currently — crashes come from
    the worker via ``on_task_crashed``). The ``OrchestratorCallbacks``
    dataclass carries all three so callers don't need to remember which
    subset each function needs.
    """
    ctx = store.get_task_context(task_id)
    if not ctx:
        logger.error("Task %s not found", task_id)
        return

    run_id = ctx.run_id
    node_index = ctx.node_index

    run_status, total_tasks, completed_tasks = store.get_barrier_state(
        run_id, node_index
    )
    if run_status in RUN_TERMINAL_STATUSES:
        logger.info("Run %s is %s, skipping orchestration", run_id, run_status)
        return

    if completed_tasks < total_tasks:
        logger.info(
            "Node %d: %d/%d tasks done, waiting for siblings",
            node_index,
            completed_tasks,
            total_tasks,
        )
        return

    # Reuse cached topology when available — nodes are immutable per run,
    # so the structural graph data never changes between task completions.
    # Lazy fetcher avoids a DB round-trip on every call (cache hit path).
    topology = _get_or_build_topology(
        run_id, lambda: store.get_run_nodes(run_id), resolve_passthrough
    )
    if topology is None:
        logger.error("Run %s pipeline not found", run_id)
        return

    completed_nodes = store.get_completed_node_indices(run_id)
    launched_nodes = store.get_launched_node_indices(run_id)

    node_completion_map = store.get_node_summary(run_id)

    total_new_tasks = 0
    all_launched = set(launched_nodes)
    ready_node_indices: list[int] = []
    # Track which nodes just completed for targeted children lookup
    changed_indices: set[int] = {node_index}

    while True:
        ready = find_ready_nodes(
            topology.deps,
            completed_nodes,
            all_launched,
            children=topology.children,
            changed_indices=changed_indices,
        )
        if not ready:
            break

        # Reset for next iteration — only nodes processed in this iteration
        # can unlock further downstream nodes.
        changed_indices = set()

        for node_idx in ready:
            if not store.try_claim_node_launch(run_id, node_idx):
                logger.info("Node %d claimed by another worker, skipping", node_idx)
                all_launched.add(node_idx)
                continue

            result = _launch_node(
                store,
                run_id=run_id,
                node_index=node_idx,
                topology=topology,
                node_completion_map=node_completion_map,
            )
            total_new_tasks += result.tasks_created
            all_launched.add(node_idx)
            ready_node_indices.append(node_idx)

            if result.disposition == "failed":
                # Fan-out limit exceeded or similar fatal error — insert a
                # pre-failed marker task atomically (no queue job) so the
                # error is visible in the UI without a dispatch→fail race.
                store.dispatch_failed_task(
                    run_id, node_idx, topology.nodes[node_idx].plugin, result.error
                )
                if store.try_claim_run_terminal(run_id, RunStatus.FAILED, result.error):
                    callbacks.on_run_failed(run_id)
                    _evict_topology(run_id)
                    store.cancel_remaining_tasks(run_id, f"Cancelled: {result.error}")
                logger.error(
                    "Run %s failed at node %d: %s", run_id, node_idx, result.error
                )
                return

            if result.tasks_created == 0:
                # Skip/filter: update in-memory state so the next iteration
                # sees this node as completed without re-querying the DB.
                completed_nodes.add(node_idx)
                node_completion_map[node_idx] = (
                    NodeSummaryStatus.SKIPPED
                    if result.disposition == "skipped"
                    else NodeSummaryStatus.COMPLETED
                )
                changed_indices.add(node_idx)

    if len(completed_nodes) >= len(topology.nodes):
        if not store.try_claim_run_terminal(run_id, RunStatus.COMPLETED):
            logger.info("Run %s already terminal, skipping completion", run_id)
            return
        callbacks.on_run_completed(run_id)
        _evict_topology(run_id)
        logger.info("Run %s completed", run_id)
        return

    if not ready_node_indices:
        return

    # Track completed node count — not max(launched) which conflates
    # node index with progress and misleads when indices aren't topological.
    store.set_run_progress(run_id, len(completed_nodes))
    logger.info(
        "Run %s: launched %d node(s) %s with %d total task(s)",
        run_id,
        len(ready_node_indices),
        ready_node_indices,
        total_new_tasks,
    )


def on_task_failed(
    store: DagStore,
    *,
    task_id: str,
    error_message: str,
    callbacks: OrchestratorCallbacks,
) -> None:
    """Handle task failure. Marks the run as failed and cancels siblings."""
    run_id = store.mark_task_failed(task_id, error_message)
    if not run_id:
        return

    if not store.try_claim_run_terminal(run_id, RunStatus.FAILED, error_message):
        logger.info("Run %s already terminal, skipping failure handling", run_id)
        return

    callbacks.on_run_failed(run_id)
    _evict_topology(run_id)
    store.cancel_remaining_tasks(run_id, "Cancelled: sibling task failed")
    logger.info("Run %s failed due to task %s: %s", run_id, task_id, error_message)


def on_task_crashed(
    store: DagStore,
    *,
    task_id: str,
    error_message: str,
    callbacks: OrchestratorCallbacks,
) -> None:
    """Handle infra failure. Marks run as crashed — not billed."""
    run_id = store.mark_task_crashed(task_id, error_message)
    if not run_id:
        return

    if not store.try_claim_run_terminal(run_id, RunStatus.CRASHED, error_message):
        logger.info("Run %s already terminal, skipping crash handling", run_id)
        return

    callbacks.on_run_crashed(run_id)
    _evict_topology(run_id)
    store.cancel_remaining_tasks(run_id, "Cancelled: worker crashed")
    logger.info("Run %s crashed due to task %s: %s", run_id, task_id, error_message)


def abort_run(
    store: DagStore,
    *,
    run_id: str,
    reason: str,
    callbacks: OrchestratorCallbacks,
    status: RunStatus = RunStatus.FAILED,
) -> bool:
    """Terminate a run externally (deadline, cancellation, admin kill).

    Run-level counterpart to on_task_failed / on_task_crashed. Use when
    no single task triggered the termination. For task-level timeouts,
    use on_task_failed instead.
    """
    if status not in (RunStatus.FAILED, RunStatus.CRASHED, RunStatus.CANCELLED):
        raise ValueError(
            f"abort_run status must be FAILED, CRASHED, or CANCELLED, got {status}"
        )

    if not store.try_claim_run_terminal(run_id, status, reason):
        logger.info("Run %s already terminal, skipping abort", run_id)
        return False

    match status:
        case RunStatus.CRASHED:
            callbacks.on_run_crashed(run_id)
        case RunStatus.CANCELLED:
            callbacks.on_run_cancelled(run_id)
        case _:
            callbacks.on_run_failed(run_id)

    _evict_topology(run_id)
    store.cancel_remaining_tasks(run_id, f"Cancelled: {reason}")
    logger.info("Run %s aborted (%s): %s", run_id, status.value, reason)
    return True
