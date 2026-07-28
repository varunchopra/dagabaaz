"""Task retries and run replanning."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from dagabaaz.constants import NodeDisposition
from dagabaaz.graph import downstream_closure, validate_graph
from dagabaaz.store import DagRetryStore, StoreContractError


@dataclass(frozen=True, slots=True)
class RunRetryResult:
    """The node indices, generations and retry count committed by a run retry."""

    affected_nodes: tuple[int, ...]
    generations: Mapping[int, int]
    retry_count: int


class RetryRejectedError(RuntimeError):
    """The run no longer permits the requested retry."""


def retry_task(
    store: DagRetryStore,
    task_id: str,
    *,
    expected_attempt_id: str,
) -> str | None:
    """A current crashed attempt may be replaced while its run remains active.

    The store creates a replacement only if ``expected_attempt_id`` is still
    current, and reuses the task's plan and generation. Success also creates the
    replacement queue outbox row; rejection returns ``None``.
    """

    return store.try_create_task_retry(task_id, expected_attempt_id=expected_attempt_id)


def retry_run(
    store: DagRetryStore,
    run_id: str,
    boundary_indices: Iterable[int] | None = None,
) -> RunRetryResult:
    """A failed, crashed or cancelled run is reopened.

    Failed and crashed runs require active failure boundaries. A cancelled run
    is retried without a boundary. Every incomplete active launch is restarted
    in addition to the supplied boundaries, so work cancelled during
    terminalisation cannot remain attached to the reopened run.

    The store validates the proposal against locked state and commits the
    retry count, generations and invalidations together. Reconciliation and
    application cleanup may begin only after that operation succeeds.
    """

    nodes = store.get_run_nodes(run_id)
    if nodes is None:
        raise RetryRejectedError(f"run {run_id!r} has no stored topology")
    dependencies = validate_graph(nodes)

    boundary: tuple[int, ...] | None = None
    restart_roots = {
        index
        for index, launch in store.list_node_launches(run_id).items()
        if not launch.complete or launch.disposition == NodeDisposition.FAILED
    }
    if boundary_indices is not None:
        supplied = tuple(boundary_indices)
        if not supplied:
            raise ValueError("a supplied retry boundary must not be empty")
        wrong_types = [index for index in supplied if type(index) is not int]
        if wrong_types:
            raise ValueError(
                f"boundary node indices must be integers: {wrong_types!r}"
            )
        requested = set(supplied)
        invalid = sorted(index for index in requested if index < 0 or index >= len(nodes))
        if invalid:
            raise ValueError(f"boundary contains invalid node indices {invalid!r}")
        restart_roots.update(requested)
        boundary = tuple(sorted(requested))

    invalid_roots = sorted(index for index in restart_roots if index < 0 or index >= len(nodes))
    if invalid_roots:
        raise StoreContractError(f"active launches contain invalid node indices {invalid_roots!r}")

    affected = tuple(sorted(downstream_closure(dependencies, restart_roots)))
    reopened = store.try_reopen_run(run_id, boundary, affected)
    if reopened is None:
        raise RetryRejectedError(f"run {run_id!r} is not retryable or its retry state became stale")
    generation_indices = set(reopened.generations)
    if generation_indices != set(affected):
        raise StoreContractError(
            "run retry returned a different generation set; "
            f"expected={list(affected)!r}, actual={sorted(generation_indices)!r}"
        )
    return RunRetryResult(
        affected_nodes=affected,
        generations=reopened.generations,
        retry_count=reopened.retry_count,
    )
