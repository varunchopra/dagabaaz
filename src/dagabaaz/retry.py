"""Task retries and run replanning."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from dagabaaz.store import DagRetryStore, RunReopenResult, StoreContractError


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
    """Replace a crashed attempt without changing its planned work.

    The expected ID prevents an old retry from replacing newer work. The
    replacement keeps its wait correlation so it can load the same application
    response. Its attempt and outbox row must commit together.
    """

    return store.try_create_task_retry(task_id, expected_attempt_id=expected_attempt_id)


def retry_run(
    store: DagRetryStore,
    run_id: str,
    boundary_indices: Iterable[int] | None = None,
) -> RunRetryResult:
    """A failed, crashed or cancelled run is reopened.

    The optional boundary is an assertion about active failed launches. The
    store discovers affected launches and descendants after acquiring the run
    lock, then commits the retry count, generations and invalidations together.
    Reconciliation and application cleanup may begin only after that operation
    succeeds.
    """

    boundary: tuple[int, ...] | None = None
    if boundary_indices is not None:
        supplied = tuple(boundary_indices)
        if not supplied:
            raise ValueError("a supplied retry boundary must not be empty")
        wrong_types = [index for index in supplied if type(index) is not int]
        if wrong_types:
            raise ValueError(f"boundary node indices must be integers: {wrong_types!r}")
        invalid = sorted({index for index in supplied if index < 0})
        if invalid:
            raise ValueError(f"boundary contains invalid node indices {invalid!r}")
        boundary = tuple(sorted(set(supplied)))

    reopened = store.try_reopen_run(run_id, boundary)
    if reopened is None:
        raise RetryRejectedError(f"run {run_id!r} is not retryable or its retry state became stale")
    if not isinstance(reopened, RunReopenResult):
        raise StoreContractError("run retry did not return a RunReopenResult")
    generation_indices = set(reopened.generations)
    if generation_indices != set(reopened.affected_nodes):
        raise StoreContractError(
            "run retry returned a different generation set; "
            f"expected={list(reopened.affected_nodes)!r}, "
            f"actual={sorted(generation_indices)!r}"
        )
    return RunRetryResult(
        affected_nodes=reopened.affected_nodes,
        generations=reopened.generations,
        retry_count=reopened.retry_count,
    )
