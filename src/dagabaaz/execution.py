"""Claim task execution and replace abandoned attempts."""

from __future__ import annotations

from dagabaaz.store import (
    DagStore,
    StoreContractError,
    TaskClaimResult,
    TaskRecoveryResult,
)


def _validate_request(
    task_id: str,
    expected_attempt_id: str,
    expected_generation: int,
) -> None:
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task ID must be a non-empty string")
    if not isinstance(expected_attempt_id, str) or not expected_attempt_id:
        raise ValueError("expected attempt ID must be a non-empty string")
    if type(expected_generation) is not int or expected_generation < 0:
        raise ValueError("expected generation must be a non-negative integer")


def claim_task(
    store: DagStore,
    task_id: str,
    *,
    expected_attempt_id: str,
    expected_generation: int,
) -> TaskClaimResult | None:
    """Return execution context only when this delivery wins the claim.

    ``None`` forbids execution, but does not say whether another worker owns
    the attempt, the task has moved on or recovery is safe.
    """

    _validate_request(task_id, expected_attempt_id, expected_generation)
    result = store.try_claim_task(
        task_id,
        expected_attempt_id=expected_attempt_id,
        expected_generation=expected_generation,
    )
    if result is None:
        return None
    if not isinstance(result, TaskClaimResult):
        raise StoreContractError("task claim did not return a TaskClaimResult")
    mismatches: list[str] = []
    if result.task_id != task_id:
        mismatches.append("task_id")
    if result.attempt_id != expected_attempt_id:
        mismatches.append("attempt_id")
    if result.generation != expected_generation:
        mismatches.append("generation")
    if mismatches:
        raise StoreContractError(
            f"task claim returned mismatched fields: {', '.join(mismatches)}"
        )
    return result


def recover_task(
    store: DagStore,
    task_id: str,
    *,
    expected_attempt_id: str,
    expected_generation: int,
) -> TaskRecoveryResult | None:
    """Replace abandoned execution only after repetition is known to be safe.

    An expired queue lease does not stop the old worker. Before recovery, the
    application must prevent that worker from changing shared state, make its
    actions safe to repeat, or check what happened. The result records the
    transition; only a later delivery that wins a claim may execute.
    """

    _validate_request(task_id, expected_attempt_id, expected_generation)
    result = store.try_recover_task(
        task_id,
        expected_attempt_id=expected_attempt_id,
        expected_generation=expected_generation,
    )
    if result is None:
        return None
    if not isinstance(result, TaskRecoveryResult):
        raise StoreContractError("task recovery did not return a TaskRecoveryResult")
    mismatches: list[str] = []
    if result.task_id != task_id:
        mismatches.append("task_id")
    if result.abandoned_attempt_id != expected_attempt_id:
        mismatches.append("abandoned_attempt_id")
    if result.generation != expected_generation:
        mismatches.append("generation")
    if mismatches:
        raise StoreContractError(
            f"task recovery returned mismatched fields: {', '.join(mismatches)}"
        )
    return result
