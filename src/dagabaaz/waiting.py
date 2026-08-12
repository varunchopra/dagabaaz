"""Leave a task unfinished, then start it again from stored state."""

from __future__ import annotations

from dagabaaz.execution import _validate_request
from dagabaaz.store import (
    DagWaitStore,
    StoreContractError,
    TaskResumeResult,
    TaskWaitResult,
    validate_wait_id,
)


def wait_task(
    store: DagWaitStore,
    task_id: str,
    wait_id: str,
    *,
    expected_attempt_id: str,
    expected_generation: int,
) -> TaskWaitResult | None:
    """Release the worker while leaving the logical task unfinished."""

    _validate_request(task_id, expected_attempt_id, expected_generation)
    validate_wait_id(wait_id)
    result = store.try_wait_task(
        task_id,
        wait_id,
        expected_attempt_id=expected_attempt_id,
        expected_generation=expected_generation,
    )
    if result is None:
        return None
    if not isinstance(result, TaskWaitResult):
        raise StoreContractError("task wait did not return a TaskWaitResult")
    mismatches: list[str] = []
    if result.task_id != task_id:
        mismatches.append("task_id")
    if result.attempt_id != expected_attempt_id:
        mismatches.append("attempt_id")
    if result.generation != expected_generation:
        mismatches.append("generation")
    if result.wait_id != wait_id:
        mismatches.append("wait_id")
    if mismatches:
        raise StoreContractError(
            f"task wait returned mismatched fields: {', '.join(mismatches)}"
        )
    return result


def resume_task(
    store: DagWaitStore,
    task_id: str,
    wait_id: str,
    *,
    expected_attempt_id: str,
    expected_generation: int,
) -> TaskResumeResult | None:
    """Create at most one replacement for an active wait.

    The result records what the store committed; it does not authorise
    execution. The replacement may run only after its queue delivery wins a
    claim.
    """

    _validate_request(task_id, expected_attempt_id, expected_generation)
    validate_wait_id(wait_id)
    result = store.try_resume_task(
        task_id,
        wait_id,
        expected_attempt_id=expected_attempt_id,
        expected_generation=expected_generation,
    )
    if result is None:
        return None
    if not isinstance(result, TaskResumeResult):
        raise StoreContractError("task resume did not return a TaskResumeResult")
    mismatches: list[str] = []
    if result.task_id != task_id:
        mismatches.append("task_id")
    if result.waiting_attempt_id != expected_attempt_id:
        mismatches.append("waiting_attempt_id")
    if result.generation != expected_generation:
        mismatches.append("generation")
    if result.wait_id != wait_id:
        mismatches.append("wait_id")
    if mismatches:
        raise StoreContractError(
            f"task resume returned mismatched fields: {', '.join(mismatches)}"
        )
    return result
