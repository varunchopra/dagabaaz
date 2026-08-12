import pytest

from dagabaaz.constants import RunStatus, TaskStatus
from dagabaaz.execution import claim_task, recover_task
from dagabaaz.models import DagNode, EmittedOutput
from dagabaaz.retry import retry_run
from dagabaaz.store import (
    DagStore,
    StoreContractError,
    TaskAttemptRef,
    TaskClaimResult,
    TaskRecoveryResult,
)

from .helpers import FakeStore, claim_current


def queued_task() -> tuple[FakeStore, str, str, int]:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    launch = store.seed_launch(0)
    task_id = launch.task_ids[0]
    return store, task_id, store.current_attempts[task_id], launch.generation


def test_claim_is_a_mandatory_store_operation_and_changes_only_status() -> None:
    store, task_id, attempt_id, generation = queued_task()
    plan = store.plans[task_id]
    queue_payloads = list(store.queue_payloads)

    result = claim_task(
        store,
        task_id,
        expected_attempt_id=attempt_id,
        expected_generation=generation,
    )

    assert isinstance(store, DagStore)
    assert result == TaskClaimResult(
        task_id=task_id,
        attempt_id=attempt_id,
        run_id="run",
        node_index=0,
        generation=0,
        resumed_from_wait_id=None,
    )
    assert store.task_statuses[task_id] == TaskStatus.RUNNING
    assert store.current_attempts[task_id] == attempt_id
    assert store.plans[task_id] is plan
    assert store.queue_payloads == queue_payloads


def test_only_one_delivery_can_claim_an_attempt() -> None:
    store, task_id, attempt_id, generation = queued_task()

    assert claim_task(
        store,
        task_id,
        expected_attempt_id=attempt_id,
        expected_generation=generation,
    )
    assert (
        claim_task(
            store,
            task_id,
            expected_attempt_id=attempt_id,
            expected_generation=generation,
        )
        is None
    )
    assert store.task_statuses[task_id] == TaskStatus.RUNNING


@pytest.mark.parametrize(
    "status",
    tuple(status for status in TaskStatus if status != TaskStatus.QUEUED),
)
def test_claim_rejects_every_nonqueued_state(status: TaskStatus) -> None:
    store, task_id, attempt_id, generation = queued_task()
    store.task_statuses[task_id] = status

    assert (
        claim_task(
            store,
            task_id,
            expected_attempt_id=attempt_id,
            expected_generation=generation,
        )
        is None
    )
    assert store.task_statuses[task_id] == status


def test_claim_rejects_wrong_attempt_generation_and_inactive_work() -> None:
    store, task_id, attempt_id, generation = queued_task()

    assert (
        claim_task(
            store,
            task_id,
            expected_attempt_id="stale",
            expected_generation=generation,
        )
        is None
    )
    assert (
        claim_task(
            store,
            task_id,
            expected_attempt_id=attempt_id,
            expected_generation=generation + 1,
        )
        is None
    )

    store.plans.pop(task_id)
    assert (
        claim_task(
            store,
            task_id,
            expected_attempt_id=attempt_id,
            expected_generation=generation,
        )
        is None
    )

    store, task_id, attempt_id, generation = queued_task()
    store.launches.clear()
    assert (
        claim_task(
            store,
            task_id,
            expected_attempt_id=attempt_id,
            expected_generation=generation,
        )
        is None
    )

    store, task_id, attempt_id, generation = queued_task()
    assert store.try_finalize_run("run", RunStatus.CANCELLED, reason="cancelled")
    assert (
        claim_task(
            store,
            task_id,
            expected_attempt_id=attempt_id,
            expected_generation=generation,
        )
        is None
    )


@pytest.mark.parametrize(
    ("task_id", "attempt_id", "generation", "message"),
    [
        ("", "attempt", 0, "task ID"),
        ("task", "", 0, "attempt ID"),
        ("task", "attempt", -1, "generation"),
        ("task", "attempt", True, "generation"),
    ],
)
def test_execution_wrappers_validate_requests(
    task_id: str,
    attempt_id: str,
    generation: int,
    message: str,
) -> None:
    store, _, _, _ = queued_task()
    with pytest.raises(ValueError, match=message):
        claim_task(
            store,
            task_id,
            expected_attempt_id=attempt_id,
            expected_generation=generation,
        )


@pytest.mark.parametrize("mismatch", ["task", "attempt", "generation"])
def test_claim_rejects_contradictory_store_results(mismatch: str) -> None:
    class WrongClaimStore(FakeStore):
        def try_claim_task(
            self,
            task_id: str,
            *,
            expected_attempt_id: str,
            expected_generation: int,
        ) -> TaskClaimResult | None:
            values = {
                "task_id": task_id,
                "attempt_id": expected_attempt_id,
                "run_id": "run",
                "node_index": 0,
                "generation": expected_generation,
                "resumed_from_wait_id": None,
            }
            if mismatch == "task":
                values["task_id"] = "another-task"
            elif mismatch == "attempt":
                values["attempt_id"] = "another-attempt"
            else:
                values["generation"] = expected_generation + 1
            return TaskClaimResult(**values)  # type: ignore[arg-type]

    store = WrongClaimStore([DagNode(slug="root", plugin="p")])
    launch = store.seed_launch(0)
    task_id = launch.task_ids[0]
    with pytest.raises(StoreContractError, match="mismatched fields"):
        claim_task(
            store,
            task_id,
            expected_attempt_id=store.current_attempts[task_id],
            expected_generation=0,
        )


def test_claim_rejects_a_result_outside_the_protocol() -> None:
    class WrongClaimStore(FakeStore):
        def try_claim_task(self, *args, **kwargs) -> TaskClaimResult | None:
            return object()  # type: ignore[return-value]

    store = WrongClaimStore([DagNode(slug="root", plugin="p")])
    task_id = store.seed_launch(0).task_ids[0]
    with pytest.raises(StoreContractError, match="TaskClaimResult"):
        claim_task(
            store,
            task_id,
            expected_attempt_id=store.current_attempts[task_id],
            expected_generation=0,
        )


def test_recovery_replaces_a_running_attempt_with_a_new_delivery() -> None:
    store, task_id, abandoned, generation = queued_task()
    plan = store.plans[task_id]
    claim_current(store, task_id)
    before = len(store.queue_payloads)

    assert (
        claim_task(
            store,
            task_id,
            expected_attempt_id=abandoned,
            expected_generation=generation,
        )
        is None
    )
    recovered = recover_task(
        store,
        task_id,
        expected_attempt_id=abandoned,
        expected_generation=generation,
    )

    assert recovered is not None
    assert recovered.recovered_attempt_id != abandoned
    assert recovered.resumed_from_wait_id is None
    assert store.task_statuses[task_id] == TaskStatus.QUEUED
    assert store.current_attempts[task_id] == recovered.recovered_attempt_id
    assert store.plans[task_id] is plan
    assert len(store.queue_payloads) == before + 1
    assert store.queue_payloads[-1]["attempt_id"] == recovered.recovered_attempt_id

    assert (
        store.try_complete_task(
            task_id,
            (EmittedOutput(id="not-claimed"),),
            expected_attempt_id=recovered.recovered_attempt_id,
        )
        is None
    )


def test_duplicate_recovery_returns_one_replacement_and_outbox_entry() -> None:
    store, task_id, abandoned, generation = queued_task()
    claim_current(store, task_id)

    first = recover_task(
        store,
        task_id,
        expected_attempt_id=abandoned,
        expected_generation=generation,
    )
    queue_count = len(store.queue_payloads)
    second = recover_task(
        store,
        task_id,
        expected_attempt_id=abandoned,
        expected_generation=generation,
    )

    assert first is not None
    assert second == first
    assert len(store.queue_payloads) == queue_count


def test_each_recovery_keeps_its_original_replacement() -> None:
    store, task_id, attempt_a, generation = queued_task()
    claim_current(store, task_id)
    recovery_ab = recover_task(
        store,
        task_id,
        expected_attempt_id=attempt_a,
        expected_generation=generation,
    )
    assert recovery_ab is not None

    claim_current(store, task_id)
    recovery_bc = recover_task(
        store,
        task_id,
        expected_attempt_id=recovery_ab.recovered_attempt_id,
        expected_generation=generation,
    )
    assert recovery_bc is not None
    queue_count = len(store.queue_payloads)

    replay_ab = recover_task(
        store,
        task_id,
        expected_attempt_id=attempt_a,
        expected_generation=generation,
    )
    assert replay_ab == recovery_ab
    assert replay_ab.recovered_attempt_id != recovery_bc.recovered_attempt_id
    assert len(store.queue_payloads) == queue_count


@pytest.mark.parametrize("later_state", ["completed", "cancelled", "invalidated"])
def test_repeated_recovery_returns_its_original_result_after_later_state(
    later_state: str,
) -> None:
    store, task_id, attempt_a, generation = queued_task()
    claim_current(store, task_id)
    recovery = recover_task(
        store,
        task_id,
        expected_attempt_id=attempt_a,
        expected_generation=generation,
    )
    assert recovery is not None

    if later_state == "completed":
        claim_current(store, task_id)
        assert store.try_complete_task(
            task_id,
            (),
            expected_attempt_id=recovery.recovered_attempt_id,
        )
    elif later_state == "cancelled":
        assert store.try_finalize_run("run", RunStatus.CANCELLED, reason="cancelled")
    else:
        claim_current(store, task_id)
        failed = store.mark_task_failed(
            task_id,
            "failed",
            expected_attempt_id=recovery.recovered_attempt_id,
        )
        assert failed is not None
        assert store.try_finalize_run(
            "run",
            RunStatus.FAILED,
            cause=TaskAttemptRef(task_id=task_id, attempt_id=recovery.recovered_attempt_id),
        )
        retry_run(store, "run", [0])

    queue_count = len(store.queue_payloads)
    assert (
        recover_task(
            store,
            task_id,
            expected_attempt_id=attempt_a,
            expected_generation=generation,
        )
        == recovery
    )
    assert len(store.queue_payloads) == queue_count


@pytest.mark.parametrize("terminal", [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CRASHED])
def test_terminal_transition_winning_rejects_recovery(terminal: TaskStatus) -> None:
    store, task_id, attempt_id, generation = queued_task()
    claim_current(store, task_id)

    if terminal == TaskStatus.COMPLETED:
        assert store.try_complete_task(task_id, (), expected_attempt_id=attempt_id)
    elif terminal == TaskStatus.FAILED:
        assert store.mark_task_failed(task_id, "failed", expected_attempt_id=attempt_id)
    else:
        assert store.mark_task_crashed(task_id, "crashed", expected_attempt_id=attempt_id)

    assert (
        recover_task(
            store,
            task_id,
            expected_attempt_id=attempt_id,
            expected_generation=generation,
        )
        is None
    )


@pytest.mark.parametrize("callback", ["complete", "fail", "crash"])
def test_recovery_makes_callbacks_from_the_abandoned_attempt_stale(callback: str) -> None:
    store, task_id, attempt_id, generation = queued_task()
    claim_current(store, task_id)
    recovery = recover_task(
        store,
        task_id,
        expected_attempt_id=attempt_id,
        expected_generation=generation,
    )
    assert recovery is not None

    if callback == "complete":
        result = store.try_complete_task(
            task_id,
            (EmittedOutput(id="late"),),
            expected_attempt_id=attempt_id,
        )
    elif callback == "fail":
        result = store.mark_task_failed(task_id, "late", expected_attempt_id=attempt_id)
    else:
        result = store.mark_task_crashed(task_id, "late", expected_attempt_id=attempt_id)

    assert result is None
    assert store.task_statuses[task_id] == TaskStatus.QUEUED
    assert store.current_attempts[task_id] == recovery.recovered_attempt_id
    assert store.outputs == {}


def test_cancellation_and_recovery_are_decided_by_which_commits_first() -> None:
    cancelled_first, task_id, attempt_id, generation = queued_task()
    claim_current(cancelled_first, task_id)
    assert cancelled_first.try_finalize_run("run", RunStatus.CANCELLED, reason="cancelled")
    assert (
        recover_task(
            cancelled_first,
            task_id,
            expected_attempt_id=attempt_id,
            expected_generation=generation,
        )
        is None
    )

    recovery_first, task_id, attempt_id, generation = queued_task()
    claim_current(recovery_first, task_id)
    recovery = recover_task(
        recovery_first,
        task_id,
        expected_attempt_id=attempt_id,
        expected_generation=generation,
    )
    assert recovery is not None
    assert recovery_first.try_finalize_run("run", RunStatus.CANCELLED, reason="cancelled")
    assert recovery_first.task_statuses[task_id] == TaskStatus.CANCELLED
    queue_count = len(recovery_first.queue_payloads)
    assert (
        recover_task(
            recovery_first,
            task_id,
            expected_attempt_id=attempt_id,
            expected_generation=generation,
        )
        == recovery
    )
    assert len(recovery_first.queue_payloads) == queue_count


def test_recovery_requires_the_current_running_attempt_and_generation() -> None:
    store, task_id, attempt_id, generation = queued_task()
    claim_current(store, task_id)
    assert (
        recover_task(
            store,
            task_id,
            expected_attempt_id=attempt_id,
            expected_generation=generation + 1,
        )
        is None
    )
    assert store.current_attempts[task_id] == attempt_id

    store.task_statuses[task_id] = TaskStatus.WAITING
    assert (
        recover_task(
            store,
            task_id,
            expected_attempt_id=attempt_id,
            expected_generation=generation,
        )
        is None
    )


@pytest.mark.parametrize("mismatch", ["task", "attempt", "generation"])
def test_recovery_rejects_contradictory_store_results(mismatch: str) -> None:
    class WrongRecoveryStore(FakeStore):
        def try_recover_task(
            self,
            task_id: str,
            *,
            expected_attempt_id: str,
            expected_generation: int,
        ) -> TaskRecoveryResult | None:
            values = {
                "task_id": task_id,
                "abandoned_attempt_id": expected_attempt_id,
                "recovered_attempt_id": "replacement",
                "run_id": "run",
                "node_index": 0,
                "generation": expected_generation,
                "resumed_from_wait_id": None,
            }
            if mismatch == "task":
                values["task_id"] = "another-task"
            elif mismatch == "attempt":
                values["abandoned_attempt_id"] = "another-attempt"
            else:
                values["generation"] = expected_generation + 1
            return TaskRecoveryResult(**values)  # type: ignore[arg-type]

    store = WrongRecoveryStore([DagNode(slug="root", plugin="p")])
    task_id = store.seed_launch(0).task_ids[0]
    with pytest.raises(StoreContractError, match="mismatched fields"):
        recover_task(
            store,
            task_id,
            expected_attempt_id=store.current_attempts[task_id],
            expected_generation=0,
        )
