import pytest

import dagabaaz.store as store_module
from dagabaaz.constants import (
    MAX_WAIT_ID_LENGTH,
    TASK_TERMINAL_STATUSES,
    InputMode,
    RunStatus,
    TaskStatus,
)
from dagabaaz.execution import recover_task
from dagabaaz.models import DagNode, EmittedOutput, InputEdge
from dagabaaz.orchestrator import (
    on_task_complete,
    on_task_crashed,
    on_task_failed,
    reconcile_run,
)
from dagabaaz.retry import retry_run, retry_task
from dagabaaz.store import (
    DagWaitStore,
    StoreContractError,
    TaskAttemptRef,
    TaskResumeResult,
    TaskWaitResult,
)
from dagabaaz.waiting import resume_task, wait_task

from .helpers import CallbackRecorder, FakeStore, claim_current


def running_task() -> tuple[FakeStore, str, str, int]:
    store = FakeStore([DagNode(slug="root", plugin="p")])
    launch = store.seed_launch(0)
    task_id = launch.task_ids[0]
    attempt_id = store.current_attempts[task_id]
    claim_current(store, task_id)
    return store, task_id, attempt_id, launch.generation


def park(
    store: FakeStore,
    task_id: str,
    attempt_id: str,
    generation: int,
    wait_id: str = "wait-1",
) -> TaskWaitResult:
    result = wait_task(
        store,
        task_id,
        wait_id,
        expected_attempt_id=attempt_id,
        expected_generation=generation,
    )
    assert result is not None
    return result


def test_waiting_is_nonterminal_and_wait_support_is_optional() -> None:
    store = FakeStore([DagNode(slug="root", plugin="p")])

    assert TaskStatus.WAITING not in TASK_TERMINAL_STATUSES
    assert isinstance(store, DagWaitStore)


def test_running_attempt_enters_waiting_without_a_new_dispatch() -> None:
    store, task_id, attempt_id, generation = running_task()
    plan = store.plans[task_id]
    queue_payloads = list(store.queue_payloads)
    delivered_payload = store.active_deliveries[attempt_id]

    waited = park(store, task_id, attempt_id, generation)

    assert waited == TaskWaitResult(
        task_id=task_id,
        attempt_id=attempt_id,
        run_id="run",
        node_index=0,
        generation=generation,
        wait_id="wait-1",
    )
    assert store.task_statuses[task_id] == TaskStatus.WAITING
    assert store.current_attempts[task_id] == attempt_id
    assert store.plans[task_id] is plan
    assert store.launches[0].complete is False
    assert store.queue_payloads == queue_payloads

    # wait_task does not acknowledge the delivery that entered the wait.
    assert store.active_deliveries[attempt_id] is delivered_payload


def test_repeated_wait_returns_the_same_result_and_cannot_change_its_id() -> None:
    store, task_id, attempt_id, generation = running_task()
    first = park(store, task_id, attempt_id, generation)
    revision = store._revision

    assert (
        wait_task(
            store,
            task_id,
            "wait-1",
            expected_attempt_id=attempt_id,
            expected_generation=generation,
        )
        == first
    )
    assert store._revision == revision
    assert (
        wait_task(
            store,
            task_id,
            "another-wait",
            expected_attempt_id=attempt_id,
            expected_generation=generation,
        )
        is None
    )
    assert set(store.waits) == {(task_id, "wait-1")}


@pytest.mark.parametrize(
    ("wait_id", "message"),
    [
        ("", "non-empty"),
        ("x" * (MAX_WAIT_ID_LENGTH + 1), "512-character"),
        ("before\x00after", "NUL"),
    ],
)
def test_wait_identifiers_are_validated(wait_id: str, message: str) -> None:
    store, task_id, attempt_id, generation = running_task()

    with pytest.raises(ValueError, match=message):
        wait_task(
            store,
            task_id,
            wait_id,
            expected_attempt_id=attempt_id,
            expected_generation=generation,
        )


def test_wait_identifier_preserves_whitespace_and_may_reach_the_limit() -> None:
    store, task_id, attempt_id, generation = running_task()
    wait_id = " " + "x" * (MAX_WAIT_ID_LENGTH - 2) + " "

    result = park(store, task_id, attempt_id, generation, wait_id)

    assert result.wait_id == wait_id


def test_wait_rejects_wrong_attempt_generation_and_nonrunning_states() -> None:
    store, task_id, attempt_id, generation = running_task()
    assert (
        wait_task(
            store,
            task_id,
            "wrong-attempt",
            expected_attempt_id="stale",
            expected_generation=generation,
        )
        is None
    )
    assert (
        wait_task(
            store,
            task_id,
            "wrong-generation",
            expected_attempt_id=attempt_id,
            expected_generation=generation + 1,
        )
        is None
    )

    store.task_statuses[task_id] = TaskStatus.QUEUED
    assert (
        wait_task(
            store,
            task_id,
            "not-running",
            expected_attempt_id=attempt_id,
            expected_generation=generation,
        )
        is None
    )


def test_matching_wait_resumes_to_one_new_unclaimed_attempt() -> None:
    store, task_id, waiting_attempt, generation = running_task()
    plan = store.plans[task_id]
    park(store, task_id, waiting_attempt, generation)
    before = len(store.queue_payloads)

    resumed = resume_task(
        store,
        task_id,
        "wait-1",
        expected_attempt_id=waiting_attempt,
        expected_generation=generation,
    )

    assert resumed is not None
    assert resumed.waiting_attempt_id == waiting_attempt
    assert resumed.resumed_attempt_id != waiting_attempt
    assert store.task_statuses[task_id] == TaskStatus.QUEUED
    assert store.current_attempts[task_id] == resumed.resumed_attempt_id
    assert store.plans[task_id] is plan
    assert store.task_contexts[task_id].generation == generation
    assert store.attempt_wait_ids[resumed.resumed_attempt_id] == "wait-1"
    assert len(store.queue_payloads) == before + 1
    assert store.queue_payloads[-1]["attempt_id"] == resumed.resumed_attempt_id

    assert (
        store.try_complete_task(
            task_id,
            (),
            expected_attempt_id=resumed.resumed_attempt_id,
        )
        is None
    )


def test_claim_exposes_the_wait_that_resumed_execution() -> None:
    store, task_id, waiting_attempt, generation = running_task()
    park(store, task_id, waiting_attempt, generation, "checkpoint")
    resumed = resume_task(
        store,
        task_id,
        "checkpoint",
        expected_attempt_id=waiting_attempt,
        expected_generation=generation,
    )
    assert resumed is not None

    claimed = claim_current(store, task_id)

    assert claimed.attempt_id == resumed.resumed_attempt_id
    assert claimed.resumed_from_wait_id == "checkpoint"


def test_duplicate_resume_returns_one_attempt_and_outbox_entry() -> None:
    store, task_id, waiting_attempt, generation = running_task()
    park(store, task_id, waiting_attempt, generation)
    first = resume_task(
        store,
        task_id,
        "wait-1",
        expected_attempt_id=waiting_attempt,
        expected_generation=generation,
    )
    queue_count = len(store.queue_payloads)

    second = resume_task(
        store,
        task_id,
        "wait-1",
        expected_attempt_id=waiting_attempt,
        expected_generation=generation,
    )

    assert first is not None
    assert second == first
    assert len(store.queue_payloads) == queue_count


@pytest.mark.parametrize("callback", ["complete", "fail", "crash"])
def test_callbacks_from_the_waiting_attempt_are_stale_after_resume(callback: str) -> None:
    store, task_id, waiting_attempt, generation = running_task()
    park(store, task_id, waiting_attempt, generation)
    resumed = resume_task(
        store,
        task_id,
        "wait-1",
        expected_attempt_id=waiting_attempt,
        expected_generation=generation,
    )
    assert resumed is not None

    if callback == "complete":
        result = store.try_complete_task(
            task_id,
            (EmittedOutput(id="late"),),
            expected_attempt_id=waiting_attempt,
        )
    elif callback == "fail":
        result = store.mark_task_failed(
            task_id,
            "late",
            expected_attempt_id=waiting_attempt,
        )
    else:
        result = store.mark_task_crashed(
            task_id,
            "late",
            expected_attempt_id=waiting_attempt,
        )

    assert result is None
    assert store.task_statuses[task_id] == TaskStatus.QUEUED
    assert store.current_attempts[task_id] == resumed.resumed_attempt_id
    assert store.outputs == {}


def test_resume_rejects_the_wrong_wait_attempt_or_generation() -> None:
    store, task_id, waiting_attempt, generation = running_task()
    park(store, task_id, waiting_attempt, generation)

    assert (
        resume_task(
            store,
            task_id,
            "unknown",
            expected_attempt_id=waiting_attempt,
            expected_generation=generation,
        )
        is None
    )
    assert (
        resume_task(
            store,
            task_id,
            "wait-1",
            expected_attempt_id="stale",
            expected_generation=generation,
        )
        is None
    )
    assert (
        resume_task(
            store,
            task_id,
            "wait-1",
            expected_attempt_id=waiting_attempt,
            expected_generation=generation + 1,
        )
        is None
    )
    assert store.task_statuses[task_id] == TaskStatus.WAITING


def test_cancellation_invalidates_an_unresolved_wait() -> None:
    store, task_id, waiting_attempt, generation = running_task()
    wait_result = park(store, task_id, waiting_attempt, generation)
    assert store.try_finalize_run("run", RunStatus.CANCELLED, reason="cancelled")

    assert store.waits[(task_id, "wait-1")].state == "invalidated"
    assert store.task_statuses[task_id] == TaskStatus.CANCELLED
    assert (
        resume_task(
            store,
            task_id,
            "wait-1",
            expected_attempt_id=waiting_attempt,
            expected_generation=generation,
        )
        is None
    )
    assert (
        wait_task(
            store,
            task_id,
            "wait-1",
            expected_attempt_id=waiting_attempt,
            expected_generation=generation,
        )
        == wait_result
    )


@pytest.mark.parametrize("later_state", ["completed", "cancelled", "invalidated"])
def test_repeated_resume_returns_its_original_result_after_later_state(
    later_state: str,
) -> None:
    store, task_id, waiting_attempt, generation = running_task()
    wait_result = park(store, task_id, waiting_attempt, generation)
    resume_result = resume_task(
        store,
        task_id,
        "wait-1",
        expected_attempt_id=waiting_attempt,
        expected_generation=generation,
    )
    assert resume_result is not None

    if later_state == "completed":
        claim_current(store, task_id)
        assert store.try_complete_task(
            task_id,
            (),
            expected_attempt_id=resume_result.resumed_attempt_id,
        )
    elif later_state == "cancelled":
        assert store.try_finalize_run("run", RunStatus.CANCELLED, reason="cancelled")
    else:
        claim_current(store, task_id)
        assert store.mark_task_failed(
            task_id,
            "failed",
            expected_attempt_id=resume_result.resumed_attempt_id,
        )
        assert store.try_finalize_run(
            "run",
            RunStatus.FAILED,
            cause=TaskAttemptRef(
                task_id=task_id,
                attempt_id=resume_result.resumed_attempt_id,
            ),
        )
        retry_run(store, "run", [0])

    queue_count = len(store.queue_payloads)
    assert (
        resume_task(
            store,
            task_id,
            "wait-1",
            expected_attempt_id=waiting_attempt,
            expected_generation=generation,
        )
        == resume_result
    )
    assert (
        wait_task(
            store,
            task_id,
            "wait-1",
            expected_attempt_id=waiting_attempt,
            expected_generation=generation,
        )
        == wait_result
    )
    assert len(store.queue_payloads) == queue_count


def test_resume_rejects_a_wait_invalidated_before_resolution() -> None:
    store, task_id, waiting_attempt, generation = running_task()
    park(store, task_id, waiting_attempt, generation)
    assert store.try_finalize_run("run", RunStatus.FAILED, reason="other branch failed")
    retry_run(store, "run")

    assert (
        resume_task(
            store,
            task_id,
            "wait-1",
            expected_attempt_id=waiting_attempt,
            expected_generation=generation,
        )
        is None
    )


def test_resumed_wait_id_survives_task_retry_and_recovery() -> None:
    retry_store, task_id, waiting_attempt, generation = running_task()
    park(retry_store, task_id, waiting_attempt, generation, "answer")
    resumed = resume_task(
        retry_store,
        task_id,
        "answer",
        expected_attempt_id=waiting_attempt,
        expected_generation=generation,
    )
    assert resumed is not None
    claim_current(retry_store, task_id)
    assert retry_store.mark_task_crashed(
        task_id,
        "crashed",
        expected_attempt_id=resumed.resumed_attempt_id,
    )
    retried_attempt = retry_task(
        retry_store,
        task_id,
        expected_attempt_id=resumed.resumed_attempt_id,
    )
    assert retried_attempt is not None
    assert claim_current(retry_store, task_id).resumed_from_wait_id == "answer"

    recovery_store, task_id, waiting_attempt, generation = running_task()
    park(recovery_store, task_id, waiting_attempt, generation, "callback")
    resumed = resume_task(
        recovery_store,
        task_id,
        "callback",
        expected_attempt_id=waiting_attempt,
        expected_generation=generation,
    )
    assert resumed is not None
    claim_current(recovery_store, task_id)
    recovered = recover_task(
        recovery_store,
        task_id,
        expected_attempt_id=resumed.resumed_attempt_id,
        expected_generation=generation,
    )
    assert recovered is not None
    assert recovered.resumed_from_wait_id == "callback"
    assert claim_current(recovery_store, task_id).resumed_from_wait_id == "callback"


def test_task_can_wait_more_than_once_but_wait_ids_are_lifetime_unique() -> None:
    store, task_id, attempt_a, generation = running_task()
    park(store, task_id, attempt_a, generation, "first")
    resumed = resume_task(
        store,
        task_id,
        "first",
        expected_attempt_id=attempt_a,
        expected_generation=generation,
    )
    assert resumed is not None
    claim_current(store, task_id)

    assert (
        wait_task(
            store,
            task_id,
            "first",
            expected_attempt_id=resumed.resumed_attempt_id,
            expected_generation=generation,
        )
        is None
    )
    second = park(
        store,
        task_id,
        resumed.resumed_attempt_id,
        generation,
        "second",
    )
    resumed_again = resume_task(
        store,
        task_id,
        "second",
        expected_attempt_id=second.attempt_id,
        expected_generation=generation,
    )
    assert resumed_again is not None
    assert resumed_again.resumed_attempt_id not in {
        attempt_a,
        resumed.resumed_attempt_id,
    }
    assert claim_current(store, task_id).resumed_from_wait_id == "second"


def test_parallel_tasks_wait_and_resume_independently() -> None:
    store = FakeStore(
        [
            DagNode(slug="left", plugin="p"),
            DagNode(slug="right", plugin="p"),
        ]
    )
    left = store.seed_launch(0).task_ids[0]
    right = store.seed_launch(1).task_ids[0]
    left_attempt = store.current_attempts[left]
    right_attempt = store.current_attempts[right]
    claim_current(store, left)
    claim_current(store, right)
    park(store, left, left_attempt, 0, "left-wait")
    park(store, right, right_attempt, 0, "right-wait")

    left_resume = resume_task(
        store,
        left,
        "left-wait",
        expected_attempt_id=left_attempt,
        expected_generation=0,
    )
    assert left_resume is not None
    claim_current(store, left)
    callbacks = CallbackRecorder()
    on_task_complete(
        store,
        task_id=left,
        expected_attempt_id=left_resume.resumed_attempt_id,
        callbacks=callbacks.callbacks(),
    )

    assert store.status == RunStatus.RUNNING
    assert store.task_statuses[right] == TaskStatus.WAITING
    assert not callbacks.completed

    right_resume = resume_task(
        store,
        right,
        "right-wait",
        expected_attempt_id=right_attempt,
        expected_generation=0,
    )
    assert right_resume is not None
    claim_current(store, right)
    on_task_complete(
        store,
        task_id=right,
        expected_attempt_id=right_resume.resumed_attempt_id,
        callbacks=callbacks.callbacks(),
    )
    assert store.status == RunStatus.COMPLETED
    assert callbacks.completed == ["run"]


def test_waiting_launch_blocks_descendants_and_run_completion() -> None:
    nodes = [
        DagNode(slug="source", plugin="p"),
        DagNode(
            slug="child",
            plugin="p",
            input_mode=InputMode.EACH,
            edges=(InputEdge(name="source", source="source"),),
        ),
    ]
    store = FakeStore(nodes)
    source = store.seed_launch(0).task_ids[0]
    attempt = store.current_attempts[source]
    claim_current(store, source)
    park(store, source, attempt, 0)
    callbacks = CallbackRecorder()
    queue_count = len(store.queue_payloads)

    reconcile_run(store, "run", callbacks=callbacks.callbacks())

    assert 1 not in store.launches
    assert len(store.queue_payloads) == queue_count
    assert store.status == RunStatus.RUNNING
    assert not store.try_finalize_run("run", RunStatus.COMPLETED)
    assert not callbacks.completed


def test_callbacks_from_the_waiting_attempt_are_all_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, task_id, waiting_attempt, generation = running_task()
    wait_result = park(store, task_id, waiting_attempt, generation)
    queue_payloads = list(store.queue_payloads)
    callbacks = CallbackRecorder()

    # A zero limit would fail this callback if WAITING were checked too late.
    monkeypatch.setattr(store_module, "MAX_OUTPUTS_PER_TASK_COMPLETION", 0)
    on_task_complete(
        store,
        task_id=task_id,
        expected_attempt_id=waiting_attempt,
        outputs=(EmittedOutput(id="oversized"),),
        callbacks=callbacks.callbacks(),
    )
    on_task_failed(
        store,
        task_id=task_id,
        expected_attempt_id=waiting_attempt,
        error_message="late failure",
        callbacks=callbacks.callbacks(),
    )
    on_task_crashed(
        store,
        task_id=task_id,
        expected_attempt_id=waiting_attempt,
        error_message="late crash",
        callbacks=callbacks.callbacks(),
    )

    assert store.task_statuses[task_id] == TaskStatus.WAITING
    assert store.status == RunStatus.RUNNING
    assert store.launches[0].complete is False
    assert store.waits[(task_id, "wait-1")].wait_result == wait_result
    assert store.waits[(task_id, "wait-1")].state == "active"
    assert store.outputs == {}
    assert store.attempt_errors == {}
    assert store.queue_payloads == queue_payloads
    assert not callbacks.completed
    assert not callbacks.failed
    assert not callbacks.crashed


def test_failure_on_another_branch_invalidates_a_wait() -> None:
    store = FakeStore(
        [
            DagNode(slug="waiting", plugin="p"),
            DagNode(slug="failing", plugin="p"),
        ]
    )
    waiting_task = store.seed_launch(0).task_ids[0]
    failing_task = store.seed_launch(1).task_ids[0]
    waiting_attempt = store.current_attempts[waiting_task]
    failing_attempt = store.current_attempts[failing_task]
    claim_current(store, waiting_task)
    claim_current(store, failing_task)
    park(store, waiting_task, waiting_attempt, 0)

    on_task_failed(
        store,
        task_id=failing_task,
        expected_attempt_id=failing_attempt,
        error_message="failed",
        callbacks=CallbackRecorder().callbacks(),
    )

    assert store.status == RunStatus.FAILED
    assert store.task_statuses[waiting_task] == TaskStatus.CANCELLED
    assert store.waits[(waiting_task, "wait-1")].state == "invalidated"
    assert (
        resume_task(
            store,
            waiting_task,
            "wait-1",
            expected_attempt_id=waiting_attempt,
            expected_generation=0,
        )
        is None
    )


def test_wait_and_recovery_are_decided_by_which_commits_first() -> None:
    wait_first, task_id, attempt_id, generation = running_task()
    park(wait_first, task_id, attempt_id, generation)
    assert (
        recover_task(
            wait_first,
            task_id,
            expected_attempt_id=attempt_id,
            expected_generation=generation,
        )
        is None
    )

    recovery_first, task_id, attempt_id, generation = running_task()
    recovered = recover_task(
        recovery_first,
        task_id,
        expected_attempt_id=attempt_id,
        expected_generation=generation,
    )
    assert recovered is not None
    assert (
        wait_task(
            recovery_first,
            task_id,
            "wait-1",
            expected_attempt_id=attempt_id,
            expected_generation=generation,
        )
        is None
    )


@pytest.mark.parametrize("terminal", [TaskStatus.FAILED, TaskStatus.CRASHED])
def test_wait_and_terminal_result_are_decided_by_which_commits_first(
    terminal: TaskStatus,
) -> None:
    wait_first, task_id, attempt_id, generation = running_task()
    park(wait_first, task_id, attempt_id, generation)
    if terminal == TaskStatus.FAILED:
        result = wait_first.mark_task_failed(
            task_id,
            "late",
            expected_attempt_id=attempt_id,
        )
    else:
        result = wait_first.mark_task_crashed(
            task_id,
            "late",
            expected_attempt_id=attempt_id,
        )
    assert result is None
    assert wait_first.task_statuses[task_id] == TaskStatus.WAITING

    terminal_first, task_id, attempt_id, generation = running_task()
    if terminal == TaskStatus.FAILED:
        result = terminal_first.mark_task_failed(
            task_id,
            "failed",
            expected_attempt_id=attempt_id,
        )
    else:
        result = terminal_first.mark_task_crashed(
            task_id,
            "crashed",
            expected_attempt_id=attempt_id,
        )
    assert result is not None
    assert (
        wait_task(
            terminal_first,
            task_id,
            "wait-1",
            expected_attempt_id=attempt_id,
            expected_generation=generation,
        )
        is None
    )


@pytest.mark.parametrize("operation", ["wait", "resume"])
@pytest.mark.parametrize("mismatch", ["task", "attempt", "generation", "wait"])
def test_wait_wrappers_reject_contradictory_store_results(
    operation: str,
    mismatch: str,
) -> None:
    class WrongWaitStore(FakeStore):
        def try_wait_task(
            self,
            task_id: str,
            wait_id: str,
            *,
            expected_attempt_id: str,
            expected_generation: int,
        ) -> TaskWaitResult | None:
            result = TaskWaitResult(
                task_id="another" if mismatch == "task" else task_id,
                attempt_id="another" if mismatch == "attempt" else expected_attempt_id,
                run_id="run",
                node_index=0,
                generation=(
                    expected_generation + 1
                    if mismatch == "generation"
                    else expected_generation
                ),
                wait_id="another" if mismatch == "wait" else wait_id,
            )
            return result

        def try_resume_task(
            self,
            task_id: str,
            wait_id: str,
            *,
            expected_attempt_id: str,
            expected_generation: int,
        ) -> TaskResumeResult | None:
            return TaskResumeResult(
                task_id="another" if mismatch == "task" else task_id,
                waiting_attempt_id=(
                    "another" if mismatch == "attempt" else expected_attempt_id
                ),
                resumed_attempt_id="replacement",
                run_id="run",
                node_index=0,
                generation=(
                    expected_generation + 1
                    if mismatch == "generation"
                    else expected_generation
                ),
                wait_id="another" if mismatch == "wait" else wait_id,
            )

    store = WrongWaitStore([DagNode(slug="root", plugin="p")])
    task_id = store.seed_launch(0).task_ids[0]
    attempt_id = store.current_attempts[task_id]
    operation_fn = wait_task if operation == "wait" else resume_task

    with pytest.raises(StoreContractError, match="mismatched fields"):
        operation_fn(
            store,
            task_id,
            "wait-1",
            expected_attempt_id=attempt_id,
            expected_generation=0,
        )


@pytest.mark.parametrize("operation", ["wait", "resume"])
def test_wait_wrappers_reject_results_outside_the_protocol(operation: str) -> None:
    class WrongWaitStore(FakeStore):
        def try_wait_task(self, *args, **kwargs) -> TaskWaitResult | None:
            return object()  # type: ignore[return-value]

        def try_resume_task(self, *args, **kwargs) -> TaskResumeResult | None:
            return object()  # type: ignore[return-value]

    store = WrongWaitStore([DagNode(slug="root", plugin="p")])
    task_id = store.seed_launch(0).task_ids[0]
    attempt_id = store.current_attempts[task_id]
    operation_fn = wait_task if operation == "wait" else resume_task

    with pytest.raises(StoreContractError, match="Task(?:Wait|Resume)Result"):
        operation_fn(
            store,
            task_id,
            "wait-1",
            expected_attempt_id=attempt_id,
            expected_generation=0,
        )
