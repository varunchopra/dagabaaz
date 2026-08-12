# dagabaaz

A Python library that orchestrates multi-step workflows as directed acyclic graphs. You define the steps and their dependencies; the engine handles scheduling, data routing and failures.

```shell
pip install dagabaaz
```

Python 3.12 or later is required. The optional `re2` extra protects pipe expressions that use regular expressions from ReDoS attacks.

## Purpose

Dagabaaz is intended for applications that already provide their own database, queue and worker runtime. The application supplies storage and queueing through Python protocols.

## Pipeline definition

A pipeline is a list of `DagNode` objects. Each node has a plugin name and a non-empty slug that is unique within the pipeline. Non-root nodes have named input edges and must declare an input mode. A root node may omit its input mode, which resolves to `ALL`.

```python
from dagabaaz.constants import InputMode, InputRole
from dagabaaz.models import DagNode, EdgeSource, InputEdge

nodes = [
    DagNode(slug="load", plugin="load"),
    DagNode(slug="settings", plugin="load_settings"),
    DagNode(
        slug="transform",
        plugin="transform",
        input_mode=InputMode.EACH,
        edges=(InputEdge(name="records", source="load"),),
        bindings={"title": EdgeSource(edge="records", field="title")},
    ),
    DagNode(
        slug="publish",
        plugin="publish",
        input_mode=InputMode.ALL,
        edges=(
            InputEdge(name="results", source="transform"),
            InputEdge(
                name="settings",
                source="settings",
                role=InputRole.SIDE,
                required=False,
            ),
        ),
    ),
]
```

`InputMode.EACH` creates one task for each selected output on its main edge. `InputMode.ALL` creates one task with all selected outputs. `InputMode.BY_CORRELATION` creates one task for each retained correlation ID.

Filters and selection belong to an `InputEdge`. Bindings refer to an edge by name.

## Store integration

The application provides storage and queueing through a `DagStore` implementation. The excerpt below shows the planning and finalisation methods; a complete store must implement the full protocol.

```python
class AppStore:
    def get_run_nodes(self, run_id): ...

    def try_start_run(self, run_id): ...

    def get_planning_snapshot(
        self,
        run_id,
        node_index,
        source_indices,
        *,
        max_outputs_per_edge,
        max_outputs_total,
        max_routing_bytes,
    ): ...

    def try_create_node_launch(
        self,
        run_id,
        node_index,
        plugin_name,
        snapshot_token,
        plans,
        zero_task_disposition,
        error="",
    ): ...

    def try_claim_task(
        self,
        task_id,
        *,
        expected_attempt_id,
        expected_generation,
    ): ...

    def try_recover_task(
        self,
        task_id,
        *,
        expected_attempt_id,
        expected_generation,
    ): ...

    def try_complete_task(
        self,
        task_id,
        outputs,
        *,
        expected_attempt_id,
    ): ...

    def try_finalize_run(
        self,
        run_id,
        status,
        *,
        cause=None,
        reason=None,
    ): ...
```

Each edge and the combined snapshot may contain one output beyond the supplied count limit to signal overflow. Snapshot loading also stops after the first output that exceeds the routing-byte limit. `try_create_node_launch` checks the snapshot token and writes the node launch, tasks, plans and queue outbox entries in one transaction. `try_complete_task` rejects a first publication above the output-count or routing-byte limit before it writes the task status, outputs, launch state and progress. Materialised data is outside this byte limit. The store protocol is in [`store.py`](src/dagabaaz/store.py).

Claim and recovery belong to `DagStore` because every store dispatches work and accepts its result. Waiting remains optional through `DagWaitStore`. A replacement attempt and its queue outbox row must commit together; otherwise a task could be marked `QUEUED` with nothing to run.

## Run lifecycle

The application stores the run and its node definitions before it calls `start_run`. Node definitions must remain unchanged after the run record is created. Start requests must be delivered at least once, or the application must reconcile running runs after a process failure.

```python
from dagabaaz.orchestrator import OrchestratorCallbacks, start_run

callbacks = OrchestratorCallbacks(
    on_run_completed=lambda run_id: print("completed", run_id),
    on_run_failed=lambda run_id: print("failed", run_id),
    on_run_crashed=lambda run_id: print("crashed", run_id),
    on_run_cancelled=lambda run_id: print("cancelled", run_id),
)

start_run(store, "run-1", callbacks=callbacks)
```

Before a worker acts on a queue delivery, it must claim the attempt named in that delivery:

```python
from dagabaaz.execution import claim_task

claimed = claim_task(
    store,
    delivery.task_id,
    expected_attempt_id=delivery.attempt_id,
    expected_generation=delivery.generation,
)
if claimed is None:
    return
```

Only one delivery can claim an attempt. If `claim_task` returns `None`, that delivery must not run. The return value does not say why the claim failed: another worker may own it, the task may have finished or entered `WAITING`, or the delivery may be stale. Use the queue and provider records to decide whether to acknowledge it, leave its owner alone or recover abandoned work.

After claiming, the worker reports completion with the same attempt ID. The store commits the outputs and task state before Dagabaaz looks for newly ready nodes. Repeating a committed callback returns the first result; it cannot replace the outputs. An invalid first output batch fails the task and run.

The first completion, failure or crash callback succeeds only for the attempt that currently owns a `RUNNING` task. This makes callbacks from earlier attempts harmless. An exact repeat after a terminal transition returns the stored result. Run finalisation checks the attempt again so a concurrent retry cannot fail or crash its replacement.

## Waiting and re-entry

The task status describes the logical task. The attempt ID names the execution allowed to change it.

```text
PENDING --> QUEUED -- claim --> RUNNING -- wait --> WAITING
              ^                         |                    |
              |                         |                    |
              +-- recovery, new attempt +                    |
              |                                              |
              +----------- resume, new attempt --------------+

RUNNING --> COMPLETED | FAILED | CRASHED

Any unfinished task may become CANCELLED when its run is finalised.
```

`WAITING` means unfinished but deliberately not runnable. The task keeps its plan and generation, creates no new queue delivery and cannot be claimed. The delivery that entered the wait may remain leased until the application acknowledges it. Other branches may continue, but dependent nodes remain blocked and the run stays `RUNNING`.

Dagabaaz needs only a key that links the wait to application data. The application keeps the prompt, answer, secret, approval or provider payload. The key must be non-empty, contain no NUL character and be no longer than 512 characters.

```python
from dagabaaz.waiting import resume_task, wait_task

waited = wait_task(
    store,
    task_id,
    checkpoint_id,
    expected_attempt_id=attempt_id,
    expected_generation=generation,
)

# In the transaction that records the response:
resumed = resume_task(
    store,
    task_id,
    checkpoint_id,
    expected_attempt_id=attempt_id,
    expected_generation=generation,
)
```

Resumption uses a new attempt ID so callbacks from the execution that entered the wait are stale. When the new attempt is claimed, `TaskClaimResult.resumed_from_wait_id` tells the worker which application checkpoint to load. Retry and recovery keep this value so replacement attempts load the same response.

A timeout can hide whether a store operation committed. Repeating the same wait, resume or recovery therefore returns its original result without creating another attempt or queue entry. That result remains available after later state changes. Trying to resume a wait that cancellation or replanning has already invalidated returns `None`, and a task never reuses a wait ID.

The result of `resume_task` or `recover_task` records what the store committed; it is not permission to execute. Work starts only when the outbox produces a delivery and that delivery wins `claim_task`.

> Waiting, resumption and recovery start the task again from stored state; they do not continue a suspended Python stack. Earlier work may run again unless the application records it or makes it safe to repeat.

### Recovery safety

An expired queue lease means that the broker may deliver the message again. It does not prove that the old worker stopped; the worker may still be running after losing its heartbeat. Recover an attempt only when at least one of these statements is true:

- The old execution can no longer change anything outside Dagabaaz.
- The application has confirmed that the worker or provider stopped.
- The external service rejects a repeated action with the same idempotency key.
- The application checked the uncertain outcome and knows that retrying is safe.

Otherwise, treat the outcome as uncertain or report a crash instead of recovering the attempt automatically. Dagabaaz rejects late database callbacks from the abandoned attempt, but it cannot prevent a repeated browser action, purchase, message or submission.

### Application transactions

Two separate transactions can disagree after a crash: the application may store a response without resuming the task, or resume the task without storing the response. Where the database and queue allow it, use one transaction for:

- The application checkpoint, `wait_task` and acknowledgement of the current delivery.
- The application response and `resume_task`, including the replacement outbox row.

The `wait_task` and `resume_task` wrappers do not create this surrounding transaction. They rely on each store method to make its own state and outbox writes atomic. If the resources cannot share a transaction, the application needs a recovery procedure that is safe to repeat. Dagabaaz does not account for worker time or billing.

`QUEUED` promises that a delivery exists or that an outbox row will create one. A queued task with no dispatch work breaks that promise and looks like lost publication. Represent an external wait as `WAITING`, not as a null job ID or an application-only phase.

Every `DagStore` implementation must add claim and recovery before upgrading. `DagWaitStore` remains optional. A wait-capable store must keep resolved and invalidated waits, plus recovery results, because the current task row cannot answer a repeated call after the task changes. Database-backed stores need their own transaction and concurrency tests; the in-memory suite cannot prove database atomicity.

## Task inputs

Each task stores a `TaskInputPlan`, which records the selected output IDs under their edge names. Workers use the routing decision made during planning.

```python
from dagabaaz.task_input import resolve_task_inputs

inputs = resolve_task_inputs(output_store, run_id="run-1", plan=plan)
record = inputs.edges["records"][0]
print(record.fields, record.data, inputs.parameters)
```

The output resolver must return the requested IDs and no others. Dagabaaz restores the order recorded in the plan. A missing or extra ID raises `MaterializationError`.

`OutputRef.fields` contains the JSON used for filtering, selection, expressions and bindings. `ResolvedOutput.data` contains values needed by the plugin, such as paths, URLs or records. The routing decision is fixed before the worker materialises those values.

Each root `TaskInputPlan.parameters` stores the complete merged run input. The worker receives the same mapping as `TaskInputs.parameters`. An active binding replaces a parameter with the same name. Non-root nodes receive runtime values only through bindings and expressions.

JSON equality in filters preserves JSON types at every nesting level: booleans do not equal numbers, arrays retain order and object key order is ignored. Ordered filters coerce numbers and numeric strings to numbers. Text predicates retain scalar string coercion. Output and correlation IDs are opaque store identities of at most 512 characters. Paths and URIs belong in materialised data.

Run input is limited to 256 KiB. A node may declare up to 64 bindings, an edge filter may contain up to 64 rules and an expression may perform up to 64 reference lookups and pipe calls. Binding evaluation may process up to 16 MiB of JSON per node launch. The remaining execution limits are defined in [`constants.py`](src/dagabaaz/constants.py).

`SecretSource` stores a reference name in `TaskInputPlan.secret_refs`; the worker adapter resolves that reference. Secret values are not stored in task plans.

## Expressions

An `ExpressionSource` uses `{namespace.field | pipe}` syntax. Edge names form namespaces, while `input` refers to run input. A standalone reference retains its JSON type; interpolation with other text returns a string.

```python
from dagabaaz.models import ExpressionSource

ExpressionSource(expression="{records.title | trim}")
ExpressionSource(expression="record-{input.sequence}")
ExpressionSource(expression="{list(left.url, right.url) | compact | join(;)}")
```

Graph validation checks expression syntax, pipe names and edge references. Planning evaluates the expression and stores its result in the task plan. Workers do not evaluate it again. The pipe implementations are listed in [`pipes.py`](src/dagabaaz/pipes.py).

## Correlation

A correlation ID identifies outputs that belong to the same routing group. `BY_CORRELATION` uses the ID to combine outputs from parallel branches.

Workers do not supply correlation IDs. The store derives each ID from the stored node and task plan when it accepts an `EmittedOutput`. With `CorrelationMode.DEFAULT`, each root output starts a group, outputs from `EACH` and `BY_CORRELATION` inherit the task plan's group, and outputs from `ALL` have no correlation ID. An explicit `CorrelationMode` overrides these defaults.

Applications may store parent-child output links when they need an execution history.

## Retries

A retry of a resumed attempt must keep `resumed_from_wait_id`; otherwise its replacement cannot find the response that resumed the crashed attempt. The expected attempt ID prevents a stale retry from replacing newer work. A waiting task has not crashed, so it must be resumed or cancelled.

`retry_run` works with failed, crashed and cancelled runs. The store finds unsuccessful launches and their descendants while it holds the run lock. A caller may supply a boundary for a failed or crashed run; the boundary asserts that the listed nodes still contain active failures.

Replanning replaces a node generation as a unit. Dagabaaz invalidates its tasks, plans, unresolved waits and outputs, even when one task had succeeded, so a late response cannot revive old work. Results from earlier wait, resume and recovery calls remain available only so callers can recognise a repeated request. Completed launches outside the affected closure remain in place, and invalidated output IDs remain reserved for the life of the run.

The store increments the retry count, advances the affected plan generations, invalidates the affected launches and reopens the run in one transaction. The application must not call `retry_run` and a terminal callback for the same run at the same time. Cleanup and `reconcile_run` may follow only after the retry succeeds.

## Licence

MIT
