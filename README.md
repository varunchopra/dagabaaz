# dagabaaz

A Python library that orchestrates multi-step workflows as directed acyclic graphs. You define the steps and their dependencies; the engine handles scheduling, data routing, and failures.

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

A worker adapter first checks the queued attempt ID and plan generation. It then passes the attempt ID and a tuple of `EmittedOutput` objects to `on_task_complete`. The store commits the task and its outputs before Dagabaaz checks whether another node is ready. The attempt ID is the completion idempotency key, so a repeated callback cannot replace committed outputs. An invalid first output batch fails the task and run.

`on_task_failed` and `on_task_crashed` receive the queued attempt ID. The store updates the task only if that attempt is still current. `try_finalize_run` repeats the attempt check before changing the run status, which protects a replacement attempt created by a concurrent retry. The same transaction rejects invalid terminal transitions and cancels unfinished tasks.

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

`retry_task` creates another attempt for a crashed task with its existing plan while the run remains active. The caller supplies the current attempt ID so a stale retry cannot replace a later attempt.

`retry_run` works with failed, crashed and cancelled runs. The store finds unsuccessful launches and their descendants while it holds the run lock. A caller may supply a boundary for a failed or crashed run; the boundary asserts that the listed nodes still contain active failures.

Replanning invalidates a node generation as a unit. Its tasks, plans and outputs are removed even when one task in that generation had succeeded. Invalidated output IDs remain reserved for the lifetime of the run. Completed launches outside the affected closure remain in place. A cancelled run may be reopened before any node has launched.

The store increments the retry count, advances the affected plan generations, invalidates the affected launches and reopens the run in one transaction. The application must not call `retry_run` and a terminal callback for the same run at the same time. Cleanup and `reconcile_run` may follow only after the retry succeeds.

## Licence

MIT
