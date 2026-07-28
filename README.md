# dagabaaz

A Python library that orchestrates multi-step workflows as directed acyclic graphs. You define the steps and their dependencies; the engine handles scheduling, data routing, and failures.

```shell
pip install dagabaaz
```

Python 3.12 or later is required. The optional `re2` extra protects pipe expressions that use regular expressions from ReDoS attacks.

## Purpose

Dagabaaz is intended for applications that already provide their own database, queue and worker runtime. The application supplies storage and queueing through Python protocols.

## Pipeline definition

A pipeline is a list of `DagNode` objects. Each node has a slug and a plugin name. Non-root nodes have named input edges that connect them to upstream nodes.

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

    def get_planning_snapshot(self, run_id, node_index, edges): ...

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

    def try_finalize_run(self, run_id, status, error, *, cause=None): ...
```

`try_create_node_launch` checks the snapshot token and writes the node launch, tasks, plans and queue outbox entries in one transaction. The orchestrator's store protocol is in [`store.py`](src/dagabaaz/store.py).

## Run lifecycle

The application stores the run and its node definitions before it calls `start_run`.

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

A worker adapter first checks the queued attempt ID and plan generation. It then records task completion and the task's outputs in one application-defined transaction. After that transaction commits, `on_task_complete` checks whether another node is ready.

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

JSON equality in filters preserves JSON types at every nesting level: booleans do not equal numbers, arrays retain order and object key order is ignored.

`SecretSource` stores a reference name in `TaskInputPlan.secret_refs`; the worker adapter resolves that reference. Secret values are not stored in task plans.

## Correlation

A correlation ID identifies outputs that belong to the same routing group. `BY_CORRELATION` uses the ID to combine outputs from parallel branches.

The application calls `correlation_id_for_output` when it creates an `OutputRef`. With `CorrelationMode.DEFAULT`, each root output starts a group, outputs from `EACH` and `BY_CORRELATION` inherit the task plan's group, and outputs from `ALL` have no correlation ID. An explicit `CorrelationMode` overrides these defaults.

Applications may store parent-child output links when they need an execution history.

## Retries

`retry_task` creates another attempt for a crashed task with its existing plan while the run remains active. The caller supplies the current attempt ID so a stale retry cannot replace a later attempt.

For a failed or crashed run, `retry_run` takes a node boundary. It restarts the boundary and every incomplete or failed launch, together with their descendants. Completed siblings outside this set remain in place.

For a run with `CANCELLED` status, `retry_run` takes no boundary and restarts its incomplete work. This also covers a run cancelled before any node was launched.

The store checks the run status and affected nodes while holding the run lock. It increments the retry count, advances the affected plan generations and invalidates the affected launches in the same transaction. The application must serialise a retry with terminal handlers for the same run until the terminal callback has returned. Cleanup and `reconcile_run` may follow only after the retry succeeds.

## Licence

MIT
