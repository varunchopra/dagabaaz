# dagabaaz

A Python library that orchestrates multi-step workflows as directed acyclic graphs. You define the steps and their dependencies; the engine handles scheduling, data routing, and failures.

```
pip install dagabaaz
```

Requires Python 3.12+. Optional: `google-re2` for ReDoS-safe regex in pipe expressions.

## Why This Exists

Most DAG engines (Airflow, Prefect, Dagster) are platforms. They own the scheduler, the database, the UI, and the execution runtime. If you're building a product where pipelines are a *feature* rather than the whole product, you don't want a platform. You want a library you call from your own code.

Persistence and dispatch are behind a `Protocol`. Bring your own database and queue.

## Quick Start

### 1. Define a pipeline

A pipeline is a list of `DagNode` objects. Each node has a slug (unique ID), a plugin name, and optional dependencies.

```python
from dagabaaz.models import DagNode
from dagabaaz.constants import FanMode

nodes = [
    DagNode(slug="source", plugin="fetch"),
    DagNode(slug="process", plugin="transform", depends_on=["source"]),
    DagNode(
        slug="export",
        plugin="export",
        depends_on=["process"],
        fan_mode=FanMode.AGGREGATE,
    ),
]
```

### 2. Implement `DagStore`

The engine talks to your infrastructure through the `DagStore` protocol (see `store.py`). The three most important methods:

```python
class MyStore:
    def get_barrier_state(self, run_id, node_index):
        # Return (run_status, total_tasks, completed_tasks)
        ...

    def try_claim_node_launch(self, run_id, node_index) -> bool:
        # Returns True if this call claimed the node
        ...

    def dispatch_task(self, run_id, node_index, plugin_name, input_artifact_id) -> str:
        # Create task record, push to your job queue, return task_id
        ...
```

### 3. Start a run

```python
from dagabaaz.orchestrator import start_run

root_indices = start_run(store, run_id="run-1", nodes=nodes)
```

### 4. Build task input

On the worker side, use `build_task_input` to assemble the data your plugin needs:

```python
from dagabaaz.task_input import build_task_input

input_data = build_task_input(
    store,
    run_id="run-1",
    node_index=1,
    input_artifact_id="artifact-xyz",
    nodes=nodes,
)
```

### 5. Handle task completion

After your worker executes a task, call back into the engine so it can dispatch the next steps:

```python
from dagabaaz.orchestrator import on_task_complete, OrchestratorCallbacks

callbacks = OrchestratorCallbacks(
    on_run_completed=lambda run_id: print(f"Run {run_id} done"),
    on_run_failed=lambda run_id: print(f"Run {run_id} failed"),
    on_run_crashed=lambda run_id: print(f"Run {run_id} crashed"),
    on_run_cancelled=lambda run_id: print(f"Run {run_id} cancelled"),
)

on_task_complete(
    store,
    task_id="task-1",
    callbacks=callbacks,
    resolve_passthrough=lambda plugin: False,
)
```

`on_task_complete` must be serialized per run (e.g. with a lock per run ID). `try_claim_node_launch` and `try_claim_run_terminal` must be atomic.

## Pipeline Patterns

### Linear

```python
nodes = [
    DagNode(slug="fetch", plugin="fetch"),
    DagNode(slug="transform", plugin="transform", depends_on=["fetch"]),
    DagNode(slug="export", plugin="export", depends_on=["transform"]),
]
```

### Fan-out / scatter-gather

A source produces multiple files. Two branches process each file in parallel. The merge node collects results that came from the same original file. If the source produced 10 files and there are 2 branches, the merge node gets 10 tasks, each with 2 results.

```python
nodes = [
    DagNode(slug="source", plugin="fetch"),
    DagNode(slug="branch_a", plugin="process_a", depends_on=["source"]),
    DagNode(slug="branch_b", plugin="process_b", depends_on=["source"]),
    DagNode(
        slug="merge",
        plugin="merge",
        depends_on=["branch_a", "branch_b"],
        fan_mode=FanMode.GROUPED,
    ),
]
```

### Conditional routing with edge filters

Edge filters route artifacts to different branches by type. Video files go to one branch, subtitles to another.

```python
from dagabaaz.models import DagNode, EdgeFilter, FilterRule
from dagabaaz.constants import FanMode, FilterOperator

nodes = [
    DagNode(slug="source", plugin="fetch"),
    DagNode(
        slug="video",
        plugin="transcode",
        depends_on=["source"],
        fan_mode=FanMode.AGGREGATE,
        edge_filters={
            "source": EdgeFilter(
                rules=[
                    FilterRule(
                        field="file_type", operator=FilterOperator.EQ, value="video"
                    )
                ]
            )
        },
    ),
    DagNode(
        slug="subtitle",
        plugin="parse_subs",
        depends_on=["source"],
        edge_filters={
            "source": EdgeFilter(
                rules=[
                    FilterRule(
                        field="file_type", operator=FilterOperator.EQ, value="subtitle"
                    )
                ]
            )
        },
    ),
]
```

When `source` produces a mix of `.mp4` and `.srt` files, the engine routes each type to the correct branch. If a branch receives no artifacts (e.g., no subtitles), it is marked `filtered` and does not block downstream nodes.

## Concepts

A pipeline is a graph of **nodes**. Each node wraps a plugin and declares which other nodes it depends on. When you execute a pipeline, that execution is called a **run**.

A node doesn't run until all its dependencies have finished -- this is **barrier sync**. Once a node runs, each execution of it is a **task**, and each task produces **artifacts** (files with optional metadata). A node's **fan mode** controls how many tasks it spawns: one per upstream artifact (single), one for all of them (aggregate), or one per group of related artifacts (grouped). Grouped mode uses **origin artifact** tracking to know which artifacts belong together -- if 10 files fan out through 3 branches, the merge node gets 10 tasks, each with 3 results.

**Edge filters** sit between nodes and decide which artifacts pass through. All rules must match (AND logic). **Input bindings** control how a task gets its data: from an upstream artifact field, a literal config value, user-provided run input, or an expression template.

When a node's upstream is dead, the node is **skipped** and that cascades to everything downstream. When a node simply has no artifacts to work with (edge filter rejected all), it is **filtered** -- this does not cascade. If a filtered node is a routing node (**passthrough**), the engine walks past it to find artifacts from further upstream. Processing nodes block this walk.

A task or run that can't transition further is in a **terminal state**. Tasks end as `completed`, `failed`, `crashed`, `cancelled`, `skipped`, or `filtered`. Runs end as `completed`, `failed`, `crashed`, or `cancelled`.

## Expression Language

Input bindings can use `{namespace.key | pipe}` expressions:

```python
"{source.file_path}"  # artifact field
"{source.title | upper | truncate(50)}"  # with transforms
"{list(branch_a.url, branch_b.url) | join(,)}"  # multiple sources
"{input.api_url | required}"  # run input
"{config.output_format | default(mp4)}"  # config value
```

Built-in pipes include `upper`, `lower`, `default`, `required`, `join`, `basename`, `match`, and others. See [`pipes.py`](src/dagabaaz/pipes.py) for the full list.

Expressions are validated at pipeline save time and evaluated at task execution time.

## License

MIT
