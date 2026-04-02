"""Centralized domain constants for the DAG execution engine."""

from enum import StrEnum


class RunStatus(StrEnum):
    """Lifecycle states for a pipeline run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CRASHED = "crashed"
    CANCELLED = "cancelled"


class TaskStatus(StrEnum):
    """Lifecycle states for a single task within a run.

    SKIPPED vs FILTERED: both are terminal non-execution states, but they
    have different cascade semantics. SKIPPED cascades — if a node is
    skipped, all downstream nodes are also skipped (upstream is dead).
    FILTERED does NOT cascade — the node had no artifacts (edge filter
    rejected all, or BFS stopped at a non-passthrough upstream). Downstream
    nodes still attempt artifact collection. Whether they succeed depends
    on the ``passthrough`` flag on upstream plugins: BFS walks through
    passthrough nodes (Gate) to reach earlier ancestors, but stops at
    non-passthrough nodes (processors).
    """

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CRASHED = "crashed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    FILTERED = "filtered"


class NodeSummaryStatus(StrEnum):
    """Per-node aggregate status for skip cascade decisions."""

    COMPLETED = "completed"
    SKIPPED = "skipped"
    PARTIAL = "partial"


class FilterOperator(StrEnum):
    """Predicate operators for edge filter rules."""

    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


class FileType(StrEnum):
    """Virtual file type classification for artifact filtering."""

    VIDEO = "video"
    SUBTITLE = "subtitle"
    IMAGE = "image"
    AUDIO = "audio"
    ARCHIVE = "archive"
    OTHER = "other"


class ArtifactSelector(StrEnum):
    """Post-filter reduction strategies for edge filters."""

    LARGEST = "largest"
    SMALLEST = "smallest"


class FanMode(StrEnum):
    """How a node aggregates upstream artifacts into tasks.

    A boolean fan_in couldn't express scatter-gather correlation — the
    pattern where N items fan out to M parallel branches and a downstream
    node must receive N groups of M correlated results.

    SINGLE:    One task per artifact (fan-out). The default.
    AGGREGATE: One task receiving ALL upstream artifacts.
    GROUPED:   One task per origin group. For scatter-gather correlation —
               10 items x 3 branches -> 10 tasks, each receiving 3 results.
    """

    SINGLE = "single"
    AGGREGATE = "aggregate"
    GROUPED = "grouped"


class BindingSource(StrEnum):
    """Where a binding gets its value from."""

    PREVIOUS_NODE = "previous_node"
    CONFIG = "config"
    RUNTIME = "runtime"
    EXPRESSION = "expression"


RUN_TERMINAL_STATUSES = frozenset(
    {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CRASHED, RunStatus.CANCELLED}
)
"""Run statuses where no further state transitions are allowed."""

TASK_TERMINAL_STATUSES = frozenset(
    {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CRASHED,
        TaskStatus.CANCELLED,
        TaskStatus.SKIPPED,
        TaskStatus.FILTERED,
    }
)
"""Task statuses where no further state transitions are allowed."""

RUN_RETRYABLE_STATUSES = RUN_TERMINAL_STATUSES - {RunStatus.COMPLETED}
"""Terminal run statuses that can be retried."""

TASK_RETRYABLE_STATUSES = TASK_TERMINAL_STATUSES - {
    TaskStatus.COMPLETED,
    TaskStatus.SKIPPED,
}
"""Terminal task statuses that warrant retry."""

ARTIFACT_STANDARD_FIELDS = frozenset(
    {"file_path", "file_name", "file_size", "mime_type"}
)
"""Fields resolved directly from artifact attributes, not metadata.

Note: ``file_path`` is only present on ``TaskArtifact`` (worker-side),
not ``DagArtifact`` (orchestrator-side). ``extract_artifact_field``
handles the absence gracefully via getattr with debug logging.
"""

MAX_FAN_OUT = 200
"""Fan-out guard: fail loudly rather than silently truncating results.
Airflow caps at 1024; 200 is generous for our use case."""

MAX_BFS_DEPTH = 50
"""BFS depth cap for dependency traversal — detects cycles."""
