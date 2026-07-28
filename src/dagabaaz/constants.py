"""Enums and limits used by the execution model."""

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
    """Lifecycle states for task attempts.

    Node outcomes such as skipped or filtered belong to ``NodeDisposition``.
    """

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CRASHED = "crashed"
    CANCELLED = "cancelled"


class NodeDisposition(StrEnum):
    """The outcome of planning and launching one node.

    A required edge whose source was skipped gives the consumer a ``SKIPPED``
    disposition. A filtered source supplies an empty edge, after which the
    consumer's required-edge rules determine its disposition.
    """

    LAUNCHED = "launched"
    SKIPPED = "skipped"
    FILTERED = "filtered"
    FAILED = "failed"


class InputMode(StrEnum):
    """How selected main-edge outputs become task plans.

    ``EACH`` creates one plan per output. ``ALL`` creates one plan containing
    every selected output. ``BY_CORRELATION`` creates one plan per retained
    correlation ID.
    """

    EACH = "each"
    ALL = "all"
    BY_CORRELATION = "by_correlation"


class InputRole(StrEnum):
    """Whether an edge participates in input-mode grouping or supplies a side input."""

    MAIN = "main"
    SIDE = "side"


class CorrelationMode(StrEnum):
    """How emitted outputs receive correlation IDs.

    ``DEFAULT`` is resolved from whether the node is a root and from its input
    mode. ``INHERIT`` copies the task plan's correlation ID, ``NEW`` uses the
    emitted output's ID and ``NONE`` leaves the output uncorrelated.
    """

    DEFAULT = "default"
    INHERIT = "inherit"
    NEW = "new"
    NONE = "none"


class FilterOperator(StrEnum):
    """Predicate operators for routing fields."""

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


class SortOrder(StrEnum):
    """Ordering used by edge selection."""

    ASC = "asc"
    DESC = "desc"


class LaunchCreateStatus(StrEnum):
    """The outcome of an attempted node-launch creation.

    ``CREATED`` identifies a newly written launch, ``ALREADY_EXISTS`` identifies
    a launch for the same generation, and ``STALE`` means that the snapshot no
    longer represents the active source state.
    """

    CREATED = "created"
    ALREADY_EXISTS = "already_exists"
    STALE = "stale"


RUN_TERMINAL_STATUSES = frozenset(
    {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CRASHED, RunStatus.CANCELLED}
)
TASK_TERMINAL_STATUSES = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CRASHED, TaskStatus.CANCELLED}
)
RUN_RETRYABLE_STATUSES = RUN_TERMINAL_STATUSES - {RunStatus.COMPLETED}

# These limits bound memory, storage and queueing at execution boundaries.
MAX_OUTPUT_FIELDS_BYTES = 64 * 1024
MAX_JSON_DEPTH = 16
MAX_JSON_KEYS = 512
MAX_JSON_VALUES = 10_000
MAX_INPUT_EDGES_PER_NODE = 64
MAX_NODE_BINDINGS = 64
MAX_NODE_BINDING_EVALUATIONS = 64
MAX_LAUNCH_BINDING_EVALUATIONS = 10_000
MAX_LAUNCH_BINDING_EVALUATED_BYTES = 16 * 1024 * 1024
MAX_FILTER_RULES_PER_EDGE = 64
MAX_FILTER_MEMBERSHIP_VALUES = 256
MAX_OUTPUT_ID_LENGTH = 512
# Snapshot readers return at most each limit plus one output. The extra output
# only proves that planning must reject the snapshot.
MAX_SNAPSHOT_OUTPUTS_PER_EDGE = 10_000
MAX_SNAPSHOT_OUTPUTS_PER_NODE = 50_000
MAX_SNAPSHOT_ROUTING_BYTES = 16 * 1024 * 1024
# A first task publication above either limit fails before output persistence.
MAX_OUTPUTS_PER_TASK_COMPLETION = MAX_SNAPSHOT_OUTPUTS_PER_NODE
MAX_TASK_COMPLETION_ROUTING_BYTES = MAX_SNAPSHOT_ROUTING_BYTES
MAX_TASK_PLANS_PER_LAUNCH = 200
MAX_OUTPUT_REFS_PER_PLAN = 5_000
MAX_TASK_PLAN_BYTES = 512 * 1024
# Root plans include the run input and the plan envelope.
MAX_RUN_INPUT_BYTES = MAX_TASK_PLAN_BYTES // 2
MAX_EXPRESSION_LENGTH = 16_384
MAX_EXPRESSION_RESULT_BYTES = MAX_TASK_PLAN_BYTES
MAX_EXPRESSION_EVALUATION_STEPS = 64
MAX_PIPE_ARGUMENT_LENGTH = 4_096
MAX_PIPE_INTEGER_DIGITS = 4_096
MAX_SNAPSHOT_RETRIES = 3
