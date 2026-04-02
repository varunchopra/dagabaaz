"""Core DAG models — node, artifact, edge filter, binding sources, and lifecycle constants.

``DagArtifact`` is frozen — artifacts are immutable. Binding source models
live here so this module is a leaf dependency (imports only from ``constants.py``).
"""

from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

from dagabaaz.constants import ArtifactSelector, FanMode, FilterOperator


class ExpressionError(ValueError):
    """Syntax error or required-value violation in an expression."""


# Binding source models -- define how data moves between nodes.
# Resolution logic lives in bindings.py.
class NodeSource(BaseModel):
    """Read a field from a specific upstream node's artifacts."""

    source: Literal["previous_node"] = "previous_node"
    node: str  # slug of the dependency node
    key: str = ""  # artifact field: file_path, url, or metadata key


class ConfigSource(BaseModel):
    """Literal value passthrough (or $SECRET_NAME for secret resolution)."""

    source: Literal["config"] = "config"
    value: str = ""  # the literal value


class RuntimeSource(BaseModel):
    """Read from the run's user-provided input at execution time."""

    source: Literal["runtime"] = "runtime"
    key: str = ""  # run_input key
    label: str = ""
    placeholder: str = ""
    required: bool = False
    default: str = ""


class ExpressionSource(BaseModel):
    """Inline expression combining multiple sources with pipe transforms."""

    source: Literal["expression"] = "expression"
    expression: str  # e.g. "{step_1.file_path | basename}"


InputBinding = Annotated[
    Annotated[NodeSource, Tag("previous_node")]
    | Annotated[ConfigSource, Tag("config")]
    | Annotated[RuntimeSource, Tag("runtime")]
    | Annotated[ExpressionSource, Tag("expression")],
    Discriminator("source"),
]


class FilterRule(BaseModel):
    """A single predicate in an edge filter's rule list.

    Evaluated against each artifact independently. All rules in an
    EdgeFilter are AND-composed — an artifact must pass every rule.

    Fields can be:
    - Virtual (computed from artifact data): file_type, extension,
      file_size, file_name, mime_type
    - Dynamic (from upstream plugin output_vars metadata): custom_field,
      external_id, category, confidence, etc.
    """

    field: str
    operator: FilterOperator
    value: str | float | list[str]


class EdgeFilter(BaseModel):
    """Filter configuration for a single DAG edge (dependency → node).

    Two composable operations:
      1. Rules (predicates): AND-composed, narrow N artifacts to M
      2. Select (reduction): pick largest/smallest from survivors (M→1)

    When the result is empty, the orchestrator marks the downstream node
    as ``filtered`` (non-cascading) for SINGLE fan mode, or ``skipped``
    (cascading) for AGGREGATE fan mode where a poisoned edge corrupts
    the entire merged input. See ``_launch_node`` for the full decision
    tree.
    """

    rules: list[FilterRule] = Field(default_factory=list)
    select: ArtifactSelector | None = None


class DagNode(BaseModel):
    """A node in a DAG pipeline."""

    name: str = ""
    slug: str = ""
    plugin: str
    depends_on: list[str] = Field(default_factory=list)
    edge_filters: dict[str, EdgeFilter] = Field(default_factory=dict)
    bindings: dict[str, InputBinding] = Field(default_factory=dict)
    # Graph-level aggregation mode — see FanMode docstring for the three modes.
    # This is a graph property, not a plugin property: the pipeline designer
    # decides how a node aggregates, not the plugin author. Plugins declare
    # fan_mode_default as a UI hint; the orchestrator reads only this field.
    fan_mode: FanMode = FanMode.SINGLE


@runtime_checkable
class DagArtifactLike(Protocol):
    """Structural interface for artifacts used in filtering and binding resolution.

    Both ``DagArtifact`` (engine) and ``ArtifactWorkerRow`` (application)
    satisfy this.
    """

    file_name: str
    file_size: int | None
    mime_type: str | None
    metadata: dict[str, Any]
    origin_artifact_id: str | None


class DagArtifact(BaseModel):
    """Lightweight artifact representation for DAG routing decisions.

    Carries just enough data for edge filtering and task routing — not
    the full artifact record. ``metadata`` is included so edge filter
    rules on dynamic fields (e.g. media_type, confidence) can inspect
    plugin output without a separate DB round-trip.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    file_name: str
    file_size: int | None
    mime_type: str | None
    metadata: dict[str, Any] = Field(default_factory=dict)
    origin_artifact_id: str | None = None


class TaskArtifact(BaseModel):
    """Worker-side artifact with file path for physical access.

    Richer than ``DagArtifact`` — carries ``file_path`` needed for actual
    file operations. Used by ``TaskInputStore`` implementations to return
    artifacts with enough data for plugin execution, not just routing.
    """

    model_config = ConfigDict(frozen=True)

    file_path: str
    file_name: str
    file_size: int | None
    mime_type: str | None
    metadata: dict[str, Any] = Field(default_factory=dict)
    origin_artifact_id: str | None = None

    def to_input_dict(self) -> dict[str, object]:
        """Convert to a dict suitable for plugin input_data."""
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "metadata": self.metadata,
        }


class TaskContext(BaseModel):
    """Minimal task context for orchestrator routing decisions."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    node_index: int
