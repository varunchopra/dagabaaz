"""Models used by task planning and execution."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Literal, Protocol, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from dagabaaz.constants import (
    MAX_EXPRESSION_LENGTH,
    MAX_FILTER_MEMBERSHIP_VALUES,
    MAX_FILTER_RULES_PER_EDGE,
    MAX_INPUT_EDGES_PER_NODE,
    MAX_NODE_BINDINGS,
    MAX_OUTPUT_FIELDS_BYTES,
    MAX_OUTPUT_ID_LENGTH,
    MAX_OUTPUT_REFS_PER_PLAN,
    MAX_TASK_PLAN_BYTES,
    CorrelationMode,
    FilterOperator,
    InputMode,
    InputRole,
    LaunchCreateStatus,
    NodeDisposition,
    SortOrder,
)
from dagabaaz.json import (
    FrozenDict,
    JsonValue,
    bounded_json_size,
    canonical_json_bytes,
    freeze_json,
    freeze_object,
    thaw_json,
)

_EDGE_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_-]*$"


def _validate_edge_name(value: str) -> str:
    if value == "input":
        raise ValueError("reserved edge name 'input' conflicts with runtime input")
    return value


_EdgeName = Annotated[
    str,
    Field(pattern=_EDGE_NAME_PATTERN),
    AfterValidator(_validate_edge_name),
]

_ExpressionText = Annotated[str, Field(min_length=1, max_length=MAX_EXPRESSION_LENGTH)]
OutputId = Annotated[str, Field(min_length=1, max_length=MAX_OUTPUT_ID_LENGTH)]


def _copy_field_value(value: object, *, deep: bool) -> object:
    if isinstance(value, Mapping):
        return {
            key: _copy_field_value(item, deep=deep) if deep else item for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_copy_field_value(item, deep=deep) for item in value) if deep else value
    if isinstance(value, list):
        return [_copy_field_value(item, deep=deep) for item in value] if deep else value
    if deep and isinstance(value, BaseModel):
        return value.model_copy(deep=True)
    return copy.deepcopy(value) if deep else value


def _validate_output_fields_size(fields: FrozenDict) -> None:
    size = bounded_json_size(fields, MAX_OUTPUT_FIELDS_BYTES)
    if size > MAX_OUTPUT_FIELDS_BYTES:
        raise ValueError(f"output fields exceed the {MAX_OUTPUT_FIELDS_BYTES}-byte limit")


class _FrozenModel(BaseModel):
    """A frozen Pydantic model whose ``model_copy`` updates are revalidated."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    def model_copy(
        self,
        *,
        update: Mapping[str, object] | None = None,
        deep: bool = False,
    ) -> Self:
        """The copy contains only updates that pass model validation."""

        values = {
            name: _copy_field_value(getattr(self, name), deep=deep)
            for name in type(self).model_fields
        }
        if update:
            values.update(update)
        return type(self).model_validate(values)

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        copied = self.model_copy(deep=True)
        memo[id(self)] = copied
        return copied


class ExpressionError(ValueError):
    """An expression could not be parsed or evaluated."""


class PlanningError(ValueError):
    """Node inputs could not be planned from the supplied snapshot."""


class MaterializationError(RuntimeError):
    """A worker could not resolve every output named by its task plan."""


class EdgeSource(_FrozenModel):
    """An edge binding reads a routing field from the selected outputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Literal["edge"] = "edge"
    edge: _EdgeName
    field: str = Field(min_length=1)
    when: _ExpressionText | None = None


class LiteralSource(_FrozenModel):
    """A JSON literal stored as a task parameter."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    source: Literal["literal"] = "literal"
    value: JsonValue = None
    when: _ExpressionText | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _freeze_value(cls, value: object) -> JsonValue:
        return freeze_json(value)

    @field_serializer("value")
    def _serialize_value(self, value: JsonValue) -> object:
        return thaw_json(value)


class RuntimeSource(_FrozenModel):
    """A run input resolved while constructing a task plan."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    source: Literal["runtime"] = "runtime"
    key: str = Field(min_length=1)
    label: str = ""
    placeholder: str = ""
    required: bool = False
    default: JsonValue = None
    when: _ExpressionText | None = None

    @field_validator("default", mode="before")
    @classmethod
    def _freeze_default(cls, value: object) -> JsonValue:
        return freeze_json(value)

    @field_serializer("default")
    def _serialize_default(self, value: JsonValue) -> object:
        return thaw_json(value)


class SecretSource(_FrozenModel):
    """A secret reference resolved by the application's worker adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Literal["secret"] = "secret"
    name: str = Field(min_length=1)
    when: _ExpressionText | None = None


class ExpressionSource(_FrozenModel):
    """An expression evaluated against routing fields and run input."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Literal["expression"] = "expression"
    expression: _ExpressionText
    when: _ExpressionText | None = None


InputBinding = Annotated[
    EdgeSource | LiteralSource | RuntimeSource | SecretSource | ExpressionSource,
    Field(discriminator="source"),
]


class FilterRule(_FrozenModel):
    """One predicate evaluated against an ``OutputRef.fields`` value."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    field: str = Field(min_length=1)
    operator: FilterOperator
    value: JsonValue = None

    @field_validator("value", mode="before")
    @classmethod
    def _freeze_value(cls, value: object) -> JsonValue:
        return freeze_json(value)

    @field_serializer("value")
    def _serialize_value(self, value: JsonValue) -> object:
        return thaw_json(value)

    @model_validator(mode="after")
    def _check_membership_size(self) -> FilterRule:
        if (
            self.operator in (FilterOperator.IN, FilterOperator.NOT_IN)
            and isinstance(self.value, tuple)
            and len(self.value) > MAX_FILTER_MEMBERSHIP_VALUES
        ):
            raise ValueError(
                f"membership filter contains {len(self.value)} values; maximum is "
                f"{MAX_FILTER_MEMBERSHIP_VALUES}"
            )
        return self


class EdgeFilter(_FrozenModel):
    """Predicates that must all match for an output to remain on an edge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rules: tuple[FilterRule, ...] = Field(
        default=(),
        max_length=MAX_FILTER_RULES_PER_EDGE,
    )


class Selection(_FrozenModel):
    """Ordering and limit applied after an edge filter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: str = Field(min_length=1)
    order: SortOrder = SortOrder.ASC
    limit: int = Field(default=1, ge=1, strict=True)


class InputEdge(_FrozenModel):
    """A named connection to an upstream node and its routing rules.

    The name identifies the edge in plans, bindings and worker inputs.
    Filtering, selection, role and required status belong to the edge.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: _EdgeName
    source: str = Field(min_length=1)
    role: InputRole = InputRole.MAIN
    required: bool = True
    filter: EdgeFilter | None = None
    selection: Selection | None = None


class DagNode(BaseModel):
    """A pipeline node whose named edges define its dependencies and planned inputs.

    The slug is the node's stable identity within the pipeline. Input and
    correlation modes are configured on the node.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = ""
    slug: str = Field(min_length=1)
    plugin: str = Field(min_length=1)
    edges: tuple[InputEdge, ...] = Field(default=(), max_length=MAX_INPUT_EDGES_PER_NODE)
    bindings: dict[str, InputBinding] = Field(
        default_factory=dict,
        max_length=MAX_NODE_BINDINGS,
    )
    input_mode: InputMode | None = None
    correlation_mode: CorrelationMode = CorrelationMode.DEFAULT

    @field_validator("bindings")
    @classmethod
    def _validate_binding_names(cls, value: dict[str, InputBinding]) -> dict[str, InputBinding]:
        if any(not name for name in value):
            raise ValueError("binding names must not be empty")
        return value

    def model_copy(
        self,
        *,
        update: Mapping[str, object] | None = None,
        deep: bool = False,
    ) -> Self:
        """The copy contains only updates that pass model validation."""

        values = {
            name: _copy_field_value(getattr(self, name), deep=deep)
            for name in type(self).model_fields
        }
        if update:
            values.update(update)
        return type(self).model_validate(values)


class NodeDefinition(Protocol):
    """Read-only node data used after graph validation."""

    @property
    def name(self) -> str: ...

    @property
    def slug(self) -> str: ...

    @property
    def plugin(self) -> str: ...

    @property
    def edges(self) -> tuple[InputEdge, ...]: ...

    @property
    def bindings(self) -> Mapping[str, InputBinding]: ...

    @property
    def input_mode(self) -> InputMode: ...

    @property
    def correlation_mode(self) -> CorrelationMode: ...


class OutputRef(_FrozenModel):
    """The routing metadata for one run-scoped output.

    ``fields`` contains bounded, immutable JSON used during planning, and
    ``correlation_id`` identifies the output's routing group. ``ResolvedOutput``
    supplies the materialised data used by a worker.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    id: OutputId
    fields: FrozenDict = Field(default_factory=FrozenDict)
    correlation_id: OutputId | None = None

    @field_validator("fields", mode="before")
    @classmethod
    def _freeze_fields(cls, value: object) -> FrozenDict:
        if not isinstance(value, Mapping):
            raise ValueError("output fields must be a JSON object")
        return freeze_object(value)

    @model_validator(mode="after")
    def _check_size(self) -> OutputRef:
        _validate_output_fields_size(self.fields)
        return self

    @field_serializer("fields")
    def _serialize_fields(self, value: FrozenDict) -> object:
        return thaw_json(value)


def output_ref_routing_size(output: OutputRef) -> int:
    """Return the canonical JSON size of an output ID, fields and correlation ID."""

    return len(canonical_json_bytes(output.model_dump(mode="json")))


class PlannedEdgeInput(_FrozenModel):
    """The output IDs selected for one input edge, in task order.

    The plan retains the edge name used by bindings and worker inputs, together
    with the source index and role from the stored topology.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    edge: _EdgeName
    source_index: int = Field(ge=0, strict=True)
    role: InputRole
    output_ids: tuple[OutputId, ...] = ()

    @model_validator(mode="after")
    def _check_output_ids(self) -> PlannedEdgeInput:
        if len(self.output_ids) != len(set(self.output_ids)):
            raise ValueError(f"planned edge {self.edge!r} contains duplicate output IDs")
        return self


class TaskInputPlan(_FrozenModel):
    """A stored routing decision for one task.

    ``schema_version`` identifies the serialised shape, while ``generation``
    identifies the current set of plans after boundary replanning. A task retry
    reuses the selected output IDs, parameters, secret references and generation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    schema_version: Literal[1] = 1
    generation: int = Field(ge=0, strict=True)
    edges: tuple[PlannedEdgeInput, ...] = Field(
        default=(),
        max_length=MAX_INPUT_EDGES_PER_NODE,
    )
    parameters: FrozenDict = Field(default_factory=FrozenDict)
    secret_refs: Mapping[str, str] = Field(default_factory=dict)
    correlation_id: OutputId | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def _validate_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @field_validator("parameters", mode="before")
    @classmethod
    def _freeze_parameters(cls, value: object) -> FrozenDict:
        if not isinstance(value, Mapping):
            raise ValueError("plan parameters must be a JSON object")
        return freeze_object(value)

    @field_validator("secret_refs", mode="after")
    @classmethod
    def _freeze_secret_refs(cls, value: object) -> Mapping[str, str]:
        if not isinstance(value, Mapping):
            raise ValueError("secret_refs must be an object")
        refs = dict(value)
        if any(not key for key in refs):
            raise ValueError("secret reference targets must not be empty")
        if any(not item for item in refs.values()):
            raise ValueError("secret reference names must not be empty")
        return FrozenDict(refs)  # type: ignore[arg-type]

    @field_serializer("parameters")
    def _serialize_parameters(self, value: FrozenDict) -> object:
        return thaw_json(value)

    @field_serializer("secret_refs")
    def _serialize_secret_refs(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    @model_validator(mode="after")
    def _check_plan(self) -> TaskInputPlan:
        edge_names = [edge.edge for edge in self.edges]
        if len(edge_names) != len(set(edge_names)):
            raise ValueError("task plan edge names must be unique")

        shared_targets = set(self.parameters) & set(self.secret_refs)
        if shared_targets:
            raise ValueError(
                f"task plan parameter and secret targets overlap: {sorted(shared_targets)!r}"
            )

        reference_count = sum(len(edge.output_ids) for edge in self.edges)
        if reference_count > MAX_OUTPUT_REFS_PER_PLAN:
            raise ValueError(
                f"task plan references {reference_count} outputs; maximum is "
                f"{MAX_OUTPUT_REFS_PER_PLAN}"
            )

        serialised = {
            "schema_version": self.schema_version,
            "generation": self.generation,
            "edges": tuple(
                {
                    "edge": edge.edge,
                    "source_index": edge.source_index,
                    "role": edge.role.value,
                    "output_ids": edge.output_ids,
                }
                for edge in self.edges
            ),
            "parameters": self.parameters,
            "secret_refs": self.secret_refs,
            "correlation_id": self.correlation_id,
        }
        size = bounded_json_size(serialised, MAX_TASK_PLAN_BYTES)
        if size > MAX_TASK_PLAN_BYTES:
            raise ValueError(f"task plan exceeds the {MAX_TASK_PLAN_BYTES}-byte limit")
        return self


class EmittedOutput(_FrozenModel):
    """One output supplied when a worker completes a task.

    The store derives correlation from the stored node and task plan. Routing
    fields remain separate from materialised data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    id: OutputId
    fields: FrozenDict = Field(default_factory=FrozenDict)
    data: Mapping[str, object] = Field(default_factory=dict)

    @field_validator("fields", mode="before")
    @classmethod
    def _freeze_fields(cls, value: object) -> FrozenDict:
        if not isinstance(value, Mapping):
            raise ValueError("emitted output fields must be an object")
        return freeze_object(value)

    @field_validator("data", mode="after")
    @classmethod
    def _copy_data(cls, value: object) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise ValueError("emitted output data must be an object")
        return MappingProxyType(dict(value))

    @model_validator(mode="after")
    def _check_size(self) -> EmittedOutput:
        _validate_output_fields_size(self.fields)
        return self

    @field_serializer("fields")
    def _serialize_fields(self, value: FrozenDict) -> object:
        return thaw_json(value)

    @field_serializer("data")
    def _serialize_data(self, value: Mapping[str, object]) -> dict[str, object]:
        return dict(value)


class ResolvedOutput(_FrozenModel):
    """An output resolved for worker execution.

    Routing fields remain separate from data such as paths, URIs or records.
    Materialised data must remain valid for the duration of task execution.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    id: OutputId
    fields: FrozenDict = Field(default_factory=FrozenDict)
    data: Mapping[str, object] = Field(default_factory=dict)

    @field_validator("fields", mode="before")
    @classmethod
    def _freeze_fields(cls, value: object) -> FrozenDict:
        if not isinstance(value, Mapping):
            raise ValueError("resolved output fields must be an object")
        return freeze_object(value)

    @field_validator("data", mode="after")
    @classmethod
    def _copy_data(cls, value: object) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise ValueError("resolved output data must be an object")
        return MappingProxyType(dict(value))

    @model_validator(mode="after")
    def _check_size(self) -> ResolvedOutput:
        _validate_output_fields_size(self.fields)
        return self

    @field_serializer("fields")
    def _serialize_fields(self, value: FrozenDict) -> object:
        return thaw_json(value)

    @field_serializer("data")
    def _serialize_data(self, value: Mapping[str, object]) -> dict[str, object]:
        return dict(value)


class TaskInputs(_FrozenModel):
    """Namespaced worker inputs reconstructed from a task plan.

    Each edge retains its persisted output order. Parameters remain separate
    from routing fields and materialised data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    edges: Mapping[str, tuple[ResolvedOutput, ...]]
    parameters: FrozenDict = Field(default_factory=FrozenDict)

    @field_validator("edges", mode="after")
    @classmethod
    def _copy_edges(cls, value: object) -> Mapping[str, tuple[ResolvedOutput, ...]]:
        if not isinstance(value, Mapping):
            raise ValueError("task input edges must be an object")
        return MappingProxyType({str(key): tuple(items) for key, items in value.items()})

    @field_validator("parameters", mode="before")
    @classmethod
    def _freeze_parameters(cls, value: object) -> FrozenDict:
        if not isinstance(value, Mapping):
            raise ValueError("task parameters must be an object")
        return freeze_object(value)

    @field_serializer("edges")
    def _serialize_edges(
        self, value: Mapping[str, tuple[ResolvedOutput, ...]]
    ) -> dict[str, tuple[ResolvedOutput, ...]]:
        return dict(value)

    @field_serializer("parameters")
    def _serialize_parameters(self, value: FrozenDict) -> object:
        return thaw_json(value)


class PlanningSnapshot(_FrozenModel):
    """A consistent view of the inputs used to plan one node generation.

    The store uses ``token`` to reject a launch if its source launches or
    outputs have changed. The edge mappings correspond to the node's declared
    edges, and ``runtime_inputs`` contains the run input used by bindings and
    root plans.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    token: str = Field(min_length=1)
    generation: int = Field(ge=0, strict=True)
    outputs_by_edge: Mapping[str, tuple[OutputRef, ...]]
    source_dispositions: Mapping[str, NodeDisposition]
    runtime_inputs: FrozenDict = Field(default_factory=FrozenDict)

    @field_validator("outputs_by_edge", mode="after")
    @classmethod
    def _copy_outputs(cls, value: object) -> Mapping[str, tuple[OutputRef, ...]]:
        if not isinstance(value, Mapping):
            raise ValueError("outputs_by_edge must be an object")
        return MappingProxyType({str(key): tuple(items) for key, items in value.items()})

    @field_validator("source_dispositions", mode="after")
    @classmethod
    def _copy_dispositions(cls, value: object) -> Mapping[str, NodeDisposition]:
        if not isinstance(value, Mapping):
            raise ValueError("source_dispositions must be an object")
        return MappingProxyType({str(key): NodeDisposition(item) for key, item in value.items()})

    @field_validator("runtime_inputs", mode="before")
    @classmethod
    def _freeze_runtime_inputs(cls, value: object) -> FrozenDict:
        if not isinstance(value, Mapping):
            raise ValueError("runtime_inputs must be an object")
        return freeze_object(value)

    @field_serializer("outputs_by_edge", "source_dispositions")
    def _serialize_snapshot_mappings(self, value: Mapping[str, object]) -> dict[str, object]:
        return dict(value)

    @field_serializer("runtime_inputs")
    def _serialize_runtime_inputs(self, value: FrozenDict) -> object:
        return thaw_json(value)


class NodeLaunch(_FrozenModel):
    """The persisted state for one node generation.

    A ``LAUNCHED`` node owns task IDs. ``SKIPPED``, ``FILTERED`` and ``FAILED``
    launches contain no tasks and are complete when created.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    node_index: int = Field(ge=0, strict=True)
    plugin_name: str = Field(min_length=1)
    generation: int = Field(ge=0, strict=True)
    disposition: NodeDisposition
    task_ids: tuple[str, ...] = ()
    complete: bool = False
    error: str = ""

    @model_validator(mode="after")
    def _check_disposition(self) -> NodeLaunch:
        if any(not task_id for task_id in self.task_ids):
            raise ValueError("node launch task IDs must not be empty")
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("node launch task IDs must be unique")

        if self.disposition == NodeDisposition.LAUNCHED:
            if not self.task_ids:
                raise ValueError("a launched node must contain task IDs")
            return self

        if self.task_ids:
            raise ValueError(f"a {self.disposition.value} node cannot contain task IDs")
        if not self.complete:
            raise ValueError(f"a {self.disposition.value} node must be complete")
        if self.disposition == NodeDisposition.FAILED and not self.error:
            raise ValueError("a failed node must contain an error")
        return self


class LaunchCreateResult(_FrozenModel):
    """A store response to launch creation from a planning snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: LaunchCreateStatus
    launch: NodeLaunch | None = None

    @model_validator(mode="after")
    def _check_launch(self) -> LaunchCreateResult:
        if self.status == LaunchCreateStatus.STALE:
            if self.launch is not None:
                raise ValueError("a stale result cannot contain a launch")
        elif self.launch is None:
            raise ValueError(f"a {self.status.value} result must contain a launch")
        return self


class TaskContext(_FrozenModel):
    """Run, node and generation data used to reject stale task deliveries."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    node_index: int = Field(ge=0, strict=True)
    generation: int = Field(ge=0, strict=True)
