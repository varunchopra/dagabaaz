"""Binding resolution logic for DAG data flow.

Resolution functions accept artifact objects via duck typing
(getattr + metadata dict), so both ``DagArtifact`` and application-level
artifact rows work without adapters.
"""

import logging
from typing import Any

from dagabaaz.constants import ARTIFACT_STANDARD_FIELDS
from dagabaaz.expressions import Lookup, extract_refs, resolve_expression
from dagabaaz.models import (
    ConfigSource,
    DagArtifactLike,
    ExpressionError,
    ExpressionSource,
    InputBinding,
    NodeSource,
    RuntimeSource,
)

logger = logging.getLogger(__name__)


def extract_artifact_field(artifacts: list[DagArtifactLike], key: str) -> list[object]:
    """Extract a field from artifact attributes or metadata, preserving native types.

    The pipe layer handles type-specific behavior (string pipes coerce
    on entry via ``str()``; structured pipes ``isinstance``-check), so
    pre-coercing here would silently break ``json_get`` and friends.
    """
    values: list[object] = []
    for artifact in artifacts:
        if key in ARTIFACT_STANDARD_FIELDS:
            field_value = getattr(artifact, key)
            if field_value is not None:
                values.append(field_value)
        elif key in artifact.metadata:
            values.append(artifact.metadata[key])
        else:
            # Surface binding typos (e.g. "file_paht") — silent drops
            # produce empty results with zero diagnostic signal.
            logger.debug(
                "Artifact '%s' has no field '%s' (standard or metadata)",
                getattr(artifact, "file_name", "?"),
                key,
            )
    return values


def _unwrap_field(
    artifacts: list[DagArtifactLike], key: str
) -> object | list[object] | None:
    """Extract a field from artifacts, returning scalar for single values.

    Shared by resolve_binding (NodeSource) and build_expression_lookup
    to avoid duplicating the extract → unwrap logic.
    """
    resolved = extract_artifact_field(artifacts, key)
    if len(resolved) == 1:
        return resolved[0]
    return resolved if resolved else None


def resolve_binding(
    binding: NodeSource | ConfigSource | RuntimeSource | ExpressionSource,
    artifacts_by_node: dict[int, list[DagArtifactLike]],
    slug_to_node_index: dict[str, int],
    run_input: dict[str, object],
    node_config: dict[str, Any] | None = None,
) -> object | None:
    """Resolve a single binding to its value."""
    match binding:
        case NodeSource(node=node, key=key):
            idx = slug_to_node_index.get(node)
            if idx is None:
                return None
            return _unwrap_field(artifacts_by_node.get(idx, []), key)

        case RuntimeSource(key=key, default=default):
            if key in run_input:
                return run_input[key]
            return default or None

        case ConfigSource(value=value):
            return value if value else None

        case ExpressionSource(expression=expression):
            return resolve_expression(
                expression,
                build_expression_lookup(
                    artifacts_by_node,
                    slug_to_node_index,
                    run_input,
                    node_config or {},
                ),
            )

        case _:
            return None


def build_expression_lookup(
    artifacts_by_node: dict[int, list[DagArtifactLike]],
    slug_to_node_index: dict[str, int],
    run_input: dict[str, object],
    node_config: dict[str, Any],
) -> Lookup:
    """Build a lookup function for expression resolution.

    Maps (namespace, key) to values from three sources:
      - node slug -> artifact field via extract_artifact_field
      - "input"  -> run_input dict
      - "config" -> node config dict

    **Caller must not mutate inputs after creating the lookup** — the
    closure captures references, not snapshots.
    """

    def lookup(namespace: str, key: str) -> object | None:
        if namespace == "input":
            return run_input.get(key)
        if namespace == "config":
            return node_config.get(key)
        idx = slug_to_node_index.get(namespace)
        if idx is None:
            return None
        arts = artifacts_by_node.get(idx, [])
        return _unwrap_field(arts, key)

    return lookup


def extract_node_indices_from_bindings(
    bindings: dict[str, InputBinding],
    slug_to_node_index: dict[str, int],
) -> set[int]:
    """Collect all node indices needed by NodeSource/ExpressionSource bindings."""
    indices: set[int] = set()
    for binding in bindings.values():
        match binding:
            case NodeSource(node=node):
                idx = slug_to_node_index.get(node)
                if idx is not None:
                    indices.add(idx)
            case ExpressionSource(expression=expression):
                slugs, _ = extract_refs(expression)
                for slug in slugs:
                    idx = slug_to_node_index.get(slug)
                    if idx is not None:
                        indices.add(idx)
    return indices


def any_binding_requires_run_input(
    bindings: dict[str, InputBinding],
) -> bool:
    """Check if any binding needs run input (RuntimeSource or expression with input.*)."""
    for binding in bindings.values():
        match binding:
            case RuntimeSource():
                return True
            case ExpressionSource(expression=expression):
                try:
                    _, runtime_keys = extract_refs(expression)
                except ExpressionError:
                    logger.warning(
                        "Bad expression in binding, skipping: %s", expression
                    )
                    continue
                if runtime_keys:
                    return True
    return False
