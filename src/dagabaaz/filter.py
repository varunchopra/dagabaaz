"""Artifact filter engine for pipeline edge routing.

Rules are AND-composed predicates: each artifact must pass ALL rules.
Selection is a post-filter reduction: pick one artifact from survivors.

Virtual fields (file_type, extension, file_size, file_name, mime_type)
are computed from artifact data. Dynamic fields come from upstream
plugin output_vars stored in artifact metadata.
"""

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from dagabaaz.constants import ArtifactSelector, FileType, FilterOperator
from dagabaaz.models import DagArtifact, DagArtifactLike, EdgeFilter, FilterRule

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = frozenset(
    {
        ".mp4",
        ".mkv",
        ".avi",
        ".mov",
        ".wmv",
        ".flv",
        ".webm",
        ".m4v",
        ".mpg",
        ".mpeg",
        ".ts",  # MPEG transport stream, not TypeScript
    }
)

SUBTITLE_EXTENSIONS = frozenset(
    {
        ".srt",
        ".sub",
        ".ass",
        ".ssa",
        ".vtt",
        ".idx",
    }
)

IMAGE_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".webp",
        ".svg",
        ".tiff",
    }
)

AUDIO_EXTENSIONS = frozenset(
    {
        ".mp3",
        ".flac",
        ".aac",
        ".ogg",
        ".wav",
        ".wma",
        ".m4a",
        ".opus",
    }
)

ARCHIVE_EXTENSIONS = frozenset(
    {
        ".zip",
        ".rar",
        ".7z",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
    }
)


def _file_extension(file_name: str) -> str:
    """Return lowercase extension including the dot, e.g. '.mp4'.

    Uses PurePosixPath for platform-consistent behavior — os.path.splitext
    follows the host OS conventions, which differ on Windows.
    """
    return PurePosixPath(file_name).suffix.lower()


def _size_key(a: DagArtifactLike) -> int:
    """Key function for size-based artifact selection.

    Pre-filtered to non-None file_size by caller (_select_by_size).
    """
    assert a.file_size is not None  # guaranteed by _select_by_size pre-filter
    return a.file_size


def _select_by_size[A: DagArtifactLike](
    artifacts: list[A], *, largest: bool
) -> A | None:
    """Return the artifact with the largest or smallest file_size, or None.

    Skips artifacts where file_size is None — they can't participate
    in size-based selection. Returns None if no artifact has a size.
    """
    sized = [a for a in artifacts if a.file_size is not None]
    if not sized:
        return None
    return max(sized, key=_size_key) if largest else min(sized, key=_size_key)


def classify_file_type(artifact: DagArtifactLike) -> FileType:
    """Compute the virtual file_type enum for an artifact.

    Extension-only types (subtitle, archive) are checked BEFORE MIME-based
    types to avoid misclassification when a file has a misleading MIME type.
    For example, a .srt file served with ``video/x-matroska`` MIME type
    would otherwise be classified as VIDEO instead of SUBTITLE.
    """
    mime = artifact.mime_type or ""
    ext = _file_extension(artifact.file_name)

    # Extension-only types first — subtitles and archives have no standard
    # MIME prefix, so they must be checked before MIME-based types.
    if ext in SUBTITLE_EXTENSIONS:
        return FileType.SUBTITLE
    if ext in ARCHIVE_EXTENSIONS:
        return FileType.ARCHIVE
    if mime.startswith("video/") or ext in VIDEO_EXTENSIONS:
        return FileType.VIDEO
    if mime.startswith("image/") or ext in IMAGE_EXTENSIONS:
        return FileType.IMAGE
    if mime.startswith("audio/") or ext in AUDIO_EXTENSIONS:
        return FileType.AUDIO
    return FileType.OTHER


# Maps virtual field names to accessor functions. Dynamic fields
# (plugin output_vars in metadata) fall through to artifact.metadata.
_VIRTUAL_FIELD_RESOLVERS: dict[str, Callable[[DagArtifactLike], Any]] = {
    "file_type": classify_file_type,
    "extension": lambda a: _file_extension(a.file_name),
    "file_size": lambda a: a.file_size,
    "file_name": lambda a: a.file_name,
    "mime_type": lambda a: a.mime_type or "",
}


def _resolve_field(artifact: DagArtifactLike, field: str) -> Any:
    """Resolve a field name to its value on an artifact.

    Virtual fields are resolved via ``_VIRTUAL_FIELD_RESOLVERS``.
    Everything else is looked up in the artifact's metadata dict
    (dynamic fields from upstream plugin output_vars).
    """
    resolver = _VIRTUAL_FIELD_RESOLVERS.get(field)
    if resolver is not None:
        return resolver(artifact)

    # Dynamic field — look in artifact metadata (plugin output_vars)
    return artifact.metadata.get(field)


_BOOL_LITERALS = {"true": True, "false": False}


def _coerce_bool_literal(value: Any) -> bool | None:
    """True/False or "true"/"false" (case-insensitive, whitespace-stripped);
    None otherwise so the caller falls through to the next path.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _BOOL_LITERALS.get(value.strip().lower())
    return None


def _coerce_numeric(value: Any) -> float | None:
    """Coerce to float for numeric comparisons; None if not meaningful.
    Bool is explicitly excluded – see _coerce_bool_literal.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _values_equal(actual: Any, expected: Any) -> bool:
    """Shared equality for EQ/NEQ/IN/NOT_IN."""
    a_bool = _coerce_bool_literal(actual)
    e_bool = _coerce_bool_literal(expected)
    if a_bool is not None and e_bool is not None:
        return a_bool == e_bool

    actual_num = _coerce_numeric(actual)
    expected_num = _coerce_numeric(expected)
    if actual_num is not None and expected_num is not None:
        return actual_num == expected_num

    return str(actual) == str(expected)


# Each function evaluates one FilterOperator against actual/expected values.
def _eval_eq(actual: Any, expected: Any) -> bool:
    return _values_equal(actual, expected)


def _eval_neq(actual: Any, expected: Any) -> bool:
    return not _values_equal(actual, expected)


def _eval_gt(actual: Any, expected: Any) -> bool:
    actual_num = _coerce_numeric(actual)
    expected_num = _coerce_numeric(expected)
    if actual_num is None or expected_num is None:
        return False
    return actual_num > expected_num


def _eval_gte(actual: Any, expected: Any) -> bool:
    actual_num = _coerce_numeric(actual)
    expected_num = _coerce_numeric(expected)
    if actual_num is None or expected_num is None:
        return False
    return actual_num >= expected_num


def _eval_lt(actual: Any, expected: Any) -> bool:
    actual_num = _coerce_numeric(actual)
    expected_num = _coerce_numeric(expected)
    if actual_num is None or expected_num is None:
        return False
    return actual_num < expected_num


def _eval_lte(actual: Any, expected: Any) -> bool:
    actual_num = _coerce_numeric(actual)
    expected_num = _coerce_numeric(expected)
    if actual_num is None or expected_num is None:
        return False
    return actual_num <= expected_num


def _eval_in(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_values_equal(actual, v) for v in expected)
    return _values_equal(actual, expected)


def _eval_not_in(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return not any(_values_equal(actual, v) for v in expected)
    return not _values_equal(actual, expected)


def _eval_contains(actual: Any, expected: Any) -> bool:
    return str(expected) in str(actual)


def _eval_not_contains(actual: Any, expected: Any) -> bool:
    return str(expected) not in str(actual)


def _eval_starts_with(actual: Any, expected: Any) -> bool:
    return str(actual).startswith(str(expected))


def _eval_ends_with(actual: Any, expected: Any) -> bool:
    return str(actual).endswith(str(expected))


def _eval_exists(actual: Any, _expected: Any) -> bool:
    # Falsy values like 0, False, [] count as "existing" — only None and
    # empty string are treated as absent. This is intentional: metadata
    # fields may legitimately hold zero or false.
    return actual is not None and str(actual) != ""


def _eval_not_exists(actual: Any, _expected: Any) -> bool:
    return actual is None or str(actual) == ""


# Maps each FilterOperator to its evaluation function. Adding a new
# operator requires only adding a function + one dict entry.
_OPERATOR_DISPATCH: dict[FilterOperator, Callable[[Any, Any], bool]] = {
    FilterOperator.EQ: _eval_eq,
    FilterOperator.NEQ: _eval_neq,
    FilterOperator.GT: _eval_gt,
    FilterOperator.GTE: _eval_gte,
    FilterOperator.LT: _eval_lt,
    FilterOperator.LTE: _eval_lte,
    FilterOperator.IN: _eval_in,
    FilterOperator.NOT_IN: _eval_not_in,
    FilterOperator.CONTAINS: _eval_contains,
    FilterOperator.NOT_CONTAINS: _eval_not_contains,
    FilterOperator.STARTS_WITH: _eval_starts_with,
    FilterOperator.ENDS_WITH: _eval_ends_with,
    FilterOperator.EXISTS: _eval_exists,
    FilterOperator.NOT_EXISTS: _eval_not_exists,
}


def _artifact_satisfies_rule(artifact: DagArtifactLike, rule: FilterRule) -> bool:
    actual = _resolve_field(artifact, rule.field)
    evaluator = _OPERATOR_DISPATCH.get(rule.operator)
    if evaluator is None:
        logger.warning(
            "Unknown filter operator %r on field %r", rule.operator, rule.field
        )
        return False
    return evaluator(actual, rule.value)


def filter_artifacts[A: DagArtifactLike](
    artifacts: list[A],
    edge_filter: EdgeFilter,
) -> list[A]:
    """Apply an EdgeFilter to a list of artifacts.

    Two-phase pipeline:
      1. Rules (predicates): AND-composed, each artifact must pass all rules.
         This narrows N artifacts to M (where M <= N).
      2. Select (reduction): picks one artifact from the survivors.
         "largest"/"smallest" compare by file_size.

    Returns the filtered list (may be empty — caller decides whether to
    skip the downstream node).
    """
    if not artifacts:
        return []

    if not edge_filter.rules and not edge_filter.select:
        # No rules and no selection — return a shallow copy to avoid
        # aliasing the caller's input list.
        return list(artifacts)

    result = [
        artifact
        for artifact in artifacts
        if all(_artifact_satisfies_rule(artifact, rule) for rule in edge_filter.rules)
    ]

    if edge_filter.select and result:
        if edge_filter.select == ArtifactSelector.LARGEST:
            picked = _select_by_size(result, largest=True)
        elif edge_filter.select == ArtifactSelector.SMALLEST:
            picked = _select_by_size(result, largest=False)
        else:
            raise ValueError(f"Unhandled artifact selector: {edge_filter.select}")
        result = [picked] if picked else []

    return result


@dataclass(frozen=True, slots=True)
class OriginGroups:
    groups: dict[str, list[DagArtifact]]
    broadcast: list[DagArtifact]


def group_by_origin(artifacts: list[DagArtifact]) -> OriginGroups:
    groups: dict[str, list[DagArtifact]] = defaultdict(list)
    broadcast: list[DagArtifact] = []

    for art in artifacts:
        if art.origin_artifact_id is None:
            broadcast.append(art)
        else:
            groups[art.origin_artifact_id].append(art)

    return OriginGroups(groups=dict(groups), broadcast=broadcast)


def filter_artifacts_by_origin[A: DagArtifactLike](
    artifacts_by_node: dict[int, list[A]],
    origin_artifact_id: str,
) -> dict[int, list[A]]:
    """Filter each node's artifacts to only those sharing a given origin.

    For grouped tasks: each task must see only its own correlated data.
    Artifacts with ``origin_artifact_id=None`` (broadcast) pass through —
    they're shared context that every group needs.

    Consumer-side counterpart of ``group_by_origin()``.
    """
    return {
        node_idx: [
            a
            for a in artifacts
            if a.origin_artifact_id is None
            or a.origin_artifact_id == origin_artifact_id
        ]
        for node_idx, artifacts in artifacts_by_node.items()
    }
