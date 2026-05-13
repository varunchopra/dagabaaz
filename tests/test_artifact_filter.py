import pytest
from pydantic import ValidationError

from dagabaaz.filter import filter_artifacts, group_by_origin
from dagabaaz.models import EdgeFilter, FilterRule
from tests.helpers import make_dag_artifact as _art

SAMPLE_ARTIFACTS = [
    _art("project/main.mp4", 129_000_000, "video/mp4"),
    _art("project/extra.mp4", 15_000_000, "video/mp4"),
    _art("project/cover.jpg", 45_000, "image/jpeg"),
    _art("project/main.en.srt", 1_500, "text/plain"),
    _art("project/main.nl.srt", 1_500, "text/plain"),
    _art("project/info.nfo", 200, None),
]


def test_empty_filter_passes_all() -> None:
    ef = EdgeFilter()
    result = filter_artifacts(SAMPLE_ARTIFACTS, ef)
    assert len(result) == 6


def test_empty_artifacts_returns_empty() -> None:
    ef = EdgeFilter(rules=[FilterRule(field="file_type", operator="eq", value="video")])
    result = filter_artifacts([], ef)
    assert result == []


def test_rule_file_type_eq_video() -> None:
    ef = EdgeFilter(rules=[FilterRule(field="file_type", operator="eq", value="video")])
    result = filter_artifacts(SAMPLE_ARTIFACTS, ef)
    assert len(result) == 2
    names = {a.file_name for a in result}
    assert names == {"project/main.mp4", "project/extra.mp4"}


def test_rule_file_type_in_video_and_subtitle() -> None:
    ef = EdgeFilter(
        rules=[
            FilterRule(field="file_type", operator="in", value=["video", "subtitle"])
        ]
    )
    result = filter_artifacts(SAMPLE_ARTIFACTS, ef)
    assert len(result) == 4
    names = {a.file_name for a in result}
    assert "project/main.mp4" in names
    assert "project/main.en.srt" in names
    assert "project/cover.jpg" not in names


def test_rule_file_type_neq() -> None:
    ef = EdgeFilter(
        rules=[FilterRule(field="file_type", operator="neq", value="video")]
    )
    result = filter_artifacts(SAMPLE_ARTIFACTS, ef)
    # 4 non-video: poster, 2 subs, nfo
    assert len(result) == 4


def test_rule_file_type_not_in() -> None:
    ef = EdgeFilter(
        rules=[
            FilterRule(field="file_type", operator="not_in", value=["video", "image"])
        ]
    )
    result = filter_artifacts(SAMPLE_ARTIFACTS, ef)
    # 2 subs + 1 nfo
    assert len(result) == 3


def test_rule_extension_not_in() -> None:
    ef = EdgeFilter(
        rules=[FilterRule(field="extension", operator="not_in", value=[".jpg", ".nfo"])]
    )
    result = filter_artifacts(SAMPLE_ARTIFACTS, ef)
    assert len(result) == 4
    names = {a.file_name for a in result}
    assert "project/cover.jpg" not in names
    assert "project/info.nfo" not in names


def test_rule_extension_eq() -> None:
    ef = EdgeFilter(rules=[FilterRule(field="extension", operator="eq", value=".mp4")])
    result = filter_artifacts(SAMPLE_ARTIFACTS, ef)
    assert len(result) == 2


def test_rule_file_size_gt() -> None:
    ef = EdgeFilter(
        rules=[FilterRule(field="file_size", operator="gt", value=50_000_000)]
    )
    result = filter_artifacts(SAMPLE_ARTIFACTS, ef)
    assert len(result) == 1
    assert result[0].file_name == "project/main.mp4"


def test_rule_file_size_gte() -> None:
    ef = EdgeFilter(
        rules=[FilterRule(field="file_size", operator="gte", value=15_000_000)]
    )
    result = filter_artifacts(SAMPLE_ARTIFACTS, ef)
    assert len(result) == 2


def test_rule_file_size_lt() -> None:
    ef = EdgeFilter(rules=[FilterRule(field="file_size", operator="lt", value=2000)])
    result = filter_artifacts(SAMPLE_ARTIFACTS, ef)
    # 2 subs (1500) + 1 nfo (200)
    assert len(result) == 3


def test_rule_file_size_lte() -> None:
    ef = EdgeFilter(rules=[FilterRule(field="file_size", operator="lte", value=1500)])
    result = filter_artifacts(SAMPLE_ARTIFACTS, ef)
    assert len(result) == 3


def test_rule_file_name_contains() -> None:
    ef = EdgeFilter(
        rules=[FilterRule(field="file_name", operator="contains", value="extra")]
    )
    result = filter_artifacts(SAMPLE_ARTIFACTS, ef)
    assert len(result) == 1
    assert result[0].file_name == "project/extra.mp4"


def test_rule_file_name_not_contains() -> None:
    ef = EdgeFilter(
        rules=[FilterRule(field="file_name", operator="not_contains", value="extra")]
    )
    result = filter_artifacts(SAMPLE_ARTIFACTS, ef)
    assert len(result) == 5


def test_rule_starts_with() -> None:
    arts = [
        _art("sample.mp4", 100),
        _art("sample_extra.mp4", 100),
        _art("BigBuck.mp4", 100),
    ]
    ef = EdgeFilter(
        rules=[FilterRule(field="file_name", operator="starts_with", value="sample")]
    )
    result = filter_artifacts(arts, ef)
    assert len(result) == 2
    assert {a.file_name for a in result} == {"sample.mp4", "sample_extra.mp4"}


def test_rule_ends_with() -> None:
    arts = [
        _art("main.mp4", 100),
        _art("extra.mp4", 100),
        _art("poster.jpg", 100),
    ]
    ef = EdgeFilter(
        rules=[FilterRule(field="file_name", operator="ends_with", value=".mp4")]
    )
    result = filter_artifacts(arts, ef)
    assert len(result) == 2
    assert {a.file_name for a in result} == {"main.mp4", "extra.mp4"}


def test_rule_exists_metadata_present() -> None:
    """exists passes when a metadata field has a non-empty value."""
    arts = [
        _art("a.mp4", 100, None, {"external_id": "id_123"}),
        _art("b.mp4", 100, None, {}),
        _art("c.mp4", 100, None, {"external_id": ""}),
    ]
    ef = EdgeFilter(
        rules=[FilterRule(field="external_id", operator="exists", value="")]
    )
    result = filter_artifacts(arts, ef)
    assert len(result) == 1
    assert result[0].file_name == "a.mp4"


def test_rule_not_exists_metadata_missing() -> None:
    """not_exists passes when a metadata field is missing or empty."""
    arts = [
        _art("a.mp4", 100, None, {"external_id": "id_123"}),
        _art("b.mp4", 100, None, {}),
        _art("c.mp4", 100, None, {"external_id": ""}),
    ]
    ef = EdgeFilter(
        rules=[FilterRule(field="external_id", operator="not_exists", value="")]
    )
    result = filter_artifacts(arts, ef)
    assert len(result) == 2
    assert {a.file_name for a in result} == {"b.mp4", "c.mp4"}


def test_rule_exists_on_virtual_field() -> None:
    """exists on mime_type — passes when non-empty."""
    arts = [
        _art("main.mp4", 100, "video/mp4"),
        _art("unknown.xyz", 100, None),
    ]
    ef = EdgeFilter(rules=[FilterRule(field="mime_type", operator="exists", value="")])
    result = filter_artifacts(arts, ef)
    assert len(result) == 1
    assert result[0].file_name == "main.mp4"


def test_rule_exists_numeric_zero_is_present() -> None:
    """A numeric 0 is a present value — exists should pass."""
    arts = [
        _art("a.mp4", 100, None, {"index": 0}),
        _art("b.mp4", 100, None, {}),
    ]
    ef = EdgeFilter(rules=[FilterRule(field="index", operator="exists", value="")])
    result = filter_artifacts(arts, ef)
    assert len(result) == 1
    assert result[0].file_name == "a.mp4"


def test_rule_exists_boolean_false_is_present() -> None:
    """Boolean False is a present value — exists should pass."""
    arts = [
        _art("a.mp4", 100, None, {"enabled": False}),
        _art("b.mp4", 100, None, {}),
    ]
    ef = EdgeFilter(rules=[FilterRule(field="enabled", operator="exists", value="")])
    result = filter_artifacts(arts, ef)
    assert len(result) == 1
    assert result[0].file_name == "a.mp4"


def test_rule_exists_empty_list_is_present() -> None:
    """An empty list is a present value — exists should pass."""
    arts = [
        _art("a.mp4", 100, None, {"tags": []}),
        _art("b.mp4", 100, None, {}),
    ]
    ef = EdgeFilter(rules=[FilterRule(field="tags", operator="exists", value="")])
    result = filter_artifacts(arts, ef)
    assert len(result) == 1
    assert result[0].file_name == "a.mp4"


def test_rule_metadata_eq() -> None:
    arts = [
        _art("main.mp4", 100, "video/mp4", {"category": "type_a"}),
        _art("show.mp4", 100, "video/mp4", {"category": "tv"}),
    ]
    ef = EdgeFilter(rules=[FilterRule(field="category", operator="eq", value="tv")])
    result = filter_artifacts(arts, ef)
    assert len(result) == 1
    assert result[0].file_name == "show.mp4"


def test_rule_metadata_neq() -> None:
    arts = [
        _art("main.mp4", 100, "video/mp4", {"category": "type_a"}),
        _art("show.mp4", 100, "video/mp4", {"category": "tv"}),
    ]
    ef = EdgeFilter(rules=[FilterRule(field="category", operator="neq", value="tv")])
    result = filter_artifacts(arts, ef)
    assert len(result) == 1
    assert result[0].file_name == "main.mp4"


def test_rule_metadata_numeric_gt() -> None:
    """Numeric comparison on metadata values (stored as mixed types)."""
    arts = [
        _art("a.mp4", 100, None, {"confidence": 0.8}),
        _art("b.mp4", 100, None, {"confidence": "0.3"}),  # string from JSON
        _art("c.mp4", 100, None, {"confidence": 0.95}),
    ]
    ef = EdgeFilter(rules=[FilterRule(field="confidence", operator="gt", value=0.5)])
    result = filter_artifacts(arts, ef)
    assert len(result) == 2
    names = {a.file_name for a in result}
    assert names == {"a.mp4", "c.mp4"}


def test_rule_metadata_missing_field_fails() -> None:
    """If a metadata field doesn't exist on an artifact, the rule should fail."""
    arts = [
        _art("a.mp4", 100, None, {"category": "type_a"}),
        _art("b.mp4", 100, None, {}),  # no category
    ]
    ef = EdgeFilter(rules=[FilterRule(field="category", operator="eq", value="type_a")])
    result = filter_artifacts(arts, ef)
    assert len(result) == 1
    assert result[0].file_name == "a.mp4"


def test_multiple_rules_are_and_composed() -> None:
    """Artifact must pass ALL rules (AND logic)."""
    ef = EdgeFilter(
        rules=[
            FilterRule(field="file_type", operator="eq", value="video"),
            FilterRule(field="file_size", operator="gt", value=50_000_000),
        ]
    )
    result = filter_artifacts(SAMPLE_ARTIFACTS, ef)
    assert len(result) == 1
    assert result[0].file_name == "project/main.mp4"


def test_select_largest() -> None:
    ef = EdgeFilter(select="largest")
    result = filter_artifacts(SAMPLE_ARTIFACTS, ef)
    assert len(result) == 1
    assert result[0].file_name == "project/main.mp4"


def test_select_smallest() -> None:
    ef = EdgeFilter(select="smallest")
    result = filter_artifacts(SAMPLE_ARTIFACTS, ef)
    assert len(result) == 1
    assert result[0].file_name == "project/info.nfo"


def test_rules_plus_select() -> None:
    """Rules narrow first, then selector picks from survivors."""
    ef = EdgeFilter(
        rules=[FilterRule(field="file_type", operator="eq", value="video")],
        select="smallest",
    )
    result = filter_artifacts(SAMPLE_ARTIFACTS, ef)
    assert len(result) == 1
    assert result[0].file_name == "project/extra.mp4"


def test_video_detection_by_extension_no_mime() -> None:
    """Video detection falls back to extension when mime_type is None."""
    arts = [_art("main.mkv", 500_000, None)]
    ef = EdgeFilter(rules=[FilterRule(field="file_type", operator="eq", value="video")])
    result = filter_artifacts(arts, ef)
    assert len(result) == 1


def test_select_on_single_item() -> None:
    """Selector with a single artifact just returns it."""
    arts = [_art("readme.txt", 100, "text/plain")]
    ef = EdgeFilter(select="largest")
    result = filter_artifacts(arts, ef)
    assert len(result) == 1
    assert result[0].file_name == "readme.txt"


@pytest.mark.parametrize(
    "select,expected_name",
    [("smallest", "small.mp4"), ("largest", "big.mp4")],
)
def test_select_skips_none_file_size(select: str, expected_name: str) -> None:
    """Artifacts with None file_size are excluded from selection."""
    arts = [
        _art("no_size.mp4", size=None, mime="video/mp4"),
        _art("big.mp4", 5000, "video/mp4"),
        _art("small.mp4", 100, "video/mp4"),
    ]
    ef = EdgeFilter(select=select)
    result = filter_artifacts(arts, ef)
    assert len(result) == 1
    assert result[0].file_name == expected_name


def test_select_all_none_file_size_returns_empty() -> None:
    """When all artifacts have None file_size, selector returns empty."""
    arts = [_art("a.mp4", size=None), _art("b.mp4", size=None)]
    ef = EdgeFilter(select="smallest")
    result = filter_artifacts(arts, ef)
    assert result == []


def test_unknown_operator_rejects_artifact() -> None:
    """Unknown operators are rejected at model validation time."""
    with pytest.raises(ValidationError):
        FilterRule(field="file_name", operator="fuzzy_match", value="a")


@pytest.mark.parametrize(
    "operator,expected_count",
    [("eq", 1), ("neq", 0)],
)
def test_numeric_coercion(operator: str, expected_count: int) -> None:
    """Numeric comparison between int (1000) and float (1000.0)."""
    arts = [_art("a.mp4", 1000)]
    ef = EdgeFilter(
        rules=[FilterRule(field="file_size", operator=operator, value=1000.0)]
    )
    result = filter_artifacts(arts, ef)
    assert len(result) == expected_count


def test_eq_string_fallback() -> None:
    """eq falls back to string comparison for non-numeric values."""
    arts = [
        _art("a.mp4", 100, None, {"tag": "release"}),
        _art("b.mp4", 100, None, {"tag": "draft"}),
    ]
    ef = EdgeFilter(rules=[FilterRule(field="tag", operator="eq", value="release")])
    result = filter_artifacts(arts, ef)
    assert len(result) == 1
    assert result[0].file_name == "a.mp4"


@pytest.mark.parametrize("ext", ["srt", "sub", "ass", "ssa", "vtt", "idx"])
def test_subtitle_extension_recognized(ext: str) -> None:
    """Subtitle extension is correctly classified as subtitle file_type."""
    arts = [_art(f"subs.{ext}", 1000, None)]
    ef = EdgeFilter(
        rules=[
            FilterRule(field="file_type", operator="in", value=["video", "subtitle"])
        ]
    )
    result = filter_artifacts(arts, ef)
    assert len(result) == 1


@pytest.mark.parametrize(
    "metadata_value,rule_value,expected_match",
    [
        # Direct bool ↔ lowercase-str match
        (False, "false", True),
        (True, "true", True),
        (False, "true", False),
        (True, "false", False),
        # Case-insensitive + whitespace tolerance
        (False, "False", True),
        (True, "TRUE", True),
        (False, " false ", True),
        # Intentional change: no bool ↔ "0"/"1" match
        (False, "0", False),
        (True, "1", False),
    ],
)
def test_eq_bool_metadata(
    metadata_value: bool, rule_value: str, expected_match: bool
) -> None:
    arts = [_art("a.mp4", 100, None, {"was_created": metadata_value})]
    ef = EdgeFilter(
        rules=[FilterRule(field="was_created", operator="eq", value=rule_value)]
    )
    assert (len(filter_artifacts(arts, ef)) == 1) == expected_match


@pytest.mark.parametrize(
    "metadata_value,rule_value,expected_match",
    [
        (False, "false", False),
        (True, "false", True),
        (False, "true", True),
        (False, "False", False),
    ],
)
def test_neq_bool_metadata(
    metadata_value: bool, rule_value: str, expected_match: bool
) -> None:
    arts = [_art("a.mp4", 100, None, {"was_created": metadata_value})]
    ef = EdgeFilter(
        rules=[FilterRule(field="was_created", operator="neq", value=rule_value)]
    )
    assert (len(filter_artifacts(arts, ef)) == 1) == expected_match


def test_in_bool_metadata() -> None:
    arts = [
        _art("a.mp4", 100, None, {"flag": False}),
        _art("b.mp4", 100, None, {"flag": True}),
    ]
    ef = EdgeFilter(rules=[FilterRule(field="flag", operator="in", value=["false"])])
    result = filter_artifacts(arts, ef)
    assert len(result) == 1
    assert result[0].file_name == "a.mp4"


def test_not_in_bool_metadata_case_insensitive() -> None:
    arts = [_art("a.mp4", 100, None, {"flag": True})]
    ef = EdgeFilter(
        rules=[FilterRule(field="flag", operator="not_in", value=["FALSE"])]
    )
    assert len(filter_artifacts(arts, ef)) == 1


def test_gt_numeric_path_unchanged() -> None:
    """GT on non-bool numeric values."""
    arts = [_art("a.mp4", 1000)]
    ef = EdgeFilter(rules=[FilterRule(field="file_size", operator="gt", value=999)])
    assert len(filter_artifacts(arts, ef)) == 1


def test_gt_on_bool_returns_false() -> None:
    """GT/GTE/LT/LTE on bool fields return False."""
    arts = [_art("a.mp4", 100, None, {"flag": True})]
    ef = EdgeFilter(rules=[FilterRule(field="flag", operator="gt", value=0)])
    assert filter_artifacts(arts, ef) == []


class TestGroupByOrigin:
    """Direct tests for group_by_origin broadcast and grouping semantics."""

    def test_groups_by_origin_id(self) -> None:
        arts = [
            _art("a.dat", origin_artifact_id="o1"),
            _art("b.dat", origin_artifact_id="o1"),
            _art("c.dat", origin_artifact_id="o2"),
        ]
        result = group_by_origin(arts)
        assert len(result.groups) == 2
        assert len(result.groups["o1"]) == 2
        assert len(result.groups["o2"]) == 1
        assert result.broadcast == []

    def test_broadcast_appended_to_every_group(self) -> None:
        arts = [
            _art("a.dat", origin_artifact_id="o1"),
            _art("b.dat", origin_artifact_id="o2"),
            _art("shared.dat", origin_artifact_id=None),
        ]
        result = group_by_origin(arts)
        assert len(result.groups) == 2
        for group_arts in result.groups.values():
            names = {a.file_name for a in group_arts}
            assert "shared.dat" in names

    def test_all_broadcast_returns_empty_groups(self) -> None:
        arts = [
            _art("x.dat", origin_artifact_id=None),
            _art("y.dat", origin_artifact_id=None),
        ]
        result = group_by_origin(arts)
        assert result.groups == {}
        assert len(result.broadcast) == 2

    def test_empty_input(self) -> None:
        result = group_by_origin([])
        assert result.groups == {}
        assert result.broadcast == []
