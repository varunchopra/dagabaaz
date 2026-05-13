import pytest

from dagabaaz.models import DagNode, NodeSource
from dagabaaz.schema import (
    InputFieldSpec,
    merge_run_input,
    validate_binding_references,
)


class TestMergeRunInput:
    def test_binding_defaults_lowest_precedence(self) -> None:
        fields = [InputFieldSpec(name="url", label="URL", default="http://binding")]
        result = merge_run_input(fields, {"url": "http://pipeline"}, {})
        assert result["url"] == "http://pipeline"

    def test_pipeline_defaults_mid_precedence(self) -> None:
        fields = [InputFieldSpec(name="url", label="URL", default="http://binding")]
        result = merge_run_input(fields, {}, {})
        assert result["url"] == "http://binding"

    def test_run_input_highest_precedence(self) -> None:
        fields = [InputFieldSpec(name="url", label="URL", default="http://binding")]
        result = merge_run_input(
            fields, {"url": "http://pipeline"}, {"url": "http://user"}
        )
        assert result["url"] == "http://user"

    def test_empty_string_run_input_does_not_override(self) -> None:
        fields = [InputFieldSpec(name="url", label="URL")]
        result = merge_run_input(fields, {"url": "http://pipeline"}, {"url": ""})
        assert result["url"] == "http://pipeline"

    def test_none_run_input_does_not_override(self) -> None:
        fields = [InputFieldSpec(name="url", label="URL")]
        result = merge_run_input(fields, {"url": "http://pipeline"}, {"url": None})
        assert result["url"] == "http://pipeline"

    def test_required_missing_raises(self) -> None:
        fields = [InputFieldSpec(name="url", label="URL", required=True)]
        with pytest.raises(ValueError, match="Required input missing"):
            merge_run_input(fields, {}, {})

    def test_required_empty_string_raises(self) -> None:
        fields = [InputFieldSpec(name="url", label="URL", required=True)]
        with pytest.raises(ValueError, match="Required input missing"):
            merge_run_input(fields, {}, {"url": ""})

    def test_required_none_raises(self) -> None:
        fields = [InputFieldSpec(name="url", label="URL", required=True)]
        with pytest.raises(ValueError, match="Required input missing"):
            merge_run_input(fields, {}, {"url": None})

    def test_required_satisfied_by_run_input(self) -> None:
        fields = [InputFieldSpec(name="url", label="URL", required=True)]
        result = merge_run_input(fields, {}, {"url": "http://example.com"})
        assert result["url"] == "http://example.com"

    def test_required_satisfied_by_default(self) -> None:
        fields = [
            InputFieldSpec(
                name="url", label="URL", required=True, default="http://default"
            )
        ]
        result = merge_run_input(fields, {}, {})
        assert result["url"] == "http://default"

    def test_multiple_fields(self) -> None:
        fields = [
            InputFieldSpec(name="url", label="URL", default="http://default"),
            InputFieldSpec(name="quality", label="Quality"),
        ]
        result = merge_run_input(fields, {"quality": "720p"}, {"url": "http://custom"})
        assert result["url"] == "http://custom"
        assert result["quality"] == "720p"

    def test_extra_run_input_preserved(self) -> None:
        fields: list[InputFieldSpec] = []
        result = merge_run_input(fields, {}, {"extra": "value"})
        assert result["extra"] == "value"

    def test_zero_string_default_preserved(self) -> None:
        fields = [InputFieldSpec(name="count", label="Count", default="0")]
        result = merge_run_input(fields, {}, {})
        assert result["count"] == "0"

    def test_required_zero_int_passes(self) -> None:
        fields = [InputFieldSpec(name="count", label="Count", required=True)]
        result = merge_run_input(fields, {}, {"count": 0})
        assert result["count"] == 0

    def test_required_zero_string_passes(self) -> None:
        fields = [InputFieldSpec(name="count", label="Count", required=True)]
        result = merge_run_input(fields, {}, {"count": "0"})
        assert result["count"] == "0"


class TestValidateBindingWhenClause:
    def _nodes_with_when(self, when_expr: str) -> tuple[list[DagNode], dict[str, int]]:
        nodes = [
            DagNode(plugin="src", slug="a"),
            DagNode(
                plugin="proc",
                slug="b",
                depends_on=["a"],
                bindings={
                    "field": NodeSource(node="a", key="x", when=when_expr),
                },
            ),
        ]
        return nodes, {"a": 0, "b": 1}

    def test_bad_syntax_rejected(self) -> None:
        nodes, idx = self._nodes_with_when("{unclosed")
        err = validate_binding_references(nodes, idx)
        assert err is not None
        assert "when-clause error" in err

    def test_unknown_slug_rejected(self) -> None:
        nodes, idx = self._nodes_with_when("{ghost.flag}")
        err = validate_binding_references(nodes, idx)
        assert err is not None
        assert "not in depends_on" in err

    def test_valid_when_accepted(self) -> None:
        nodes, idx = self._nodes_with_when("{a.flag | not}")
        assert validate_binding_references(nodes, idx) is None
