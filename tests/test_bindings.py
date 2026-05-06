"""Binding resolution preserves native metadata types – required by structured
pipes (``json_get``, ``first``, ``flatten``, ...) which ``isinstance``-check their input.
"""

from dagabaaz.bindings import resolve_binding
from dagabaaz.models import ExpressionSource, NodeSource
from tests.helpers import make_dag_artifact


def _resolve(binding, *artifacts):
    return resolve_binding(
        binding=binding,
        artifacts_by_node={0: list(artifacts)},
        slug_to_node_index={"upstream": 0},
        run_input={},
    )


def test_json_get_on_dict_metadata() -> None:
    art = make_dag_artifact("x", metadata={"shareable_urls": {"embed_urls": "https://x"}})
    expr = ExpressionSource(expression="{upstream.shareable_urls | json_get(embed_urls)}")
    assert _resolve(expr, art) == "https://x"


def test_first_pipe_on_list_metadata() -> None:
    art = make_dag_artifact("x", metadata={"tags": ["a", "b"]})
    expr = ExpressionSource(expression="{upstream.tags | first}")
    assert _resolve(expr, art) == "a"


def test_standard_field_returns_native_type() -> None:
    art = make_dag_artifact("x", size=2048)
    assert _resolve(NodeSource(node="upstream", key="file_size"), art) == 2048


def test_multiple_artifacts_resolve_to_list() -> None:
    a1 = make_dag_artifact("a", metadata={"x": {"k": 1}})
    a2 = make_dag_artifact("b", metadata={"x": {"k": 2}})
    assert _resolve(NodeSource(node="upstream", key="x"), a1, a2) == [{"k": 1}, {"k": 2}]
