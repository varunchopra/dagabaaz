import pytest

from dagabaaz.constants import CorrelationMode, InputMode, InputRole
from dagabaaz.graph import (
    assign_slugs,
    downstream_closure,
    find_ready_nodes,
    validate_graph,
)
from dagabaaz.models import (
    DagNode,
    EdgeSource,
    ExpressionSource,
    InputEdge,
    LiteralSource,
)
from dagabaaz.topology import RunTopology


def test_removed_depends_on_and_fan_mode_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="depends_on"):
        DagNode.model_validate(
            {"plugin": "consumer", "depends_on": ["source"], "fan_mode": "single"}
        )


def test_assign_slugs_is_stable_and_unique() -> None:
    nodes = [
        DagNode(slug="fetch_1", plugin="fetch"),
        DagNode(plugin="fetch"),
        DagNode(plugin="fetch"),
    ]
    assign_slugs(nodes)
    assert [node.slug for node in nodes] == ["fetch_1", "fetch_2", "fetch_3"]


def test_dependencies_come_from_direct_edges_and_support_forward_order() -> None:
    nodes = [
        DagNode(
            slug="consumer",
            plugin="p",
            edges=(InputEdge(name="source", source="producer"),),
            input_mode=InputMode.EACH,
        ),
        DagNode(slug="producer", plugin="p"),
    ]
    assert validate_graph(nodes) == [[1], []]


def test_edges_from_the_same_source_share_one_barrier_dependency() -> None:
    nodes = [
        DagNode(slug="source", plugin="p"),
        DagNode(
            slug="consumer",
            plugin="p",
            edges=(
                InputEdge(name="records", source="source"),
                InputEdge(name="settings", source="source", role=InputRole.SIDE),
            ),
        ),
    ]

    assert validate_graph(nodes) == [[], [0]]


def test_run_topology_copies_nodes_and_protects_its_index() -> None:
    nodes = [
        DagNode(
            slug="source",
            plugin="p",
            bindings={"value": LiteralSource(value={"enabled": True})},
        )
    ]
    topology = RunTopology.build(nodes)
    nodes[0].slug = "changed"
    nodes[0].bindings["other"] = LiteralSource(value=False)

    assert topology.nodes[0].slug == "source"
    assert "other" not in topology.nodes[0].bindings
    with pytest.raises(AttributeError):
        topology.nodes[0].slug = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        topology.nodes[0].bindings["other"] = LiteralSource(value=False)  # type: ignore[index]
    with pytest.raises(TypeError):
        topology.slug_to_index["changed"] = 0  # type: ignore[index]


def test_cycle_and_unknown_source_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        validate_graph(
            [
                DagNode(
                    slug="a",
                    plugin="p",
                    edges=(InputEdge(name="x", source="missing"),),
                    input_mode=InputMode.EACH,
                )
            ]
        )
    with pytest.raises(ValueError, match="cycle"):
        validate_graph(
            [
                DagNode(
                    slug="a",
                    plugin="p",
                    edges=(InputEdge(name="b", source="b"),),
                    input_mode=InputMode.EACH,
                ),
                DagNode(
                    slug="b",
                    plugin="p",
                    edges=(InputEdge(name="a", source="a"),),
                    input_mode=InputMode.EACH,
                ),
            ]
        )


def test_edge_names_are_unique_and_bindings_are_edge_addressed() -> None:
    with pytest.raises(ValueError, match="duplicate edge"):
        validate_graph(
            [
                DagNode(slug="root", plugin="p"),
                DagNode(
                    slug="child",
                    plugin="p",
                    input_mode=InputMode.EACH,
                    edges=(
                        InputEdge(name="same", source="root"),
                        InputEdge(
                            name="same",
                            source="root",
                            role=InputRole.SIDE,
                            required=False,
                        ),
                    ),
                ),
            ]
        )
    with pytest.raises(ValueError, match="unknown edge"):
        validate_graph(
            [
                DagNode(slug="root", plugin="p"),
                DagNode(
                    slug="child",
                    plugin="p",
                    input_mode=InputMode.EACH,
                    edges=(InputEdge(name="actual", source="root"),),
                    bindings={"x": EdgeSource(edge="typo", field="x")},
                ),
            ]
        )


def test_mode_validation() -> None:
    root = DagNode(slug="root", plugin="p")
    with pytest.raises(ValueError, match="root"):
        validate_graph([root.model_copy(update={"input_mode": InputMode.BY_CORRELATION})])
    with pytest.raises(ValueError, match="exactly one"):
        validate_graph(
            [
                root,
                DagNode(
                    slug="child",
                    plugin="p",
                    input_mode=InputMode.EACH,
                    edges=(
                        InputEdge(name="one", source="root"),
                        InputEdge(name="two", source="root"),
                    ),
                ),
            ]
        )
    with pytest.raises(ValueError, match="cannot inherit"):
        validate_graph(
            [
                DagNode(
                    slug="root",
                    plugin="p",
                    correlation_mode=CorrelationMode.INHERIT,
                )
            ]
        )


def test_ready_nodes_and_downstream_closure() -> None:
    dependencies = ((), (0,), (0,), (1, 2))
    assert find_ready_nodes(dependencies, set(), set()) == [0]
    assert find_ready_nodes(dependencies, {0}, {0}) == [1, 2]
    assert downstream_closure(dependencies, (1,)) == {1, 3}


def test_large_forward_ordered_dag_does_not_use_python_recursion() -> None:
    nodes = [
        DagNode(
            slug=f"node-{index}",
            plugin="p",
            edges=(InputEdge(name="upstream", source=f"node-{index + 1}"),),
            input_mode=InputMode.EACH,
        )
        for index in range(1_499)
    ]
    nodes.append(DagNode(slug="node-1499", plugin="p"))
    dependencies = validate_graph(nodes)
    assert dependencies[0] == [1]
    assert dependencies[-1] == []


def test_large_cycle_is_rejected_without_recursion_error() -> None:
    nodes = [
        DagNode(
            slug=f"node-{index}",
            plugin="p",
            edges=(
                InputEdge(
                    name="upstream",
                    source=f"node-{(index + 1) % 1_500}",
                ),
            ),
            input_mode=InputMode.EACH,
        )
        for index in range(1_500)
    ]
    with pytest.raises(ValueError, match="cycle detected"):
        validate_graph(nodes)


def test_expression_and_when_references_are_part_of_graph_validation() -> None:
    root = DagNode(slug="root", plugin="p")
    for binding in (
        ExpressionSource(expression="{ghost.value}"),
        EdgeSource(edge="actual", field="value", when="{ghost.enabled}"),
    ):
        child = DagNode(
            slug="child",
            plugin="p",
            input_mode=InputMode.EACH,
            edges=(InputEdge(name="actual", source="root"),),
            bindings={"value": binding},
        )
        with pytest.raises(ValueError, match="unknown edges"):
            validate_graph([root, child])
