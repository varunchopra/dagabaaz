import pytest

from dagabaaz.models import (
    MaterializationError,
    PlannedEdgeInput,
    ResolvedOutput,
    TaskInputPlan,
)
from dagabaaz.task_input import resolve_task_inputs


class Resolver:
    def __init__(self) -> None:
        self.result = {
            "a": ResolvedOutput(id="a", fields={"n": 1}, data={"path": "/a"}),
            "b": ResolvedOutput(id="b", fields={"n": 2}, data={"path": "/b"}),
        }
        self.requested: tuple[str, ...] = ()
        self.run_id = ""

    def resolve_outputs(self, run_id: str, output_ids: tuple[str, ...]):
        self.run_id = run_id
        self.requested = output_ids
        return dict(self.result)


def plan() -> TaskInputPlan:
    return TaskInputPlan(
        generation=2,
        edges=(
            PlannedEdgeInput(edge="first", source_index=0, role="main", output_ids=("b", "a")),
            PlannedEdgeInput(edge="second", source_index=1, role="side", output_ids=("a",)),
        ),
        parameters={"limit": 2},
    )


def test_resolution_is_bulk_unique_and_restores_plan_order() -> None:
    resolver = Resolver()
    inputs = resolve_task_inputs(resolver, run_id="run", plan=plan())
    assert resolver.run_id == "run"
    assert resolver.requested == ("b", "a")
    assert [item.id for item in inputs.edges["first"]] == ["b", "a"]
    assert [item.id for item in inputs.edges["second"]] == ["a"]
    assert inputs.parameters == {"limit": 2}


def test_missing_extra_and_key_identity_are_failures() -> None:
    resolver = Resolver()
    resolver.result.pop("b")
    with pytest.raises(MaterializationError, match="missing"):
        resolve_task_inputs(resolver, run_id="run", plan=plan())

    resolver = Resolver()
    resolver.result["extra"] = ResolvedOutput(id="extra", fields={}, data={})
    with pytest.raises(MaterializationError, match="extra"):
        resolve_task_inputs(resolver, run_id="run", plan=plan())

    resolver = Resolver()
    resolver.result["a"] = ResolvedOutput(id="wrong", fields={}, data={})
    with pytest.raises(MaterializationError, match="contains output"):
        resolve_task_inputs(resolver, run_id="run", plan=plan())


def test_empty_plan_is_resolved_as_an_exact_empty_request() -> None:
    resolver = Resolver()
    resolver.result = {}

    inputs = resolve_task_inputs(
        resolver,
        run_id="run",
        plan=TaskInputPlan(generation=2),
    )

    assert resolver.requested == ()
    assert inputs.edges == {}


def test_materializer_failure_is_reported_as_a_materialization_error() -> None:
    class FailingResolver:
        def resolve_outputs(self, run_id: str, output_ids: tuple[str, ...]):
            raise OSError("storage unavailable")

    with pytest.raises(MaterializationError, match="materializer failed") as error:
        resolve_task_inputs(FailingResolver(), run_id="run", plan=plan())

    assert isinstance(error.value.__cause__, OSError)


@pytest.mark.parametrize(
    "result",
    [None, [], {"a": object(), "b": object()}, {1: ResolvedOutput(id="1")}],
)
def test_materializer_must_return_the_output_mapping_shape(
    result: object,
) -> None:
    class InvalidResolver:
        def resolve_outputs(self, run_id: str, output_ids: tuple[str, ...]):
            return result

    with pytest.raises(MaterializationError, match="mapping|protocol"):
        resolve_task_inputs(InvalidResolver(), run_id="run", plan=plan())
