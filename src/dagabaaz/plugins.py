"""Plugin metadata protocols for the DAG execution engine."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from dagabaaz.json import JsonInput, freeze_json


@dataclass(frozen=True, slots=True)
class PluginInputMeta:
    """Input metadata used to generate a run-input schema.

    ``default`` accepts JSON input and is frozen during construction.
    """

    name: str
    description: str
    source: str
    required: bool = False
    placeholder: str = ""
    default: JsonInput = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("plugin input name must not be empty")
        if not self.source:
            raise ValueError("plugin input source must not be empty")
        object.__setattr__(self, "default", freeze_json(self.default))


@runtime_checkable
class PluginMeta(Protocol):
    """Metadata required to construct a run-input schema."""

    def get_effective_inputs(self) -> list[PluginInputMeta]: ...


PluginLookup = Callable[[str], PluginMeta | None]
"""Given a plugin name, return its metadata or None if not found."""
