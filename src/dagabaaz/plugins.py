"""Plugin metadata protocols for the DAG execution engine.

``core.plugins.BasePlugin`` satisfies ``PluginMeta`` via structural subtyping.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class PluginInputMeta:
    """Minimal input metadata the engine needs for schema generation."""

    name: str
    description: str
    source: str
    required: bool = False
    placeholder: str = ""


@runtime_checkable
class PluginMeta(Protocol):
    """Structural interface for plugin metadata used by the engine.

    ``worker_only`` determines server resolution. ``get_effective_inputs()``
    is used by pipeline schema generation to build the run input form.
    """

    name: str
    worker_only: bool

    def get_effective_inputs(self) -> list[PluginInputMeta]: ...


PluginLookup = Callable[[str], PluginMeta | None]
"""Given a plugin name, return its metadata or None if not found."""
