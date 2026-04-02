import pytest


@pytest.fixture()
def reset_topology_cache():
    """Reset the topology module's global cache before and after each test."""
    from dagabaaz import topology

    topology._topology_cache.clear()
    original_max = topology._MAX_TOPOLOGY_CACHE
    topology._MAX_TOPOLOGY_CACHE = 500
    yield
    topology._topology_cache.clear()
    topology._MAX_TOPOLOGY_CACHE = original_max
