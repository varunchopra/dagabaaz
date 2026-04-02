from dagabaaz.retry import RetryBoundary, compute_retry_boundary


class TestComputeRetryBoundary:
    """compute_retry_boundary partitions nodes into failed vs downstream."""

    def test_single_failed_node(self) -> None:
        boundary = compute_retry_boundary(
            completed_nodes={0, 1},
            launched_nodes={0, 1, 2},
            node_count=4,
            node_has_retryable_failure={2: True},
        )
        assert boundary.failed == [2]
        assert boundary.downstream == [3]  # unlaunched

    def test_all_completed(self) -> None:
        boundary = compute_retry_boundary(
            completed_nodes={0, 1, 2},
            launched_nodes={0, 1, 2},
            node_count=3,
            node_has_retryable_failure={},
        )
        assert boundary == RetryBoundary(failed=[], downstream=[])

    def test_mixed_failed_and_downstream(self) -> None:
        boundary = compute_retry_boundary(
            completed_nodes={0, 1},
            launched_nodes={0, 1, 2, 3},
            node_count=5,
            node_has_retryable_failure={2: True, 3: False},
        )
        assert boundary.failed == [2]
        assert boundary.downstream == [3, 4]  # 3 = launched no failure, 4 = unlaunched

    def test_all_nodes_failed(self) -> None:
        boundary = compute_retry_boundary(
            completed_nodes=set(),
            launched_nodes={0, 1, 2},
            node_count=3,
            node_has_retryable_failure={0: True, 1: True, 2: True},
        )
        assert boundary.failed == [0, 1, 2]
        assert boundary.downstream == []

    def test_unlaunched_nodes_are_downstream(self) -> None:
        boundary = compute_retry_boundary(
            completed_nodes={0},
            launched_nodes={0, 1},
            node_count=4,
            node_has_retryable_failure={1: True},
        )
        assert boundary.failed == [1]
        assert boundary.downstream == [2, 3]

    def test_empty_graph(self) -> None:
        boundary = compute_retry_boundary(
            completed_nodes=set(),
            launched_nodes=set(),
            node_count=0,
            node_has_retryable_failure={},
        )
        assert boundary == RetryBoundary(failed=[], downstream=[])

    def test_non_completed_without_failure_is_downstream(self) -> None:
        """Launched nodes that aren't completed but have no retryable failures
        are downstream (e.g., cancelled sibling tasks)."""
        boundary = compute_retry_boundary(
            completed_nodes={0},
            launched_nodes={0, 1, 2},
            node_count=3,
            node_has_retryable_failure={1: False, 2: False},
        )
        assert boundary.failed == []
        assert boundary.downstream == [1, 2]
