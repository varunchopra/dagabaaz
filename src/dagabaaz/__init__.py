"""DAG execution engine — graph readiness, artifact routing, and orchestration.

Given a graph of nodes with dependencies, the engine determines readiness,
routes artifacts through edges with filtering, handles fan-out/fan-in,
skip cascades, and barrier synchronization.

Persistence and job dispatch are abstracted behind the ``DagStore`` protocol.
"""
