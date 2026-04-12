"""
Task definitions for DevOps Incident Response OpenEnv.

Three tasks ordered by difficulty:
  1. easy_cpu_spike       – single service CPU spike
  2. medium_memory_db     – memory leak + database degradation
  3. hard_cascading_failure – multi-service cascading failure
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Internal task registry
# ---------------------------------------------------------------------------

_TASKS: List[Dict[str, Any]] = [
    {
        "id": "easy_cpu_spike",
        "difficulty": "easy",
        "severity": "high",
        "description": (
            "The 'api-gateway' service is experiencing a CPU spike above 95%. "
            "Latency has jumped to 2 400 ms and the on-call alert fired 3 minutes ago. "
            "Identify the root cause and restore normal CPU usage below 60%."
        ),
        "max_steps": 15,
        "initial_state": {
            "services": [
                {
                    "name": "api-gateway",
                    "status": "degraded",
                    "cpu_percent": 97.3,
                    "memory_percent": 45.0,
                    "error_rate": 0.12,
                    "latency_ms": 2400.0,
                    "replicas_desired": 3,
                    "replicas_ready": 3,
                    "metadata": {"version": "v2.4.1", "namespace": "production"},
                },
                {
                    "name": "auth-service",
                    "status": "healthy",
                    "cpu_percent": 22.0,
                    "memory_percent": 38.0,
                    "error_rate": 0.0,
                    "latency_ms": 80.0,
                    "replicas_desired": 2,
                    "replicas_ready": 2,
                    "metadata": {"version": "v1.9.0", "namespace": "production"},
                },
            ],
            "alerts": [
                "CRITICAL: api-gateway CPU > 95% for 3 min",
                "WARNING: api-gateway p99 latency > 2 000 ms",
            ],
            "flags": {
                "runaway_process_pid": 4821,
                "runaway_process_name": "log-aggregator",
                "hpa_enabled": False,
            },
        },
        "success_criteria": {
            "target_service": "api-gateway",
            "cpu_below": 60.0,
            "error_rate_below": 0.02,
            "required_flags": ["process_killed_or_hpa_scaled"],
        },
        "bad_actions": ["delete deployment", "kubectl delete pod --all", "rm -rf"],
    },
    {
        "id": "medium_memory_db",
        "difficulty": "medium",
        "severity": "critical",
        "description": (
            "The 'order-service' pods are OOMKilled repeatedly (memory usage at 98%). "
            "Simultaneously, 'postgres-primary' shows query latency of 8 500 ms and "
            "connection pool exhaustion. Orders are failing at 34% error rate. "
            "Stabilise both services and restore order processing."
        ),
        "max_steps": 20,
        "initial_state": {
            "services": [
                {
                    "name": "order-service",
                    "status": "degraded",
                    "cpu_percent": 55.0,
                    "memory_percent": 98.1,
                    "error_rate": 0.34,
                    "latency_ms": 5200.0,
                    "replicas_desired": 4,
                    "replicas_ready": 2,
                    "metadata": {
                        "version": "v3.1.0",
                        "namespace": "production",
                        "memory_limit": "512Mi",
                        "oom_count": 7,
                    },
                },
                {
                    "name": "postgres-primary",
                    "status": "degraded",
                    "cpu_percent": 88.0,
                    "memory_percent": 76.0,
                    "error_rate": 0.18,
                    "latency_ms": 8500.0,
                    "replicas_desired": 1,
                    "replicas_ready": 1,
                    "metadata": {
                        "version": "14.5",
                        "namespace": "production",
                        "max_connections": 100,
                        "active_connections": 99,
                        "slow_queries": 43,
                    },
                },
                {
                    "name": "cache-redis",
                    "status": "healthy",
                    "cpu_percent": 12.0,
                    "memory_percent": 41.0,
                    "error_rate": 0.0,
                    "latency_ms": 1.5,
                    "replicas_desired": 1,
                    "replicas_ready": 1,
                    "metadata": {"version": "7.0", "namespace": "production"},
                },
            ],
            "alerts": [
                "CRITICAL: order-service OOMKilled x7 in last 10 min",
                "CRITICAL: postgres-primary connection pool exhausted (99/100)",
                "WARNING: order-service error rate 34%",
                "WARNING: postgres slow query count 43",
            ],
            "flags": {
                "memory_leak_root": "unbounded_result_set",
                "db_culprit_query": "SELECT * FROM orders WHERE created_at > '2020-01-01'",
                "connection_pool_size": 100,
                "pgbouncer_available": True,
            },
        },
        "success_criteria": {
            "targets": ["order-service", "postgres-primary"],
            "memory_below": 80.0,
            "db_latency_below_ms": 500.0,
            "error_rate_below": 0.05,
            "required_flags": ["memory_limit_raised_or_leak_fixed", "slow_query_killed_or_index_added"],
        },
        "bad_actions": [
            "drop table",
            "kubectl delete pod --all",
            "truncate orders",
            "rm -rf",
        ],
    },
    {
        "id": "hard_cascading_failure",
        "difficulty": "hard",
        "severity": "critical",
        "description": (
            "A cascading failure has brought down 60% of the production stack. "
            "Root cause appears to be a bad config-map deployment that caused 'config-service' "
            "to return 500s, which propagated through 'user-service', 'payment-service', and "
            "'notification-service'. The ingress is rate-limiting legitimate traffic. "
            "Revenue impact: ~$12 000/min. Restore all services in the correct dependency order "
            "and prevent re-occurrence."
        ),
        "max_steps": 30,
        "initial_state": {
            "services": [
                {
                    "name": "config-service",
                    "status": "down",
                    "cpu_percent": 0.0,
                    "memory_percent": 0.0,
                    "error_rate": 1.0,
                    "latency_ms": 0.0,
                    "replicas_desired": 2,
                    "replicas_ready": 0,
                    "metadata": {
                        "version": "v1.2.3-bad",
                        "namespace": "production",
                        "last_good_version": "v1.2.2",
                        "bad_configmap": "app-config-v7",
                    },
                },
                {
                    "name": "user-service",
                    "status": "down",
                    "cpu_percent": 4.0,
                    "memory_percent": 20.0,
                    "error_rate": 0.98,
                    "latency_ms": 30000.0,
                    "replicas_desired": 3,
                    "replicas_ready": 0,
                    "metadata": {"version": "v5.0.1", "namespace": "production", "depends_on": "config-service"},
                },
                {
                    "name": "payment-service",
                    "status": "degraded",
                    "cpu_percent": 30.0,
                    "memory_percent": 55.0,
                    "error_rate": 0.76,
                    "latency_ms": 12000.0,
                    "replicas_desired": 3,
                    "replicas_ready": 1,
                    "metadata": {
                        "version": "v4.3.0",
                        "namespace": "production",
                        "depends_on": "user-service",
                        "circuit_breaker": "open",
                    },
                },
                {
                    "name": "notification-service",
                    "status": "degraded",
                    "cpu_percent": 15.0,
                    "memory_percent": 30.0,
                    "error_rate": 0.55,
                    "latency_ms": 8000.0,
                    "replicas_desired": 2,
                    "replicas_ready": 1,
                    "metadata": {"version": "v2.1.0", "namespace": "production", "depends_on": "user-service"},
                },
                {
                    "name": "ingress-nginx",
                    "status": "degraded",
                    "cpu_percent": 75.0,
                    "memory_percent": 60.0,
                    "error_rate": 0.30,
                    "latency_ms": 3000.0,
                    "replicas_desired": 2,
                    "replicas_ready": 2,
                    "metadata": {
                        "version": "1.9.0",
                        "namespace": "ingress-nginx",
                        "rate_limit_active": True,
                        "rate_limit_rps": 10,
                    },
                },
            ],
            "alerts": [
                "CRITICAL: config-service 0/2 replicas ready",
                "CRITICAL: user-service 0/3 replicas ready",
                "CRITICAL: payment-service circuit breaker OPEN",
                "CRITICAL: payment error rate 76%",
                "WARNING: notification-service error rate 55%",
                "WARNING: ingress rate-limiting active (10 rps)",
                "INFO: bad configmap app-config-v7 deployed 8 min ago",
            ],
            "flags": {
                "bad_configmap": "app-config-v7",
                "last_good_configmap": "app-config-v6",
                "rollback_available": True,
                "dependency_order": ["config-service", "user-service", "payment-service", "notification-service"],
                "postmortem_required": True,
            },
        },
        "success_criteria": {
            "targets": ["config-service", "user-service", "payment-service", "notification-service"],
            "all_error_rate_below": 0.05,
            "all_replicas_ready": True,
            "ingress_rate_limit_lifted": True,
            "required_flags": [
                "configmap_rolled_back",
                "services_restarted_in_order",
                "circuit_breaker_reset",
                "postmortem_filed",
            ],
        },
        "bad_actions": [
            "kubectl delete namespace production",
            "kubectl delete pod --all",
            "rm -rf",
            "drop database",
        ],
    },
]

_TASK_MAP: Dict[str, Dict[str, Any]] = {t["id"]: t for t in _TASKS}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_tasks() -> List[Dict[str, Any]]:
    """Return all task definitions (deep-copied so callers cannot mutate registry)."""
    return deepcopy(_TASKS)


def get_task(task_id: str) -> Dict[str, Any]:
    """Return a specific task by ID, raising KeyError if not found."""
    if task_id not in _TASK_MAP:
        available = ", ".join(_TASK_MAP.keys())
        raise KeyError(f"Unknown task_id '{task_id}'. Available: {available}")
    return deepcopy(_TASK_MAP[task_id])


def default_task_id() -> str:
    return _TASKS[0]["id"]
