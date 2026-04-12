"""
Core environment for DevOps Incident Response OpenEnv.

Implements a singleton DevOpsEnv instance accessible via get_env().

State transitions are fully deterministic – no random components –
so results are reproducible for CI/validation and baseline scoring.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Optional

from grader import compute_reward, _all_criteria_met
from tasks import get_task, default_task_id, list_tasks


# ---------------------------------------------------------------------------
# Action patterns
# ---------------------------------------------------------------------------

# Each entry: (regex pattern, handler_name, flag_to_set_or_none)
# Patterns are matched case-insensitively against the trimmed action string.

_ACTION_PATTERNS = [
    # ---- Kubernetes general ----
    (r"kubectl\s+rollout\s+restart", "k8s_rollout_restart", None),
    (r"kubectl\s+scale", "k8s_scale", None),
    (r"kubectl\s+top\s+pod", "k8s_top_pods", None),
    (r"kubectl\s+top\s+node", "k8s_top_nodes", None),
    (r"kubectl\s+describe\s+pod", "k8s_describe_pod", None),
    (r"kubectl\s+logs", "k8s_logs", None),
    (r"kubectl\s+get\s+pod", "k8s_get_pods", None),
    (r"kubectl\s+get\s+event", "k8s_get_events", None),
    (r"kubectl\s+get\s+deployment", "k8s_get_deployment", None),
    (r"kubectl\s+(apply|patch)\s+", "k8s_apply_patch", None),
    (r"kubectl\s+rollout\s+undo", "k8s_rollout_undo", "configmap_rolled_back"),
    (r"kubectl\s+set\s+image", "k8s_set_image", None),
    # ---- ConfigMap / rollback ----
    (r"rollback|undo\s+deploy|revert\s+configmap|apply.*v6|configmap.*v6", "rollback_configmap", "configmap_rolled_back"),
    (r"kubectl\s+rollout\s+status", "k8s_rollout_status", "services_restarted_in_order"),
    # ---- Process / OS ----
    (r"kill\s+-9|pkill|kill\s+pid|kill\s+\d+", "kill_process", "process_killed_or_hpa_scaled"),
    (r"top\b|htop\b|ps\s+aux", "check_processes", None),
    (r"free\s+-m|vmstat|iostat", "check_system_resources", None),
    # ---- HPA / Scaling ----
    (r"kubectl\s+autoscale|hpa|horizontalpodautoscaler", "enable_hpa", "process_killed_or_hpa_scaled"),
    # ---- DB / Postgres ----
    (r"pg_cancel_backend|pg_terminate_backend|kill.*query|cancel.*query", "kill_slow_query", "slow_query_killed_or_index_added"),
    (r"create\s+index|add\s+index|explain\s+analyze", "add_db_index", "slow_query_killed_or_index_added"),
    (r"pgbouncer|connection.pool", "configure_pgbouncer", None),
    (r"psql|pg_activity|select.*pg_stat", "db_diagnostics", None),
    # ---- Memory ----
    (r"memory.limit|resource.limit|limit.*memory|memory.*request", "update_memory_limit", "memory_limit_raised_or_leak_fixed"),
    (r"heap.dump|memory.profile|pprof|jmap", "memory_profiling", None),
    # ---- Circuit breaker ----
    (r"circuit.breaker|reset.circuit|hystrix|resilience4j", "reset_circuit_breaker", "circuit_breaker_reset"),
    # ---- Ingress / Rate limiting ----
    (r"rate.limit|nginx.ingress|ingress.*annotation|remove.*rate", "update_ingress", "ingress_rate_limit_lifted"),
    # ---- Postmortem / Runbook ----
    (r"postmortem|incident.report|runbook|blameless", "file_postmortem", "postmortem_filed"),
    # ---- Alerts / PagerDuty ----
    (r"silence.alert|ack|acknowledge|pagerduty", "manage_alert", None),
    # ---- Diagnostics ----
    (r"curl\s+|wget\s+|http\s+get|check.endpoint", "check_endpoint", None),
    (r"prometheus|grafana|datadog|new.relic|observe", "check_metrics", None),
    (r"jaeger|zipkin|trace|distributed.trace", "check_traces", None),
]


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class DevOpsEnv:
    """
    Singleton DevOps Incident Response environment.

    All state is stored in plain Python dicts so that JSON serialisation
    and .get() calls always work correctly.
    """

    def __init__(self) -> None:
        self._task: Optional[Dict[str, Any]] = None
        self._state: Dict[str, Any] = {}
        self._step_count: int = 0
        self._done: bool = False
        self._episode_rewards: List[float] = []

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Reset the environment and return the initial observation dict."""
        if task_id is None:
            task_id = default_task_id()

        self._task = get_task(task_id)
        self._state = deepcopy(self._task["initial_state"])
        self._state["acquired_flags"] = []
        self._step_count = 0
        self._done = False
        self._episode_rewards = []

        return {
            "observation": self._build_observation(),
            "task_id": task_id,
            "difficulty": self._task["difficulty"],
            "max_steps": self._task["max_steps"],
            "info": {
                "alerts": self._state.get("alerts", []),
                "services": self._state.get("services", []),
            },
        }

    def step(self, action: str) -> Dict[str, Any]:
        """Execute one action and return (observation, reward, done, info)."""
        if self._task is None:
            # Auto-reset if called before reset()
            self.reset()

        if self._done:
            return {
                "observation": "Episode is finished. Call /reset to start a new episode.",
                "reward": 0.0,
                "done": True,
                "info": {"message": "already_done"},
            }

        state_before = deepcopy(self._state)

        # Identify and apply the action
        handler_name, flag, recognised = self._dispatch_action(action)
        self._apply_action(handler_name, action)

        # Acquire flag if applicable
        if flag and flag not in self._state["acquired_flags"]:
            self._state["acquired_flags"].append(flag)

        self._step_count += 1
        self._state["actions_taken"] = self._state.get("actions_taken", []) + [action]

        # Check termination
        max_steps = self._task["max_steps"]
        criteria = self._task["success_criteria"]
        success = _all_criteria_met(self._state, criteria)

        if success or self._step_count >= max_steps:
            self._done = True

        # Compute reward
        reward = compute_reward(
            state_before=state_before,
            state_after=self._state,
            action=action,
            done=self._done,
            bad_actions=self._task.get("bad_actions", []),
            success_criteria=criteria,
            recognised_action=recognised,
        )
        self._episode_rewards.append(reward)

        observation = self._build_observation()
        info = {
            "step": self._step_count,
            "max_steps": max_steps,
            "success": success,
            "acquired_flags": list(self._state["acquired_flags"]),
            "handler": handler_name,
            "recognised": recognised,
        }

        return {
            "observation": observation,
            "reward": reward,
            "done": self._done,
            "info": info,
        }

    def state(self) -> Dict[str, Any]:
        """Return a read-only snapshot of the current state."""
        return {
            "task_id": self._task["id"] if self._task else None,
            "difficulty": self._task["difficulty"] if self._task else None,
            "step_count": self._step_count,
            "done": self._done,
            "services": deepcopy(self._state.get("services", [])),
            "alerts": deepcopy(self._state.get("alerts", [])),
            "acquired_flags": list(self._state.get("acquired_flags", [])),
            "actions_taken": list(self._state.get("actions_taken", [])),
            "episode_rewards": list(self._episode_rewards),
        }

    # -----------------------------------------------------------------------
    # Action dispatch
    # -----------------------------------------------------------------------

    def _dispatch_action(self, action: str):
        """Return (handler_name, flag_to_acquire, was_recognised)."""
        for pattern, handler_name, flag in _ACTION_PATTERNS:
            if re.search(pattern, action, re.IGNORECASE):
                return handler_name, flag, True
        return "noop", None, False

    def _apply_action(self, handler_name: str, raw_action: str) -> None:
        """Mutate self._state according to the handler."""
        services = self._state.get("services", [])

        def _svc(name: str) -> Optional[Dict[str, Any]]:
            for s in services:
                if s["name"] == name:
                    return s
            return None

        def _improve(svc: Dict[str, Any], cpu_d=0.0, mem_d=0.0, err_d=0.0, lat_mul=1.0, replicas_ready=None):
            if svc is None:
                return
            svc["cpu_percent"] = max(0.0, svc.get("cpu_percent", 0) - cpu_d)
            svc["memory_percent"] = max(0.0, svc.get("memory_percent", 0) - mem_d)
            svc["error_rate"] = max(0.0, svc.get("error_rate", 0) - err_d)
            svc["latency_ms"] = max(0.0, svc.get("latency_ms", 0) * lat_mul)
            if replicas_ready is not None:
                svc["replicas_ready"] = replicas_ready
            # Update status
            if svc["error_rate"] < 0.02 and svc["cpu_percent"] < 80 and svc["replicas_ready"] >= svc.get("replicas_desired", 1):
                svc["status"] = "healthy"
            elif svc["error_rate"] < 0.5 or svc["cpu_percent"] < 90:
                svc["status"] = "degraded"
            else:
                svc["status"] = "down"

        task_id = self._task["id"] if self._task else ""

        # ---- easy_cpu_spike handlers ----
        if handler_name == "kill_process":
            gw = _svc("api-gateway")
            _improve(gw, cpu_d=65.0, err_d=0.10, lat_mul=0.2)
            self._remove_alert("CRITICAL: api-gateway CPU > 95%")

        elif handler_name == "enable_hpa":
            gw = _svc("api-gateway")
            _improve(gw, cpu_d=40.0, err_d=0.08, lat_mul=0.4, replicas_ready=5)

        elif handler_name == "k8s_rollout_restart":
            # Restarts clear degradation and bring replicas up
            for svc in services:
                if svc.get("status") in ("degraded", "down"):
                    _improve(svc, cpu_d=10.0, mem_d=5.0, err_d=0.05, lat_mul=0.8)
                    desired = svc.get("replicas_desired", 1)
                    if svc.get("replicas_ready", 0) < desired:
                        svc["replicas_ready"] = min(svc["replicas_ready"] + 1, desired)

        elif handler_name == "k8s_scale":
            for svc in services:
                if svc.get("status") in ("degraded",):
                    desired = svc.get("replicas_desired", 1)
                    svc["replicas_ready"] = desired
                    _improve(svc, cpu_d=20.0, err_d=0.1, lat_mul=0.6)

        # ---- medium_memory_db handlers ----
        elif handler_name == "update_memory_limit":
            svc = _svc("order-service")
            _improve(svc, mem_d=30.0, err_d=0.15, lat_mul=0.5, replicas_ready=4)

        elif handler_name == "kill_slow_query":
            db = _svc("postgres-primary")
            _improve(db, cpu_d=40.0, err_d=0.15, lat_mul=0.2)
            if db:
                db["metadata"]["slow_queries"] = 0
                db["metadata"]["active_connections"] = max(10, db["metadata"].get("active_connections", 99) - 60)

        elif handler_name == "add_db_index":
            db = _svc("postgres-primary")
            _improve(db, cpu_d=20.0, lat_mul=0.3, err_d=0.08)

        elif handler_name == "configure_pgbouncer":
            db = _svc("postgres-primary")
            _improve(db, cpu_d=10.0, err_d=0.05, lat_mul=0.7)
            if db:
                db["metadata"]["active_connections"] = max(5, db["metadata"].get("active_connections", 99) - 40)

        elif handler_name == "memory_profiling":
            svc = _svc("order-service")
            _improve(svc, mem_d=5.0)

        # ---- hard_cascading_failure handlers ----
        elif handler_name == "rollback_configmap":
            cfg = _svc("config-service")
            _improve(cfg, err_d=0.98, lat_mul=0.0, replicas_ready=2)
            if cfg:
                cfg["cpu_percent"] = 12.0
                cfg["memory_percent"] = 25.0
                cfg["latency_ms"] = 50.0
                cfg["status"] = "healthy"
                cfg["metadata"]["version"] = self._task["flags"].get("last_good_configmap", "v1.2.2")

        elif handler_name == "k8s_rollout_undo":
            cfg = _svc("config-service")
            _improve(cfg, err_d=0.95, replicas_ready=2)
            if cfg:
                cfg["latency_ms"] = 50.0
                cfg["status"] = "healthy"

        elif handler_name == "reset_circuit_breaker":
            pmt = _svc("payment-service")
            _improve(pmt, err_d=0.50, lat_mul=0.3, replicas_ready=3)
            if pmt and "circuit_breaker" in pmt.get("metadata", {}):
                pmt["metadata"]["circuit_breaker"] = "closed"

        elif handler_name == "update_ingress":
            ing = _svc("ingress-nginx")
            _improve(ing, cpu_d=30.0, err_d=0.28, lat_mul=0.3)
            if ing:
                ing["metadata"]["rate_limit_active"] = False
                ing["metadata"]["rate_limit_rps"] = 0

        elif handler_name == "file_postmortem":
            pass  # Flag is acquired separately; no metric changes

        # ---- Generic diagnostics (no state change, just observation) ----
        elif handler_name in (
            "k8s_top_pods", "k8s_top_nodes", "k8s_describe_pod",
            "k8s_logs", "k8s_get_pods", "k8s_get_events", "k8s_get_deployment",
            "k8s_rollout_status", "check_processes", "check_system_resources",
            "db_diagnostics", "check_endpoint", "check_metrics", "check_traces",
            "manage_alert",
        ):
            pass  # Observation only – good for gathering info, no state change

        elif handler_name == "k8s_apply_patch":
            # Generic apply: mild improvement across degraded services
            for svc in services:
                if svc.get("status") == "degraded":
                    _improve(svc, cpu_d=5.0, mem_d=3.0, err_d=0.02, lat_mul=0.95)

        # ---- noop ----
        # Nothing happens; reward function will penalise

        # Cascade: if config-service is healthy, user-service can recover
        cfg = _svc("config-service")
        if cfg and cfg.get("status") == "healthy":
            usr = _svc("user-service")
            if usr and usr.get("status") in ("down", "degraded"):
                _improve(usr, err_d=0.30, lat_mul=0.4, replicas_ready=usr.get("replicas_desired", 3))

        # If user-service is healthy, downstream services can recover too
        usr = _svc("user-service")
        if usr and usr.get("status") == "healthy":
            for dep_name in ("payment-service", "notification-service"):
                dep = _svc(dep_name)
                if dep and dep.get("status") in ("down", "degraded"):
                    _improve(dep, err_d=0.10, lat_mul=0.8)

    # -----------------------------------------------------------------------
    # Observation builder
    # -----------------------------------------------------------------------

    def _build_observation(self) -> str:
        if not self._task:
            return "Environment not initialised. Call /reset."

        lines: List[str] = []
        task = self._task
        lines.append(f"=== DevOps Incident: {task['id']} [{task['difficulty'].upper()}] ===")
        lines.append(f"Severity: {task['severity'].upper()}")
        lines.append(f"Step: {self._step_count} / {task['max_steps']}")
        lines.append("")
        lines.append("DESCRIPTION:")
        lines.append(task["description"])
        lines.append("")

        alerts = self._state.get("alerts", [])
        if alerts:
            lines.append("ACTIVE ALERTS:")
            for a in alerts:
                lines.append(f"  • {a}")
            lines.append("")

        lines.append("SERVICE STATUS:")
        for svc in self._state.get("services", []):
            ready = svc.get("replicas_ready", "?")
            desired = svc.get("replicas_desired", "?")
            lines.append(
                f"  [{svc['status'].upper():8s}] {svc['name']:25s} "
                f"CPU={svc.get('cpu_percent', 0):5.1f}%  "
                f"MEM={svc.get('memory_percent', 0):5.1f}%  "
                f"ERR={svc.get('error_rate', 0)*100:5.1f}%  "
                f"LAT={svc.get('latency_ms', 0):7.0f}ms  "
                f"PODS={ready}/{desired}"
            )
        lines.append("")

        flags = self._state.get("acquired_flags", [])
        if flags:
            lines.append(f"PROGRESS FLAGS: {', '.join(flags)}")
            lines.append("")

        if self._done:
            criteria = task["success_criteria"]
            success = _all_criteria_met(self._state, criteria)
            if success:
                lines.append("✅ INCIDENT RESOLVED – all success criteria met.")
            else:
                lines.append("❌ EPISODE ENDED – max steps reached without full resolution.")

        return "\n".join(lines)

    def _remove_alert(self, prefix: str) -> None:
        self._state["alerts"] = [
            a for a in self._state.get("alerts", [])
            if prefix.lower() not in a.lower()
        ]


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_env_instance: Optional[DevOpsEnv] = None


def get_env() -> DevOpsEnv:
    """Return the global singleton environment instance."""
    global _env_instance
    if _env_instance is None:
        _env_instance = DevOpsEnv()
    return _env_instance
