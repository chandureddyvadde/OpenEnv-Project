"""
Deterministic grader for DevOps Incident Response OpenEnv.

Reward is always in [0.0, 1.0].

Reward components
-----------------
1. Progress shaping  – partial credit for measurable improvement each step.
2. Completion bonus  – large bonus when all success criteria are satisfied.
3. Penalties         – deducted for harmful, wasteful, or no-op actions.
"""

from __future__ import annotations

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMPLETION_BONUS = 0.40
STEP_PENALTY = 0.005        # small penalty for every step (encourages efficiency)
BAD_ACTION_PENALTY = 0.15   # large penalty for harmful commands
NOOP_PENALTY = 0.02         # small penalty for unrecognised / useless commands

MAX_PROGRESS_REWARD_PER_STEP = 0.10   # cap per-step progress reward


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_reward(
    state_before: Dict[str, Any],
    state_after: Dict[str, Any],
    action: str,
    done: bool,
    bad_actions: List[str],
    success_criteria: Dict[str, Any],
    recognised_action: bool,
) -> float:
    """
    Return a scalar reward in [0.0, 1.0].

    Parameters
    ----------
    state_before      : env state snapshot before the action
    state_after       : env state snapshot after the action
    action            : raw action string supplied by the agent
    done              : whether the episode is complete
    bad_actions       : list of destructive command substrings for this task
    success_criteria  : task-specific dict of targets
    recognised_action : True if the action matched a known command pattern
    """
    reward = 0.0

    # 1. Penalise harmful actions immediately
    if _is_bad_action(action, bad_actions):
        return max(0.0, -BAD_ACTION_PENALTY + reward)  # clamp

    # 2. Penalise unrecognised no-ops
    if not recognised_action:
        reward -= NOOP_PENALTY

    # 3. Progress shaping – reward improvement in key metrics
    reward += _progress_reward(state_before, state_after, success_criteria)

    # 4. Step cost (encourages fewer steps)
    reward -= STEP_PENALTY

    # 5. Completion bonus
    if done and _all_criteria_met(state_after, success_criteria):
        reward += COMPLETION_BONUS

    # Clamp to [0.0, 1.0]
    return float(min(1.0, max(0.0, reward)))


def score_episode(rewards: List[float]) -> float:
    """
    Aggregate a list of per-step rewards into a single episode score in [0.0, 1.0].
    Uses a discounted cumulative sum normalised to [0, 1].
    """
    if not rewards:
        return 0.0
    gamma = 0.95
    discounted = 0.0
    for i, r in enumerate(rewards):
        discounted += (gamma ** i) * r
    # Normalise: max possible discounted reward if every step got 1.0
    n = len(rewards)
    max_possible = sum(gamma ** i for i in range(n))
    return float(min(1.0, max(0.0, discounted / max_possible)))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_bad_action(action: str, bad_actions: List[str]) -> bool:
    action_lower = action.lower()
    return any(bad.lower() in action_lower for bad in bad_actions)


def _progress_reward(
    before: Dict[str, Any],
    after: Dict[str, Any],
    criteria: Dict[str, Any],
) -> float:
    """
    Compute improvement reward based on measurable metric deltas.
    Returns value in [0.0, MAX_PROGRESS_REWARD_PER_STEP].
    """
    reward = 0.0

    services_before: List[Dict[str, Any]] = before.get("services", [])
    services_after: List[Dict[str, Any]] = after.get("services", [])

    before_map = {s["name"]: s for s in services_before}
    after_map = {s["name"]: s for s in services_after}

    for name, svc_after in after_map.items():
        svc_before = before_map.get(name, {})

        # CPU improvement
        cpu_delta = svc_before.get("cpu_percent", 0) - svc_after.get("cpu_percent", 0)
        if cpu_delta > 0:
            reward += (cpu_delta / 100.0) * 0.04

        # Memory improvement
        mem_delta = svc_before.get("memory_percent", 0) - svc_after.get("memory_percent", 0)
        if mem_delta > 0:
            reward += (mem_delta / 100.0) * 0.04

        # Error rate improvement
        err_delta = svc_before.get("error_rate", 0) - svc_after.get("error_rate", 0)
        if err_delta > 0:
            reward += err_delta * 0.06

        # Latency improvement
        lat_before = svc_before.get("latency_ms", 0)
        lat_after = svc_after.get("latency_ms", 0)
        if lat_before > 0 and lat_after < lat_before:
            improvement_ratio = (lat_before - lat_after) / lat_before
            reward += improvement_ratio * 0.04

        # Replica readiness improvement
        rep_desired = svc_after.get("replicas_desired", 1)
        rep_ready_after = svc_after.get("replicas_ready", 0)
        rep_ready_before = svc_before.get("replicas_ready", 0)
        if rep_desired > 0 and rep_ready_after > rep_ready_before:
            reward += ((rep_ready_after - rep_ready_before) / rep_desired) * 0.04

    # Bonus for flags acquired
    flags_before = set(before.get("acquired_flags", []))
    flags_after = set(after.get("acquired_flags", []))
    new_flags = flags_after - flags_before
    reward += len(new_flags) * 0.03

    return min(MAX_PROGRESS_REWARD_PER_STEP, reward)


def _all_criteria_met(state: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """Return True when every success criterion is satisfied."""
    services_map = {s["name"]: s for s in state.get("services", [])}
    acquired_flags = set(state.get("acquired_flags", []))
    required_flags = set(criteria.get("required_flags", []))

    # Check flags
    if not required_flags.issubset(acquired_flags):
        return False

    # Check per-service metrics
    if "target_service" in criteria:
        svc = services_map.get(criteria["target_service"], {})
        if "cpu_below" in criteria and svc.get("cpu_percent", 999) >= criteria["cpu_below"]:
            return False
        if "error_rate_below" in criteria and svc.get("error_rate", 1) >= criteria["error_rate_below"]:
            return False

    if "targets" in criteria:
        for name in criteria["targets"]:
            svc = services_map.get(name, {})
            if "memory_below" in criteria and svc.get("memory_percent", 999) >= criteria["memory_below"]:
                return False
            if "error_rate_below" in criteria and svc.get("error_rate", 1) >= criteria["error_rate_below"]:
                return False
            if "all_error_rate_below" in criteria and svc.get("error_rate", 1) >= criteria["all_error_rate_below"]:
                return False
            if criteria.get("all_replicas_ready") and svc.get("replicas_ready", 0) < svc.get("replicas_desired", 1):
                return False

    return True
