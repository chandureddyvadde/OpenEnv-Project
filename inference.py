"""
inference.py – Baseline agent for DevOps Incident Response OpenEnv.

Uses the OpenAI Python client pointed at either:
  • A local server (default: http://localhost:7860)
  • An external Hugging Face Inference Endpoint / OpenAI-compatible API

Credentials are read from environment variables:
  HF_TOKEN   – Hugging Face token (used as bearer token for HF Inference)
  OPENAI_API_KEY – alternative OpenAI key
  API_BASE_URL   – override the base URL (default: http://localhost:7860)
  TASK_ID        – which task to run (default: easy_cpu_spike)
  MODEL_NAME     – model to use (default: meta-llama/Llama-3-8b-instruct)

If the model call fails, a rule-based fallback policy is used so that
the script always completes and produces reproducible step logs.
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:7860")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", HF_TOKEN)
MODEL_NAME: str = os.environ.get("MODEL_NAME", "meta-llama/Llama-3-8b-instruct")
TASK_ID: str = os.environ.get("TASK_ID", "easy_cpu_spike")
MAX_RETRIES: int = 2

# For OpenAI client – target local env server (not the LLM API)
ENV_BASE = API_BASE_URL.rstrip("/")

# ---------------------------------------------------------------------------
# Optional OpenAI client import
# ---------------------------------------------------------------------------

try:
    from openai import OpenAI  # type: ignore

    _llm_client = OpenAI(
        api_key=OPENAI_API_KEY or "dummy",
        base_url=os.environ.get("LLM_BASE_URL", "https://api-inference.huggingface.co/v1"),
    )
    _HAS_OPENAI = True
except ImportError:
    _llm_client = None
    _HAS_OPENAI = False
    print("[inference] openai package not installed; using rule-based fallback only.")


# ---------------------------------------------------------------------------
# Rule-based fallback policy
# ---------------------------------------------------------------------------

_RULE_BASED_POLICIES: Dict[str, List[str]] = {
    "easy_cpu_spike": [
        "kubectl top pod -n production",
        "kubectl describe pod -n production -l app=api-gateway",
        "kubectl logs -n production -l app=api-gateway --tail=100",
        "kill -9 4821",  # known runaway PID from task description
        "kubectl rollout restart deployment/api-gateway -n production",
        "kubectl autoscale deployment api-gateway --min=3 --max=10 --cpu-percent=70 -n production",
        "kubectl top pod -n production",
        "silence alert CRITICAL api-gateway",
    ],
    "medium_memory_db": [
        "kubectl top pod -n production",
        "kubectl describe pod -n production -l app=order-service",
        "kubectl logs -n production -l app=order-service --tail=200",
        "kubectl patch deployment order-service -n production --patch '{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"order-service\",\"resources\":{\"limits\":{\"memory\":\"1Gi\"}}}]}}}}'",
        "psql -c 'SELECT pid, query, state FROM pg_stat_activity WHERE state != ''idle'' ORDER BY duration DESC LIMIT 10;'",
        "psql -c 'SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE duration > interval ''30 seconds'' AND state = ''active'';'",
        "psql -c 'CREATE INDEX CONCURRENTLY idx_orders_created_at ON orders(created_at);'",
        "kubectl rollout restart deployment/order-service -n production",
        "configure pgbouncer max_client_conn=200 pool_size=25",
        "kubectl get pod -n production",
    ],
    "hard_cascading_failure": [
        "kubectl get events -n production --sort-by='.lastTimestamp' | tail -20",
        "kubectl describe pod -n production -l app=config-service",
        "kubectl rollout undo deployment/config-service -n production",
        "kubectl rollout status deployment/config-service -n production",
        "kubectl rollout restart deployment/user-service -n production",
        "kubectl rollout status deployment/user-service -n production",
        "kubectl rollout restart deployment/payment-service -n production",
        "circuit breaker reset payment-service",
        "kubectl rollout restart deployment/notification-service -n production",
        "kubectl annotate ingress main-ingress nginx.ingress.kubernetes.io/limit-rps='' --overwrite -n production",
        "kubectl get pod -n production",
        "postmortem incident-2024-01 config-service bad configmap rollback resolved",
    ],
}


def _rule_based_action(task_id: str, step: int, observation: str) -> str:
    """Return the next action from the rule-based policy."""
    policy = _RULE_BASED_POLICIES.get(task_id, _RULE_BASED_POLICIES["easy_cpu_spike"])
    if step < len(policy):
        return policy[step]
    return "kubectl get pod -n production"


# ---------------------------------------------------------------------------
# LLM-based action selection
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) responding to a production incident.

Your job is to diagnose and resolve the incident described in the observation.

Rules:
- Issue ONE shell/kubectl command per turn.
- Prefer safe, targeted commands (e.g. kubectl rollout restart, kill specific PID).
- Never run destructive commands like 'kubectl delete namespace', 'rm -rf', or 'drop table'.
- Respond with ONLY the command, nothing else – no explanation, no markdown fences.
- Work systematically: diagnose first, then remediate, then verify.
"""


def _llm_action(observation: str, history: List[Dict[str, str]]) -> Optional[str]:
    """Call the LLM and return the action string, or None on failure."""
    if not _HAS_OPENAI or not _llm_client:
        return None
    if not OPENAI_API_KEY or OPENAI_API_KEY == "dummy":
        return None

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-10:])  # keep last 10 turns for context
    messages.append({"role": "user", "content": observation})

    try:
        resp = _llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=128,
            temperature=0.0,
        )
        action = resp.choices[0].message.content.strip()
        # Strip markdown fences if model added them
        action = re.sub(r"^```[a-z]*\n?", "", action)
        action = re.sub(r"\n?```$", "", action)
        return action.strip() or None
    except Exception as exc:
        print(f"  [llm] call failed: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Environment HTTP client
# ---------------------------------------------------------------------------

def _env_post(endpoint: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{ENV_BASE}/{endpoint.lstrip('/')}"
    resp = requests.post(url, json=payload or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _env_get(endpoint: str) -> Dict[str, Any]:
    url = f"{ENV_BASE}/{endpoint.lstrip('/')}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

def run_episode(task_id: str = TASK_ID) -> Dict[str, Any]:
    """Run a single episode and return summary statistics."""
    print(f"\n{'='*70}")
    print(f"  DevOps Incident Response OpenEnv – Baseline Agent")
    print(f"  Task: {task_id}")
    print(f"  Model: {MODEL_NAME if (_HAS_OPENAI and OPENAI_API_KEY and OPENAI_API_KEY != 'dummy') else 'rule-based fallback'}")
    print(f"{'='*70}\n")

    # Health check
    try:
        health = _env_get("/health")
        assert health.get("status") == "ok", f"Unexpected health response: {health}"
        print("✓ Health check passed\n")
    except Exception as exc:
        print(f"✗ Health check failed: {exc}")
        sys.exit(1)

    # Reset
    reset_result = _env_post("/reset", {"task_id": task_id})
    observation: str = reset_result.get("observation", "")
    max_steps: int = reset_result.get("max_steps", 20)

    print(observation)
    print(f"\n--- Starting episode (max_steps={max_steps}) ---\n")

    history: List[Dict[str, str]] = []
    total_reward = 0.0
    steps_taken = 0
    done = False

    for step_num in range(max_steps):
        # Select action
        action = _llm_action(observation, history)
        if action is None:
            action = _rule_based_action(task_id, step_num, observation)
            print(f"[Step {step_num+1:02d}] [FALLBACK] > {action}")
        else:
            print(f"[Step {step_num+1:02d}] [LLM    ] > {action}")

        # Execute action
        try:
            step_result = _env_post("/step", {"action": action})
        except Exception as exc:
            print(f"  ✗ Step request failed: {exc}")
            break

        obs_new: str = step_result.get("observation", "")
        reward: float = step_result.get("reward", 0.0)
        done = step_result.get("done", False)
        info: Dict[str, Any] = step_result.get("info", {})

        total_reward += reward
        steps_taken += 1

        print(f"         reward={reward:.4f}  done={done}  flags={info.get('acquired_flags', [])}")

        # Update history for LLM context
        history.append({"role": "user", "content": observation})
        history.append({"role": "assistant", "content": action})

        observation = obs_new

        if done:
            print(f"\n[Episode complete at step {step_num+1}]")
            break

        time.sleep(0.1)  # be polite to the server

    # Final observation
    print("\n--- Final State ---")
    print(observation)

    summary = {
        "task_id": task_id,
        "steps_taken": steps_taken,
        "total_reward": round(total_reward, 4),
        "mean_reward": round(total_reward / max(steps_taken, 1), 4),
        "done": done,
        "success": info.get("success", False) if steps_taken > 0 else False,
    }

    print(f"\n{'='*70}")
    print(f"  EPISODE SUMMARY")
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")
    print(f"{'='*70}\n")

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else TASK_ID
    results = []
    for t in [task] if task != "all" else ["easy_cpu_spike", "medium_memory_db", "hard_cascading_failure"]:
        result = run_episode(task_id=t)
        results.append(result)
        time.sleep(1)

    if len(results) > 1:
        avg_reward = sum(r["total_reward"] for r in results) / len(results)
        print(f"Overall mean episode reward: {avg_reward:.4f}")
