---
title: DevOps Incident Response OpenEnv
emoji: 🚨
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
license: apache-2.0
tags:
  - openenv
  - devops
  - sre
  - incident-response
  - reinforcement-learning
  - kubernetes
---

# 🚨 DevOps Incident Response OpenEnv

> A production-grade OpenEnv environment for training and evaluating AI agents on real-world DevOps / SRE incident response workflows.

---

## Overview & Motivation

Modern SRE teams face high-pressure, time-sensitive incidents where every minute of downtime costs money and user trust. This environment simulates three realistic production incidents — from a simple CPU spike to a multi-service cascading failure — and challenges an agent to diagnose root causes and apply remediation actions in the correct order.

Unlike toy grid-world environments, every task here mirrors patterns directly observed in production Kubernetes deployments:

- Runaway processes starving a gateway CPU.
- Memory leaks combined with database connection pool exhaustion.
- Bad ConfigMap deployments cascading through service dependency chains.

The goal: build agents that can outperform or assist on-call engineers under time pressure.

---

## Action Space

| Type | Description |
|------|-------------|
| Text | A single shell or `kubectl` command string |

**Examples:**
```
kubectl top pod -n production
kubectl rollout restart deployment/api-gateway -n production
kill -9 4821
psql -c 'SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE duration > interval ''30 seconds'';'
kubectl rollout undo deployment/config-service -n production
circuit breaker reset payment-service
postmortem incident-2024-01 config-service resolved
```

Good actions are diagnostic (e.g. `kubectl logs`) or targeted remediation (`kill -9 <pid>`). Destructive commands such as `kubectl delete namespace production` are penalised.

---

## Observation Space

| Type | Description |
|------|-------------|
| Text | Structured plain-text snapshot of the current incident state |

Each observation contains:
- Task ID, difficulty, severity, and description.
- Active alert list (PagerDuty-style).
- Per-service table: CPU %, Memory %, Error rate, P99 latency, Pod readiness.
- Acquired progress flags (e.g. `process_killed_or_hpa_scaled`).
- Step counter vs. max steps.

---

## Task Descriptions

### Task 1 — `easy_cpu_spike` (Easy, 15 steps)

**Severity:** HIGH

The `api-gateway` is experiencing a CPU spike above 95%, driven by a runaway `log-aggregator` process (PID 4821). P99 latency has jumped to 2 400 ms.

**Goal:** Kill the runaway process and/or enable HPA to bring CPU below 60% and error rate below 2%.

**Key actions:** `kubectl top pod`, `kill -9 4821`, `kubectl autoscale`

---

### Task 2 — `medium_memory_db` (Medium, 20 steps)

**Severity:** CRITICAL

The `order-service` is OOMKilled repeatedly (memory at 98%) due to an unbounded result set query. Simultaneously, `postgres-primary` has exhausted its connection pool (99/100) with 43 slow queries, causing 8 500 ms latency.

**Goal:** Raise memory limits, kill slow queries, add a DB index, configure PgBouncer, and restore error rate below 5%.

**Key actions:** `kubectl patch` (memory limit), `pg_terminate_backend`, `CREATE INDEX CONCURRENTLY`, `configure pgbouncer`

---

### Task 3 — `hard_cascading_failure` (Hard, 30 steps)

**Severity:** CRITICAL · Revenue impact ~$12 000/min

A bad ConfigMap (`app-config-v7`) deployment caused `config-service` to crash (0/2 replicas), which cascaded through `user-service` (0/3 replicas), `payment-service` (circuit breaker OPEN, 76% error rate), and `notification-service`. The ingress is rate-limiting to 10 rps.

**Goal:** Roll back the ConfigMap, restart services in dependency order, reset the circuit breaker, lift rate limiting, and file a postmortem.

**Key actions:** `kubectl rollout undo`, `kubectl rollout status`, `circuit breaker reset`, ingress annotation removal, `postmortem`

---

## Reward Structure

| Component | Value |
|-----------|-------|
| Per-step progress (CPU/MEM/ERR/LAT improvement) | 0.0 – 0.10 |
| Completion bonus (all criteria met) | +0.40 |
| Step cost | –0.005/step |
| Harmful action penalty | –0.15 |
| Unrecognised no-op penalty | –0.02 |
| **Total range** | **[0.0, 1.0]** |

---

## Setup & Usage

### Local Development

```bash
# 1. Clone repository
git clone https://huggingface.co/spaces/<your-username>/devops-incident-response-openenv
cd devops-incident-response-openenv

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server (PYTHONPATH must include repo root)
PYTHONPATH=. python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# 5. Verify health
curl http://localhost:7860/health
# → {"status":"ok"}
```

### Docker

```bash
# Build
docker build -t devops-openenv .

# Run
docker run -p 7860:7860 devops-openenv

# Test
curl http://localhost:7860/health
```

### API Quick-start

```bash
# Reset to easy task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_cpu_spike"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "kubectl top pod -n production"}'

# Check state
curl http://localhost:7860/state

# List tasks
curl http://localhost:7860/tasks
```

### Running the Baseline Agent

```bash
# Rule-based fallback (no API key needed)
PYTHONPATH=. python inference.py easy_cpu_spike

# All tasks
PYTHONPATH=. python inference.py all

# LLM-powered agent
export HF_TOKEN=hf_xxx
export LLM_BASE_URL=https://api-inference.huggingface.co/v1
PYTHONPATH=. python inference.py medium_memory_db
```

### OpenEnv Validation

```bash
openenv validate --url http://localhost:7860
```

---

## Baseline Performance

Results from the rule-based fallback agent (deterministic, no LLM):

| Task | Steps Used | Total Reward | Resolved |
|------|-----------|-------------|---------|
| easy_cpu_spike | 8 / 15 | 0.61 | ✅ |
| medium_memory_db | 10 / 20 | 0.54 | ✅ |
| hard_cascading_failure | 12 / 30 | 0.49 | ✅ |

An LLM-powered agent (Llama-3-8B-Instruct via HF Inference) achieves similar success rates with slightly fewer steps on the easy task.

---

## Project Structure

```
project-root/
├── server/
│   ├── __init__.py
│   └── app.py          # FastAPI app, all endpoints
├── env.py              # Singleton DevOpsEnv, state transitions
├── grader.py           # Deterministic reward computation
├── models.py           # Pydantic models
├── tasks.py            # Task registry (3 tasks)
├── inference.py        # Baseline agent (OpenAI client + rule-based fallback)
├── Dockerfile          # Hugging Face Space docker build
├── requirements.txt
├── openenv.yaml        # OpenEnv manifest
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## License

Apache 2.0
