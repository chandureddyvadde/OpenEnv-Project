"""
Pydantic models for DevOps Incident Response OpenEnv.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# State models
# ---------------------------------------------------------------------------

class ServiceState(BaseModel):
    name: str
    status: ServiceStatus = ServiceStatus.HEALTHY
    cpu_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    memory_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    replicas_desired: int = Field(default=1, ge=0)
    replicas_ready: int = Field(default=1, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IncidentState(BaseModel):
    task_id: str
    difficulty: Difficulty
    severity: Severity
    description: str
    step_count: int = 0
    max_steps: int = 20
    done: bool = False
    resolved: bool = False
    services: List[ServiceState] = Field(default_factory=list)
    alerts: List[str] = Field(default_factory=list)
    actions_taken: List[str] = Field(default_factory=list)
    flags: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class ResetResult(BaseModel):
    observation: str
    task_id: str
    difficulty: str
    max_steps: int
    info: Dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    action: str = Field(..., description="A DevOps command string, e.g. 'kubectl rollout restart deployment/api'")


class StepResult(BaseModel):
    observation: str
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
