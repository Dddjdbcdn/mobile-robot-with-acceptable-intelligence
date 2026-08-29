from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ActionResult:
    """Structured lifecycle event shared by every robot action."""

    action_id: str
    action_type: str
    status: str # running / succeeded / failed / cancelled
    target: str | None = None
    outcome: str | None = None
    reason_code: str | None = None
    retryable: bool = False
    data: dict[str, Any] = field(default_factory=dict)
