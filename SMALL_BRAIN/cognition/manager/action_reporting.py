"""Compact, LLM-facing action result reporting."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass

from actions.action_result import ActionResult


def serialize(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            key: serialize(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


def compact_result(result):
    """Keep model context actionable; retain full results only internally."""
    if not isinstance(result, ActionResult):
        return serialize(result)

    compact = {
        "action_type": result.action_type,
        "status": result.status,
    }
    for key, value in (
        ("target", result.target),
        ("reason_code", result.reason_code),
    ):
        if value is not None:
            compact[key] = value
    if result.outcome is not None and (
        result.status == "succeeded" or result.reason_code is None
    ):
        compact["outcome"] = result.outcome
    if result.retryable:
        compact["retryable"] = True

    data = result.data if isinstance(result.data, dict) else {}
    for key in ("goal", "direction", "failed_step", "camera_owner"):
        value = data.get(key)
        if value is not None:
            compact[key] = value

    step_result = data.get("step_result")
    if isinstance(step_result, dict):
        step_data = step_result.get("data")
        if isinstance(step_data, dict):
            for key in ("message", "error", "error_type", "error_stage"):
                if step_data.get(key):
                    compact[key] = step_data[key]
        step_reason = step_result.get("reason_code")
        if step_reason and step_reason != result.reason_code:
            compact["step_reason_code"] = step_reason
    elif data.get("message"):
        compact["message"] = data["message"]
    elif data.get("error"):
        compact["error"] = data["error"]

    guidance = data.get("continuation_guidance")
    if isinstance(guidance, dict):
        retry_answers = guidance.get("retry_answers")
        compact_guidance = {}
        if guidance.get("unresolved_action_type"):
            compact_guidance["action"] = guidance["unresolved_action_type"]
        if isinstance(retry_answers, dict):
            compact_guidance["retry"] = retry_answers
        if guidance.get("out_of_frame_requires_no_retry"):
            compact_guidance["out_of_frame_stops"] = True
        if compact_guidance:
            compact["guidance"] = compact_guidance

    if "verified" in data:
        compact["verified"] = bool(data["verified"])
    return compact


def decision_instruction(result):
    if (
        isinstance(result, ActionResult)
        and result.reason_code in {
            "APPROACH_VERIFICATION_REQUIRED",
            "APPROACH_VERIFICATION_TIMEOUT",
            "APPROACH_RANGE_INVALID",
            "STABLE_SEED_TIMEOUT",
            "TARGET_NOT_TRACKED_AFTER_NAVIGATION",
        }
    ):
        return (
            "Briefly report that approach was not safely confirmed. Ask whether "
            "to continue carefully or reposition the target; retry only if asked."
        )
    if (
        isinstance(result, ActionResult)
        and result.reason_code
        in {"PERSON_DETECTION_FAILED", "OBJECT_DETECTION_FAILED"}
        and result.data.get("user_confirms_visible") is True
    ):
        return (
            "Report that direct detection failed. Do not retry or move without "
            "new user guidance."
        )
    if (
        isinstance(result, ActionResult)
        and result.reason_code == "SEARCH_GUIDANCE_REQUIRED"
    ):
        action_type = result.data.get(
            "continuation_guidance", {}
        ).get("unresolved_action_type", result.action_type)
        target = result.target
        if result.data.get("direction") == "behind":
            choices = "high, low, elsewhere out of frame, or best effort"
            retry_rule = "Do not retry behind or elsewhere out of frame."
        else:
            choices = (
                "high, low, behind DJ, elsewhere out of frame, or best effort"
            )
            retry_rule = "Do not retry if it is elsewhere out of frame."
        return (
            f"Ask one short question: is {target!r} {choices}? Retry "
            f"{action_type} using the matching guidance. {retry_rule}"
        )
    if isinstance(result, ActionResult) and result.action_type == "watch_target":
        if result.status == "succeeded":
            return (
                "The robot is now continuously watching the target. Confirm this "
                "briefly and do not call another tool."
            )
        return "Report the watch-target failure briefly."
    if (
        isinstance(result, ActionResult)
        and result.action_type in {"navigate_action", "explicit_navigation"}
    ):
        if result.status == "succeeded":
            return (
                "Navigation completed. Report completion briefly. Do not call "
                "explicit_navigation again unless the user gives a new movement goal."
            )
        return (
            "Report the navigation reason briefly. Do not substitute blind motion. "
            "Retry only if new sensor evidence or user guidance materially changes "
            "the situation."
        )
    if isinstance(result, ActionResult) and result.action_type == "find_target":
        if result.status != "succeeded":
            return (
                "The autonomous object/place search did not complete. Report the "
                "reason and checked-space summary briefly. Do not retry unless the "
                "user supplies new information or explicitly asks."
            )
        return (
            "The robot completed the requested find goal. An object was approached "
            "and reacquired, or a place was reached with its one-shot contextual "
            "move. Report completion briefly and do not call another tool."
        )
    if isinstance(result, ActionResult) and result.action_type == "follow_person":
        if result.status == "cancelled":
            return "Briefly confirm that following has stopped."
        return (
            "Report why continuous following ended. Do not restart follow_person "
            "unless the user explicitly asks."
        )
    return (
        "Continue the unresolved goal if needed; do not repeat the finished action."
    )


def action_envelope(event_type, result, action_state, decision=None):
    envelope = {
        "event_type": event_type,
        "result": compact_result(result),
        "decision_instruction": (
            decision if decision is not None else decision_instruction(result)
        ),
    }
    if action_state["active_actions"] or action_state["autonomy_active"]:
        envelope["decision_state"] = action_state
    return envelope

