import json
import os

def load_json(file):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: Json file not found!")
        return {}

def save_json(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=4)

def update_memory(memory_file,category, new_info):
    memory = load_json(memory_file)
    
    if category in ["facts", "identity updates"]:
        if new_info not in memory[category]:
            memory[category].append(new_info)
            save_json(memory_file,memory)
            return f"Successfully saved to {category}: {new_info}"
        return f"I already have that in my {category}."
    return "Invalid memory category."

def _format_section(title, items):
    """Format a list of strings into readable lines."""
    if isinstance(items, list):
        return "\n".join(f"- {item}" for item in items)
    return str(items)

def build_system_prompt(identity, memory):
    """Combines the static identity and dynamic memory into DJ's system prompt."""

    sections = []

    sections.append("=== WHO YOU ARE ===")
    sections.append(_format_section("Identity", identity.get("identity", [])))

    sections.append("\n=== YOUR CREATOR ===")
    sections.append(_format_section("Owner", identity.get("owner", [])))

    sections.append("\n=== HOW YOU ARE ===")
    sections.append(_format_section("Character", identity.get("character", [])))

    sections.append("\n=== HARD RULES (NEVER BREAK THESE) ===")
    sections.append(_format_section("Rules", identity["rules"]))

    has_memory = any(v for v in memory.values() if v)
    if has_memory:
        sections.append("\n=== WHAT YOU REMEMBER ===")
        for key, value in memory.items():
            if isinstance(value, list) and value:
                joined = "; ".join(value)
                sections.append(f"{key}: {joined}")

    sections.append("\nUse the 'update_memory' tool to save new facts or things you learn about your owner.")

    sections.append(
        """
=== ROBOT ACTION CONTROL ===
- You control the robot with atomic, interruptible actions and decide the next action after each result.
- A result with event_type=command_result and result.status=running means the action started; it does not mean the physical goal finished.
- Wait for a matching [ROBOT ACTION EVENT] with event_type=action_finished and the same action_id before relying on completion.
- Re-evaluate the user's current goal after every terminal action event. You may act, speak, wait, recover differently, or stop.
- Use result.outcome, result.reason_code, result.retryable, result.data, robot_state, and constraints as evidence.
- Do not repeat an identical failed action without changing the viewpoint, parameters, or evidence.
- A retryable not_found search may justify rotating or moving to a new viewpoint and searching again; do not retry indefinitely.
- Call approach_action only when robot_state.constraints.can_approach is true and the tracked target matches the intended target.
- New user instructions take priority. Stop an incompatible running action before starting the replacement.
- Never claim physical success from a started/running event.
""".strip()
    )

    return "\n".join(sections)
