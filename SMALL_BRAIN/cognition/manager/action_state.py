"""Live, read-only view of the robot's globally active actions."""

from __future__ import annotations

from actions.tracking.target_catalog import normalize_human_target


class ActionState:
    """Derive one consistent action context from the actual action objects."""

    def __init__(
        self,
        *,
        search,
        tracking,
        navigation,
        find_target,
        follow_person,
    ):
        self.search = search
        self.tracking = tracking
        self.navigation = navigation
        self.find_target = find_target
        self.follow_person = follow_person

    @property
    def person_tracking(self):
        return bool(
            self.tracking.active
            and normalize_human_target(self.tracking.target) is not None
        )

    @property
    def gesture_context(self):
        map_navigation = getattr(
            self.find_target, "map_navigation_action", None
        )
        find_navigation_active = bool(
            getattr(map_navigation, "active", False)
        )
        navigation_active = bool(
            self.navigation.active or find_navigation_active
        )
        movement_active = bool(
            self.follow_person.active
            or navigation_active
        )
        if self.follow_person.active:
            mode = "following"
            action = {
                "action_id": self.follow_person.action_id,
                "action_type": "follow_person",
                "target": self.follow_person.target,
                "state": "running",
            }
        elif self.navigation.active and self.navigation.mode == "approach":
            mode = "approaching"
            action = {
                "action_id": self.navigation.action_id,
                "action_type": "approach_action",
                "target": self.navigation.target,
                "state": "running",
            }
        elif self.navigation.active:
            mode = "navigating"
            command = self.navigation.target
            target = command
            if command == "go_to_room":
                room_id = self.navigation.room_id
                if room_id is not None:
                    target = f"room {room_id}"
            action = {
                "action_id": self.navigation.action_id,
                "action_type": "explicit_navigation",
                "target": target,
                "command": command,
                "state": "running",
            }
        elif getattr(map_navigation, "active", False):
            mode = "navigating"
            action = {
                "action_id": map_navigation.action_id,
                "action_type": "map_navigation",
                "target": map_navigation.target,
                "state": getattr(map_navigation, "stage", "running"),
            }
        elif self.search.active:
            mode = "searching"
            action = {
                "action_id": self.search.action_id,
                "action_type": "search_action",
                "target": self.search.target,
                "state": "running",
            }
        elif self.find_target.active:
            mode = "finding_target"
            action = {
                "action_id": self.find_target.action_id,
                "action_type": self.find_target.action_type,
                "target": self.find_target.target,
                "state": "running",
            }
        elif self.person_tracking:
            mode = "tracking_person"
            action = {
                "action_id": getattr(
                    self.tracking, "session_id", self.tracking.action_id
                ),
                "action_type": "track_action",
                "target": self.tracking.target,
                "state": "running",
            }
        elif self.tracking.active:
            mode = "tracking_target"
            action = {
                "action_id": getattr(
                    self.tracking, "session_id", self.tracking.action_id
                ),
                "action_type": "track_action",
                "target": self.tracking.target,
                "state": "running",
            }
        else:
            mode = "idle"
            action = {
                "action_id": None,
                "action_type": None,
                "target": None,
                "state": "idle",
            }
        return {
            "mode": mode,
            "action": action,
            "tracking_active": bool(self.tracking.active),
            "tracking_person": self.person_tracking,
            "following_active": bool(self.follow_person.active),
            "approach_active": bool(
                self.navigation.active and self.navigation.mode == "approach"
            ),
            "navigation_active": navigation_active,
            "movement_active": movement_active,
        }

    def snapshot(self, autonomy_task=None):
        active_actions = []
        for action_type, action in (
            ("search_action", self.search),
            (self.navigation.action_type, self.navigation),
            ("follow_person", self.follow_person),
        ):
            if action.active:
                active_actions.append({
                    "action_id": action.action_id,
                    "action_type": action_type,
                    "target": action.target,
                })

        if self.find_target.active:
            active_actions.append({
                "action_id": self.find_target.action_id,
                "action_type": self.find_target.action_type,
                "target": self.find_target.target,
            })

        return {
            "active_actions": active_actions,
            "tracking": {
                "active": bool(self.tracking.active),
                "session_id": getattr(
                    self.tracking, "session_id", self.tracking.action_id
                ),
                "target": self.tracking.target,
            },
            "autonomy_active": bool(
                autonomy_task is not None and not autonomy_task.done()
            ),
        }
