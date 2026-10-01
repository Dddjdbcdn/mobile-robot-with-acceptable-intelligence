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
        approach,
        explicit_navigation,
        find_target,
        follow_person,
    ):
        self.search = search
        self.tracking = tracking
        self.approach = approach
        self.explicit_navigation = explicit_navigation
        self.find_target = find_target
        self.follow_person = follow_person

    @property
    def person_tracking(self):
        return bool(
            self.tracking.active
            and normalize_human_target(self.tracking.target) is not None
        )

    @property
    def camera_owner(self):
        if self.follow_person.active:
            return "follow_person"
        if self.find_target.active:
            return self.find_target.action_type
        if (
            self.explicit_navigation is not None
            and self.explicit_navigation.active
        ):
            return "explicit_navigation"
        if self.search.active:
            return "search_action"
        if self.tracking.active:
            return "target_tracking"
        if self.approach.active:
            return "approach_action"
        return None

    @property
    def gesture_context(self):
        if self.follow_person.active:
            mode = "following"
        elif self.approach.active:
            mode = "approaching"
        elif self.person_tracking:
            mode = "tracking_person"
        else:
            mode = "watching_person"
        return {
            "mode": mode,
            "tracking_active": bool(self.tracking.active),
            "tracking_person": self.person_tracking,
            "following_active": bool(self.follow_person.active),
            "approach_active": bool(self.approach.active),
            "navigation_active": bool(
                self.explicit_navigation is not None
                and self.explicit_navigation.active
            ),
        }

    def snapshot(self, autonomy_task=None):
        active_actions = []
        for action_type, action in (
            ("search_action", self.search),
            ("track_action", self.tracking),
            ("approach_action", self.approach),
            ("explicit_navigation", self.explicit_navigation),
            ("follow_person", self.follow_person),
        ):
            if action is not None and getattr(action, "active", False):
                active_actions.append({
                    "action_id": getattr(action, "action_id", None),
                    "action_type": action_type,
                    "target": getattr(action, "target", None),
                })

        if self.find_target.active:
            active_actions.append({
                "action_id": self.find_target.action_id,
                "action_type": self.find_target.action_type,
                "target": self.find_target.target,
            })

        return {
            "active_actions": active_actions,
            "autonomy_active": bool(
                autonomy_task is not None and not autonomy_task.done()
            ),
        }
