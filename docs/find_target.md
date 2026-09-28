# Find-target action

`find_target(target, target_kind)` handles objects and visually recognizable
places. It intentionally does not accept people or abstract relations such as
“middle of the room” or “between a chair and a table.”

- `object`: search, track, approach, and reacquire the object.
- `place`: search for the place, then perform one final map-and-vision context
  navigation from the confirmed frame.

Person acquisition remains an internal capability shared by
automatic hand guidance, `watch_target`, `follow_person`, and startup initialization. It is
not exposed as a `find_target` target kind.

Moving next to a named object belongs to this action because approach is already
part of the object flow.

Search assessment has one contextual category: `context`. The old local versus
destination split is gone. A contextual frame receives up to four movements.
Each movement sends its 85-degree observation cone, sampled reachable poses,
and the furthest reachable center-ray pose to `MapNavigationAction`; no frontier
is included. `visual` candidates are verified as possible targets, while
`speculative` evidence does not trigger contextual motion.

When there is no useful clue, exploration ranks uncovered known-free views. The
whole goal remains bounded to 20 exploration waypoints, 50 total physical
waypoints, and 600 seconds by default.

The lidar person tracker runs continuously. Visual ToF seeds refine or hand off
the association, but ending a camera tracking lease does not erase the lidar
track. The bridge publishes the latest transformed person pose with robot state
so human-related commands can react without starting a fresh person search.
