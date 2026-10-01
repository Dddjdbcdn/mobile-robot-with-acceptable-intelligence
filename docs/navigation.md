# Navigation actions

Navigation is split into three small actions.

## Explicit navigation

`explicit_navigation` accepts one `local_command`:

- nudges: forward, backward, left, or right;
- rotations: left, right, or around;
- furthest reachable point forward or backward;
- middle of local open space;
- previous position; or
- face the continuously tracked person;
- stop when an exit crossing is identified;
- cross into and register another room;
- return to Room 1; or
- navigate to a known numbered room.

The realtime model never selects coordinates or a marker from a rendered map.
`BIG_BRAIN/src/robot/robot/map/map_logic.py` resolves the named command against the current
reachable occupancy grid and the bridge sends the resulting pose to Nav2.

Map logic treats sufficiently open reachable regions separated by narrow
clearance bands as geometric chambers. The first observed chamber is retained
as `Room 1`. Room-core labels expand through reachable floor by path distance,
so a frontier can belong to a distant part of the same room but cannot be
assigned through a wall; meaningful competing room cores divide doorway and
corridor space between them. `exit_room` visits unprocessed frontiers belonging
to the current room, then stops as soon as the `cross_room_boundary` phase and
its crossing goal are available. It deliberately does not dispatch that final
goal.
After each frontier arrival, the next room decision waits for an occupancy-grid
message received after Nav2 reported success. Re-analyzing an older map is not
treated as a map refresh.
If startup noise leaves no usable room core, or exploration temporarily leaves
no frontier or adjacent-room evidence, the same action may perform up to three
short map-stabilization moves. Each endpoint is already connected and safe in
the current costmap; moves are limited to 0.35--1.0 metres, avoid prior recovery
points, and prefer progress toward an unvisited frontier when one exists.
When several current-room frontiers are available, selection maximizes expected
unknown-area gain minus a modest travel penalty (0.15 square metres of gain per
metre travelled). Distance therefore resolves similar choices without forcing
the robot toward a nearby low-value pocket instead of a more informative open
boundary.

`go_to_another_room` runs the same exploration, dispatches the crossing goal,
and succeeds after a fresh map places the robot in a different open component.
That component receives the next stable runtime ID (`Room 2`, `Room 3`, and so
on), and its connection to the origin room is recorded. `go_to_room` sends
Nav2 to a known room's stored anchor; `return_to_initial_place` is the explicit
shortcut for Room 1. Multi-pass traversal is limited to 12 movements. Room
identity is geometric and lasts for the bridge process; it is not a semantic
room-name classifier or disk-backed map annotation.

## Watching and automatic hand-guided navigation

`watch_target` runs the find-target acquisition flow with approach disabled,
then leaves camera tracking active for the requested person or object.

While a person is tracked, `HandGestureInterface` automatically runs MediaPipe
on a dynamic crop around the YOLO right wrist. Welcome and push poses
temporarily aim the camera at the palm; every other pose immediately resumes
the current person target. A hand ToF seed is accepted only when it remains
close to the lidar person pose, and validated hand and person seeds are both
forwarded to lidar tracking. Welcome followed by fingers-up approaches the
saved hand seed. Push followed by fingers-down returns to and faces the person.
Recognized commands enter CognitionManager as ordinary `ToolRequest` events
with LLM and voice reporting disabled, so they use the same arbitration and
lifecycle path as spoken commands.

## Find-loop map navigation

`MapNavigationAction` is internal to `find_target`. It requests one map crop,
uses the out-of-band selector when a pose choice is needed, executes at most one
Nav2 goal, and returns control to the find loop.

Context candidates are sampled within the observation's 85-degree cone and
include `CF`, the furthest reachable point on its center ray. Context mode never
offers a frontier. Exploration remains bounded by the find loop and may use its
own coverage candidate and furthest forward extension. When map logic offers
only one exploration pose, it is dispatched directly with its sampled viewing
heading; vision selection is used only when there is a choice.

`stop_navigation` cancels explicit navigation or the active approach. Approach
and follow cancellation leave their independent person-tracking action running;
stopping watch-target tracking ends that tracking action.
`stop_goal` cancels the find loop and its internal map navigation.
