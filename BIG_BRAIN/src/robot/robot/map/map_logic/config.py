"""Configuration for map analysis and navigation."""

from dataclasses import dataclass


@dataclass
class LogicConfig:
    # Occupancy / costmap
    occupied: int = 50
    cost_limit: int = 99
    clearance: float = 0.10

    # ToF approach ray
    approach_ray_radius_m: float = 0.15
    approach_obstacle_min_cells: int = 3
    approach_obstacle_standoff_m: float = 0.25
    approach_cost_threshold: int = 253

    # Local navigation
    sample_radius: float = 3.0
    nudge_distance_m: float = 0.25
    candidate_min_separation_m: float = 0.30
    local_person_pose_max_age_seconds: float = 1.0

    # Frontiers
    frontier_hole_min_area: float = 0.5
    frontier_min_length: float = 0.30
    frontier_gain_radius: float = 1.0
    frontier_min_gain: float = 0.25
    frontier_min_separation: float = 0.75
    frontier_rays: int = 120
    frontier_samples_per_component: int = 5

    # Geometric exit-room navigation
    exit_open_clearance_m: float = 0.65
    exit_component_min_area_m2: float = 1.0
    exit_frontier_revisit_m: float = 0.60
    exit_component_lookup_radius_m: float = 1.50
    exit_wall_min_length_m: float = 1.00
    exit_recovery_max_moves: int = 3
    exit_recovery_min_move_m: float = 0.35
    exit_recovery_max_move_m: float = 1.00
    exit_recovery_revisit_m: float = 0.40
    exit_frontier_travel_penalty_m2_per_m: float = 0.15

    # Search views
    camera_center_pan_deg: float = 95.0
    camera_horizontal_fov_deg: float = 85.0
    camera_fov_max_range_m: float = 2.0
    context_radius_samples: int = 3
    context_bearing_fractions: tuple = (-0.75, 0.0, 0.75)
    exploration_step: float = 0.5
    exploration_headings: int = 8
    exploration_shortlist: int = 16
    exploration_ray_m: float = 4.0
    visibility_min_rays: int = 31
