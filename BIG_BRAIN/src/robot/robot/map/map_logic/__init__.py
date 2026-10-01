"""Compact facade for map analysis, navigation, and search planning."""

from .analysis import MapAnalysisMixin
from .approach import ApproachNavigationMixin
from .common import GeometryMixin
from .config import LogicConfig
from .grid import Grid, yaw_of
from .local_navigation import LocalNavigationMixin
from .room_navigation import RoomNavigationMixin
from .search import SearchPlanningMixin
from .visibility import VisibilityMixin


class MapLogic(
    MapAnalysisMixin,
    ApproachNavigationMixin,
    LocalNavigationMixin,
    RoomNavigationMixin,
    SearchPlanningMixin,
    VisibilityMixin,
    GeometryMixin,
):
    """Unified compatibility facade over focused map-logic components."""

    def __init__(self, config=None):
        self.cfg = config if config is not None else LogicConfig()


__all__ = ["Grid", "LogicConfig", "MapLogic", "yaw_of"]

