"""Reset accumulated BIG_BRAIN world state from one topic pulse."""

import rclpy
from nav2_msgs.srv import ClearEntireCostmap
from rclpy.node import Node
from slam_toolbox.srv import Reset
from std_msgs.msg import Empty


class ResetNode(Node):
    def __init__(self):
        super().__init__("big_brain_reset")
        self.slam = self.create_client(Reset, "/slam_toolbox/reset")
        self.local_costmap = self.create_client(
            ClearEntireCostmap,
            "/local_costmap/clear_entirely_local_costmap",
        )
        self.global_costmap = self.create_client(
            ClearEntireCostmap,
            "/global_costmap/clear_entirely_global_costmap",
        )
        self.create_subscription(Empty, "/big_brain/reset", self.reset, 10)
        self.get_logger().info("Listening for reset pulses on /big_brain/reset")

    def _call_if_available(self, name, client, request) -> None:
        if client.service_is_ready():
            client.call_async(request)
        else:
            self.get_logger().warning(f"Skipping unavailable {name} service")

    def reset(self, _message: Empty) -> None:
        self.get_logger().info("Resetting BIG_BRAIN state")
        slam_request = Reset.Request()
        slam_request.pause_new_measurements = False
        self._call_if_available("SLAM", self.slam, slam_request)
        self._call_if_available(
            "local costmap",
            self.local_costmap,
            ClearEntireCostmap.Request(),
        )
        self._call_if_available(
            "global costmap",
            self.global_costmap,
            ClearEntireCostmap.Request(),
        )


def main(args=None):
    rclpy.init(args=args)
    node = ResetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
