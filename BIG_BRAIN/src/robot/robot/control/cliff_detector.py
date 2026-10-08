"""Convert downward ToF returns into obstacle points for Nav2."""

import math
from typing import Iterable, Sequence

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener


GRID_WIDTH = 8
GRID_HEIGHT = 8
TOF_VERTICAL_CENTER = 3.9
TOF_FIELD_OF_VIEW = math.radians(60.0)
TOF_RAY_STEP = 2.0 * math.tan(TOF_FIELD_OF_VIEW / 2.0) / GRID_WIDTH
OBSTACLE_OUTPUT_HEIGHT = 0.1
CLIFF_X_STANDOFF = 0.1
INVALID_MIN_ADJACENT_COLUMNS = 2
INVALID_CONFIRMATION_FRAMES = 3


def rotate_and_translate(
    point: Sequence[float], transform: TransformStamped
) -> tuple[float, float, float]:
    """Apply a geometry_msgs transform without requiring tf2_geometry_msgs."""
    x, y, z = point
    q = transform.transform.rotation
    t = transform.transform.translation

    # Unit-quaternion rotation matrix.
    xx, yy, zz = q.x * q.x, q.y * q.y, q.z * q.z
    xy, xz, yz = q.x * q.y, q.x * q.z, q.y * q.z
    wx, wy, wz = q.w * q.x, q.w * q.y, q.w * q.z
    return (
        (1.0 - 2.0 * (yy + zz)) * x + 2.0 * (xy - wz) * y
        + 2.0 * (xz + wy) * z + t.x,
        2.0 * (xy + wz) * x + (1.0 - 2.0 * (xx + zz)) * y
        + 2.0 * (yz - wx) * z + t.y,
        2.0 * (xz - wy) * x + 2.0 * (yz + wx) * y
        + (1.0 - 2.0 * (xx + yy)) * z + t.z,
    )


def set_base_height_in_source_frame(
    point: Sequence[float],
    height: float,
    transform: TransformStamped,
    x_standoff: float = 0.0,
) -> tuple[float, float, float]:
    """Apply a base-frame X standoff and height, then return to the source frame."""
    base_x, base_y, _ = rotate_and_translate(point, transform)
    t = transform.transform.translation
    q = transform.transform.rotation
    dx = base_x - x_standoff - t.x
    dy = base_y - t.y
    dz = height - t.z

    # Apply the inverse rotation (the transpose of the unit-quaternion matrix).
    xx, yy, zz = q.x * q.x, q.y * q.y, q.z * q.z
    xy, xz, yz = q.x * q.y, q.x * q.z, q.y * q.z
    wx, wy, wz = q.w * q.x, q.w * q.y, q.w * q.z
    return (
        (1.0 - 2.0 * (yy + zz)) * dx + 2.0 * (xy + wz) * dy
        + 2.0 * (xz - wy) * dz,
        2.0 * (xy - wz) * dx + (1.0 - 2.0 * (xx + zz)) * dy
        + 2.0 * (yz + wx) * dz,
        2.0 * (xz + wy) * dx + 2.0 * (yz - wx) * dy
        + (1.0 - 2.0 * (xx + yy)) * dz,
    )


def bottom_row_points(
    points: Sequence[Sequence[float]], row_count: int
) -> Iterable[Sequence[float]]:
    """Return pixels from the physically lowest rows of the bridge's grid."""
    count = max(0, min(row_count, GRID_HEIGHT)) * GRID_WIDTH
    return points[:count]


def invalid_ray_floor_intersection(
    point_index: int,
    ray_x: float,
    ray_y: float,
    transform: TransformStamped,
) -> tuple[float, float, float] | None:
    """Project an invalid ToF pixel along its ray to the base Z=0 floor."""
    row = point_index // GRID_WIDTH
    ray_z = -(TOF_VERTICAL_CENTER - row) * TOF_RAY_STEP

    origin_in_base = rotate_and_translate((0.0, 0.0, 0.0), transform)
    ray_in_base = rotate_and_translate((ray_x, ray_y, ray_z), transform)
    direction_z_in_base = ray_in_base[2] - origin_in_base[2]

    if direction_z_in_base >= 0.0:
        return None

    scale = -origin_in_base[2] / direction_z_in_base
    if scale <= 0.0 or not math.isfinite(scale):
        return None

    return (scale * ray_x, scale * ray_y, scale * ray_z)


def spatially_supported_invalid_indices(indices: set[int]) -> set[int]:
    """Keep invalid pixels only when invalidity spans adjacent grid columns."""
    columns = {index % GRID_WIDTH for index in indices}
    supported_columns: set[int] = set()
    for start in range(GRID_WIDTH - INVALID_MIN_ADJACENT_COLUMNS + 1):
        run = set(range(start, start + INVALID_MIN_ADJACENT_COLUMNS))
        if run.issubset(columns):
            supported_columns.update(run)
    return {
        index for index in indices if index % GRID_WIDTH in supported_columns
    }


class CliffDetector(Node):
    def __init__(self) -> None:
        """Set up the ToF subscription, TF listener, and cliff publisher."""
        super().__init__('cliff_detector')

        self.declare_parameter('input_topic', '/tof_pointcloud')
        self.declare_parameter('output_topic', '/cliff_layer')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('lowest_rows', 2)
        self.declare_parameter('cliff_z_threshold', -0.1)
        self.declare_parameter('transform_timeout', 0.1)

        self.base_frame = str(self.get_parameter('base_frame').value)
        self.lowest_rows = int(self.get_parameter('lowest_rows').value)
        self.cliff_z_threshold = float(
            self.get_parameter('cliff_z_threshold').value
        )
        self.transform_timeout = float(
            self.get_parameter('transform_timeout').value
        )

        output_topic = str(self.get_parameter('output_topic').value)
        input_topic = str(self.get_parameter('input_topic').value)
        self.publisher = self.create_publisher(
            PointCloud2, output_topic, qos_profile_sensor_data
        )
        self.subscription = self.create_subscription(
            PointCloud2,
            input_topic,
            self.pointcloud_callback,
            qos_profile_sensor_data,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.invalid_streaks = [0] * (GRID_WIDTH * GRID_HEIGHT)

    def pointcloud_callback(self, message: PointCloud2) -> None:
        expected_points = GRID_WIDTH * GRID_HEIGHT
        if message.width * message.height != expected_points:
            self.get_logger().warning(
                f'Expected an organized 8x8 ToF cloud; got '
                f'{message.width}x{message.height}. The bridge must preserve '
                'invalid pixels.',
                throttle_duration_sec=5.0,
            )
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                message.header.frame_id,
                Time.from_msg(message.header.stamp),
                timeout=Duration(seconds=self.transform_timeout),
            )
        except TransformException as error:
            self.get_logger().warning(
                f'Cannot transform ToF cloud from {message.header.frame_id} '
                f'to {self.base_frame}: {error}',
                throttle_duration_sec=5.0,
            )
            return

        points = list(
            point_cloud2.read_points(
                message,
                field_names=('x', 'y', 'z'),
                skip_nans=False,
            )
        )
        obstacles: list[tuple[float, float, float]] = []
        invalid_indices: set[int] = set()
        selected_points = list(bottom_row_points(points, self.lowest_rows))

        for point_index, point in enumerate(selected_points):
            x, y, z = (float(point[0]), float(point[1]), float(point[2]))
            valid = math.isfinite(x) and math.isfinite(y) and math.isfinite(z)

            if valid:
                point_in_base = rotate_and_translate((x, y, z), transform)
                if point_in_base[2] >= self.cliff_z_threshold:
                    continue
                obstacle = (x, y, z)
            else:
                if not math.isfinite(x) or not math.isfinite(y):
                    continue
                invalid_indices.add(point_index)
                continue

            obstacles.append(
                set_base_height_in_source_frame(
                    obstacle,
                    OBSTACLE_OUTPUT_HEIGHT,
                    transform,
                    CLIFF_X_STANDOFF,
                )
            )

        supported_invalid = spatially_supported_invalid_indices(invalid_indices)
        for point_index in range(len(selected_points)):
            if point_index in supported_invalid:
                self.invalid_streaks[point_index] += 1
            else:
                self.invalid_streaks[point_index] = 0

            if self.invalid_streaks[point_index] < INVALID_CONFIRMATION_FRAMES:
                continue

            point = selected_points[point_index]
            obstacle = invalid_ray_floor_intersection(
                point_index,
                float(point[0]),
                float(point[1]),
                transform,
            )
            if obstacle is None:
                continue
            obstacles.append(
                set_base_height_in_source_frame(
                    obstacle,
                    OBSTACLE_OUTPUT_HEIGHT,
                    transform,
                    CLIFF_X_STANDOFF,
                )
            )

        header = Header()
        header.stamp = message.header.stamp
        header.frame_id = message.header.frame_id
        self.publisher.publish(point_cloud2.create_cloud_xyz32(header, obstacles))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CliffDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
