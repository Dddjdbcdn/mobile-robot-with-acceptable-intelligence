import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int16MultiArray, Header, Float32
from geometry_msgs.msg import Point32
from sensor_msgs.msg import Imu, Range, PointCloud2
from sensor_msgs_py import point_cloud2
import math
from rclpy.duration import Duration
from rclpy.time import Time

class Stm32SensorBridge(Node):
    def __init__(self):
        super().__init__('stm32_sensor_bridge')
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        # --- IMU SETUP ---
        self.imu_subscription = self.create_subscription(
            Point32, 
            'stm32/imu_msg', 
            self.imu_callback, 
            qos
        )
        self.imu_publisher = self.create_publisher(Imu, '/imu', qos)
        
        # --- ULTRASONIC SETUP ---
        self.ultrasonic_subscription = self.create_subscription(
            Point32,
            'stm32/ultrasonic_msg',
            self.listener_callback,
            qos
        )
        self.pub_left = self.create_publisher(Range, 'ultrasonic/left', qos)
        self.pub_center = self.create_publisher(Range, 'camera_tof', qos)
        self.pub_right = self.create_publisher(Range, 'ultrasonic/right', qos)

        # --- TOF (VL53L7CX) SETUP ---
        self.tof_subscription = self.create_subscription(
            Int16MultiArray,
            'stm32/tof_raw_data',
            self.tof_callback,
            qos
        )
        self.tof_pc_publisher = self.create_publisher(PointCloud2, '/tof_pointcloud', qos)

        self.tof_cx = 3.5
        self.tof_cy = 3.9 # calibrated

        self.compute_ray_directions()

    def compute_ray_directions(self):
        self.ray_dirs = []
        fov_rad = math.radians(60.0) 
        sensor_width = 2.0 * math.tan(fov_rad / 2.0)
        
        for i in range(64):
            x_idx = i % 8
            y_idx = i // 8
            
            x_dir = 1.0
            y_dir = (self.tof_cx - x_idx) * (sensor_width / 8.0)
            z_dir = -(self.tof_cy - y_idx) * (sensor_width / 8.0)
            
            self.ray_dirs.append((x_dir, y_dir, z_dir))

    # ==========================================
    # TOF FUNCTIONS
    # ==========================================
    def tof_callback(self, msg):
        # Validate data size
        if len(msg.data) != 64:
            self.get_logger().warn(f"Expected 64 ToF values, got {len(msg.data)}")
            return
            
        points = []
        
        for i, dist_mm in enumerate(msg.data):
            if dist_mm <= 0: continue

            dist_m = dist_mm / 1000.0
            x_dir, y_dir, z_dir = self.ray_dirs[i]
            
            # Calculate the 3D point
            x = dist_m * x_dir
            y = dist_m * y_dir
            z = dist_m * z_dir
            
            points.append([x, y, z])
            
        # Create header
        header = Header()
        header.stamp = Time(seconds=0, nanoseconds=0).to_msg()
        header.frame_id = 'tof_link'
        
        # Generate and publish PointCloud2 message
        pc2_msg = point_cloud2.create_cloud_xyz32(header, points)
        self.tof_pc_publisher.publish(pc2_msg)

    # ==========================================
    # IMU FUNCTIONS
    # ==========================================
    def imu_callback(self, msg):
        imu_msg = Imu()
        
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_link'

        imu_msg.linear_acceleration.x = float(msg.x) * 9.81
        imu_msg.linear_acceleration.y = float(msg.y) * 9.81
        imu_msg.linear_acceleration.z = 9.81 
        
        imu_msg.angular_velocity.x = 0.0
        imu_msg.angular_velocity.y = 0.0
        imu_msg.angular_velocity.z = float(msg.z) * 3.141592653 / 180

        imu_msg.linear_acceleration_covariance[0] = 0.01 # Variance for Ax
        imu_msg.linear_acceleration_covariance[4] = 0.01 # Variance for Ay
        imu_msg.angular_velocity_covariance[8] = 0.01    # Variance for Gz (Vyaw)
        
        # Mark orientation as unknown
        imu_msg.orientation_covariance[0] = -1.0 

        self.imu_publisher.publish(imu_msg)

    # ==========================================
    # ULTRASONIC FUNCTIONS
    # ==========================================
    def listener_callback(self, msg):
        self.publish_ultrasonic_range(self.pub_left, msg.x, 'ultrasonic_left_link')
        self.publish_tof_range(self.pub_center, msg.y, 'camera_tof_link')
        self.publish_ultrasonic_range(self.pub_right, msg.z, 'ultrasonic_right_link')

    def publish_ultrasonic_range(self, publisher, distance, frame_id):
        range_msg = Range()
        range_msg.header.stamp = self.get_clock().now().to_msg()
        range_msg.header.frame_id = frame_id
        range_msg.radiation_type = Range.ULTRASOUND
        range_msg.field_of_view = 0.2
        range_msg.min_range = 0.01    
        range_msg.max_range = 0.3

        if distance <= 0.0 or distance > range_msg.max_range:
            range_msg.range = range_msg.max_range 
        else:
            range_msg.range = float(distance)
            
        publisher.publish(range_msg)

    def publish_tof_range(self, publisher, distance, frame_id):
        range_msg = Range()
        range_msg.header.stamp = self.get_clock().now().to_msg()
        range_msg.header.frame_id = frame_id
        range_msg.radiation_type = Range.INFRARED
        range_msg.field_of_view = 0.2
        range_msg.min_range = 0.01    
        range_msg.max_range = 12.0
        
        if distance <= 0.0 or distance > range_msg.max_range:
            range_msg.range = range_msg.max_range
        else:
            range_msg.range = float(distance)
            
        publisher.publish(range_msg)

def main(args=None):
    rclpy.init(args=args)
    node = Stm32SensorBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()