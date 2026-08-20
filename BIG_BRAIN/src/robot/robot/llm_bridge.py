import math
import threading
import json
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, TwistStamped
from std_msgs.msg import String
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu, Range, PointCloud2
from nav2_msgs.action import NavigateToPose
import zmq
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs
import time
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy

class CameraServo():
    def __init__(self, pan_pub, tilt_pub):
        self.servo_pan_pub = pan_pub
        self.servo_tilt_pub = tilt_pub
        self.reset_pan_angle = 95.0
        self.reset_tilt_angle = 90.0
        
        self.pan_angle = self.reset_pan_angle
        self.tilt_angle = self.reset_tilt_angle 
        
        self.min_pan_angle = 30.0
        self.max_pan_angle = 160.0
        self.min_tilt_angle = 30.0
        self.max_tilt_angle = 120.0

        self.delta_pan_angle = 0.0
        self.delta_tilt_angle = 0.0
        self.Kp = 0.1

        self.deadband_degrees = 0.5
        self.max_delta_angle = 90.0
        self.max_step_degrees = 5

    def publish_servo_command(self,tracking=False):
        remaining_pan_angle = 0.0

        if abs(self.delta_pan_angle) < self.deadband_degrees:
            self.delta_pan_angle = 0.0
        if abs(self.delta_tilt_angle) < self.deadband_degrees:
            self.delta_tilt_angle = 0.0

        step_pan = min(self.delta_pan_angle, self.max_delta_angle) if self.delta_pan_angle > 0 else max(self.delta_pan_angle, -self.max_delta_angle)
        step_tilt = min(self.delta_tilt_angle, self.max_delta_angle) if self.delta_tilt_angle > 0 else max(self.delta_tilt_angle, -self.max_delta_angle)

        if tracking:
            step_pan = self.delta_pan_angle * self.Kp
            step_tilt = self.delta_tilt_angle * self.Kp

            step_pan = max(min(step_pan, self.max_step_degrees), -self.max_step_degrees)
            step_tilt = max(min(step_tilt, self.max_step_degrees), -self.max_step_degrees)

        self.pan_angle = self.pan_angle + step_pan
        self.tilt_angle = self.tilt_angle + step_tilt

        if self.pan_angle < self.min_pan_angle: 
            remaining_pan_angle = self.pan_angle - self.min_pan_angle
            self.pan_angle = self.min_pan_angle

        if self.pan_angle > self.max_pan_angle: 
            remaining_pan_angle = self.pan_angle - self.max_pan_angle
            self.pan_angle = self.max_pan_angle

        if self.tilt_angle < self.min_tilt_angle: 
            self.tilt_angle = self.min_tilt_angle

        if self.tilt_angle > self.max_tilt_angle: 
            self.tilt_angle = self.max_tilt_angle

        pan_msg = Float32()
        tilt_msg = Float32()
        reset_tilt_msg = Float32()

        pan_msg.data = self.pan_angle
        tilt_msg.data = self.tilt_angle
        reset_tilt_msg.data = self.reset_tilt_angle

        self.servo_tilt_pub.publish(tilt_msg)
        self.servo_pan_pub.publish(pan_msg)

        if not tracking and (step_pan != self.delta_pan_angle or step_tilt != self.delta_tilt_angle):
            self.delta_pan_angle = self.delta_pan_angle - step_pan
            self.delta_tilt_angle = self.delta_tilt_angle - step_tilt
            time.sleep(0.5)
            self.publish_servo_command()
        else:
            self.delta_pan_angle = 0.0
            self.delta_tilt_angle = 0.0

        return remaining_pan_angle
        

class LLMRosBridge(Node):
    def __init__(self):
        super().__init__('llm_ros_bridge')
        self.get_logger().info("Starting DJ-ROS Bridge (Async Mode)...")

        self.zmq_context = zmq.Context()
        
        self.rep_socket = self.zmq_context.socket(zmq.REP)
        self.rep_socket.bind("tcp://*:5555")

        self.pub_socket = self.zmq_context.socket(zmq.PUB)
        self.pub_socket.bind("tcp://*:5556")

        self.sub_socket = self.zmq_context.socket(zmq.SUB)
        self.sub_socket.bind("tcp://*:5557")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        self.pub_lock = threading.Lock()

        self.cmd_pub = self.create_publisher(TwistStamped, '/diff_drive_controller/cmd_vel', 10)
        self.servo_pan_pub = self.create_publisher(Float32, '/stm32/servo_pan', 10)
        self.servo_tilt_pub = self.create_publisher(Float32, '/stm32/servo_tilt', 10)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.camera_tof_range = 0.0
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Range, '/camera_tof', self.camera_tof_callback, qos)
        self.create_subscription(String, '/yolo/detections', self.yolo_callback, 10)

        self.state_timer = self.create_timer(0.05, self.state_pub_loop)

        self.current_nav_goal_handle = None
        self.blind_move_timer = None
        self.track_object_timer = None
        self.navigation_active = False
        self.max_tracking_time = 10.0

        self.zmq_thread = threading.Thread(target=self.listen_for_llm, daemon=True)
        self.zmq_thread.start()

        self.background_listener_thread = threading.Thread(target=self.background_listener, daemon=True)
        self.background_listener_thread.start()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.servo = CameraServo(self.servo_pan_pub,self.servo_tilt_pub)

    def publish_cmd(self, linear_x, angular_z):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.linear.x = float(linear_x)
        msg.twist.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def state_pub_loop(self):
        with self.pub_lock:
            self.pub_socket.send_json(
                {
                "type": "state",
                "camera_tof_range": self.camera_tof_range,
                "servo_pan_angle": self.servo.pan_angle,
                "servo_tilt_angle": self.servo.tilt_angle
                }
            )

    def camera_tof_callback(self, msg):
        self.camera_tof_range = msg.range

    def yolo_callback(self, msg):
        try:
            payload = json.loads(msg.data)

            with self.pub_lock:
                self.pub_socket.send_json({
                    "type": "yolo",
                    **payload
                })

        except Exception as e:
            self.get_logger().error(
                f"Failed to forward YOLO detections: {e}"
            )

    def send_telemetry(self, event_name, status_msg):
        with self.pub_lock:
            payload = {
                "type": "event",
                "event": event_name, 
                "status": status_msg}
            self.pub_socket.send_json(payload)
            self.get_logger().info(f"Broadcasted to LLM: {payload}")

    def blind_move_loop(self,lin_vel,ang_vel,fwd_dur,rot_dur,start_time,send_telemetry=False):
        elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
        
        if elapsed >= max(fwd_dur,rot_dur):
            self.publish_cmd(0.0, 0.0)
            self.blind_move_timer.cancel()
            self.blind_move_timer = None
            if send_telemetry: self.send_telemetry("blind_move", "completed")
            return

        lin = lin_vel if elapsed < fwd_dur else 0.0
        ang = ang_vel if elapsed < rot_dur else 0.0
        self.publish_cmd(lin, ang)

    def nav_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.navigation_active = False
            self.send_telemetry("navigation", "Goal Rejected by Nav2")
            return
        
        self.current_nav_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.nav_result_cb)

    def nav_result_cb(self, future):
        status = future.result().status
        self.navigation_active = False
        self.current_nav_goal_handle = None
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.send_telemetry("navigation", "Goal Reached")
        else:
            self.send_telemetry("navigation", f"Failed or Canceled (Status {status})")

    def stop_all_motion(self):
        if self.blind_move_timer:
            self.blind_move_timer.cancel()
            self.blind_move_timer = None
        if self.current_nav_goal_handle:
            self.current_nav_goal_handle.cancel_goal_async()
            self.current_nav_goal_handle = None

        self.publish_cmd(0.0, 0.0)

    def track_object_loop(self, start_time):
        remaining_pan_angle = self.servo.publish_servo_command(
            tracking=True
        )

        if self.navigation_active: return

        remaining_rad = math.radians(remaining_pan_angle)

        body_kp = 100.0
        ang_vel = body_kp * remaining_rad

        max_ang_vel = 1.0
        ang_vel = max(min(ang_vel, max_ang_vel),-max_ang_vel)

        self.publish_cmd(0.0, ang_vel)

    def background_listener(self):
        while rclpy.ok():
            try:
                message = self.sub_socket.recv_json()
                self.servo.delta_pan_angle = message.get("delta_pan_angle", 0.0)
                self.servo.delta_tilt_angle = message.get("delta_tilt_angle", 0.0)
            except Exception as e:
                self.get_logger().error(f"Internal Loop Error: {e}")

    def shutdown(self):
        self.get_logger().info("Shutting down LLM ROS Bridge...")

        if self.blind_move_timer is not None:
            self.blind_move_timer.cancel()

        if self.track_object_timer is not None:
            self.track_object_timer.cancel()

        self.publish_cmd(0.0, 0.0)

        self.rep_socket.close(linger=0)
        self.pub_socket.close(linger=0)
        self.sub_socket.close(linger=0)

        self.zmq_context.term()
        
    def listen_for_llm(self):
        while rclpy.ok():
            try:
                request = self.rep_socket.recv_json()
                cmd = request.get("command")

                if cmd == 'move_camera':
                    self.servo.delta_pan_angle = float(request.get("delta_pan_angle", 0.0))
                    self.servo.delta_tilt_angle = float(request.get("delta_tilt_angle", 0.0))

                    self.servo.publish_servo_command(tracking=False)
                    self.rep_socket.send_json(
                        {
                            "status": "accepted",
                            "pan_angle": self.servo.pan_angle,
                            "tilt_angle": self.servo.tilt_angle,
                        }
                    )
                
                elif cmd == "blind_move":
                    self.stop_all_motion()
                    
                    lin_vel = float(request.get("linear_velocity", 0.0))
                    dist = float(request.get("distance", 0.0))
                    ang_vel = float(request.get("angular_velocity", 0.0))
                    angle = float(request.get("angle", 0.0))
                    fwd_dur = abs(dist / lin_vel) if lin_vel != 0 else 0.0
                    rot_dur = abs(angle / ang_vel) if ang_vel != 0 else 0.0
                    start_time = self.get_clock().now()
        
                    self.blind_move_timer = self.create_timer(0.05, lambda: self.blind_move_loop(lin_vel, ang_vel, fwd_dur, rot_dur, start_time, True))
                    self.rep_socket.send_json({"status": "accepted", "message": "Blind move started"})

                elif cmd == "track_object":
                    start_time = self.get_clock().now()
                    self.track_object_timer = self.create_timer(0.05, lambda: self.track_object_loop(start_time))
                    self.rep_socket.send_json({"status": "accepted", "message": "Object is being tracked"})

                elif cmd == "stop_tracking_object":
                    if self.track_object_timer is not None:
                        self.track_object_timer.cancel()
                        self.track_object_timer = None

                    self.publish_cmd(0.0, 0.0)

                    self.servo.delta_pan_angle = self.servo.reset_pan_angle - self.servo.pan_angle
                    self.servo.delta_tilt_angle = self.servo.reset_tilt_angle - self.servo.tilt_angle

                    self.servo.publish_servo_command(tracking=False)

                    self.rep_socket.send_json({"status": "accepted", "message": "Object tracking is stopped"})

                elif cmd == "navigate_to_pose":
                    self.stop_all_motion()

                    self.navigation_active = True

                    x = float(request.get("x", 0.0))
                    y = float(request.get("y", 0.0))
                    angle = float(request.get("angle", 0.0))

                    pose_in = PoseStamped()
                    pose_in.header.frame_id = 'base_footprint' 
                    pose_in.header.stamp = self.get_clock().now().to_msg()
                    
                    pose_in.pose.position.x = x
                    pose_in.pose.position.y = y
                    pose_in.pose.position.z = 0.0
                    
                    # Apply the local yaw angle
                    pose_in.pose.orientation.z = math.sin(angle / 2.0)
                    pose_in.pose.orientation.w = math.cos(angle / 2.0)

                    try:
                        timeout = rclpy.duration.Duration(seconds=0.1)
                        pose_map = self.tf_buffer.transform(pose_in, 'map', timeout=timeout)
                    except Exception as e:
                        self.rep_socket.send_json({"status": "error", "message": f"TF Transform failed: {e}"})
                        continue

                    goal = NavigateToPose.Goal()
                    goal.pose = pose_map
                    
                    if not self.nav_client.wait_for_server(timeout_sec=1.0):
                        self.rep_socket.send_json({"status": "error", "message": "Nav2 not available"})
                        continue

                    send_goal_future = self.nav_client.send_goal_async(goal)
                    send_goal_future.add_done_callback(self.nav_goal_response_cb)

                    self.rep_socket.send_json({"status": "accepted", "message": "Nav2 Goal Dispatched"})
                
                elif cmd == "stop_motors":
                    self.stop_all_motion()
                    self.rep_socket.send_json({"status": "accepted", "message": "All motion stopped"})

                else:
                    self.rep_socket.send_json({"status": "error", "message": "Unknown command"})

            except Exception as e:
                self.get_logger().error(f"Internal Loop Error: {e}")
                try:
                    self.rep_socket.send_json({"status": "error", "message": f"Internal exception: {str(e)}"})
                except Exception as zmq_e:
                    self.get_logger().error(f"Could not recover ZMQ state: {zmq_e}")

def main(args=None):
    rclpy.init(args=args)
    node = LLMRosBridge()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("Ctrl-C received")

    finally:
        node.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()