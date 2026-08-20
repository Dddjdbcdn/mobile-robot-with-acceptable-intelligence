import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import sys, select, termios, tty
import math

msg = """
Control Your Pan/Tilt Camera!
---------------------------
Moving around:
        w    
   a    s    d

w/s : Tilt Up/Down
a/d : Pan Left/Right
r   : Reset to Center
q   : Quit

CTRL-C to quit
"""

class ServoTeleop(Node):
    def __init__(self):
        super().__init__('servo_teleop')
        self.pan_pub = self.create_publisher(Float32, '/stm32/servo_pan', 10)
        self.tilt_pub = self.create_publisher(Float32, '/stm32/servo_tilt', 10)

        # Servo Settings (Adjust these based on your actual servo limits)
        self.reset_pan_angle = 95.0
        self.reset_tilt_angle = 90.0
        
        self.pan_angle = self.reset_pan_angle
        self.tilt_angle = self.reset_tilt_angle
        self.step = 10.0  # Degrees to move per key press
        self.min_pan_angle = 30.0
        self.max_pan_angle = 160.0
        self.min_tilt_angle = 30.0
        self.max_tilt_angle = 120.0

    def publish_angles(self):
        pan_msg = Float32()
        tilt_msg = Float32()
        pan_msg.data = self.pan_angle
        tilt_msg.data = self.tilt_angle
        
        self.pan_pub.publish(pan_msg)
        self.tilt_pub.publish(tilt_msg)
        print(f"\rPan: {self.pan_angle}° | Tilt: {self.tilt_angle}°    ", end='')

def getKey(settings):
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def main():
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init()
    node = ServoTeleop()
    
    print(msg)
    node.publish_angles()

    try:
        while True:
            key = getKey(settings)
            
            if key == 'w':
                node.tilt_angle = max(node.min_tilt_angle, node.tilt_angle - node.step)
            elif key == 's':
                node.tilt_angle = min(node.max_tilt_angle, node.tilt_angle + node.step)
            elif key == 'a':
                node.pan_angle = min(node.max_pan_angle, node.pan_angle + node.step)
            elif key == 'd':
                node.pan_angle = max(node.min_pan_angle, node.pan_angle - node.step)
            elif key == 'r':
                node.pan_angle = node.reset_pan_angle
                node.tilt_angle = node.reset_tilt_angle
            elif key == 'q' or key == '\x03': # CTRL+C
                break
                
            node.publish_angles()

    except Exception as e:
        print(e)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()