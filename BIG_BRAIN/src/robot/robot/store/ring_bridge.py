import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
import asyncio
from bleak import BleakClient, BleakScanner

# These MUST match the Arduino code exactly
CHARACTERISTIC_UUID = "19b10001-e8f2-537e-4f3c-d72f28a30000"
DEVICE_NAME = "Xiao-Ring"

class RingBridgeNode(Node):
    def __init__(self):
        super().__init__('ring_bridge_node')
        # Create a publisher that broadcasts to the stamped velocity topic
        self.publisher_ = self.create_publisher(TwistStamped, '/diff_drive_controller/cmd_vel', 10)
        self.get_logger().info("Ring Bridge Node Started. Waiting for BLE data...")

    def publish_command(self, command_byte):
        msg = TwistStamped()
        
        # Populate the Header with current ROS time and frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link' # Change this if your TF tree uses a different base frame
        
        # Translate the byte into robot velocity speeds
        # Notice how the vectors are now nested inside '.twist'
        if command_byte == 1:
            msg.twist.linear.x = 0.5   # Forward
        elif command_byte == 2:
            msg.twist.linear.x = -0.5  # Backward
        elif command_byte == 3:
            msg.twist.angular.z = 1.0  # Turn Left
        elif command_byte == 4:
            msg.twist.angular.z = -1.0 # Turn Right
        else:
            msg.twist.linear.x = 0.0   # Stop
            msg.twist.angular.z = 0.0
            
        self.publisher_.publish(msg)

async def run_ble_loop(args=None, node=None):
    # This is the async loop that handles Bluetooth
    print(f"Scanning for {DEVICE_NAME}...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME)

    if not device:
        print(f"Could not find {DEVICE_NAME}. Is the ring powered on?")
        return

    print(f"Found {DEVICE_NAME} at {device.address}. Connecting...")
    
    # Connect to the Ring and keep the connection open
    async with BleakClient(device) as client:
        print("Connected to Ring! Translating gestures to /diff_drive_controller/cmd_vel...")
        
        # Subscribe to the custom characteristic
        def notification_handler(sender, data):
            command = data[0] # Extract the first byte from the BLE packet
            node.publish_command(command)

        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        
        # Keep the ROS 2 node spinning so it can publish messages
        try:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
                await asyncio.sleep(0.01)
        except KeyboardInterrupt:
            print("Disconnecting...")

def main(args=None):
    # This is the synchronous entry point ROS 2 expects
    rclpy.init(args=args)
    node = RingBridgeNode()
    
    try:
        # We manually trigger the async loop from inside the sync function
        asyncio.run(run_ble_loop(args, node))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()