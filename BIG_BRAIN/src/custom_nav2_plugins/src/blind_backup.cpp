#include <string>
#include <memory>
#include "behaviortree_cpp/action_node.h"
#include "rclcpp/rclcpp.hpp"
// 1. Updated include to TwistStamped
#include "geometry_msgs/msg/twist_stamped.hpp" 

namespace custom_nav2_plugins
{

class BlindBackUp : public BT::StatefulActionNode
{
public:
  BlindBackUp(const std::string& name, const BT::NodeConfig& config)
    : BT::StatefulActionNode(name, config),
      logger_(rclcpp::get_logger("BlindBackUp")) // Initialize default logger
  {
    auto node = config.blackboard->get<rclcpp::Node::SharedPtr>("node");
    
    // Update logger to use the parent node's logger for consistent formatting
    logger_ = node->get_logger(); 
    
    // 2. Updated publisher type
    vel_pub_ = node->create_publisher<geometry_msgs::msg::TwistStamped>("cmd_vel_nav", 10);
    clock_ = node->get_clock();

    RCLCPP_INFO(logger_, "BlindBackUp BT Node initialized");
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<double>("backup_dist", 0.30, "Distance to backup"),
      BT::InputPort<double>("backup_speed", 0.15, "Speed of backup")
    };
  }

  BT::NodeStatus onStart() override
  {
    double dist, speed;
    getInput("backup_dist", dist);
    getInput("backup_speed", speed);

    backup_duration_ = rclcpp::Duration::from_seconds(dist / speed);
    start_time_ = clock_->now();

    // Log the start of the action with its parameters
    RCLCPP_INFO(logger_, "BlindBackUp started! Backing up %.2f meters at %.2f m/s for %.2f seconds.", 
                dist, speed, backup_duration_.seconds());

    // 3. Set the frame_id once
    twist_msg_.header.frame_id = "base_footprint";

    // 4. Update stamp and nest velocities under .twist
    twist_msg_.header.stamp = clock_->now();
    twist_msg_.twist.linear.x = -std::abs(speed);
    twist_msg_.twist.angular.z = 0.0;

    vel_pub_->publish(twist_msg_);
    return BT::NodeStatus::RUNNING;
  }

  BT::NodeStatus onRunning() override
  {
    auto elapsed_time = clock_->now() - start_time_;

    if (elapsed_time >= backup_duration_) {
      RCLCPP_INFO(logger_, "BlindBackUp completed successfully. Stopping robot.");
      stopRobot();
      return BT::NodeStatus::SUCCESS;
    }

    // CRITICAL: Always update the timestamp before publishing in the loop!
    twist_msg_.header.stamp = clock_->now();
    vel_pub_->publish(twist_msg_);
    
    return BT::NodeStatus::RUNNING;
  }

  void onHalted() override
  {
    RCLCPP_WARN(logger_, "BlindBackUp halted prematurely! Stopping robot.");
    stopRobot();
  }

private:
  void stopRobot()
  {
    // Update stamp for the stop command too
    twist_msg_.header.stamp = clock_->now();
    twist_msg_.twist.linear.x = 0.0;
    twist_msg_.twist.angular.z = 0.0;
    vel_pub_->publish(twist_msg_);
  }

  // 5. Updated member variables
  rclcpp::Logger logger_; // Added logger member variable
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr vel_pub_;
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Time start_time_;
  rclcpp::Duration backup_duration_{0, 0};
  geometry_msgs::msg::TwistStamped twist_msg_;
};

}  // namespace custom_nav2_plugins

#include "behaviortree_cpp/bt_factory.h"
BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<custom_nav2_plugins::BlindBackUp>("BlindBackUp");
}