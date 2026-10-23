#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>

#include "behaviortree_cpp/action_node.h"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2/utils.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "nav2_msgs/srv/get_cost.hpp"

namespace custom_nav2_plugins
{

class DJGoalValidator : public BT::SyncActionNode
{
public:
  DJGoalValidator(const std::string & name, const BT::NodeConfig & config)
  : BT::SyncActionNode(name, config)
  {
    node = config.blackboard->get<rclcpp::Node::SharedPtr>("node");
    global_cost_client = node->create_client<nav2_msgs::srv::GetCost>("/global_costmap/get_cost_global_costmap");
    goal_pub = node->create_publisher<geometry_msgs::msg::PoseStamped>("/accepted_goal_pose",rclcpp::QoS(10));
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::BidirectionalPort<geometry_msgs::msg::PoseStamped>("goal"),
      BT::InputPort<geometry_msgs::msg::PoseStamped>("robot_pose"),
      BT::InputPort<int>("patience", 2, "Number of failed checks before replacing goal"),
    };
  }

  BT::NodeStatus tick() override
  {
    geometry_msgs::msg::PoseStamped goal, robot_pose;
    int patience;

    if (!getInput("goal", goal) || !getInput("robot_pose", robot_pose)) return BT::NodeStatus::FAILURE;
    getInput("patience", patience);

    const auto goal_cost = sampleCost(global_cost_client, goal, true);

    if (!goal_cost.success || goal_cost.cost == 253.0 || goal_cost.cost == 254.0) { 

      ++failure_count;
      RCLCPP_WARN(node->get_logger(), "[DJGoalValidator]: 🥊 Goal is inside obstacle with cost %f! Failure count: %d",goal_cost.cost, failure_count);

      if (failure_count >= patience) {

        geometry_msgs::msg::PoseStamped new_goal = chooseGoalTowardRobot(
          goal, robot_pose, 0.1);
        setOutput("goal", new_goal); 
        goal_pub->publish(new_goal);

        RCLCPP_WARN(node->get_logger(), "[DJGoalValidator]: ⭐️ New goal published at position: x = %.3f,y = %.3f!", 
        new_goal.pose.position.x,new_goal.pose.position.y);

        failure_count = 0;
      }
    }

    else failure_count = 0;
    
    goal_pub->publish(goal);
    return BT::NodeStatus::SUCCESS; 
  }

private:
  struct CostSample
  {
    bool success{false};
    double cost{0.0};
  };
  rclcpp::Node::SharedPtr node;
  rclcpp::Client<nav2_msgs::srv::GetCost>::SharedPtr global_cost_client;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub;
  int failure_count = 0;

  CostSample sampleCost(
    const rclcpp::Client<nav2_msgs::srv::GetCost>::SharedPtr & client,
    const geometry_msgs::msg::PoseStamped & pose,
    bool use_footprint = true)
  {
    CostSample result;

    if (!client->wait_for_service(std::chrono::milliseconds(50))) return result;

    auto req =
      std::make_shared<nav2_msgs::srv::GetCost::Request>();

    req->x = static_cast<float>(pose.pose.position.x);
    req->y = static_cast<float>(pose.pose.position.y);
    req->theta = static_cast<float>(tf2::getYaw(pose.pose.orientation));
    req->use_footprint = use_footprint;

    auto future = client->async_send_request(req);
    if (rclcpp::spin_until_future_complete(node,future,std::chrono::milliseconds(200))!= rclcpp::FutureReturnCode::SUCCESS) return result;

    const auto response = future.get();

    result.success = true;
    result.cost = response->cost;

    return result;
  }

  geometry_msgs::msg::PoseStamped chooseGoalTowardRobot(
    const geometry_msgs::msg::PoseStamped & goal,
    const geometry_msgs::msg::PoseStamped & robot_pose,
    const double distance_step)
  {
    const double dx = robot_pose.pose.position.x - goal.pose.position.x;
    const double dy = robot_pose.pose.position.y - goal.pose.position.y;
    const double robot_distance_to_goal = std::hypot(dx, dy);
    if (robot_distance_to_goal <= 1e-6 || distance_step <= 0.0) {
      return robot_pose;
    }

    const double goal_to_robot_heading = std::atan2(dy, dx);
    for (double offset = distance_step;
      offset <= robot_distance_to_goal + 1e-6;
      offset += distance_step)
    {
      const double sampled_offset = std::min(offset, robot_distance_to_goal);
      geometry_msgs::msg::PoseStamped candidate;
      candidate.header = goal.header;
      candidate.header.stamp = node->now();
      candidate.pose.position.x =
        goal.pose.position.x + sampled_offset * std::cos(goal_to_robot_heading);
      candidate.pose.position.y =
        goal.pose.position.y + sampled_offset * std::sin(goal_to_robot_heading);

      const double candidate_yaw = std::atan2(
        goal.pose.position.y - candidate.pose.position.y,
        goal.pose.position.x - candidate.pose.position.x);
      tf2::Quaternion q;
      q.setRPY(0.0, 0.0, candidate_yaw);
      q.normalize();
      candidate.pose.orientation = tf2::toMsg(q);

      const auto candidate_cost = sampleCost(global_cost_client, candidate, true);
      if (candidate_cost.success && candidate_cost.cost < 253.0) {
        return candidate;
      }
    }

    RCLCPP_WARN(
      node->get_logger(),
      "[DJGoalValidator]: 🥊 No valid replacement goal found between original goal and robot");
    return robot_pose;
  }
};
}

#include "behaviortree_cpp/bt_factory.h"

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<custom_nav2_plugins::DJGoalValidator>(
    "DJGoalValidator");
}
