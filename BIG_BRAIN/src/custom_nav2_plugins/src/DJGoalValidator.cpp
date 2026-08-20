#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

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

    if (!goal_cost.success || goal_cost.cost == 254.0) { 

      ++failure_count;
      RCLCPP_WARN(node->get_logger(), "[DJGoalValidator]: 🥊 Goal is inside obstacle with cost %f! Failure count: %d",goal_cost.cost, failure_count);

      if (failure_count >= patience) {

        geometry_msgs::msg::PoseStamped new_goal = chooseGoalBySampling(goal,robot_pose,0.1,1.0,0.1,M_PI/4);
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
  struct Candidate
  {
    geometry_msgs::msg::PoseStamped world_pose;
    double radius;
    double angle;
    bool is_valid{true};
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

  std::vector<Candidate> generateCandidates(
  const geometry_msgs::msg::PoseStamped & robot_pose,const geometry_msgs::msg::PoseStamped & goal,
  const double min_radius,const double max_radius,const double radius_step, const double angle_range)
  {
    const double angle_step = M_PI/12;

    std::vector<Candidate> candidates;

    const double goal_to_robot_heading = std::atan2(
      robot_pose.pose.position.y - goal.pose.position.y,
      robot_pose.pose.position.x - goal.pose.position.x);

    for (double r = min_radius; r <= max_radius + 1e-6; r += radius_step) {
      for (double a = -angle_range; a <= angle_range + 1e-6; a += angle_step)
      {
        Candidate candidate;
      
        candidate.radius = r;
        candidate.angle = a;

        const double angle = goal_to_robot_heading + a;

        candidate.world_pose.pose.position.x = goal.pose.position.x + r * std::cos(angle);
        candidate.world_pose.pose.position.y = goal.pose.position.y + r * std::sin(angle);

        const double candidate_yaw = std::atan2(
          goal.pose.position.y - candidate.world_pose.pose.position.y,
          goal.pose.position.x - candidate.world_pose.pose.position.x);

        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, candidate_yaw);
        q.normalize();

        candidate.world_pose.pose.orientation = tf2::toMsg(q);
        candidate.world_pose.header = goal.header;
        candidate.world_pose.header.stamp = node->now();

        auto candidate_cost = sampleCost(global_cost_client,candidate.world_pose,true);
        if (!candidate_cost.success || candidate_cost.cost >= 253.0) candidate.is_valid = false;

        candidates.push_back(candidate);
      }
    }

    return candidates;
  }

  geometry_msgs::msg::PoseStamped chooseGoalBySampling(
    const geometry_msgs::msg::PoseStamped & goal_,
    const geometry_msgs::msg::PoseStamped & robot_pose,
    const double min_radius,const double max_radius,const double radius_step,const double angle_range)
  {
    Candidate best_candidate;
    double shortest_distance =std::numeric_limits<double>::infinity();
    bool found = false;

    double robot_distance_to_goal = std::hypot(
      robot_pose.pose.position.x - goal_.pose.position.x,
      robot_pose.pose.position.y - goal_.pose.position.y);
  
    const auto candidates = generateCandidates(robot_pose, goal_,min_radius,max_radius,radius_step,angle_range);

    for (const auto & candidate : candidates) {

      if (!candidate.is_valid) continue;
      if (candidate.radius > robot_distance_to_goal) continue;
    
      if (candidate.radius < shortest_distance || 
        (candidate.radius == shortest_distance && fabs(candidate.angle) < fabs(best_candidate.angle))) {
        shortest_distance = candidate.radius;
        best_candidate = candidate;
        if (!found) found = true;
      }
    }

    if (!found) {
      if (max_radius >= robot_distance_to_goal - 1e-6) {
        RCLCPP_WARN(node->get_logger(),"[DJGoalValidator]: 🥊 No valid replacement goal found between original goal and robot");
        return robot_pose;
      }

      RCLCPP_WARN(node->get_logger(), "[DJGoalValidator]: 🥊 No valid new goal within radius %f! Widening the search", max_radius);
      const double next_max_radius = std::min(max_radius * 2.0, robot_distance_to_goal);

      return chooseGoalBySampling(goal_,robot_pose,max_radius,next_max_radius,radius_step,angle_range);
    }

    return best_candidate.world_pose;
  }
};
}

#include "behaviortree_cpp/bt_factory.h"

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<custom_nav2_plugins::DJGoalValidator>(
    "DJGoalValidator");
}