#include <memory>
#include <string>

#include "behaviortree_cpp/bt_factory.h"
#include "behaviortree_cpp/json_export.h"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_behavior_tree/bt_action_node.hpp"
#include "nav2_msgs/action/compute_path_to_pose.hpp"
#include "nav_msgs/msg/path.hpp"

namespace custom_nav2_plugins
{

class ComputeFollowPathToPose
  : public nav2_behavior_tree::BtActionNode<nav2_msgs::action::ComputePathToPose>
{
  using Action = nav2_msgs::action::ComputePathToPose;
  using ActionResult = Action::Result;

public:
  ComputeFollowPathToPose(
    const std::string & name,
    const BT::NodeConfiguration & config)
  : BtActionNode<Action>(name, "compute_path_to_pose", config)
  {
  }

  static BT::PortsList providedPorts()
  {
    BT::RegisterJsonDefinition<nav_msgs::msg::Path>();
    BT::RegisterJsonDefinition<geometry_msgs::msg::PoseStamped>();

    return providedBasicPorts(
      {
        BT::InputPort<geometry_msgs::msg::PoseStamped>("goal", "Destination to plan to"),
        BT::InputPort<geometry_msgs::msg::PoseStamped>(
          "start",
          "Optional planner start pose when use_start is true"),
        BT::InputPort<bool>("use_start", "Use the supplied start pose"),
        BT::InputPort<std::string>("planner_id", "", "Planner plugin to use"),
        BT::OutputPort<nav_msgs::msg::Path>("path", "Last successfully computed path"),
        BT::OutputPort<ActionResult::_error_code_type>(
          "error_code_id", "ComputePathToPose result error code"),
      });
  }

  void on_tick() override
  {
    getInput("goal", goal_.goal);
    getInput("planner_id", goal_.planner_id);

    goal_.use_start = false;
    if (getInput("use_start", goal_.use_start)) {
      if (goal_.use_start && !getInput("start", goal_.start)) {
        goal_.use_start = false;
        RCLCPP_ERROR(
          node_->get_logger(),
          "use_start is true but no start pose was supplied; using the robot pose");
      }
    } else if (getInput("start", goal_.start)) {
      goal_.use_start = true;
    }
  }

  BT::NodeStatus on_success() override
  {
    setOutput("path", result_.result->path);
    setOutput("error_code_id", ActionResult::NONE);
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus on_aborted() override
  {
    // Preserve the last successful path while the planner recovery retries.
    setOutput("error_code_id", result_.result->error_code);
    return BT::NodeStatus::FAILURE;
  }

  BT::NodeStatus on_cancelled() override
  {
    // Halting a replan must not erase the path currently used by FollowPath.
    setOutput("error_code_id", ActionResult::NONE);
    return BT::NodeStatus::SUCCESS;
  }
};

}  // namespace custom_nav2_plugins

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<custom_nav2_plugins::ComputeFollowPathToPose>(
    "ComputeFollowPathToPose");
}
