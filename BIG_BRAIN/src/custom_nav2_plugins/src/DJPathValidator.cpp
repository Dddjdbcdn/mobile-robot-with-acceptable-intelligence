#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "behaviortree_cpp/action_node.h"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav2_msgs/srv/is_path_valid.hpp"

namespace custom_nav2_plugins
{

class DJPathValidator : public BT::SyncActionNode
{
public:
  DJPathValidator(const std::string & name, const BT::NodeConfig & config)
  : BT::SyncActionNode(name, config)
  {
    node = config.blackboard->get<rclcpp::Node::SharedPtr>("node");
    is_path_valid_client = node->create_client<nav2_msgs::srv::IsPathValid>("is_path_valid");
    accepted_path_pub = node->create_publisher<nav_msgs::msg::Path>("/accepted_path",rclcpp::QoS(10));
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<geometry_msgs::msg::PoseStamped>("goal"),
      BT::InputPort<geometry_msgs::msg::PoseStamped>("robot_pose"),
      BT::InputPort<nav_msgs::msg::Path>("path"),
      BT::InputPort<int>("patience", 2, "Number of failed checks before replacing goal"),
      BT::InputPort<int>("max_corridors", 5, "Replans_per_segment"),
      BT::InputPort<int>("max_paths_per_corridors", 5, "Paths_per_corridor"),
      BT::InputPort<int>("max_replans_per_segment", 8, "Max_replans_per_segment"),
      BT::InputPort<double>("segment_length", 2.0, "Segment length"),
      BT::InputPort<double>("max_deviation", 0.5, "Maximum deviation"),
      BT::InputPort<double>("max_length_ratio", 2.0, "Max_length_ratio"),
      BT::OutputPort<nav_msgs::msg::Path>("accepted_path")
    };
  }

  BT::NodeStatus tick() override
  {
    // ===== Variables init =====
    geometry_msgs::msg::PoseStamped goal, robot_pose;
    nav_msgs::msg::Path path;

    double max_deviation,segment_length,max_length_ratio;
    int patience,max_corridors,max_paths_per_corridors,max_replans_per_segment;

    if (!getInput("goal", goal) || !getInput("robot_pose", robot_pose) || !getInput("path", path)) return BT::NodeStatus::FAILURE;

    getInput("patience", patience);
    getInput("max_corridors", max_corridors);
    getInput("max_paths_per_corridors", max_paths_per_corridors);
    getInput("max_replans_per_segment", max_replans_per_segment);
    getInput("max_deviation", max_deviation);
    getInput("segment_length", segment_length);
    getInput("max_length_ratio", max_length_ratio);

    // ===== Node init =====
    if (!initialized || distance(previous_goal,goal) > 0.5) { // if first ticked or goal changed significantly
      previous_goal = goal;

      const double direct_distance = distance(robot_pose,goal);
      max_replans = std::max(max_replans_per_segment,static_cast<int>(std::ceil(max_replans_per_segment * (direct_distance/segment_length))));

      corridors.clear();

      Corridor new_corridor;

      new_corridor.paths.push_back(path);
      new_corridor.average_path = path;
      new_corridor.observations = 1;
      new_corridor.valid_observations = validate(path).is_valid ? 1 : 0;

      corridors.push_back(new_corridor); // First corridor
      
      total_replans = 0;
      failure_count = 0;
      new_path = false;
      forcing_path = false;
      initialized = true;

      RCLCPP_INFO(node->get_logger(),"[DJPathValidator] 🌸 Initialized. Max replans: %d. Max corridors: %d",max_replans,max_corridors);
    }

    // ===== Forcing path =====
    if (forcing_path) {
      return BT::NodeStatus::SUCCESS;

      // const auto reference_path = RemainingPath(proposed_path, robot_pose);;
      // double reference_length = pathLength(reference_path);
      // double new_length = pathLength(path);
      // double length_ratio = std::max(new_length / reference_length,reference_length / new_length);
      // double deviation = pathDifference(path,reference_path);
      // bool same_corridor = length_ratio < max_length_ratio && deviation < max_deviation;

      // if (same_corridor) { // Only focus on best corridor
      //   publishAcceptedPath(path);
        
      //   if (!validate(path).is_valid) return BT::NodeStatus::FAILURE; // Tick compute_path_to_pose
      //   else return BT::NodeStatus::SUCCESS; // Don't tick compute_path_to_pose
      // }
      // else return BT::NodeStatus::FAILURE; // Tick compute_path_to_pose
    }
    
    path = RemainingPath(path,robot_pose);

    // ===== New path assessment =====
    if (new_path) {
      ++total_replans;
      RCLCPP_INFO(node->get_logger(),"[DJPathValidator] ⚡️ New path number %d was created", total_replans);

      // Assign new path to a corridor
      bool matched = false;

      for (auto & corridor : corridors) {
        const auto reference_path = RemainingPath(corridor.average_path,robot_pose);

        double reference_length = pathLength(reference_path);
        double new_length = pathLength(path);
        double length_ratio = std::max(new_length / reference_length,reference_length / new_length);
        double deviation = pathDifference(path,reference_path);

        bool same_corridor = length_ratio < max_length_ratio && deviation < max_deviation;

        // If belong to an existing corridor, add it
        if (same_corridor) {
          corridor.paths.push_back(path);

          if (corridor.paths.size() > static_cast<size_t>(max_paths_per_corridors)) {
            corridor.paths.erase(corridor.paths.begin()); // Store maximum 5 paths per corridor
          }

          ++corridor.observations;
          if (validate(path).is_valid) ++corridor.valid_observations;
          corridor.average_path = ComputeAveragePath(corridor);

          matched = true;
          break;
        }
      }

      // If unique, create new corridor
      if (!matched) {
        Corridor new_corridor;

        new_corridor.paths.push_back(path);
        new_corridor.average_path = path;
        new_corridor.observations = 1;
        new_corridor.valid_observations = validate(path).is_valid ? 1 : 0;

        corridors.push_back(new_corridor);

        RCLCPP_INFO(node->get_logger(),"[DJPathValidator] 🌸 New corridor number %zu detected", corridors.size());
      }

      new_path = false;

      // If ran out of attempts, force path
      if (corridors.size() >= static_cast<size_t>(max_corridors) || total_replans >= max_replans) {
        int highest_valid_observations = -1;
        double shortest_length = std::numeric_limits<double>::infinity();

        Corridor best_corridor;

        // // Choosing best corridors based on valid observations and observations
        // for (const auto & corridor : corridors) {
        //   bool better_corridor = (corridor.valid_observations > highest_valid_observations ||
        //     (corridor.valid_observations == highest_valid_observations && corridor.observations > best_corridor.observations));

        //   if (better_corridor) {
        //     highest_valid_observations = corridor.valid_observations;
        //     best_corridor = corridor;
        //   }
        // }

        // Choosing best corridors based on corridor length
        for (const auto & corridor : corridors) {
          double corridor_length = pathLength(corridor.average_path);

          bool better_corridor = (corridor_length < shortest_length);
          if (better_corridor) {
            shortest_length = corridor_length;
            best_corridor = corridor;
          }
        }

        forcing_path = true;
        proposed_path = RemainingPath(best_corridor.average_path,robot_pose); // Use average path for corridor reference
        auto proposed_path_validation = validate(proposed_path);

        RCLCPP_WARN(
          node->get_logger(),
          "[DJPathValidator] 🌍 Forcing corridor. Observations: %d, valid observations: %d",
          best_corridor.observations,best_corridor.valid_observations);

        if (proposed_path_validation.is_valid || best_corridor.valid_observations > (best_corridor.observations / 2.0)) {
          publishAcceptedPath(proposed_path); 
        }
        
        else {
          RCLCPP_WARN(node->get_logger(), "[DJPathValidator] 🌍 The proposed path is not valid. Using truncated path.");

          publishAcceptedPath(truncatePath(proposed_path,safeBackupIndex(proposed_path,proposed_path_validation.collision_index)));
        }
        
        return BT::NodeStatus::SUCCESS; // Don't tick compute_path_to_pose
      }
    }

    // ===== Validate input path =====
    auto path_validation = validate(path);

    if (!path_validation.is_valid) {
      ++failure_count;

      RCLCPP_INFO(node->get_logger(),"[DJPathValidator] 🍅 Path invalid at index %d. Failure count: %d",path_validation.collision_index,failure_count);

      // This path is officially invalid
      if (failure_count >= patience) {
        failure_count = 0;
        new_path = true;
        return BT::NodeStatus::FAILURE; // Tick compute_path_to_pose
      }

      publishAcceptedPath(path); // Pass the path to follow_path
      return BT::NodeStatus::SUCCESS; // Don't tick compute_path_to_pose
    }

    // Path is valid
    failure_count = 0;
    new_path = false;

    publishAcceptedPath(path); // Pass the path to follow_path
    return BT::NodeStatus::SUCCESS; // Don't tick compute_path_to_pose
  }

private:
  struct Corridor
  {
    std::vector<nav_msgs::msg::Path> paths;
    nav_msgs::msg::Path average_path;
    int observations{0};
    int valid_observations{0};
  };

  rclcpp::Node::SharedPtr node;
  rclcpp::Client<nav2_msgs::srv::IsPathValid>::SharedPtr is_path_valid_client;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr accepted_path_pub;

  std::vector<Corridor> corridors;
  nav_msgs::msg::Path proposed_path;
  geometry_msgs::msg::PoseStamped previous_goal;

  bool initialized{false};
  bool new_path{true};
  bool forcing_path{false};
  int failure_count{0};
  int max_replans{0};
  int total_replans{0};
  
  struct PathValidation
  {
    bool is_valid{false};
    int collision_index{-1};
  };
  PathValidation validate(const nav_msgs::msg::Path & path)
  {
    PathValidation out;

    auto req = std::make_shared<nav2_msgs::srv::IsPathValid::Request>();
    req->path = path;
    auto future = is_path_valid_client->async_send_request(req);
    if (rclcpp::spin_until_future_complete(node, future, std::chrono::milliseconds(500))!= rclcpp::FutureReturnCode::SUCCESS) return out;
    auto res = future.get();
    if (res->is_valid) {
      out.is_valid = true;
      return out;
    }
    if (!res->invalid_pose_indices.empty()) {
      out.collision_index =*std::min_element(res->invalid_pose_indices.begin(),res->invalid_pose_indices.end());
    }
    return out;
  }

  void publishAcceptedPath(const nav_msgs::msg::Path & path)
  {
    accepted_path_pub->publish(path);
    setOutput("accepted_path", path);
  }

  nav_msgs::msg::Path ComputeAveragePath(
    const Corridor & corridor) const
  {
    if (corridor.paths.empty()) return nav_msgs::msg::Path();
    if (corridor.paths.size() == 1) return corridor.paths.front();

    size_t best_index = 0;
    double best_average_difference = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < corridor.paths.size(); ++i) {
      double total_difference = 0.0;

      for (size_t j = 0; j < corridor.paths.size(); ++j) {
        if (i == j) continue;

        total_difference += pathDifference(corridor.paths[i],corridor.paths[j]);
      }

      double average_difference = total_difference / static_cast<double>(corridor.paths.size() - 1);

      if (average_difference < best_average_difference) {
        best_average_difference = average_difference;
        best_index = i;
      }
    }

    return corridor.paths[best_index];
  }

  int safeBackupIndex(
    const nav_msgs::msg::Path & path,
    int collision_index,
    double safety_margin = 0.25) const
  {
    if (
      path.poses.empty() ||
      collision_index < 0 ||
      static_cast<size_t>(collision_index) >= path.poses.size())
    {
      return -1;
    }

    int safe_index = collision_index;
    double backed_up_distance = 0.0;

    while (safe_index > 0 && backed_up_distance < safety_margin) {
      const auto & p1 = path.poses[safe_index].pose.position;
      const auto & p0 = path.poses[safe_index - 1].pose.position;

      backed_up_distance +=
        std::hypot(p1.x - p0.x,p1.y - p0.y);

      --safe_index;
    }

    return safe_index;
  }

  nav_msgs::msg::Path truncatePath(
  const nav_msgs::msg::Path & path,
  int last_index) const
  {
    nav_msgs::msg::Path truncated;
    truncated.header = path.header;

    if (
      path.poses.empty() ||
      last_index < 0 ||
      static_cast<size_t>(last_index) >= path.poses.size())
    {
      return truncated;
    }

    truncated.poses.insert(
      truncated.poses.end(),
      path.poses.begin(),
      path.poses.begin() + last_index + 1);

    return truncated;
  }

  nav_msgs::msg::Path RemainingPath(
    const nav_msgs::msg::Path & path,
    const geometry_msgs::msg::PoseStamped & robot_pose) const
  {
    nav_msgs::msg::Path remaining;
    remaining.header = path.header;

    if (path.poses.empty()) return remaining;

    size_t nearest_index = 0;
    double nearest_distance = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < path.poses.size(); ++i) {
      double d = distance(robot_pose,path.poses[i]);

      if (d < nearest_distance) {
        nearest_distance = d;
        nearest_index = i;
      }
    }

    remaining.poses.insert(
      remaining.poses.end(),
      path.poses.begin() + nearest_index,
      path.poses.end());

    return remaining;
  }

  double distance(const geometry_msgs::msg::PoseStamped & a,const geometry_msgs::msg::PoseStamped & b) const {
    return std::hypot(a.pose.position.x - b.pose.position.x,a.pose.position.y - b.pose.position.y);
  }

  double pathLength(const nav_msgs::msg::Path & path) const {
    double length = 0.0;
    for (size_t i = 1; i < path.poses.size(); ++i) length += distance(path.poses[i - 1], path.poses[i]);
    return length;
  }

  double oneWayDifference(const nav_msgs::msg::Path & a, const nav_msgs::msg::Path & b) const {
    if (a.poses.empty() || b.poses.empty()) return std::numeric_limits<double>::infinity();

    double total = 0.0;
    for (const auto & pa : a.poses) {
      double nearest = std::numeric_limits<double>::infinity();

      for (const auto & pb : b.poses) {
        nearest = std::min(nearest,distance(pa, pb));
      }
      total += nearest;
    }

    return total / a.poses.size();
  }

  double pathDifference(const nav_msgs::msg::Path & a,const nav_msgs::msg::Path & b) const { 
    return std::max(oneWayDifference(a, b),oneWayDifference(b, a));}
};

}

#include "behaviortree_cpp/bt_factory.h"

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<custom_nav2_plugins::DJPathValidator>(
    "DJPathValidator");
}