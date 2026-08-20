cmake_minimum_required(VERSION 3.8)
project(custom_nav2_plugins)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(behaviortree_cpp REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)


# -----------------------------------------------------------------------------
# BlindBackUp BT plugin
# -----------------------------------------------------------------------------

add_library(blind_backup_plugin SHARED
  src/blind_backup.cpp
)

ament_target_dependencies(blind_backup_plugin
  rclcpp
  behaviortree_cpp
  geometry_msgs
)

target_compile_definitions(
  blind_backup_plugin
  PRIVATE BT_PLUGIN_EXPORT
)


# -----------------------------------------------------------------------------
# DJGoalValidator BT plugin
# -----------------------------------------------------------------------------

add_library(DJGoalValidator_plugin SHARED
  src/DJGoalValidator.cpp
)

ament_target_dependencies(DJGoalValidator_plugin
  rclcpp
  behaviortree_cpp
  geometry_msgs
  nav2_msgs
  tf2
  tf2_geometry_msgs
)

target_compile_definitions(
  DJGoalValidator_plugin
  PRIVATE BT_PLUGIN_EXPORT
)


# -----------------------------------------------------------------------------
# DJPathValidator BT plugin
# -----------------------------------------------------------------------------

add_library(DJPathValidator_plugin SHARED
  src/DJPathValidator.cpp
)

ament_target_dependencies(DJPathValidator_plugin
  rclcpp
  behaviortree_cpp
  geometry_msgs
  nav_msgs
  nav2_msgs
)

target_compile_definitions(
  DJPathValidator_plugin
  PRIVATE BT_PLUGIN_EXPORT
)


# -----------------------------------------------------------------------------
# Install plugin libraries
# -----------------------------------------------------------------------------

install(
  TARGETS
    blind_backup_plugin
    DJGoalValidator_plugin
    DJPathValidator_plugin

  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)


# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()
endif()


ament_package()