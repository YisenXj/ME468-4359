cmake_minimum_required(VERSION 3.8)
project(final_project)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(custom_msgs REQUIRED)
find_package(final_project_msgs REQUIRED)

find_package(OpenCV 4.4 QUIET)
if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV > 4.4 not found.")
endif()
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

set(ORB_SLAM_DIR "/home/me468/vslam/ORB_SLAM3")
message(STATUS "print_variables------------------------------------------")
message(STATUS "ORB_SLAM_DIR = ${ORB_SLAM_DIR}")

include_directories(
${ORB_SLAM_DIR}
${ORB_SLAM_DIR}/include
${ORB_SLAM_DIR}/include/CameraModels
${ORB_SLAM_DIR}/Thirdparty/Sophus
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

set(LIBS 
${OpenCV_LIBS} 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${ORB_SLAM_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${ORB_SLAM_DIR}/Thirdparty/g2o/lib/libg2o.so
${ORB_SLAM_DIR}/lib/libORB_SLAM3.so
-lboost_system
-lcrypto
)

ament_python_install_package(${PROJECT_NAME})

#include_directories(include ${rclcpp_INCLUDE_DIRS})
#link_directories(${rclcpp_LIBRARY_DIRS})

add_executable(talker src/publisher_member_function.cpp)
ament_target_dependencies(talker rclcpp final_project_msgs)         # CHANGE

add_executable(listener src/subscriber_member_function.cpp)
target_link_libraries(listener
${LIBS}
)

ament_target_dependencies(listener rclcpp final_project_msgs)     # CHANGE

install(TARGETS
  talker
  listener
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  scripts/control_open_loop.py
  scripts/control_path_follower.py
  scripts/localizer.py
  scripts/object_detection_from_file.py
  scripts/object_recognition.py
  scripts/path_planner.py
  scripts/path_planner_from_file.py
  scripts/simulation.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(ament_cmake)
ament_export_dependencies(ament_cmake_python)
ament_export_dependencies(rclcpp)
ament_export_dependencies(rclpy)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
