#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "final_project_msgs/msg/num.hpp"     // CHANGE
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/core.hpp>

#include "/home/me468/vslam/ORB_SLAM3/include/System.h"
using std::placeholders::_1;

class MinimalSubscriber : public rclcpp::Node
{
public:
  MinimalSubscriber()
  : Node("minimal_subscriber")
  {
    subscription_ = this->create_subscription<final_project_msgs::msg::Num>(          // CHANGE
      "topic", 10, std::bind(&MinimalSubscriber::topic_callback, this, _1));
  }

private:
  void topic_callback(const final_project_msgs::msg::Num::SharedPtr msg) const       // CHANGE
  {
    RCLCPP_INFO(this->get_logger(), "I heard: '%d'", msg->num);              // CHANGE
  }
  rclcpp::Subscription<final_project_msgs::msg::Num>::SharedPtr subscription_;       // CHANGE
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());
  rclcpp::shutdown();
  return 0;
}