#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "/home/me468/vslam/ORB_SLAM3/include/System.h"

static const char WINDOW[] = "Image window";
using std::placeholders::_1;

class SlamNode : public rclcpp::Node
{
public:
    SlamNode(ORB_SLAM3::System*  pSLAM) : Node("SlamNode")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(          // CHANGE
            "image_left", 10, std::bind(&SlamNode::topic_callback, this, _1));
        mpSLAM = pSLAM;
        RCLCPP_INFO(this->get_logger(), "SLAM node ready !!!!!!!!!!!!!!!!!!!!!!!!!");
        //this->declare_parameter<std::string>("vocabulary", "p_vocabulary");
        //this->declare_parameter<std::string>("settings", "p_settings");
        //this->declare_parameter<int>("slam_ready",0);
        //init_slam();

    //cv::namedWindow(WINDOW);
    }
    void init_slam() {
        
        //std::cout<< this->get_parameter("vocabulary", path_to_vocabulary)<<std::endl;
        //std::cout<< this->get_parameter("settings", path_to_settings) << std::endl;
        //path_to_vocabulary = "/home/me468/vslam/ORB_SLAM3/Vocabulary/ORBvoc.txt";
        //path_to_settings = "/home/yisen/final/data/EuRoC.yaml";
        //RCLCPP_INFO(this->get_logger(), "vocabulary: %s", path_to_vocabulary.c_str());
        //RCLCPP_INFO(this->get_logger(), "path: %s", path_to_settings.c_str());
        
        //this->declare_parameter<int>("slam_ready", 1);
    }
private:
   std::string path_to_vocabulary, path_to_settings;
   cv_bridge::CvImagePtr cv_ptr;
   cv::Mat frame;
   ORB_SLAM3::System* mpSLAM;
   Sophus::SE3f Tcw;
  void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg)       // CHANGE
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGBA8);
    frame = cv_ptr->image;
    
    std_msgs::msg::Header h = msg->header;
    h.stamp.sec = h.stamp.sec % 100;
    double timestamp = h.stamp.sec+ h.stamp.nanosec * 1e-9;
    //RCLCPP_INFO(this->get_logger(), "----------Image received %lf!---------", timestamp);
    //std::cout <<  << std::endl;
    Tcw=mpSLAM->TrackMonocular(cv_ptr->image,timestamp);
    RCLCPP_INFO(this->get_logger(), "-----pos %f %f %f!---------",Tcw.translation().x(), Tcw.translation().y(), Tcw.translation().z());
  }
  void clock_callback(const sensor_msgs::msg::Image::SharedPtr msg)       // CHANGE
  {
      
  }
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;       // CHANGE
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  std::string path_to_vocabulary = "/home/me468/vslam/ORB_SLAM3/Vocabulary/ORBvoc.txt";
  std::string path_to_settings = "/home/yisen/final/data/mono.yaml";
  ORB_SLAM3::System SLAM(path_to_vocabulary, path_to_settings, ORB_SLAM3::System::MONOCULAR, true);
  rclcpp::spin(std::make_shared<SlamNode>(&SLAM));
  rclcpp::shutdown();
  return 0;
}