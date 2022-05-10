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
    SlamNode(ORB_SLAM3::System* pSLAM) : Node("SlamNode")
    {
        subscription_left = this->create_subscription<sensor_msgs::msg::Image>(          // CHANGE
            "image_left", 20, std::bind(&SlamNode::camera_left_callback, this, _1));
        subscription_right = this->create_subscription<sensor_msgs::msg::Image>(          // CHANGE
            "image_right", 20, std::bind(&SlamNode::camera_right_callback, this, _1));

        mpSLAM = pSLAM;
        timer_ = create_wall_timer(
            25ms, std::bind(&SlamNode::slam_timer_callback, this));
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
    cv_bridge::CvImagePtr cv_ptr_left,cv_ptr_right;
    int camera_left_ready = 0, camera_right_ready = 0;
    ORB_SLAM3::System* mpSLAM;
    double timestamp1, timestamp2;
    void camera_left_callback(const sensor_msgs::msg::Image::SharedPtr msg)       // CHANGE
    {
        
        cv_ptr_left = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGBA8);
        std_msgs::msg::Header h = msg->header;
        h.stamp.sec = h.stamp.sec;
        timestamp1 = h.stamp.sec + h.stamp.nanosec * 1e-9;
        //RCLCPP_INFO(this->get_logger(), "----------Image left received %lf!---------", timestamp1);
        camera_left_ready = 1;

        
        //
    }
    void camera_right_callback(const sensor_msgs::msg::Image::SharedPtr msg)       // CHANGE
    {
        
        cv_ptr_right = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGBA8);
        std_msgs::msg::Header h = msg->header;
        h.stamp.sec = h.stamp.sec;
        timestamp2 = h.stamp.sec + h.stamp.nanosec * 1e-9;
        //RCLCPP_INFO(this->get_logger(), "----------Image right received %lf!---------", timestamp2);
        camera_right_ready = 1;
        
        //mpSLAM->TrackMonocular(cv_ptr->image, timestamp);
    }
    void slam_timer_callback()
    {
        if (camera_right_ready==1 && camera_left_ready==1)
        {
            camera_left_ready = 0;
            camera_right_ready = 0;
            mpSLAM->TrackStereo(cv_ptr_left->image, cv_ptr_right->image, timestamp1);
            RCLCPP_INFO(this->get_logger(), "two camera ready %f %f",timestamp1,timestamp2);
        }
         
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_left;  
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_right;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    std::string path_to_vocabulary = "/home/me468/vslam/ORB_SLAM3/Vocabulary/ORBvoc.txt";
    std::string path_to_settings = "/home/yisen/final/data/stereo.yaml";
    ORB_SLAM3::System SLAM(path_to_vocabulary, path_to_settings, ORB_SLAM3::System::STEREO, true);
    rclcpp::spin(std::make_shared<SlamNode>(&SLAM));
    SLAM.Shutdown();
    rclcpp::shutdown();
    return 0;
}