#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        cv::Mat image = cv_bridge::toCvCopy(msg, "bgr8")->image;
        // Perform your blast detection logic here...

        // If a blast is detected:
        std_msgs::String detection_msg;
        detection_msg.data = "Blast detected!";
        ros::NodeHandle nh;
        ros::Publisher pub = nh.advertise<std_msgs::String>("blast_detection_topic", 1);
        pub.publish(detection_msg);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "blast_detector");
    ros::NodeHandle nh;

    ros::Subscriber sub = nh.subscribe("/camera/image_raw", 1, imageCallback);
    ros::spin();
    return 0;
}

