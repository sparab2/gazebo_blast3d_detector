#ifndef BLAST_DETECTOR_H
#define BLAST_DETECTOR_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class BlastDetector {
public:
    BlastDetector();
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);

private:
    ros::NodeHandle nh_;
    ros::Subscriber image_sub_;

    bool detectVerticalExpansion(const cv::Mat& image);
    bool detectHorizontalExpansion(const cv::Mat& image);
};

#endif // BLAST_DETECTOR_H
