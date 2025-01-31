#ifndef BLAST_DETECTOR_H
#define BLAST_DETECTOR_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include "gazebo_blast3d/EventArray.h"
#include <visualization_msgs/MarkerArray.h>
#include <numeric>
#include <vector>
#include <chrono>
#include <fstream>

class BlastDetector {
public:
    BlastDetector();
    ~BlastDetector();  // Destructor declaration
    void eventArrayCallback(const gazebo_blast3d::EventArrayConstPtr& msg);

private:
    ros::NodeHandle nh_;
    ros::Subscriber event_sub_;
    ros::Publisher marker_pub_;

    std::vector<int> x_distribution_;
    std::vector<int> y_distribution_;
//    int image_width_;
//    int image_height_;

    void publishMarkers(const gazebo_blast3d::EventArrayConstPtr& events);
    bool detectBlast(const std::vector<int>& x_dist, const std::vector<int>& y_dist);
    void updateDistributions(const std::vector<gazebo_blast3d::Event>& events);
    bool isRapidIncrease(const std::vector<int>& distribution, int threshold_percentage);
    bool isExpansion(const std::vector<int>& distribution, int threshold_percentage);
    
    std::chrono::steady_clock::time_point last_blast_time_;
    long debounce_duration_; //milliseconds
    std::ofstream detection_log;
};

#endif // BLAST_DETECTOR_H
