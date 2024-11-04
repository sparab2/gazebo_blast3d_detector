#include "gazebo_blast3d_visual_detector.h"
#include <ros/ros.h>
#include <algorithm>
#include <numeric>

BlastDetector::BlastDetector() : debounce_duration_(1000) { // Adjust these values based on your camera settings
    event_sub_ = nh_.subscribe("event_topic", 1, &BlastDetector::eventArrayCallback, this);
    x_distribution_.resize(640, 0); // Assuming width is 640
    y_distribution_.resize(480, 0); // Assuming height is 480
}

void BlastDetector::eventArrayCallback(const gazebo_blast3d::EventArrayConstPtr& msg) {
    updateDistributions(msg);
    if (detectBlast(x_distribution_, y_distribution_)) {
        ROS_INFO("Blast detected!");
    }
}

void BlastDetector::updateDistributions(const gazebo_blast3d::EventArrayConstPtr& events) {
    ROS_DEBUG("Updating distributions with %lu events", events->events.size());
    std::fill(x_distribution_.begin(), x_distribution_.end(), 0);
    std::fill(y_distribution_.begin(), y_distribution_.end(), 0);
    
    for (const auto& event : events->events) {
        x_distribution_[event.x]++;
        y_distribution_[event.y]++;
        ROS_DEBUG("Event at x: %d, y: %d", event.x, event.y);
    }
}

bool BlastDetector::detectBlast(const std::vector<int>& x_dist, const std::vector<int>& y_dist) {
    auto now = std::chrono::steady_clock::now();
    long elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_blast_time_).count();

    if (elapsed_time < debounce_duration_) return false;

    if (isRapidIncrease(y_dist, 20) && isExpansion(x_dist, 15)) {
        last_blast_time_ = now;
        return true;
    }
    return false;
}

// Detects a rapid increase along the Y-axis
bool BlastDetector::isRapidIncrease(const std::vector<int>& y_distribution, int threshold_percentage) {
    int sum = std::accumulate(y_distribution.begin(), y_distribution.end(), 0);
    double threshold = sum * threshold_percentage / 100.0;
    for (int count : y_distribution) {
        if (count > threshold) {
            ROS_INFO("Rapid vertical increase detected at count: %d", count);
            return true;
        }
    }
    return false;
}

// Checks for horizontal spread along the X-axis
bool BlastDetector::isExpansion(const std::vector<int>& x_distribution, int threshold_percentage) {
    int count_above_threshold = 0;
    int max_count = *std::max_element(x_distribution.begin(), x_distribution.end());
    double threshold = max_count * threshold_percentage / 100.0;
    for (int count : x_distribution) {
        if (count > threshold) {
            count_above_threshold++;
        }
    }
    // Check if a significant number of points exceed the calculated threshold
    if (count_above_threshold > x_distribution.size() * threshold_percentage / 100) {
        ROS_INFO("Horizontal expansion detected with %d points above threshold", count_above_threshold);
        return true;
    }
    return false;
}


//bool BlastDetector::isRapidIncrease(const std::vector<int>& distribution, int threshold_percentage) {
//    // Detects rapid increase in the distribution
//    int sum = std::accumulate(distribution.begin(), distribution.end(), 0);
//    ROS_DEBUG("Total events: %d", sum);
//    for (int count : distribution) {
//        if (count > sum * threshold_percentage / 100){
//            ROS_INFO("Rapid increase detected at count: %d", count);
//            return true;
//        }
//    }
//    return false;
//}
//
//bool BlastDetector::isExpansion(const std::vector<int>& distribution, int threshold_percentage) {
//    int count_above_threshold = 0;
//    int threshold = *std::max_element(distribution.begin(), distribution.end()) * threshold_percentage / 100;
//    ROS_DEBUG("Total events: %d", threshold);
//    for (int count : distribution) {
//        if (count > threshold){ 
//            count_above_threshold++;
//            ROS_INFO("expansion detected at count: %d", count);
//        }
//    }
//    return count_above_threshold > distribution.size() * threshold_percentage / 100;
//}

int main(int argc, char** argv) {
    ros::init(argc, argv, "blast_detector");
    BlastDetector detector;
    ros::spin();
    return 0;
}

