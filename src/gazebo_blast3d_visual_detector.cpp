#include "gazebo_blast3d_visual_detector.h"

BlastDetector::BlastDetector() : image_width_(640), image_height_(480) { // Adjust these values based on your camera settings
    event_sub_ = nh_.subscribe("event_topic", 1, &BlastDetector::eventArrayCallback, this);
    x_distribution_.resize(image_width_, 0);
    y_distribution_.resize(image_height_, 0);
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
    return isRapidIncrease(y_dist, 10) && isExpansion(x_dist, 10); // 10% is a placeholder threshold
}

bool BlastDetector::isRapidIncrease(const std::vector<int>& distribution, int threshold_percentage) {
    // Detects rapid increase in the distribution
    int sum = std::accumulate(distribution.begin(), distribution.end(), 0);
    ROS_DEBUG("Total events: %d", sum);
    for (int count : distribution) {
        if (count > sum * threshold_percentage / 100){
            ROS_INFO("Rapid increase detected at count: %d", count);
            return true;
        }
    }
    return false;
}

bool BlastDetector::isExpansion(const std::vector<int>& distribution, int threshold_percentage) {
    // Detects whether the spread has increased across the distribution
    int count_above_threshold = 0;
    int threshold = *max_element(distribution.begin(), distribution.end()) * threshold_percentage / 100;
    for (int count : distribution) {
        if (count > threshold) count_above_threshold++;
    }
    return count_above_threshold > distribution.size() * threshold_percentage / 100;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "blast_detector");
    BlastDetector detector;
    ros::spin();
    return 0;
}

