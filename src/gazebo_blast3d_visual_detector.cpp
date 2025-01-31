#include "gazebo_blast3d_visual_detector.h"
#include <ros/ros.h>
#include <algorithm>
#include <numeric>
#include <visualization_msgs/MarkerArray.h>

#include <fstream>

// Declare the ofstream as a member variable in your class
std::ofstream detection_log;

BlastDetector::~BlastDetector() {
    if (detection_log.is_open()) {
        detection_log.close();
    }
}


BlastDetector::BlastDetector() : debounce_duration_(1000) { // Adjust these values based on your camera settings
    event_sub_ = nh_.subscribe("event_topic", 1, &BlastDetector::eventArrayCallback, this);
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("visualization_marker_array", 10);
    x_distribution_.resize(640, 0); // Assuming width is 640
    y_distribution_.resize(480/20, 0); // Assuming height is 480
    
    // Open the file in the constructor
    detection_log.open("/home.md2/sparab2/wind/uncc_wind_control/ros_image/ros_ws/src/gazebo_blast3d/datasets/detection.csv", std::ios::out);
    detection_log << "Time,X,Y,Z,EventID\n"; // Assuming these are the fields you care about
}

void BlastDetector::eventArrayCallback(const gazebo_blast3d::EventArrayConstPtr& msg) {
    std::vector<gazebo_blast3d::Event> recent_events;  // Corrected to use the full namespace

    // Filter events based on time or other criteria
    for (const auto& event : msg->events) {
        recent_events.push_back(event);  // assuming you want to filter or process events here
        detection_log << ros::Time::now() << "," << event.x << "," << event.y << "," << (event.polarity ? "1" : "0") << "\n";
    }

    // Further processing...
    updateDistributions(recent_events);
    if (detectBlast(x_distribution_, y_distribution_)) {
        //ROS_INFO("Blast detected!");
        publishMarkers(msg);
    }
}

void BlastDetector::publishMarkers(const gazebo_blast3d::EventArrayConstPtr& events) {
    visualization_msgs::MarkerArray markers;
    for (const auto& event : events->events) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = ros::Time::now();
        marker.ns = "blast_events";
        marker.id = event.x + event.y * 640;  // Unique ID
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = static_cast<float>(event.x);
        marker.pose.position.y = static_cast<float>(event.y);
        marker.pose.position.z = static_cast<float>(event.y);;  // Assuming a 3D plane
        marker.scale.x = 0.2;  // Size of the marker
        marker.scale.y = 0.2;
        marker.scale.z = 0.2;
        marker.color.a = 1.0;  // Don't forget to set the alpha!
        marker.color.r = event.polarity > 0 ? 1.0 : 0.0;  // Positive events are red
        marker.color.g = 0.0;
        marker.color.b = event.polarity < 0 ? 1.0 : 0.0;  // Negative events are blue
        markers.markers.push_back(marker);
    }
    marker_pub_.publish(markers);
}

void BlastDetector::updateDistributions(const std::vector<gazebo_blast3d::Event>& events) {
    //ROS_DEBUG("Updating distributions with %lu events", events.size());
    std::fill(x_distribution_.begin(), x_distribution_.end(), 0);
    std::fill(y_distribution_.begin(), y_distribution_.end(), 0);
    
    for (const auto& event : events) {
        x_distribution_[event.x]++;
        y_distribution_[event.y]++;
        //ROS_DEBUG("Event at x: %d, y: %d", event.x, event.y);
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
    double threshold = sum * (threshold_percentage / 100.0);
    for (int count : y_distribution) {
        if (count > threshold) {
            //ROS_INFO("Rapid vertical increase detected at count: %d", count);
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
        //ROS_INFO("Horizontal expansion detected with %d points above threshold", count_above_threshold);
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


//FOR FILTERING OF BLASTS
//bool BlastDetector::isBlastEvent(const gazebo_blast3d::Event& event, const std::vector<gazebo_blast3d::Event>& allEvents) {
//    // Define what makes an event part of a blast
//    // Example: An event is considered part of a blast if there are a high number of similar events nearby
//    int count = 0;
//    for (const auto& e : allEvents) {
//        if (std::abs(e.x - event.x) < 5 && std::abs(e.y - event.y) < 5) {  // Check proximity
//            count++;
//        }
//    }
//    return count > 10;  // More than 10 events in proximity could be a blast
//}
//
//bool BlastDetector::analyzeBlastCandidates(const std::vector<gazebo_blast3d::Event>& candidates) {
//    // Further analyze the collected candidates to confirm a blast
//    // Example: Check if the spatial distribution or the intensity of events fits expected patterns
//    int maxX = 0, maxY = 0;
//    for (const auto& event : candidates) {
//        x_distribution_[event.x]++;
//        y_distribution_[event.y]++;
//        maxX = std::max(maxX, event.x);
//        maxY = std::max(maxY, event.y);
//    }
//
//    // Example check: Confirm if the distribution is sufficiently localized or widespread
//    return maxX - std::min_element(x_distribution_.begin(), x_distribution_.end()) < 50 && 
//           maxY - std::min_element(y_distribution_.begin(), y_distribution_.end()) < 50;
//}

int main(int argc, char** argv) {
    ros::init(argc, argv, "blast_detector");
    BlastDetector detector;
    ros::spin();
    return 0;
}

