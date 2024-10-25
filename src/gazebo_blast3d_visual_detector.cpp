#include "gazebo_blast3d_visual_detector.h"

BlastDetector::BlastDetector() {
    image_sub_ = nh_.subscribe("blast_image", 1, &BlastDetector::imageCallback, this);
}

void BlastDetector::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat& mat = cv_ptr->image;

        // Perform the detection
        if (detectVerticalExpansion(mat)) {
            if (detectHorizontalExpansion(mat)) {
                ROS_INFO("Blast detected!");
            }
        }
    } catch (const cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

bool BlastDetector::detectVerticalExpansion(const cv::Mat& image) {
    // Simplified logic to detect rapid changes in the Y-axis
    cv::Mat gray, edges;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 50, 150);  // Use edge detection as a simplification

    int verticalCount = cv::countNonZero(edges.colRange(0, edges.cols));
    if (verticalCount > (edges.rows * edges.cols * 0.05)) {  // 5% of the image
        return true;
    }
    return false;
}

bool BlastDetector::detectHorizontalExpansion(const cv::Mat& image) {
    // Simplified logic to detect rapid changes in the X-axis
    cv::Mat gray, edges;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 50, 150);  // Use edge detection as a simplification

    int horizontalCount = cv::countNonZero(edges.rowRange(0, edges.rows));
    if (horizontalCount > (edges.rows * edges.cols * 0.05)) {  // 5% of the image
        return true;
    }
    return false;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "blast_detector");
    BlastDetector detector;
    ros::spin();
    return 0;
}

