/*
 * @file self_flow.cpp
 * @brief Track features from frame to frame.
 */

#include <cv_bridge/cv_bridge.hpp>
#include <glog/logging.h>
#include <stdio.h>

#include <fstream>
#include <image_transport/image_transport.hpp>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video/tracking.hpp>
#include <rclcpp/rclcpp.hpp>

#include "brisk_feature_tracker.h"
#include "feature_tracker.h"
#include "lk_feature_tracker.h"
#include "orb_feature_tracker.h"
#include "sift_feature_tracker.h"

class SelfFlow : public rclcpp::Node {
 public:
  SelfFlow() : Node("self_flow") {}

  void showFlow(cv::Mat const& frame, cv::Mat const& flow, int spacing = 20) {
    cv::Mat annotatedFrame;
    if (frame.channels() == 1) {
      cv::cvtColor(frame, annotatedFrame, cv::COLOR_GRAY2BGR);
    } else {
      annotatedFrame = frame.clone();
    }

    for (int y = 0; y < flow.rows; y += spacing) {
      for (int x = 0; x < flow.cols; x += spacing) {
        const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
        cv::line(annotatedFrame, cv::Point(x, y),
                 cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),
                 cv::Scalar(0, 255, 0));
        cv::circle(annotatedFrame, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
      }
    }
    cv::imshow("view", annotatedFrame);
  }

  /** imageCallback This function is called when a new image is published.
   * For the first part of the assignment (working with a pair of images), you
   * can ignore this function.
   * @brief Tracks features from frame to frame using images received via a ROS
   *  topic.
   */
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    try {
      // Convert ROS msg type to OpenCV image type.
      cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
      cv::Mat gray;
      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

      static cv::Mat prev_gray;

      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      //  DELIVERABLE 8 | Optical Flow
      // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
      // ~ ~
      //
      // LK tracker estimates the optical flow for sparse points in the image.
      // Alternatively, dense approaches try to estimate the optical flow for
      // the whole image. Try to calculate your own optical flow using
      // Farneback’s algorithm (see OpenCV documentation).
      //
      // ~~~~ begin solution
      if (!prev_gray.empty()) {
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(prev_gray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        showFlow(gray, flow);
        cv::waitKey(10);
      }
      prev_gray = gray;
      // ~~~~ end solution
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(get_logger(),
                   "Could not convert from '%s' to 'bgr8'.",
                   msg->encoding.c_str());
    }
  }

  void run() {
    cv::namedWindow("view", cv::WINDOW_NORMAL);
    image_transport::ImageTransport it(shared_from_this());
    std::map<std::string, std::string> stats;
    image_transport::Subscriber sub =
        it.subscribe("/images_topic",
                     100,
                     std::bind(&SelfFlow::imageCallback, this, std::placeholders::_1));
  }
};

/**
 * @function main
 * @brief Main function
 */
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SelfFlow>();
  node->run();
  rclcpp::spin(node);

  return EXIT_SUCCESS;
}
