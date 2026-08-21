#include "lk_feature_tracker.h"

#include <glog/logging.h>

#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <filesystem>
#include <fstream>

#include "ring_buffer.hpp"

using namespace cv;

/**
  LK feature tracker Constructor.
*/
LKFeatureTracker::LKFeatureTracker() {
  cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
}

void LKFeatureTracker::printStats() const {
  LOG(INFO) << "Avg. Keypoints 1 Size: " << avg_num_keypoints_img1_;
  LOG(INFO) << "Avg. Keypoints 2 Size: " << avg_num_keypoints_img2_;
  LOG(INFO) << "Avg. Number of matches: " << avg_num_matches_;
  LOG(INFO) << "Avg. Number of good matches: NA";
  LOG(INFO) << "Avg. Number of Inliers: " << avg_num_inliers_;
  LOG(INFO) << "Avg. Inliers ratio: " << avg_inlier_ratio_;
  LOG(INFO) << "Num. of samples: " << num_samples_;

  if (!output_dir_.empty()) {
    std::string log_file_path = getOutputPath("logs.txt");
    std::ofstream log_file(log_file_path, std::ios::out);
    if (log_file.is_open()) {
      log_file << "Avg. Keypoints 1 Size: " << avg_num_keypoints_img1_ << "\n";
      log_file << "Avg. Keypoints 2 Size: " << avg_num_keypoints_img2_ << "\n";
      log_file << "Avg. Number of matches: " << avg_num_matches_ << "\n";
      log_file << "Avg. Number of good matches: NA\n";
      log_file << "Avg. Number of Inliers: " << avg_num_inliers_ << "\n";
      log_file << "Avg. Inliers ratio: " << avg_inlier_ratio_ << "\n";
      log_file << "Num. of samples: " << num_samples_ << "\n";
      log_file.close();
    }
  }
}

void LKFeatureTracker::setOutputDir(const std::string &output_dir) {
  output_dir_ = output_dir;
}

std::string LKFeatureTracker::getOutputPath(const std::string &filename) const {
  if (output_dir_.empty()) {
    return filename;
  }
  std::filesystem::create_directories(output_dir_);
  return (std::filesystem::path(output_dir_) / filename).string();
}

LKFeatureTracker::~LKFeatureTracker() {
  printStats();
  cv::destroyWindow(window_name_);
}

/** TODO This is the main tracking function. It takes in the current frame and
 * detects features that correspond to the previous frame.
 @param[in] frame Current image frame
*/
void LKFeatureTracker::trackFeatures(const cv::Mat& frame) {
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //  DELIVERABLE 7 | Feature Tracking: Lucas-Kanade Tracker
  // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
  //
  // For this part, you will need to:
  //
  //   1. Using OpenCV’s documentation and the C++ API for the LK tracker, track
  //   features for the video sequences we provided you by using the Harris
  //   corner detector. Show the feature tracks at a given frame
  //   extracted when using the Harris corners (consider using the 'show'
  //   function below)
  //
  //   Hint 1: take a look at cv::goodFeaturesToTrack and
  //   cv::calcOpticalFlowPyrLK
  //
  //   2. Add an extra entry in the table you made previously for the Harris +
  //   LK tracker
  //
  //   Note: LKFeatureTracker does not inherit from the base tracker like other
  //   feature trackers, so you need to also implement the statistics gathering
  //   code right here.
  //
  // ~~~~ begin solution
  if (frame.empty()) return;

  cv::Mat curr_frame_gray;
  cv::cvtColor(frame, curr_frame_gray, cv::COLOR_BGR2GRAY);

  if (prev_frame_.empty()) {
    cv::goodFeaturesToTrack(curr_frame_gray, prev_corners_, 500, 0.01, 10, cv::Mat(), 3, true, 0.04);
    curr_frame_gray.copyTo(prev_frame_);
    return;
  }

  std::vector<Point2f> curr_corners;
  std::vector<uchar> status;
  std::vector<float> err;

  cv::calcOpticalFlowPyrLK(prev_frame_, curr_frame_gray, prev_corners_, curr_corners, status, err);

  // Filter matched points
  std::vector<Point2f> matched_prev, matched_curr;
  for (size_t i = 0; i < status.size(); i++) {
    if (status[i]) {
      matched_prev.push_back(prev_corners_[i]);
      matched_curr.push_back(curr_corners[i]);
    }
  }

  // Compute inliers
  std::vector<uchar> inlier_mask;
  unsigned int num_inliers = 0;
  if (matched_prev.size() >= 8) {
    if (inlierMaskComputation(matched_prev, matched_curr, &inlier_mask)) {
      for (uchar v : inlier_mask) {
        if (v) num_inliers++;
      }
    }
  }

  // Statistics
  double new_num_samples = static_cast<double>(num_samples_) + 1.0f;
  double old_num_samples = static_cast<double>(num_samples_);
  avg_num_keypoints_img1_ = (avg_num_keypoints_img1_ * old_num_samples +
                             static_cast<double>(prev_corners_.size())) /
                            new_num_samples;
  avg_num_keypoints_img2_ = (avg_num_keypoints_img2_ * old_num_samples +
                             static_cast<double>(curr_corners.size())) /
                            new_num_samples;
  avg_num_matches_ =
      (avg_num_matches_ * old_num_samples + static_cast<double>(matched_prev.size())) /
      new_num_samples;
  avg_num_inliers_ =
      (avg_num_inliers_ * old_num_samples + static_cast<double>(num_inliers)) /
      new_num_samples;
  if (!matched_prev.empty()) {
    avg_inlier_ratio_ =
        (avg_inlier_ratio_ * old_num_samples +
         (static_cast<double>(num_inliers) / static_cast<double>(matched_prev.size()))) /
        new_num_samples;
  }
  ++num_samples_;

  // Show
  show(frame, matched_prev, matched_curr);

  // Update for next frame
  curr_frame_gray.copyTo(prev_frame_);
  prev_corners_ = matched_curr;

  // Replenish features if needed
  if (prev_corners_.size() < 100) {
    std::vector<Point2f> new_corners;
    cv::goodFeaturesToTrack(curr_frame_gray, new_corners, 500, 0.01, 10, cv::Mat(), 3, true, 0.04);

    prev_corners_ = new_corners;
  }
  // ~~~~ end solution
  // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
  //                             end deliverable 7
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}

bool LKFeatureTracker::inlierMaskComputation(const std::vector<Point2f>& pts1,
                                             const std::vector<Point2f>& pts2,
                                             std::vector<uchar>* inlier_mask) const {
  CHECK_NOTNULL(inlier_mask);

  bool mask_computed = true;  // always optimistic...
  static constexpr double max_dist_from_epi_line_in_px = 3.0;
  static constexpr double confidence_prob = 0.99;
  try {
    findFundamentalMat(pts1,
                       pts2,
                       FM_RANSAC,
                       max_dist_from_epi_line_in_px,
                       confidence_prob,
                       *inlier_mask);
  } catch (...) {
    LOG(WARNING) << "Inlier Mask could not be computed, this can happen if there"
                    "are not enough features tracked.";
    mask_computed = false;
  }
  return mask_computed;
}

/** TODO Display image with tracked features from prev to curr on the image
 * corresponding to 'frame'
 * @param[in] frame The current image frame, to draw the feature track on
 * @param[in] prev The previous set of keypoints
 * @param[in] curr The set of keypoints for the current frame
 */
void LKFeatureTracker::show(const cv::Mat& frame,
                            std::vector<cv::Point2f>& prev,
                            std::vector<cv::Point2f>& curr) {
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // ~~~~ begin solution
  cv::Mat out_img = frame.clone();
  for (size_t i = 0; i < curr.size(); i++) {
    cv::line(out_img, prev[i], curr[i], cv::Scalar(0, 255, 0), 2);
    cv::circle(out_img, curr[i], 3, cv::Scalar(0, 0, 255), -1);
  }
  cv::imshow(window_name_, out_img);
  // ~~~~ end solution
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}
