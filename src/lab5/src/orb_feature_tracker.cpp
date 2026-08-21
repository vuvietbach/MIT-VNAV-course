// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  16.485 - Fall 2024  - Lab 5 coding assignment
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//
//  In this code, we ask you to implement an ORB feature tracker derived class
//  that inherits from your FeatureTracker base class.
//
// NOTE: Deliverables for the TEAM portion of this assignment start at number 3
// and end at number 7. If you have completed the parts labeled Deliverable 3-7,
// you are done with the TEAM portion of the lab. Deliverables 1-2 are
// individual.
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  DELIVERABLE 6 (continued) | Comparing Feature Matching on Real Data
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//
// For this part, you will need to implement the same functions you've just
// implemented in the case of SIFT and AKAZE, but now for ORB features. You'll
// also implement these functions for the case of BRISK.
// For that case, see fast_feature_tracker.cpp (and its respective header)
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include "orb_feature_tracker.h"

#include <glog/logging.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

using namespace cv;

OrbFeatureTracker::OrbFeatureTracker()
    : FeatureTracker(), detector(ORB::create(500, 1.2f, 1)) {}

void OrbFeatureTracker::detectKeypoints(const cv::Mat& img,
                                        std::vector<KeyPoint>* keypoints) const {
  CHECK_NOTNULL(keypoints);
  // ~~~~ begin solution
  detector->detect(img, *keypoints);
  // ~~~~ end solution
}

void OrbFeatureTracker::describeKeypoints(const cv::Mat& img,
                                          std::vector<KeyPoint>* keypoints,
                                          cv::Mat* descriptors) const {
  CHECK_NOTNULL(keypoints);
  CHECK_NOTNULL(descriptors);
  // ~~~~ begin solution
  detector->compute(img, *keypoints, *descriptors);
  // ~~~~ end solution
}

void OrbFeatureTracker::matchDescriptors(const cv::Mat& descriptors_1,
                                         const cv::Mat& descriptors_2,
                                         std::vector<std::vector<DMatch>>* matches,
                                         std::vector<cv::DMatch>* good_matches) const {
  CHECK_NOTNULL(matches);
  // ~~~~ begin solution
  if (descriptors_1.empty() || descriptors_2.empty()) return;

  // For ORB, we use Brute-Force Hamming matcher as it is binary.
  BFMatcher matcher(NORM_HAMMING);
  matcher.knnMatch(descriptors_1, descriptors_2, *matches, 2);

  const float ratio_thresh = 0.8f;
  for (size_t i = 0; i < matches->size(); i++) {
    if ((*matches)[i].size() >= 2) {
      if ((*matches)[i][0].distance < ratio_thresh * (*matches)[i][1].distance) {
        good_matches->push_back((*matches)[i][0]);
      }
    }
  }
  // ~~~~ end solution
}
