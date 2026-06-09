#pragma once
#include <opencv2/opencv.hpp>
#include "imret.hpp"

class FeatureExtractor {
private:
    OrbConfig config;
    cv::Ptr<cv::ORB> orb_detector;

public:
    FeatureExtractor(const OrbConfig& config);
    cv::Mat extract(const cv::Mat& image);
};
