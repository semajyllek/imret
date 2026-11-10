#pragma once
#include <opencv2/opencv.hpp>
#include "imret.hpp"

class FeatureExtractor {
private:
    OrbConfig config;

public:
    FeatureExtractor(const OrbConfig& config);
    // safe to call from several threads at once
    cv::Mat extract(const cv::Mat& image) const;
};
