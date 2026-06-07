#include "extractor.hpp"

FeatureExtractor::FeatureExtractor(const OrbConfig& config) {
    // Initialize ORB with WTA_K = 2 to ensure FAISS Hamming compatibility
    this->orb_detector = cv::ORB::create(
        config.max_features, 1.2f, 8, 31, 0, 2
    );
}

cv::Mat FeatureExtractor::extract(const cv::Mat& image) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb_detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    // Anti-Padding Logic: If the background is blank, return nothing.
    if (descriptors.empty()) {
        return cv::Mat(); 
    }
    return descriptors;
}
