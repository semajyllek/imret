#include "extractor.hpp"

FeatureExtractor::FeatureExtractor(const OrbConfig& config) : config(config) {}

cv::Mat FeatureExtractor::extract(const cv::Mat& image) const {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    cv::Mat resized;
    if (config.resize_dim > 0)
        cv::resize(image, resized, cv::Size(config.resize_dim, config.resize_dim));
    const cv::Mat& input = (config.resize_dim > 0) ? resized : image;

    // A detector per call, so concurrent extractions share no OpenCV state
    // (creating one only stores its parameters). WTA_K = 2 keeps descriptors
    // FAISS Hamming compatible.
    cv::Ptr<cv::ORB> orb = cv::ORB::create(config.max_features, 1.2f, 8, 31, 0, 2);
    orb->detectAndCompute(input, cv::noArray(), keypoints, descriptors);

    // Anti-Padding Logic: If the background is blank, return nothing.
    if (descriptors.empty()) {
        return cv::Mat();
    }
    return descriptors;
}
