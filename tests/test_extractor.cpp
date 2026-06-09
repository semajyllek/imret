#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "extractor.hpp"

static OrbConfig default_config;

// Checkerboard with 10px tiles — reliable source of ORB corners
static cv::Mat make_checkerboard() {
    cv::Mat img(200, 200, CV_8UC1);
    for (int r = 0; r < 200; r++)
        for (int c = 0; c < 200; c++)
            img.at<uint8_t>(r, c) = ((r / 10 + c / 10) % 2) * 255;
    return img;
}

TEST(FeatureExtractor, BlankImageReturnsEmpty) {
    FeatureExtractor ex(default_config);
    cv::Mat blank = cv::Mat::zeros(200, 200, CV_8UC1);
    EXPECT_TRUE(ex.extract(blank).empty());
}

TEST(FeatureExtractor, TexturedImageReturnsDescriptors) {
    FeatureExtractor ex(default_config);
    EXPECT_FALSE(ex.extract(make_checkerboard()).empty());
}

TEST(FeatureExtractor, DescriptorWidthIs32Bytes) {
    FeatureExtractor ex(default_config);
    cv::Mat desc = ex.extract(make_checkerboard());
    ASSERT_FALSE(desc.empty());
    // ORB with WTA_K=2 → 256-bit = 32-byte descriptors
    EXPECT_EQ(desc.cols, 32);
    EXPECT_EQ(desc.type(), CV_8UC1);
}

TEST(FeatureExtractor, MaxFeaturesIsRespected) {
    OrbConfig cfg;
    cfg.max_features = 50;
    FeatureExtractor ex(cfg);
    cv::Mat desc = ex.extract(make_checkerboard());
    EXPECT_LE(desc.rows, 50);
}
