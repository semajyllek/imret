#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "vault.hpp"
#include <cstdio>

// --- Synthetic image helpers ---
// Each pattern produces a distinct set of ORB keypoints.

static cv::Mat make_checkerboard(int tile = 10) {
    cv::Mat img(200, 200, CV_8UC1);
    for (int r = 0; r < 200; r++)
        for (int c = 0; c < 200; c++)
            img.at<uint8_t>(r, c) = ((r / tile + c / tile) % 2) * 255;
    return img;
}

static cv::Mat make_grid(int spacing = 20) {
    cv::Mat img = cv::Mat::zeros(200, 200, CV_8UC1);
    for (int i = 0; i < 200; i += spacing) {
        img.row(i).setTo(255);
        img.col(i).setTo(255);
    }
    return img;
}

static cv::Mat make_diagonal_stripes(int width = 10) {
    cv::Mat img(200, 200, CV_8UC1);
    for (int r = 0; r < 200; r++)
        for (int c = 0; c < 200; c++)
            img.at<uint8_t>(r, c) = ((r + c) / width % 2) * 255;
    return img;
}

// --- Single-image tests ---

TEST(Vault, SearchReturnsSameImageLabel) {
    OrbConfig cfg;
    Vault vault(cfg);
    cv::Mat img = make_checkerboard();

    vault.add(img, "checkerboard");
    vault.build();

    EXPECT_EQ(vault.search(img).label, "checkerboard");
}

TEST(Vault, SameImageHasHighConfidence) {
    OrbConfig cfg;
    Vault vault(cfg);
    cv::Mat img = make_checkerboard();

    vault.add(img, "checkerboard");
    vault.build();

    EXPECT_GT(vault.search(img).confidence, 0.5f);
}

TEST(Vault, BlankQueryReturnsUnknown) {
    OrbConfig cfg;
    Vault vault(cfg);
    vault.add(make_checkerboard(), "checkerboard");
    vault.build();

    cv::Mat blank = cv::Mat::zeros(200, 200, CV_8UC1);
    MatchResult result = vault.search(blank);
    EXPECT_EQ(result.label, "Unknown");
    EXPECT_EQ(result.confidence, 0.0f);
}

// --- Multi-image tests ---

TEST(Vault, MultiImageEachLabelsCorrectly) {
    OrbConfig cfg;
    Vault vault(cfg);

    cv::Mat a = make_checkerboard();
    cv::Mat b = make_grid();
    cv::Mat c = make_diagonal_stripes();

    vault.add(a, "checkerboard");
    vault.add(b, "grid");
    vault.add(c, "diagonal");
    vault.build();

    EXPECT_EQ(vault.search(a).label, "checkerboard");
    EXPECT_EQ(vault.search(b).label, "grid");
    EXPECT_EQ(vault.search(c).label, "diagonal");
}

// --- add_batch ---

TEST(Vault, AddBatchLabelsCorrectly) {
    OrbConfig cfg;
    Vault vault(cfg);

    std::vector<cv::Mat> images = {make_checkerboard(), make_grid(), make_diagonal_stripes()};
    std::vector<std::string> labels = {"checkerboard", "grid", "diagonal"};

    vault.add_batch(images, labels);
    vault.build();

    EXPECT_EQ(vault.search(make_checkerboard()).label, "checkerboard");
    EXPECT_EQ(vault.search(make_grid()).label, "grid");
    EXPECT_EQ(vault.search(make_diagonal_stripes()).label, "diagonal");
}

TEST(Vault, AddBatchAndAddMixed) {
    OrbConfig cfg;
    Vault vault(cfg);

    vault.add(make_checkerboard(), "checkerboard");
    vault.add_batch({make_grid(), make_diagonal_stripes()}, {"grid", "diagonal"});
    vault.build();

    EXPECT_EQ(vault.search(make_checkerboard()).label, "checkerboard");
    EXPECT_EQ(vault.search(make_grid()).label, "grid");
}

// --- Incremental add + rebuild ---

TEST(Vault, AddAfterBuildAndRebuildFindsNewLabel) {
    OrbConfig cfg;
    Vault vault(cfg);

    vault.add(make_checkerboard(), "checkerboard");
    vault.build();

    vault.add(make_grid(), "grid");
    vault.build();

    EXPECT_EQ(vault.search(make_checkerboard()).label, "checkerboard");
    EXPECT_EQ(vault.search(make_grid()).label, "grid");
}

// --- Persistence + incremental add ---

TEST(Vault, SaveLoadIncrementalAddPreservesOriginal) {
    const std::string prefix = "/tmp/imret_test_incremental";

    {
        OrbConfig cfg;
        Vault vault(cfg);
        vault.add(make_checkerboard(), "checkerboard");
        vault.build();
        vault.save(prefix);
    }

    {
        OrbConfig cfg;
        Vault vault(cfg);
        vault.load(prefix);
        vault.add(make_grid(), "grid");
        vault.build();

        EXPECT_EQ(vault.search(make_checkerboard()).label, "checkerboard");
        EXPECT_EQ(vault.search(make_grid()).label, "grid");
    }

    std::remove((prefix + ".faiss").c_str());
    std::remove((prefix + ".meta").c_str());
}

TEST(Vault, SaveLoadRoundtrip) {
    const std::string prefix = "/tmp/imret_test_roundtrip";

    {
        OrbConfig cfg;
        Vault vault(cfg);
        vault.add(make_checkerboard(), "checkerboard");
        vault.build();
        vault.save(prefix);
    }

    {
        OrbConfig cfg;
        Vault vault(cfg);
        vault.load(prefix);
        EXPECT_EQ(vault.search(make_checkerboard()).label, "checkerboard");
    }

    std::remove((prefix + ".faiss").c_str());
    std::remove((prefix + ".meta").c_str());
}
