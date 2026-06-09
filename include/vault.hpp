#pragma once
#include <opencv2/opencv.hpp>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexBinaryFlat.h>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include "imret.hpp"
#include "extractor.hpp"

using imret_idx_t = int64_t;

class Vault {
private:
    OrbConfig config;
    FeatureExtractor extractor;
    
    // FAISS Indices
    std::unique_ptr<faiss::IndexBinaryFlat> quantizer;
    std::unique_ptr<faiss::IndexBinaryIVF> index;
    
    // Memory mapping
    std::unordered_map<imret_idx_t, std::string> id_to_label;
    imret_idx_t next_id = 0;
    
    // Temporary accumulation buffer for training
    std::vector<uint8_t> feature_buffer;
    std::vector<imret_idx_t> id_buffer;

    bool is_built = false;

public:
    Vault(const OrbConfig& conf);
    
    // Step 1: Accumulate images into RAM
    void add(const cv::Mat& image, const std::string& label);

    // Step 1 (parallel): extract features from multiple images concurrently via OpenMP
    void add_batch(const std::vector<cv::Mat>& images, const std::vector<std::string>& labels);

    // Step 2: Train Voronoi cells and push to index
    void build();
    
    // Step 3: Run the tiered query
    MatchResult search(const cv::Mat& image);

    void save(const std::string& prefix);
    void load(const std::string& prefix);

};
