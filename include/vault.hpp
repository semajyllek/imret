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

    struct Stats {
        int     n_images;
        int64_t n_features;
        int     nlist;
        bool    is_built;
    };

public:
    Vault(const OrbConfig& conf);

    void add(const cv::Mat& image, const std::string& label);
    void add_batch(const std::vector<cv::Mat>& images, const std::vector<std::string>& labels);
    void build();
    // Read-only: safe to call from several threads at once on a built vault
    // (not while add/build/load run).
    MatchResult search(const cv::Mat& image) const;
    Stats stats() const;

    void save(const std::string& prefix);
    // Restores a saved vault. Extraction settings (max_features, resize_dim)
    // always come from the file. Search settings (fast_cells, deep_cells,
    // max_hamming_distance, confidence_threshold) come from the file too,
    // unless keep_search_settings: then this vault's own config's are used.
    void load(const std::string& prefix, bool keep_search_settings = false);

};
