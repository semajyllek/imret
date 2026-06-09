#include "vault.hpp"
#include <iostream>
#include <algorithm>
#include <faiss/index_io.h>
#include <fstream>
#include <stdexcept>
#include <omp.h>


Vault::Vault(const OrbConfig& conf) : config(conf), extractor(conf) {}

void Vault::add(const cv::Mat& image, const std::string& label) {
    cv::Mat descriptors = extractor.extract(image);
    if (descriptors.empty()) return;

    // Record the label in our hash map
    imret_idx_t current_id = next_id++;
    id_to_label[current_id] = label;

    // Append to our accumulation buffer
    int num_features = descriptors.rows;
    feature_buffer.insert(feature_buffer.end(), 
                          descriptors.data, 
                          descriptors.data + (num_features * 32));
    
    for(int i = 0; i < num_features; i++) {
        id_buffer.push_back(current_id);
    }
}



void Vault::add_batch(const std::vector<cv::Mat>& images,
                      const std::vector<std::string>& labels) {
    int n = static_cast<int>(images.size());
    if (n == 0) return;

    // One FeatureExtractor per thread — cv::ORB is not thread-safe across instances
    int num_threads = omp_get_max_threads();
    std::vector<FeatureExtractor> thread_ex;
    thread_ex.reserve(num_threads);
    for (int t = 0; t < num_threads; t++)
        thread_ex.emplace_back(config);

    std::vector<cv::Mat> results(n);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++)
        results[i] = thread_ex[omp_get_thread_num()].extract(images[i]);

    // Sequential append — no locking needed
    for (int i = 0; i < n; i++) {
        if (results[i].empty()) continue;
        imret_idx_t current_id = next_id++;
        id_to_label[current_id] = labels[i];
        int num_features = results[i].rows;
        feature_buffer.insert(feature_buffer.end(),
                              results[i].data,
                              results[i].data + (num_features * 32));
        for (int j = 0; j < num_features; j++)
            id_buffer.push_back(current_id);
    }
}

void Vault::build() {
    int total_features = static_cast<int>(id_buffer.size());
    if (total_features == 0) return;

    const int d = 256; // 256 bits = 32 bytes (ORB descriptor dimension)

    if (is_built && index) {
        // Vault was loaded from disk: the trained index already exists.
        // Add new vectors into the existing Voronoi cells without retraining k-means.
        index->add_with_ids(total_features, feature_buffer.data(), id_buffer.data());
    } else {
        // Fresh build: train k-means centroids then populate the index.
        int nlist = std::min(4096, std::max(1, total_features / 39));
        quantizer = std::make_unique<faiss::IndexBinaryFlat>(d);
        index = std::make_unique<faiss::IndexBinaryIVF>(quantizer.get(), d, nlist);
        index->train(total_features, feature_buffer.data());
        index->add_with_ids(total_features, feature_buffer.data(), id_buffer.data());
        is_built = true;
    }

    feature_buffer.clear();
    id_buffer.clear();
}



MatchResult Vault::search(const cv::Mat& image) {
    if (!is_built)
        throw std::runtime_error("Must call build() before search().");

    cv::Mat descriptors = extractor.extract(image);
    if (descriptors.empty()) {
        return MatchResult{"Unknown", 0.0f, false};
    }

    int nq = descriptors.rows;
    std::vector<int32_t> distances(nq);
    std::vector<imret_idx_t> labels(nq);

    // --- Tier 1: Fast Search ---
    index->nprobe = config.fast_cells;
    index->search(nq, descriptors.data, 1, distances.data(), labels.data());

    auto tally_votes = [&]() -> std::pair<imret_idx_t, int> {
        std::unordered_map<imret_idx_t, int> tallies;
        int max_votes = 0;
        imret_idx_t best_id = -1;

        for(int i = 0; i < nq; i++) {
            if (distances[i] <= config.max_hamming_distance && labels[i] != -1) {
                int current_votes = ++tallies[labels[i]];
                if (current_votes > max_votes) {
                    max_votes = current_votes;
                    best_id = labels[i];
                }
            }
        }
        return {best_id, max_votes};
    };

    auto [best_id, max_votes] = tally_votes();
    float confidence = (float)max_votes / nq;
    bool fallback = false;

    // --- Tier 2: Deep Fallback ---
    if (confidence < config.confidence_threshold) {
        fallback = true;
        index->nprobe = config.deep_cells;
        index->search(nq, descriptors.data, 1, distances.data(), labels.data());
        
        auto fallback_result = tally_votes();
        best_id = fallback_result.first;
        max_votes = fallback_result.second;
        confidence = (float)max_votes / nq;
    }

    if (best_id == -1) {
        return MatchResult{"Unknown", 0.0f, fallback};
    }

    return MatchResult{id_to_label[best_id], confidence, fallback};
}


Vault::Stats Vault::stats() const {
    return Stats{
        static_cast<int>(id_to_label.size()),
        is_built && index ? static_cast<int64_t>(index->ntotal)
                          : static_cast<int64_t>(id_buffer.size()),
        index ? static_cast<int>(index->nlist) : 0,
        is_built,
    };
}

// --- SAVE ---
void Vault::save(const std::string& prefix) {
    if (!index) {
        throw std::runtime_error("Cannot save an empty or unbuilt vault.");
    }

    // 1. Save the FAISS Index
    faiss::write_index_binary(index.get(), (prefix + ".faiss").c_str());

    // 2. Save the Metadata (Binary Format)
    std::ofstream meta_out(prefix + ".meta", std::ios::binary);
    
    meta_out.write(reinterpret_cast<const char*>(&config), sizeof(OrbConfig));
    meta_out.write(reinterpret_cast<const char*>(&next_id), sizeof(imret_idx_t));

    size_t map_size = id_to_label.size();
    meta_out.write(reinterpret_cast<const char*>(&map_size), sizeof(size_t));

    for (const auto& pair : id_to_label) {
        meta_out.write(reinterpret_cast<const char*>(&pair.first), sizeof(imret_idx_t));
        
        size_t str_len = pair.second.size();
        meta_out.write(reinterpret_cast<const char*>(&str_len), sizeof(size_t));
        meta_out.write(pair.second.c_str(), str_len);
    }
    meta_out.close();
}

// --- LOAD ---
void Vault::load(const std::string& prefix) {
    // 1. Load the FAISS Index
    faiss::IndexBinary* raw_index = faiss::read_index_binary((prefix + ".faiss").c_str());
    index.reset(dynamic_cast<faiss::IndexBinaryIVF*>(raw_index));

    // 2. Load the Metadata
    std::ifstream meta_in(prefix + ".meta", std::ios::binary);
    if (!meta_in) {
        throw std::runtime_error("Could not find metadata file: " + prefix + ".meta");
    }

    meta_in.read(reinterpret_cast<char*>(&config), sizeof(OrbConfig));
    meta_in.read(reinterpret_cast<char*>(&next_id), sizeof(imret_idx_t));

    size_t map_size;
    meta_in.read(reinterpret_cast<char*>(&map_size), sizeof(size_t));

    id_to_label.clear();
    for (size_t i = 0; i < map_size; i++) {
        imret_idx_t key;
        meta_in.read(reinterpret_cast<char*>(&key), sizeof(imret_idx_t));

        size_t str_len;
        meta_in.read(reinterpret_cast<char*>(&str_len), sizeof(size_t));
        
        std::string val(str_len, '\0');
        meta_in.read(&val[0], str_len);

        id_to_label[key] = val;
    }
    meta_in.close();
    is_built = true;
}
