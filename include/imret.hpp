#pragma once
#include <string>

// The Configuration Object
struct OrbConfig {
    int max_features = 500;
    int fast_cells = 8;
    int deep_cells = 64;
    int max_hamming_distance = 45;
    float confidence_threshold = 0.15f;
    int resize_dim = 0; // 0 = no resize; positive = resize to (resize_dim x resize_dim) before extraction
};

// The Return Object for a Search
struct MatchResult {
    std::string label;
    float confidence;
    bool fallback_used;
};
