#pragma once
#include <string>

// The Configuration Object
struct OrbConfig {
    int max_features = 500;
    int fast_cells = 8;
    int deep_cells = 64;
    int max_hamming_distance = 45; 
};

// The Return Object for a Search
struct MatchResult {
    std::string label;
    float confidence;
    bool fallback_used;
};
