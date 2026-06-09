#include <iostream>
#include <opencv2/opencv.hpp>
#include "vault.hpp"

int main(int argc, char** argv) {
    // 1. Enforce the CLI rules
    if (argc < 3) {
        std::cerr << "Usage: ./imret_cli <vault_prefix> <path_to_image>" << std::endl;
        return -1;
    }

    std::string vault_path = argv[1];
    std::string image_path = argv[2];

    // 2. Read the image from disk
    cv::Mat query_image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    if (query_image.empty()) {
        std::cerr << "Error: Could not read image at " << image_path << std::endl;
        return -1;
    }

    // 3. Boot the Vault and Load the Brain
    OrbConfig config;
    Vault edge_vault(config);

    try {
        edge_vault.load(vault_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading database: " << e.what() << std::endl;
        return -1;
    }

    // 4. Search and Output
    MatchResult match = edge_vault.search(query_image);

    if (match.confidence > config.confidence_threshold) {
        // Output just the name so other programs can read it easily
        std::cout << match.label << std::endl;
        return 0; // Success
    } else {
        std::cout << "UNKNOWN" << std::endl;
        return 1; // Indicate no match found
    }
}
