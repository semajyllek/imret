#include <iostream>
#include <opencv2/opencv.hpp>
#include "vault.hpp"

int main(int argc, char** argv) {
    // 1. Enforce the CLI rules
    if (argc < 2) {
        std::cerr << "Usage: ./leaf_sorter <path_to_image>" << std::endl;
        return -1;
    }

    // Capture the file path from the terminal command
    std::string image_path = argv[1];

    // 2. Read the image from disk
    // (IMREAD_GRAYSCALE is a nice optimization so ORB doesn't have to convert it later)
    cv::Mat query_image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    
    if (query_image.empty()) {
        std::cerr << "Error: Could not read image at " << image_path << std::endl;
        return -1;
    }

    // 3. Boot the Vault and Load the Brain
    OrbConfig config;
    Vault edge_vault(config);
    
    try {
        edge_vault.load("/home/pi/botany_db");
    } catch (const std::exception& e) {
        std::cerr << "Error loading database: " << e.what() << std::endl;
        return -1;
    }

    // 4. Search and Output
    MatchResult match = edge_vault.search(query_image);

    if (match.confidence > 0.15) {
        // Output just the name so other programs can read it easily
        std::cout << match.label << std::endl;
        return 0; // Success
    } else {
        std::cout << "UNKNOWN" << std::endl;
        return 1; // Indicate no match found
    }
}
