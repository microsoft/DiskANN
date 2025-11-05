#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>

void generate_random_data(const std::string& filename, int num_points, int dim) {
    std::ofstream writer(filename, std::ios::binary);
    if (!writer) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    writer.write(reinterpret_cast<char*>(&num_points), sizeof(int));
    writer.write(reinterpret_cast<char*>(&dim), sizeof(int));

    std::mt19937 rng(42); // Seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> buffer(dim);
    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < dim; ++j) {
            buffer[j] = dist(rng);
        }
        writer.write(reinterpret_cast<char*>(buffer.data()), dim * sizeof(float));
    }

    writer.close();
    std::cout << "Successfully generated " << num_points << " points of dimension " << dim << " to " << filename << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <output_filename> <num_points> <dimensions>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int num_points = std::stoi(argv[2]);
    int dim = std::stoi(argv[3]);

    generate_random_data(filename, num_points, dim);

    return 0;
}
