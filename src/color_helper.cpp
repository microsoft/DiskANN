#include "color_helper.h"
#include "ann_exception.h"
#include <fstream>

namespace diskann
{

void color_helper::write_color_binfile(const std::string& filepath, const std::vector<uint32_t>& location_to_seller, std::uint32_t num_unique_sellers)
{
    // format: num_points, color_size, unique color count, color content
    std::ofstream outfile(filepath, std::ios::binary);
    if (outfile.fail())
    {
        throw diskann::ANNException(std::string("Failed to open file ") + filepath, -1);
    }

    std::uint32_t num_points = static_cast<std::uint32_t>(location_to_seller.size());
    std::uint32_t color_size = static_cast<std::uint32_t>(sizeof(uint32_t));
    outfile.write((char*)(&num_points), sizeof(std::uint32_t));
    outfile.write((char*)(&color_size), sizeof(std::uint32_t));
    outfile.write((char *)(&num_unique_sellers), sizeof(std::uint32_t));
    outfile.write((char*)location_to_seller.data(), location_to_seller.size() * color_size);
    outfile.close();
}

bool color_helper::load_color_binfile(const std::string& filepath, std::vector<uint32_t>& location_to_seller, std::uint32_t& num_unique_sellers)
{
    std::ifstream infile(filepath, std::ios::binary);
    if (infile.fail())
    {
        return false;
    }

    infile.seekg(0, std::ios::end);
    size_t file_size = infile.tellg();

    infile.seekg(0, std::ios::beg);

    std::uint32_t num_points_in_file = 0;
    std::uint32_t color_size = 0;
    infile.read((char*)(&num_points_in_file), sizeof(std::uint32_t));
    infile.read((char*)(&color_size), sizeof(std::uint32_t));
    infile.read((char *)(&num_unique_sellers), sizeof(std::uint32_t));
    if (color_size != sizeof(uint32_t))
    {
        return false;
    }

    size_t color_data_size = num_points_in_file * color_size;
    if (file_size != (sizeof(std::uint32_t) * 3 + color_data_size))
    {
        return false;
    }

    location_to_seller.resize(num_points_in_file);
    infile.read((char *)location_to_seller.data(), color_data_size);
    
    infile.close();

    return true;
}

}