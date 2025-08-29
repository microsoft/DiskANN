#pragma once
#include <string>
#include <vector>

namespace diskann
{

class color_helper
{
public:
    void write_color_binfile(const std::string& filepath, const std::vector<uint32_t>& location_to_seller);

    bool load_color_binfile(const std::string &filepath, std::vector<uint32_t> &location_to_seller);
};


}